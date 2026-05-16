from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ModelAlias, ProviderConfig

LOGGER = logging.getLogger("llama_bridge.providers")


@dataclass(slots=True)
class ResolvedModel:
    alias: str
    upstream_model: str
    provider: ProviderConfig


class OpenAICompatibleProvider:
    _TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
    _DEFAULT_MAX_PARALLEL_MODEL_REQUESTS = 10

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._use_fallback = False
        parallel_limit = self._configured_parallel_limit()
        self._request_semaphore = asyncio.Semaphore(parallel_limit)
        # Scale connection pool with parallel limit
        max_connections = max(parallel_limit * 4, 100)
        max_keepalive = max(parallel_limit * 2, 20)
        self._client = httpx.AsyncClient(
            timeout=config.timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self._get_api_key()
        if api_key and not api_key.startswith("${"):
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.config.headers)
        return headers

    def _payload(self, payload: dict[str, Any], stream: bool) -> dict[str, Any]:
        request = {**payload, **self.config.extra_body, "stream": stream}
        if not self.config.supports_tools:
            request.pop("tools", None)
            request.pop("tool_choice", None)
        return request

    def _configured_parallel_limit(self) -> int:
        raw = os.environ.get("LLAMA_MAX_PARALLEL_MODEL_REQUESTS")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return self._DEFAULT_MAX_PARALLEL_MODEL_REQUESTS

    @asynccontextmanager
    async def _provider_request_slot(self):
        await self._request_semaphore.acquire()
        try:
            yield
        finally:
            self._request_semaphore.release()

    def _get_base_url(self) -> str | None:
        if self._use_fallback and self.config.fallback_url:
            return self.config.fallback_url
        return self.config.base_url

    def _get_api_key(self) -> str | None:
        if self._use_fallback and self.config.fallback_api_key:
            return self.config.fallback_api_key
        return self.config.api_key

    def _try_fallback(self) -> bool:
        if not self._use_fallback and self.config.fallback_url:
            self._use_fallback = True
            return True
        return False

    def _chat_completions_url(self) -> str:
        base = self._get_base_url()
        return f"{base}/chat/completions"

    def _completions_url(self) -> str:
        base = self._get_base_url()
        return f"{base}/completions"

    def _embeddings_url(self) -> str:
        base = self._get_base_url()
        return f"{base}/embeddings"

    def _retry_delay(self, attempt: int) -> float:
        base_delay = 0.25 * (2 ** attempt)
        jitter = base_delay * 0.3 * random.random()
        return base_delay + jitter

    def _should_retry_status(self, exc: httpx.HTTPStatusError) -> bool:
        return exc.response.status_code in self._TRANSIENT_STATUS_CODES

    async def _post(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        last_exc: Exception | None = None
        tried_fallback = False
        async with self._provider_request_slot():
            for attempt in range(3):
                try:
                    response = await self._client.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                    )
                    response.raise_for_status()
                    return response
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    # Try fallback on 429 (rate limit) if not already tried
                    if exc.response.status_code == 429 and not tried_fallback:
                        if self._try_fallback():
                            tried_fallback = True
                            # Rebuild URL with fallback base
                            if "/chat/completions" in url:
                                url = self._chat_completions_url()
                            elif "/completions" in url:
                                url = self._completions_url()
                            elif "/embeddings" in url:
                                url = self._embeddings_url()
                            continue
                    if not self._should_retry_status(exc) or attempt == 2:
                        raise
                except httpx.RequestError as exc:
                    last_exc = exc
                    if attempt == 2:
                        raise
                await asyncio.sleep(self._retry_delay(attempt))
        assert last_exc is not None
        raise last_exc

    async def _stream(self, url: str, payload: dict[str, Any]) -> AsyncIterator[str]:
        """
        Safe async generator. Acquires the semaphore with raw acquire/release
        (never @asynccontextmanager). All HTTP work is delegated to a background
        task. GeneratorExit cleanly cancels the task without any re-entrant
        athrow() conflict.
        """
        await self._request_semaphore.acquire()
        queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue(maxsize=128)
        task = asyncio.create_task(self._do_stream(url, payload, queue))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        except GeneratorExit:
            LOGGER.debug("Stream abandoned by caller; cancelling background fetch")
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise
        except BaseException:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise
        finally:
            self._request_semaphore.release()
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _do_stream(
        self, url: str, payload: dict[str, Any], queue: asyncio.Queue[str | BaseException | None]
    ) -> None:
        """
        Plain coroutine — NOT an async generator.
        Performs the HTTP stream with retry logic and puts each line into queue.
        Puts None as a success sentinel or the exception object on failure.
        Never touches the semaphore.
        """
        last_exc: Exception | None = None
        tried_fallback = False
        for attempt in range(3):
            try:
                async with self._client.stream(
                    "POST", url, headers=self._headers(), json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        await queue.put(line)
                await queue.put(None)
                return
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                # Try fallback on 429 (rate limit) if not already tried
                if exc.response.status_code == 429 and not tried_fallback:
                    if self._try_fallback():
                        tried_fallback = True
                        # Rebuild URL with fallback base
                        if "/messages" in url:
                            url = self._messages_url()
                        elif "/chat/completions" in url:
                            url = self._chat_completions_url()
                        elif "/completions" in url:
                            url = self._completions_url()
                        continue
                if not self._should_retry_status(exc) or attempt == 2:
                    await queue.put(exc)
                    return
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt == 2:
                    await queue.put(exc)
                    return
            except asyncio.CancelledError:
                return
            await asyncio.sleep(self._retry_delay(attempt))
        if last_exc is not None:
            await queue.put(last_exc)

    async def create_chat_completion(
        self, payload: dict[str, Any], stream: bool = False
    ) -> httpx.Response:
        return await self._post(
            self._chat_completions_url(),
            self._payload(payload, stream),
        )

    async def create_embedding(self, payload: dict[str, Any]) -> httpx.Response:
        return await self._post(self._embeddings_url(), {**payload, **self.config.extra_body})

    async def stream_chat_completion(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        async for line in self._stream(
            self._chat_completions_url(),
            self._payload(payload, True),
        ):
            yield line


class OllamaCloudProvider(OpenAICompatibleProvider):
    def _openai_base_url(self) -> str:
        base = self._get_base_url()
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def _chat_completions_url(self) -> str:
        return f"{self._openai_base_url()}/chat/completions"

    def _completions_url(self) -> str:
        return f"{self._openai_base_url()}/completions"

    def _embeddings_url(self) -> str:
        return f"{self._openai_base_url()}/embeddings"

    def _messages_url(self) -> str:
        base = self._get_base_url()
        return f"{base}/v1/messages"

    def _anthropic_payload(self, body: dict[str, Any], model: str) -> dict[str, Any]:
        return {**body, **self.config.extra_body, "model": model}

    async def create_anthropic_message(
        self, body: dict[str, Any], model: str
    ) -> httpx.Response:
        return await self._post(
            self._messages_url(),
            self._anthropic_payload(body, model),
        )

    async def stream_anthropic_message(
        self, body: dict[str, Any], model: str
    ) -> AsyncIterator[str]:
        async for line in self._stream(
            self._messages_url(),
            self._anthropic_payload({**body, "stream": True}, model),
        ):
            if not line:
                continue
            yield f"{line}\n\n"


class AnthropicCompatibleProvider(OpenAICompatibleProvider):
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        api_key = self._get_api_key()
        if api_key and not api_key.startswith("${"):
            headers["x-api-key"] = api_key
        headers.update(self.config.headers)
        return headers

    def _anthropic_base_url(self) -> str:
        base = self._get_base_url()
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def _messages_url(self) -> str:
        return f"{self._anthropic_base_url()}/messages"

    def _anthropic_payload(self, body: dict[str, Any], model: str) -> dict[str, Any]:
        return {**body, **self.config.extra_body, "model": model}

    async def create_anthropic_message(
        self, body: dict[str, Any], model: str
    ) -> httpx.Response:
        return await self._post(
            self._messages_url(),
            self._anthropic_payload(body, model),
        )

    async def stream_anthropic_message(
        self, body: dict[str, Any], model: str
    ) -> AsyncIterator[str]:
        async for line in self._stream(
            self._messages_url(),
            self._anthropic_payload({**body, "stream": True}, model),
        ):
            if not line:
                continue
            yield f"{line}\n\n"


def build_provider(config: ProviderConfig) -> OpenAICompatibleProvider:
    if config.type not in {
        "openai_compatible",
        "nvidia_nim",
        "ollama",
        "ollama_local",
        "ollama_cloud",
        "lm_studio",
        "groq",
        "gemini",
        "openai",
        "cohere",
        "mistral",
        "deepseek",
        "openrouter",
        "sarvamai",
        "kilo",
        "opencode",
        "cline",
    }:
        raise ValueError(f"Unsupported provider type: {config.type}")
    if config.type == "opencode":
        return AnthropicCompatibleProvider(config)
    if config.type == "ollama_cloud":
        return OllamaCloudProvider(config)
    return OpenAICompatibleProvider(config)


def resolve_model(
    requested_model: str, providers: dict[str, ProviderConfig], aliases: dict[str, ModelAlias]
) -> ResolvedModel:
    alias = aliases.get(requested_model)
    if alias is None:
        alias = _resolve_claude_family_alias(requested_model, aliases)
    if alias is None:
        alias = next(
            (entry for entry in aliases.values() if entry.model == requested_model),
            None,
        )
    if alias is not None:
        upstream_model = alias.model or providers[alias.provider].default_model
        if not upstream_model:
            raise KeyError(
                f"Model alias '{requested_model}' has no model and provider "
                f"'{alias.provider}' has no default_model configured"
            )
        return ResolvedModel(
            alias=alias.alias,
            upstream_model=upstream_model,
            provider=providers[alias.provider],
        )

    # Passthrough fallback: treat requested_model as a direct upstream model name
    # Check if any provider has this as default_model
    for provider in providers.values():
        if provider.default_model == requested_model:
            return ResolvedModel(
                alias=requested_model,
                upstream_model=requested_model,
                provider=provider,
            )

    # Last resort: use the first configured provider
    fallback_provider = next(iter(providers.values()), None)
    if fallback_provider:
        import logging
        logging.warning(
            f"Model '{requested_model}' not found in aliases; forwarding directly to provider '{fallback_provider.name}'"
        )
        return ResolvedModel(
            alias=requested_model,
            upstream_model=requested_model,
            provider=fallback_provider,
        )

    available = ", ".join(sorted(aliases))
    raise KeyError(f"Unknown model '{requested_model}'. Available aliases: {available}")


def _resolve_claude_family_alias(
    requested_model: str, aliases: dict[str, ModelAlias]
) -> ModelAlias | None:
    requested = requested_model.lower()
    for family in ("haiku", "sonnet", "opus"):
        if family in requested and family in aliases:
            return aliases[family]
    if "claude" in requested and "sonnet" in aliases:
        return aliases["sonnet"]
    return None
