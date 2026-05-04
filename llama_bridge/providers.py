from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ModelAlias, ProviderConfig


@dataclass(slots=True)
class ResolvedModel:
    alias: str
    upstream_model: str
    provider: ProviderConfig


class OpenAICompatibleProvider:
    _TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
    _MAX_RETRIES = 5  # Increased from 3
    _INITIAL_RETRY_DELAY = 0.5  # seconds
    _MAX_RETRY_DELAY = 30  # seconds

    def __init__(self, config: ProviderConfig):
        self.config = config
        # Increased timeout and connection limits for better reliability
        self._client = httpx.AsyncClient(
            timeout=max(config.timeout, 60),  # Ensure minimum 60s timeout
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and not self.config.api_key.startswith("${"):
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.headers)
        return headers

    def _payload(self, payload: dict[str, Any], stream: bool) -> dict[str, Any]:
        request = {**payload, **self.config.extra_body, "stream": stream}
        if not self.config.supports_tools:
            request.pop("tools", None)
            request.pop("tool_choice", None)
        return request

    def _chat_completions_url(self) -> str:
        return f"{self.config.base_url}/chat/completions"

    def _completions_url(self) -> str:
        return f"{self.config.base_url}/completions"

    def _embeddings_url(self) -> str:
        return f"{self.config.base_url}/embeddings"

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter and cap"""
        delay = min(self._INITIAL_RETRY_DELAY * (2 ** attempt), self._MAX_RETRY_DELAY)
        return delay

    def _should_retry_status(self, exc: httpx.HTTPStatusError) -> bool:
        return exc.response.status_code in self._TRANSIENT_STATUS_CODES

    async def _post(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
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
                if not self._should_retry_status(exc) or attempt >= self._MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(self._retry_delay(attempt))
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt >= self._MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(self._retry_delay(attempt))
            await asyncio.sleep(self._retry_delay(attempt))
        assert last_exc is not None
        raise last_exc

    async def _stream(self, url: str, payload: dict[str, Any]) -> AsyncIterator[str]:
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            yielded = False
            try:
                async with self._client.stream(
                    "POST",
                    url,
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        yielded = True
                        yield line
                    return
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if not self._should_retry_status(exc) or attempt >= self._MAX_RETRIES - 1:
                    raise
                if not yielded:
                    await asyncio.sleep(self._retry_delay(attempt))
                else:
                    raise
            except httpx.RequestError as exc:
                last_exc = exc
                if yielded or attempt >= self._MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(self._retry_delay(attempt))
        assert last_exc is not None
        raise last_exc

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
        if self.config.base_url.endswith("/v1"):
            return self.config.base_url
        return f"{self.config.base_url}/v1"

    def _chat_completions_url(self) -> str:
        return f"{self._openai_base_url()}/chat/completions"

    def _completions_url(self) -> str:
        return f"{self._openai_base_url()}/completions"

    def _embeddings_url(self) -> str:
        return f"{self._openai_base_url()}/embeddings"

    def _messages_url(self) -> str:
        return f"{self.config.base_url}/v1/messages"

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
    }:
        raise ValueError(f"Unsupported provider type: {config.type}")
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
    if alias is None:
        available = ", ".join(sorted(aliases))
        raise KeyError(f"Unknown model '{requested_model}'. Available aliases: {available}")
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
