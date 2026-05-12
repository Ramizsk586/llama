from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator, AsyncIterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .anthropic import (
    anthropic_request_to_openai,
    anthropic_stream_prefix,
    estimate_input_tokens,
    openai_response_to_anthropic,
    sse_event,
)
from .config import (
    BridgeConfig,
    DEFAULT_LOG_PATH,
    DEFAULT_PID_PATH,
    load_config,
    resolve_codex_model,
    resolve_pi_model,
    resolve_vs_copilot_model,
)
from .providers import (
    OpenAICompatibleProvider,
    ProviderConfig,
    ResolvedModel,
    build_provider,
    resolve_model,
)
from .tools import ToolRegistry, select_relevant_tools
from .tool_management import ToolManager


DEV_LOG_ENABLED = os.environ.get("LLAMA_DEV_LOG") == "1"
LOGGER = logging.getLogger("uvicorn.error.llama_bridge.server")


def create_app(
    config_path: Path | None = None,
    idle_timeout_seconds: int = 0,
    idle_after_file: Path | None = None,
    config: BridgeConfig | None = None,
    include_tools: bool = True,
) -> FastAPI:
    if config is None:
        config = load_config(config_path)
    app = FastAPI(title="llama bridge", version="0.1.0")
    app.state.bridge_config = config
    app.state.providers = {
        name: build_provider(provider_config)
        for name, provider_config in config.providers.items()
    }
    app.state.tools = ToolRegistry(config) if include_tools else None
    app.state.tool_manager = (
        ToolManager(app.state.tools, config)
        if include_tools and app.state.tools is not None
        else None
    )
    app.state.anthropic_batches = {}
    app.state.anthropic_files = {}
    app.state.anthropic_skills = {}
    app.state.assistants = {}
    app.state.threads = {}
    app.state.fine_tuning_jobs = {}
    app.state.last_request_at = time.monotonic()
    app.state.telegram_task = None

    @app.middleware("http")
    async def track_activity(request: Request, call_next):
        app.state.last_request_at = time.monotonic()
        return await call_next(request)

    if idle_timeout_seconds > 0:
        @app.on_event("startup")
        async def start_idle_shutdown_watcher() -> None:
            asyncio.create_task(
                _shutdown_after_idle(app, idle_timeout_seconds, idle_after_file)
            )

    if include_tools and config.telegram.enabled and config.telegram.bot_token and not config.telegram.bot_token.startswith("${"):
        @app.on_event("startup")
        async def start_telegram_bot() -> None:
            app.state.telegram_task = asyncio.create_task(_run_embedded_telegram_bot(app))

    @app.on_event("shutdown")
    async def close_providers() -> None:
        telegram_task = getattr(app.state, "telegram_task", None)
        if telegram_task is not None:
            telegram_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await telegram_task
        for provider in app.state.providers.values():
            await provider.aclose()
        if app.state.tools is not None:
            await app.state.tools.aclose()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(config, x_api_key, authorization)
        model_ids = _available_model_ids(config)
        return {
            "data": [
                {
                    "id": model_id,
                    "type": "model",
                    "display_name": model_id,
                }
                for model_id in model_ids
            ]
        }

    @app.get("/v1/models/{model_id:path}")
    async def retrieve_model(
        model_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(config, x_api_key, authorization)
        try:
            resolved = _resolve_bridge_model(model_id, config)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": resolved.provider.name,
        }

    @app.get("/v1/v0/agents")
    @app.get("/v0/agents")
    async def poolside_list_agents(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(config, x_api_key, authorization)
        agents = _poolside_agent_records(config)
        return {
            "agents": agents,
            "data": agents,
        }

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        body = await request.json()

        try:
            _check_openai_compat_auth(config, body, x_api_key, authorization)
        except HTTPException:
            _write_dev_log(
                config,
                "openai_chat_auth_error",
                {
                    "requested_model": body.get("model"),
                    "has_x_api_key": bool(x_api_key),
                    "has_authorization": bool(authorization),
                    "body": body,
                },
            )
            raise

        try:
            resolved = _resolve_bridge_model(body["model"], config)
        except KeyError as exc:
            _write_dev_log(
                config,
                "openai_chat_model_error",
                {"requested_model": body.get("model"), "body": body, "message": str(exc)},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = _with_bridge_tools(
            {**body, "model": resolved.upstream_model},
            app.state.tools,
            config,
            tool_manager=app.state.tool_manager,
            provider_config=resolved.provider,
        )
        provider = _provider_for(app, resolved)
        _write_dev_log(
            config,
            "openai_chat_request",
            {
                "requested_model": body.get("model"),
                "upstream_model": resolved.upstream_model,
                "provider": resolved.provider.name,
                "body": body,
            },
        )
        if bool(body.get("stream")):
            if _should_buffer_streaming_tool_request(body, payload):
                try:
                    data = await _chat_completion_with_bridge_tools(app, provider, payload, config)
                except httpx.HTTPStatusError as exc:
                    _write_dev_log(
                        config,
                        "openai_chat_response_error",
                        {
                            "status_code": exc.response.status_code,
                            "body": _safe_response_text(exc.response),
                        },
                    )
                    return _upstream_error(exc.response)
                except httpx.RequestError as exc:
                    _write_dev_log(config, "openai_chat_response_error", {"message": str(exc)})
                    return _request_error(exc)
                return StreamingResponse(
                    _safe_stream(_stream_buffered_openai_completion(data)),
                    media_type="text/event-stream",
                )
            _log_streaming_tool_policy(config, "openai_chat", payload)
            return StreamingResponse(
                _safe_stream(_stream_openai_response(provider, payload, config)),
                media_type="text/event-stream",
            )

        try:
            data = await _chat_completion_with_bridge_tools(app, provider, payload, config)
        except httpx.HTTPStatusError as exc:
            _write_dev_log(
                config,
                "openai_chat_response_error",
                {
                    "status_code": exc.response.status_code,
                    "body": _safe_response_text(exc.response),
                },
            )
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            _write_dev_log(config, "openai_chat_response_error", {"message": str(exc)})
            return _request_error(exc)
        _write_dev_log(config, "openai_chat_response", data)
        return JSONResponse(data)

    @app.post("/v1/completions")
    async def create_completion(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()

        try:
            resolved = _resolve_bridge_model(body["model"], config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = _completion_request_to_chat_completion(body, resolved.upstream_model)
        payload = _with_bridge_tools(
            payload,
            app.state.tools,
            config,
            tool_manager=app.state.tool_manager,
            provider_config=resolved.provider,
        )
        provider = _provider_for(app, resolved)
        if bool(body.get("stream")):
            _log_streaming_tool_policy(config, "openai_completion", payload)
            return StreamingResponse(
                _safe_stream(_stream_completion_response(provider, payload, body, config)),
                media_type="text/event-stream",
            )

        try:
            data = await _chat_completion_with_bridge_tools(app, provider, payload, config)
        except httpx.HTTPStatusError as exc:
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            return _request_error(exc)
        return JSONResponse(_chat_completion_to_completion_response(data, body))

    @app.post("/v1/embeddings")
    async def create_embedding(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        try:
            resolved = _resolve_bridge_model(body["model"], config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = {**body, "model": resolved.upstream_model}
        provider = _provider_for(app, resolved)
        try:
            response = await provider.create_embedding(payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            return _request_error(exc)
        data = response.json()
        if "data" in data:
            return JSONResponse(data)
        return JSONResponse(_ollama_embed_to_openai_embedding(data, body))

    @app.post("/v1/moderations")
    async def create_moderation(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        return _openai_moderation_response(body)

    @app.api_route("/v1/assistants", methods=["GET", "POST"])
    async def openai_assistants(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        if request.method == "GET":
            return _openai_list_response(list(app.state.assistants.values()))
        body = await request.json()
        assistant = _openai_assistant_record(body)
        app.state.assistants[assistant["id"]] = assistant
        return JSONResponse(assistant)

    @app.api_route("/v1/assistants/{assistant_id}", methods=["GET", "POST", "DELETE"])
    async def openai_assistant(
        assistant_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        assistant = app.state.assistants.get(assistant_id)
        if assistant is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        if request.method == "DELETE":
            app.state.assistants.pop(assistant_id, None)
            return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}
        if request.method == "POST":
            body = await request.json()
            assistant.update({key: value for key, value in body.items() if key != "id"})
        return assistant

    @app.api_route("/v1/threads", methods=["GET", "POST"])
    async def openai_threads(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        if request.method == "GET":
            return _openai_list_response(list(app.state.threads.values()))
        body = await _optional_json(request)
        thread = _openai_thread_record(body)
        app.state.threads[thread["id"]] = thread
        return JSONResponse(thread)

    @app.api_route("/v1/threads/{thread_id}", methods=["GET", "POST", "DELETE"])
    async def openai_thread(
        thread_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        thread = app.state.threads.get(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        if request.method == "DELETE":
            app.state.threads.pop(thread_id, None)
            return {"id": thread_id, "object": "thread.deleted", "deleted": True}
        if request.method == "POST":
            body = await request.json()
            thread["metadata"] = body.get("metadata", thread.get("metadata", {}))
        return thread

    @app.api_route("/v1/threads/{thread_id}/messages", methods=["GET", "POST"])
    async def openai_thread_messages(
        thread_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        thread = app.state.threads.get(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        if request.method == "GET":
            return _openai_list_response(thread["messages"])
        body = await request.json()
        message = _openai_thread_message_record(thread_id, body)
        thread["messages"].append(message)
        return JSONResponse(message)

    @app.get("/v1/threads/{thread_id}/messages/{message_id}")
    async def openai_thread_message(
        thread_id: str,
        message_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        thread = app.state.threads.get(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        for message in thread["messages"]:
            if message["id"] == message_id:
                return message
        raise HTTPException(status_code=404, detail="Message not found")

    @app.api_route("/v1/threads/{thread_id}/runs", methods=["GET", "POST"])
    async def openai_thread_runs(
        thread_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        thread = app.state.threads.get(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        if request.method == "GET":
            return _openai_list_response(thread["runs"])
        body = await request.json()
        run = _openai_thread_run_record(thread_id, body)
        thread["runs"].append(run)
        return JSONResponse(run)

    @app.api_route(
        "/v1/threads/{thread_id}/runs/{run_id}",
        methods=["GET", "POST"],
    )
    async def openai_thread_run(
        thread_id: str,
        run_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        run = _find_thread_run(app.state.threads, thread_id, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if request.method == "POST":
            body = await request.json()
            run["metadata"] = body.get("metadata", run.get("metadata", {}))
        return run

    @app.post("/v1/threads/{thread_id}/runs/{run_id}/cancel")
    async def openai_cancel_thread_run(
        thread_id: str,
        run_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        run = _find_thread_run(app.state.threads, thread_id, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        run["status"] = "cancelled"
        run["cancelled_at"] = int(time.time())
        return run

    @app.post("/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs")
    async def openai_submit_tool_outputs(
        thread_id: str,
        run_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        run = _find_thread_run(app.state.threads, thread_id, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        body = await request.json()
        run["status"] = "completed"
        run["completed_at"] = int(time.time())
        run["tool_outputs"] = body.get("tool_outputs", [])
        return run

    @app.get("/v1/threads/{thread_id}/runs/{run_id}/steps")
    async def openai_thread_run_steps(
        thread_id: str,
        run_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        run = _find_thread_run(app.state.threads, thread_id, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _openai_list_response([])

    @app.api_route("/v1/fine_tuning/jobs", methods=["GET", "POST"])
    async def openai_fine_tuning_jobs(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        if request.method == "GET":
            return _openai_list_response(list(app.state.fine_tuning_jobs.values()))
        body = await request.json()
        job = _openai_fine_tuning_job_record(body)
        app.state.fine_tuning_jobs[job["id"]] = job
        return JSONResponse(job)

    @app.get("/v1/fine_tuning/jobs/{job_id}")
    async def openai_fine_tuning_job(
        job_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        job = app.state.fine_tuning_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Fine-tuning job not found")
        return job

    @app.post("/v1/fine_tuning/jobs/{job_id}/cancel")
    async def openai_cancel_fine_tuning_job(
        job_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        job = app.state.fine_tuning_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Fine-tuning job not found")
        job["status"] = "cancelled"
        return job

    @app.get("/v1/fine_tuning/jobs/{job_id}/events")
    async def openai_fine_tuning_events(
        job_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        if job_id not in app.state.fine_tuning_jobs:
            raise HTTPException(status_code=404, detail="Fine-tuning job not found")
        return _openai_list_response([])

    @app.post("/v1/images/generations")
    async def openai_image_generation(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        await request.json()
        return _unsupported_endpoint("Image generation is not implemented by llama bridge.")

    @app.post("/v1/audio/transcriptions")
    async def openai_audio_transcription(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        await request.body()
        return _unsupported_endpoint("Audio transcription is not implemented by llama bridge.")

    @app.post("/v1/audio/speech")
    async def openai_audio_speech(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        await request.body()
        return _unsupported_endpoint("Text-to-speech is not implemented by llama bridge.")

    @app.post("/v1/chat")
    async def cohere_chat(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        try:
            resolved = _resolve_bridge_model(body["model"], config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = _with_bridge_tools(
            _cohere_chat_request_to_chat_completion(body, resolved.upstream_model),
            app.state.tools,
            config,
            tool_manager=app.state.tool_manager,
            provider_config=resolved.provider,
        )
        provider = _provider_for(app, resolved)
        if bool(body.get("stream")):
            _log_streaming_tool_policy(config, "cohere_chat", payload)
            return StreamingResponse(
                _safe_stream(_stream_cohere_chat_response(provider, payload, body, config)),
                media_type="text/event-stream",
            )
        try:
            data = await _chat_completion_with_bridge_tools(app, provider, payload, config)
        except httpx.HTTPStatusError as exc:
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            return _request_error(exc)
        return JSONResponse(_chat_completion_to_cohere_chat_response(data, body))

    @app.post("/v1/embed")
    async def cohere_embed(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        try:
            resolved = _resolve_bridge_model(body["model"], config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = {
            "model": resolved.upstream_model,
            "input": body.get("texts", body.get("input", [])),
        }
        if "dimensions" in body:
            payload["dimensions"] = body["dimensions"]
        provider = _provider_for(app, resolved)
        try:
            response = await provider.create_embedding(payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            return _request_error(exc)
        return JSONResponse(_openai_embedding_to_cohere_embed(response.json(), body))

    @app.post("/v1beta/models/{model_id:path}:generateContent")
    @app.post("/v1/models/{model_id:path}:generateContent")
    async def gemini_generate_content(
        model_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        return await _gemini_generate_content(app, config, model_id, body, stream=False)

    @app.post("/v1beta/models/{model_id:path}:streamGenerateContent")
    @app.post("/v1/models/{model_id:path}:streamGenerateContent")
    async def gemini_stream_generate_content(
        model_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        return await _gemini_generate_content(app, config, model_id, body, stream=True)

    @app.post("/v1/responses")
    async def create_response(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()

        try:
            resolved = _resolve_bridge_model(body["model"], config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = _with_bridge_tools(
            _responses_request_to_chat_completion(body, resolved.upstream_model),
            app.state.tools,
            config,
            tool_manager=app.state.tool_manager,
            provider_config=resolved.provider,
        )
        provider = _provider_for(app, resolved)
        _write_dev_log(
            config,
            "codex_responses_request",
            {
                "requested_model": body.get("model"),
                "upstream_model": resolved.upstream_model,
                "provider": resolved.provider.name,
                "body": body,
            },
        )
        if bool(body.get("stream")):
            _log_streaming_tool_policy(config, "responses", payload)
            return StreamingResponse(
                _safe_stream(_stream_responses_response(provider, payload, body, config)),
                media_type="text/event-stream",
            )

        try:
            data = await _chat_completion_with_bridge_tools(app, provider, payload, config)
        except httpx.HTTPStatusError as exc:
            _write_dev_log(
                config,
                "codex_responses_error",
                {
                    "status_code": exc.response.status_code,
                    "body": _safe_response_text(exc.response),
                },
            )
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            _write_dev_log(config, "codex_responses_error", {"message": str(exc)})
            return _request_error(exc)
        _write_dev_log(config, "codex_responses_response", data)
        return JSONResponse(_chat_completion_to_responses_response(data, body))

    @app.post("/v1/complete")
    async def create_legacy_completion(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        messages_body = _anthropic_complete_to_messages_request(body)
        try:
            resolved = resolve_model(
                messages_body["model"], config.providers, config.anthropic_models
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        provider = _provider_for(app, resolved)
        payload = anthropic_request_to_openai(messages_body, resolved.upstream_model)
        if bool(body.get("stream")):
            return StreamingResponse(
                _safe_stream(_stream_anthropic_complete_response(provider, payload, body)),
                media_type="text/event-stream",
            )
        try:
            response = await provider.create_chat_completion(payload, stream=False)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            return _request_error(exc)
        data = response.json()
        return JSONResponse(_chat_completion_to_anthropic_complete_response(data, body))

    @app.post("/v1/messages/batches")
    async def create_message_batch(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        batch = _anthropic_batch_record(body)
        app.state.anthropic_batches[batch["id"]] = {
            "batch": batch,
            "requests": body.get("requests") or [],
        }
        return batch

    @app.get("/v1/messages/batches")
    async def list_message_batches(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(config, x_api_key, authorization)
        batches = [
            entry["batch"]
            for entry in app.state.anthropic_batches.values()
        ]
        return _anthropic_list_response(batches)

    @app.get("/v1/messages/batches/{batch_id}")
    async def retrieve_message_batch(
        batch_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(config, x_api_key, authorization)
        entry = app.state.anthropic_batches.get(batch_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Message batch not found")
        return entry["batch"]

    @app.post("/v1/messages/batches/{batch_id}/cancel")
    async def cancel_message_batch(
        batch_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(config, x_api_key, authorization)
        entry = app.state.anthropic_batches.get(batch_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Message batch not found")
        entry["batch"]["processing_status"] = "canceling"
        entry["batch"]["cancel_initiated_at"] = _anthropic_timestamp()
        return entry["batch"]

    @app.get("/v1/messages/batches/{batch_id}/results")
    async def message_batch_results(
        batch_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        _check_auth(config, x_api_key, authorization)
        entry = app.state.anthropic_batches.get(batch_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Message batch not found")
        return StreamingResponse(
            _safe_stream(_anthropic_batch_results(entry["requests"])),
            media_type="application/x-jsonlines",
        )

    @app.api_route("/v1/files", methods=["GET", "POST"])
    async def anthropic_files(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        if request.method == "GET":
            return _anthropic_list_response(
                [_anthropic_public_record(item) for item in app.state.anthropic_files.values()]
            )
        body = await request.body()
        record = _anthropic_file_record(request.headers.get("content-type"), body)
        app.state.anthropic_files[record["id"]] = record
        return JSONResponse(_anthropic_public_record(record))

    @app.api_route("/v1/files/{file_id}", methods=["GET", "DELETE"])
    async def anthropic_file(
        file_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        record = app.state.anthropic_files.get(file_id)
        if record is None:
            raise HTTPException(status_code=404, detail="File not found")
        if request.method == "DELETE":
            app.state.anthropic_files.pop(file_id, None)
            return {"id": file_id, "type": "file_deleted", "deleted": True}
        return _anthropic_public_record(record)

    @app.get("/v1/files/{file_id}/content")
    async def anthropic_file_content(
        file_id: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        _check_auth(config, x_api_key, authorization)
        record = app.state.anthropic_files.get(file_id)
        if record is None:
            raise HTTPException(status_code=404, detail="File not found")
        return Response(
            content=record.get("_content", b""),
            media_type=record.get("mime_type") or "application/octet-stream",
        )

    @app.api_route("/v1/skills", methods=["GET", "POST"])
    async def anthropic_skills(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        if request.method == "GET":
            return _anthropic_list_response(list(app.state.anthropic_skills.values()))
        body = await request.json()
        record = _anthropic_skill_record(body)
        app.state.anthropic_skills[record["id"]] = record
        return JSONResponse(record)

    @app.api_route("/v1/skills/{skill_id}", methods=["GET", "POST", "DELETE"])
    async def anthropic_skill(
        skill_id: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        record = app.state.anthropic_skills.get(skill_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Skill not found")
        if request.method == "DELETE":
            app.state.anthropic_skills.pop(skill_id, None)
            return {"id": skill_id, "type": "skill_deleted", "deleted": True}
        if request.method == "POST":
            body = await request.json()
            record.update({key: value for key, value in body.items() if key != "id"})
            record["updated_at"] = _anthropic_timestamp()
        return record

    @app.api_route(
        "/v1/organizations/{admin_path:path}",
        methods=["GET", "POST", "DELETE"],
    )
    async def anthropic_admin(
        admin_path: str,
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await _optional_json(request)
        return _anthropic_admin_response(request.method, admin_path, body)

    @app.get("/v1/api/tags")
    @app.get("/api/tags")
    async def ollama_tags(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_ollama_auth(config, x_api_key, authorization)
        return {
            "models": [
                _ollama_model_record(model)
                for model in config.vs_copilot_models
            ]
        }

    @app.get("/v1/api/ps")
    @app.get("/api/ps")
    async def ollama_ps(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_ollama_auth(config, x_api_key, authorization)
        return {"models": []}

    @app.post("/v1/api/show")
    @app.post("/api/show")
    async def ollama_show(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        model = body.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="model is required")
        try:
            _resolve_bridge_model(model, config)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        context_size = _vs_copilot_context_size(config, model)
        return {
            "modelfile": f"FROM {model}",
            "parameters": f"num_ctx {context_size}",
            "license": "",
            "template": "",
            "details": _ollama_model_details(model),
            "model_info": _ollama_model_info(model, context_size),
            "capabilities": ["completion", "tools"],
            "modified_at": _ollama_timestamp(),
        }

    @app.post("/v1/api/create")
    @app.post("/api/create")
    async def ollama_create(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        return _ollama_management_response(body, "create complete")

    @app.post("/v1/api/pull")
    @app.post("/api/pull")
    async def ollama_pull(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        return _ollama_management_response(body, "pull complete")

    @app.post("/v1/api/push")
    @app.post("/api/push")
    async def ollama_push(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        return _ollama_management_response(body, "push complete")

    @app.post("/v1/api/copy")
    @app.post("/api/copy")
    async def ollama_copy(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        _check_ollama_auth(config, x_api_key, authorization)
        await request.json()
        return {"status": "success"}

    @app.delete("/v1/api/delete")
    @app.delete("/api/delete")
    async def ollama_delete(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        _check_ollama_auth(config, x_api_key, authorization)
        await request.json()
        return {"status": "success"}

    @app.get("/v1/api/version")
    @app.get("/api/version")
    async def ollama_version(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        _check_ollama_auth(config, x_api_key, authorization)
        return {"version": "0.18.3"}

    @app.head("/v1/api/blobs/{digest:path}")
    @app.head("/api/blobs/{digest:path}")
    async def ollama_blob_exists(
        digest: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        _check_ollama_auth(config, x_api_key, authorization)
        return Response(status_code=404)

    @app.post("/v1/api/blobs/{digest:path}")
    @app.post("/api/blobs/{digest:path}")
    async def ollama_blob_create(
        digest: str,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        _check_ollama_auth(config, x_api_key, authorization)
        return Response(status_code=201)

    @app.post("/v1/api/chat")
    @app.post("/api/chat")
    async def ollama_chat(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        try:
            resolved = _resolve_bridge_model(body["model"], config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = _with_bridge_tools(
            _ollama_chat_request_to_chat_completion(body, resolved.upstream_model),
            app.state.tools,
            config,
            tool_manager=app.state.tool_manager,
            provider_config=resolved.provider,
        )
        provider = _provider_for(app, resolved)
        if bool(body.get("stream", True)):
            _log_streaming_tool_policy(config, "ollama_chat", payload)
            return StreamingResponse(
                _safe_stream(_stream_ollama_chat_response(provider, payload, body, config)),
                media_type="application/x-ndjson",
            )
        try:
            data = await _chat_completion_with_bridge_tools(app, provider, payload, config)
        except httpx.HTTPStatusError as exc:
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            return _request_error(exc)
        return JSONResponse(
            _chat_completion_to_ollama_chat_response(data, body)
        )

    @app.post("/v1/api/generate")
    @app.post("/api/generate")
    async def ollama_generate(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        try:
            resolved = _resolve_bridge_model(body["model"], config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = _ollama_generate_request_to_chat_completion(body, resolved.upstream_model)
        provider = _provider_for(app, resolved)
        if bool(body.get("stream", True)):
            return StreamingResponse(
                _safe_stream(_stream_ollama_generate_response(provider, payload, body, config)),
                media_type="application/x-ndjson",
            )
        try:
            response = await provider.create_chat_completion(payload, stream=False)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            return _request_error(exc)
        return JSONResponse(
            _chat_completion_to_ollama_generate_response(response.json(), body)
        )

    @app.post("/v1/api/embed")
    @app.post("/api/embed")
    async def ollama_embed(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        return await _ollama_embedding_request(app, config, body, legacy=False)

    @app.post("/v1/api/embeddings")
    @app.post("/api/embeddings")
    async def ollama_embeddings(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        return await _ollama_embedding_request(app, config, body, legacy=True)

    @app.post("/v1/api/web_search")
    @app.post("/api/web_search")
    @app.post("/v1/api/experimental/web_search")
    @app.post("/api/experimental/web_search")
    async def ollama_web_search(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        return await _ollama_web_request(app, config, "web_search", body)

    @app.post("/v1/api/web_fetch")
    @app.post("/api/web_fetch")
    @app.post("/v1/api/experimental/web_fetch")
    @app.post("/api/experimental/web_fetch")
    async def ollama_web_fetch(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_ollama_auth(config, x_api_key, authorization)
        body = await request.json()
        return await _ollama_web_request(app, config, "web_fetch", body)

    if include_tools:

        @app.get("/v1/tools")
        @app.get("/api/tools")
        async def list_bridge_tools(
            request: Request,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ):
            _check_tools_auth(config, x_api_key, authorization)
            if not config.tools.expose_http:
                raise HTTPException(status_code=404, detail="Tool endpoints are disabled")
            tool_manager = app.state.tool_manager
            full_schema = request.query_params.get("full_schema") == "true"
            management_tools = tool_manager.management_openai_tools() if tool_manager else []
            bridge_tools = app.state.tools.openai_tools()
            if full_schema:
                return _bridge_tools_response_from_list([*bridge_tools, *management_tools])
            return _bridge_tools_response_from_list([*bridge_tools, *management_tools])

        @app.post("/v1/tools/call")
        @app.post("/api/tools/call")
        async def call_bridge_tool(
            request: Request,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ):
            _check_tools_auth(config, x_api_key, authorization)
            if not config.tools.expose_http:
                raise HTTPException(status_code=404, detail="Tool endpoints are disabled")
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="tool call body must be an object")
            name = body.get("name") or body.get("tool")
            arguments = _bridge_tool_arguments(body)
            if not isinstance(name, str) or not name:
                raise HTTPException(status_code=400, detail="name is required")
            if not isinstance(arguments, dict):
                raise HTTPException(status_code=400, detail="arguments must be an object")
            return await _call_bridge_tool(app, name, arguments)

        @app.post("/v1/tools/{tool_name}")
        @app.post("/api/tools/{tool_name}")
        async def call_named_bridge_tool(
            tool_name: str,
            request: Request,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ):
            _check_tools_auth(config, x_api_key, authorization)
            if not config.tools.expose_http:
                raise HTTPException(status_code=404, detail="Tool endpoints are disabled")
            body = await _optional_json(request)
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="tool arguments must be an object")
            return await _call_bridge_tool(app, tool_name, _bridge_tool_arguments(body))

        @app.get("/v1/tools/compact")
        @app.get("/api/tools/compact")
        async def compact_tool_catalog(
            request: Request,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
            query: str = "",
            limit: int = 20,
        ):
            _check_tools_auth(config, x_api_key, authorization)
            if not config.tools.expose_http:
                raise HTTPException(status_code=404, detail="Tool endpoints are disabled")
            tool_manager = app.state.tool_manager
            compact = tool_manager.compact_manifest(query or "test", "openai")
            return {"ok": True, "data": compact[:limit], "total": len(compact)}

        @app.get("/v1/tools/{tool_name}/schema")
        @app.get("/api/tools/{tool_name}/schema")
        async def get_tool_schema(
            tool_name: str,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
            fmt: str = "openai",
        ):
            _check_tools_auth(config, x_api_key, authorization)
            if not config.tools.expose_http:
                raise HTTPException(status_code=404, detail="Tool endpoints are disabled")
            tool_manager = app.state.tool_manager
            result = await tool_manager.call_management_tool(
                "tool_schema_get", {"name": tool_name, "format": fmt}, "openai", None
            )
            return result

        @app.get("/v1/tools/{tool_name}/help")
        @app.get("/api/tools/{tool_name}/help")
        async def get_tool_help(
            tool_name: str,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ):
            _check_tools_auth(config, x_api_key, authorization)
            if not config.tools.expose_http:
                raise HTTPException(status_code=404, detail="Tool endpoints are disabled")
            tool_manager = app.state.tool_manager
            result = await tool_manager.call_management_tool(
                "tool_usage_help", {"name": tool_name}, "openai", None
            )
            return result

        @app.post("/v1/tools/search")
        @app.post("/api/tools/search")
        async def search_tools(
            request: Request,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ):
            _check_tools_auth(config, x_api_key, authorization)
            if not config.tools.expose_http:
                raise HTTPException(status_code=404, detail="Tool endpoints are disabled")
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="Request body must be an object")
            tool_manager = app.state.tool_manager
            result = await tool_manager.call_management_tool(
                "tool_catalog_search", body, "openai", None
            )
            return result

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, int]:
        _check_auth(config, x_api_key, authorization)
        body = await request.json()
        return {"input_tokens": estimate_input_tokens(body)}

    @app.post("/v1/messages")
    async def create_message(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(config, x_api_key, authorization)
        body = await request.json()

        try:
            resolved = resolve_model(
                body["model"], config.providers, config.anthropic_models
            )
        except KeyError as exc:
            _write_dev_log(
                config,
                "anthropic_messages_model_error",
                {"requested_model": body.get("model"), "body": body, "message": str(exc)},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        provider = _provider_for(app, resolved)
        stream = bool(body.get("stream"))
        _write_dev_log(
            config,
            "anthropic_messages_request",
            {
                "requested_model": body.get("model"),
                "upstream_model": resolved.upstream_model,
                "provider": resolved.provider.name,
                "stream": stream,
                "body": body,
            },
        )
        if hasattr(provider, "create_anthropic_message") and (
            stream or resolved.provider.type == "opencode"
        ):
            if stream:
                _write_dev_log(
                    config,
                    "streaming_tools_bypass",
                    {
                        "route": "anthropic_messages_native",
                        "reason": "native Anthropic streaming is proxied without bridge tool execution",
                    },
                )
                return StreamingResponse(
                    _safe_stream(_stream_anthropic_response(provider, body, resolved.upstream_model)),
                    media_type="text/event-stream",
                )
            try:
                response = await provider.create_anthropic_message(body, resolved.upstream_model)
            except httpx.HTTPStatusError as exc:
                _write_dev_log(
                    config,
                    "anthropic_messages_error",
                    {
                        "status_code": exc.response.status_code,
                        "body": _safe_response_text(exc.response),
                    },
                )
                return _upstream_error(exc.response)
            except httpx.RequestError as exc:
                _write_dev_log(config, "anthropic_messages_error", {"message": str(exc)})
                return _request_error(exc)
            return JSONResponse(response.json())

        payload = _with_bridge_tools(
            anthropic_request_to_openai(body, resolved.upstream_model),
            app.state.tools,
            config,
            tool_manager=app.state.tool_manager,
            provider_config=resolved.provider,
        )

        if stream:
            _log_streaming_tool_policy(config, "anthropic_messages", payload)
            return StreamingResponse(
                _safe_stream(_stream_response(provider, payload, body["model"], config)),
                media_type="text/event-stream",
            )

        try:
            data = await _chat_completion_with_bridge_tools(app, provider, payload, config)
        except httpx.HTTPStatusError as exc:
            _write_dev_log(
                config,
                "anthropic_messages_error",
                {
                    "status_code": exc.response.status_code,
                    "body": _safe_response_text(exc.response),
                },
            )
            return _upstream_error(exc.response)
        except httpx.RequestError as exc:
            _write_dev_log(config, "anthropic_messages_error", {"message": str(exc)})
            return _request_error(exc)
        _write_dev_log(config, "anthropic_messages_response", data)
        return JSONResponse(openai_response_to_anthropic(data, resolved.alias, body["model"]))

    return app


class _SafeAsyncStream:
    def __init__(self, source: AsyncIterable[str]):
        self._source = source.__aiter__()
        self._running = False
        self._closed = False

    def __aiter__(self) -> "_SafeAsyncStream":
        return self

    async def __anext__(self) -> str:
        if self._closed:
            raise StopAsyncIteration

        self._running = True
        try:
            item = await self._source.__anext__()
        except StopAsyncIteration:
            self._closed = True
            await self._close_source()
            raise
        except asyncio.CancelledError:
            self._closed = True
            await self._close_source()
            raise
        finally:
            self._running = False

        if self._closed:
            await self._close_source()
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self._closed = True
        if not self._running:
            await self._close_source()

    async def _close_source(self) -> None:
        close = getattr(self._source, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except RuntimeError as exc:
            if "asynchronous generator is already running" not in str(exc):
                raise


def _safe_stream(source: AsyncIterable[str]) -> _SafeAsyncStream:
    return _SafeAsyncStream(source)


def _with_bridge_tools(
    payload: dict[str, Any],
    registry: ToolRegistry | None,
    config: BridgeConfig | None = None,
    enable_filtering: bool = True,
    max_exposed_tools: int = 5,
    tool_manager: ToolManager | None = None,
    provider_config: ProviderConfig | None = None,
) -> dict[str, Any]:
    """
    Merge bridge tools into request, optionally filtering by relevance.
    Uses ToolManager for compact-first tool management when enabled.
    Returns payload unchanged when registry is None (no tools available).
    """
    if registry is None:
        return payload
    if payload.get("tools"):
        # Client-supplied tools belong to the caller (for example Claude Agent
        # SDK MCP tools). Do not merge or execute bridge tools into that request:
        # the bridge should translate the model's tool_calls back to Anthropic
        # tool_use blocks so the caller can run them.
        return _with_streaming_client_tool_instructions(payload)
    query = _latest_user_text(payload.get("messages", []))[:500]
    request_profile = _request_tool_profile(config, provider_config, payload.get("model"), query)

    if tool_manager and config and config.tools.management_enabled:
        return _with_managed_tools(payload, registry, config, tool_manager, request_profile)

    bridge_tools = registry.openai_tools()
    if not bridge_tools:
        return payload

    if config is not None:
        enable_filtering = config.tools.relevance_filter
        max_exposed_tools = max(1, int(config.tools.max_exposed or max_exposed_tools))

    if enable_filtering:
        if query:
            filtered_tools, scores = select_relevant_tools(
                bridge_tools,
                query,
                max_tools=min(max_exposed_tools, int(request_profile.get("max_tools", max_exposed_tools))),
                min_score=config.tools.confidence_threshold if config else 0.5,
                force_for_keywords=config.tools.force_for_keywords if config else True,
                default_search_provider=config.tools.default_search_provider if config else "tavily",
            )
            selected_tool_names = {t["function"]["name"] for t in filtered_tools}
            rejected_tool_names = [
                t["function"]["name"]
                for t in bridge_tools
                if t["function"]["name"] not in selected_tool_names
            ]
            _write_dev_log(
                config,
                "tool_selection",
                {
                    "query_preview": query[:100],
                    "available_tools": [t["function"]["name"] for t in bridge_tools],
                    "tool_scores": scores,
                    "selected_tools": list(selected_tool_names),
                    "rejected_tools": rejected_tool_names,
                    "config": {
                        "max_exposed": max_exposed_tools,
                        "confidence_threshold": config.tools.confidence_threshold if config else 0.5,
                        "default_search_provider": config.tools.default_search_provider if config else "tavily",
                    },
                },
            ) if config else None
            bridge_tools = filtered_tools

    merged = {**payload}
    if config is not None:
        merged = _with_tool_instructions(merged, config, bridge_tools, request_profile)
    tools = list(merged.get("tools") or [])
    existing = {
        ((tool.get("function") or {}).get("name") or tool.get("name"))
        for tool in tools
        if isinstance(tool, dict)
    }
    for tool in bridge_tools:
        name = (tool.get("function") or {}).get("name")
        if name not in existing:
            tools.append(tool)
    merged["tools"] = tools
    merged.setdefault("tool_choice", "auto")
    return merged


def _with_streaming_client_tool_instructions(payload: dict[str, Any]) -> dict[str, Any]:
    tool_names = _tool_names(payload.get("tools"))
    if not tool_names:
        return payload

    lines = [
        "Streaming tool-use rules:",
        "- Call only tool names exactly present in the supplied tool list.",
    ]
    if "shell" in tool_names:
        lines.append(
            "- For local shell commands, call `shell` with `cmd`; never call "
            "`shell_exec`, `shell_execute`, `Shell`, `Run cmd`, or pass `command`."
        )
    prefixed_bridge_tools = sorted(name for name in tool_names if name.startswith("llama_bridge_tools__"))
    if prefixed_bridge_tools:
        lines.append(
            "- For llama bridge MCP tools, use the fully prefixed "
            "`llama_bridge_tools__...` names shown in the tool list; never call "
            "unprefixed `tool_catalog_search` or `tool_schema_get`."
        )

    return _append_system_instruction(payload, "\n".join(lines))


def _append_system_instruction(payload: dict[str, Any], instruction: str) -> dict[str, Any]:
    messages = list(payload.get("messages") or [])
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": _append_content_text(messages[0].get("content"), instruction)}
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return {**payload, "messages": messages}


def _append_content_text(content: Any, text: str) -> Any:
    if isinstance(content, list):
        return [*content, {"type": "text", "text": text}]
    if isinstance(content, str):
        return f"{content}\n\n{text}".strip()
    if content is None:
        return text
    return f"{_string_content(content)}\n\n{text}".strip()


def _tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name") or tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _with_managed_tools(
    payload: dict[str, Any],
    registry: ToolRegistry,
    config: BridgeConfig,
    tool_manager: ToolManager,
    request_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Handle tool integration with compact-first management system."""
    query = _latest_user_text(payload.get("messages", []))[:500]

    # Get tools based on policy
    tools = tool_manager.schemas_for_request(query, "openai", request_profile)

    # Add compact manifest as system instruction
    compact_manifest = tool_manager.compact_manifest(query, "openai", request_profile)
    compact_text = tool_manager.compact_instruction_text(request_profile)

    if compact_manifest:
        manifest_lines = []
        for tool in compact_manifest:
            line = f"- {tool['name']}: {tool['summary']}"
            if tool.get('use_when') and not (request_profile or {}).get("minimal"):
                line += f" (use when: {', '.join(tool['use_when'][:3])})"
            if tool.get('args_hint') and not (request_profile or {}).get("minimal"):
                line += f" [args: {tool['args_hint']}]"
            manifest_lines.append(line)

        if manifest_lines:
            compact_text += "\n\nAvailable bridge tools, compact view:\n" + "\n".join(manifest_lines)
            if (request_profile or {}).get("include_management_tools", True) and tool_manager.management_openai_tools():
                compact_text += "\n\nNeed exact parameters? Call tool_schema_get with the tool name.\n"
                compact_text += "Unsure which tool? Call tool_catalog_search with your query."

    merged = {**payload}

    if not tools and not compact_text:
        merged.pop("tools", None)
        merged.pop("tool_choice", None)
        return merged

    # Add compact instructions to system message
    messages = list(merged.get("messages") or [])
    if compact_text:
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": f"{messages[0].get('content') or ''}\n\n{compact_text}".strip(),
            }
        else:
            messages.insert(0, {"role": "system", "content": compact_text})

    merged["messages"] = messages
    if tools:
        merged["tools"] = tools
        merged.setdefault("tool_choice", "auto")
    else:
        merged.pop("tools", None)
        merged.pop("tool_choice", None)

    # Log the decision
    _write_dev_log(
        config,
        "managed_tool_selection",
        {
            "query_preview": query[:100],
            "policy": config.tools.expose_full_schema_policy,
            "compact_tools_count": len(compact_manifest),
            "full_schemas_count": len([t for t in tools if t.get("type") == "function"]),
            "management_tools_count": len(tool_manager.management_openai_tools()),
        },
    )

    return merged


def _log_streaming_tool_policy(config: BridgeConfig, route: str, payload: dict[str, Any]) -> None:
    bridge_tools = [
        (tool.get("function") or {}).get("name")
        for tool in payload.get("tools") or []
        if isinstance(tool, dict)
    ]
    if not bridge_tools:
        return
    _write_dev_log(
        config,
        "streaming_tools_bypass",
        {
            "route": route,
            "policy": "streaming forwards upstream output; bridge tool calls are only executed on non-streaming requests",
            "exposed_tools": bridge_tools,
        },
    )


def _should_buffer_streaming_tool_request(body: dict[str, Any], payload: dict[str, Any]) -> bool:
    if not payload.get("tools"):
        return False
    text = _latest_user_text(body.get("messages", [])).lower()
    markers = (
        "/deep",
        "deep research",
        "deep resarch",
        "deep_research",
        "deep research tool",
        "auto plan mode",
        "report.md",
        "temp/ad.md",
        "deep_plan_agent",
        "deep_collect_agent",
        "deep_review_agent",
        "llama_bridge_tools__deep",
    )
    return any(marker in text for marker in markers)


def _request_tool_profile(
    config: BridgeConfig | None,
    provider_config: ProviderConfig | None,
    model: Any,
    query: str,
) -> dict[str, Any]:
    profile = {
        "minimal": False,
        "max_tools": max(1, int(config.tools.max_exposed)) if config else 5,
        "max_manifest_tools": max(0, int(config.tools.compact_manifest_max_tools)) if config else 5,
        "include_management_tools": bool(config.tools.always_expose_management_tools) if config else True,
        "disable_fallback": False,
        "attach_system_instructions": True,
    }
    if provider_config is None or config is None:
        return profile

    model_name = str(model or "")
    model_limits = (provider_config.model_limits or {}).get(model_name, {}) or {}
    tokens_per_minute = _limit_int(model_limits.get("tokens_per_minute"))
    low_token_budget = provider_config.type == "groq" or (
        tokens_per_minute is not None and tokens_per_minute <= 12000
    )

    if low_token_budget:
        profile.update(
            {
                "minimal": True,
                "max_tools": min(profile["max_tools"], max(1, int(model_limits.get("max_tools", 1)))),
                "max_manifest_tools": min(
                    profile["max_manifest_tools"],
                    max(0, int(model_limits.get("max_manifest_tools", 1))),
                ),
                "include_management_tools": bool(model_limits.get("include_management_tools", False)),
                "disable_fallback": True,
            }
        )

    if _is_lightweight_chat_query(query):
        profile.update(
            {
                "include_management_tools": False,
                "max_manifest_tools": 0,
            }
        )
        if low_token_budget:
            profile["attach_system_instructions"] = False

    return profile


def _limit_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_lightweight_chat_query(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return True
    words = text.split()
    if len(words) > 4:
        return False
    factual_markers = (
        "weather",
        "time",
        "date",
        "today",
        "latest",
        "news",
        "search",
        "find",
        "lookup",
        "price",
        "source",
        "verify",
        "citation",
        "wiki",
        "wikipedia",
        "fetch",
        "image",
        "research",
    )
    return not any(marker in text for marker in factual_markers)


def _latest_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
    return ""


def _with_tool_instructions(
    payload: dict[str, Any],
    config: BridgeConfig,
    selected_tools: list[dict[str, Any]],
    request_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not selected_tools:
        return payload
    if request_profile and not request_profile.get("attach_system_instructions", True):
        return payload
    instructions = config.tools.tool_system_instructions or (
        "Tool-use rules:\n"
        "- Use tools for current facts, weather, time/date, source verification, and web research.\n"
        "- Do not guess when a relevant tool is available.\n"
        "- When a tool result is returned, treat it as the source of truth unless it contains an error.\n"
        "- If a tool result has ok=false, explain the failure and answer only if enough information remains.\n"
        "- Do not invent citations, URLs, prices, dates, or weather data.\n"
        "- If sources are returned, mention them accurately."
    )
    if request_profile and request_profile.get("minimal"):
        instructions = config.tools.tool_system_instructions or (
            "Use a tool only when it is clearly needed for current facts or external data. "
            "Otherwise answer directly."
        )
    try:
        pi_model = resolve_pi_model(config)
    except Exception:
        pi_model = None
    if payload.get("model") == pi_model:
        pi_instructions = config.tools.pi_system_instructions or (
        "Pi tool policy: Use available tools for factual, current, time, weather, "
        "search, and source-backed questions. Do not guess when a suitable tool is "
        "available. Summarize tool JSON clearly, including sources or timestamps "
        "when present. Ask for clarification only when the required location, "
        "topic, or identifier is genuinely missing. When using a write/file-writing "
        "tool, include both required arguments: path and content. Example: "
        "{\"path\":\"report.md\",\"content\":\"<full markdown report>\"}. "
        "Never call write with content only."
        )
        instructions = f"{instructions}\n\n{pi_instructions}"
    messages = list(payload.get("messages") or [])
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": f"{messages[0].get('content') or ''}\n\n{instructions}".strip(),
        }
    else:
        messages.insert(0, {"role": "system", "content": instructions})
    return {**payload, "messages": messages}


def _bridge_tools_response(registry: ToolRegistry) -> dict[str, Any]:
    tools = registry.list_tools()
    openai_tools = registry.openai_tools()
    unavailable_tools = registry.unavailable_tools()
    return {
        "object": "list",
        "tools": tools,
        "data": tools,
        "unavailable_tools": unavailable_tools,
        "openai_tools": openai_tools,
        "ollama_tools": openai_tools,
        "anthropic_tools": [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters") or {"type": "object"},
            }
            for tool in tools
        ],
    }


def _bridge_tools_response_from_list(openai_tools: list[dict[str, Any]]) -> dict[str, Any]:
    tools = []
    for tool in openai_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name") or tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = function.get("description") if isinstance(function.get("description"), str) else ""
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object"}
        tools.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return {
        "object": "list",
        "tools": tools,
        "data": tools,
        "unavailable_tools": {},
        "openai_tools": openai_tools,
        "ollama_tools": openai_tools,
        "anthropic_tools": [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters") or {"type": "object"},
            }
            for tool in tools
        ],
    }


def _bridge_tool_arguments(body: dict[str, Any]) -> dict[str, Any]:
    for key in ("arguments", "args", "input", "parameters"):
        value = body.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {
        key: value
        for key, value in body.items()
        if key not in {"name", "tool", "arguments", "args", "input", "parameters"}
    }


async def _chat_completion_with_bridge_tools(
    app: FastAPI,
    provider: OpenAICompatibleProvider,
    payload: dict[str, Any],
    config: BridgeConfig | None = None,
    max_rounds: int = 4,
) -> dict[str, Any]:
    """
    Execute chat completion with automatic tool handling.
    Logs tool selection, arguments, results, and errors.
    Handles management tools internally when tool management is enabled.
    """
    request_payload = {**payload, "messages": list(payload.get("messages") or [])}
    tool_manager = getattr(app.state, "tool_manager", None)
    tools_registry = getattr(app.state, "tools", None)
    bridge_tool_names = set(tools_registry._tools) if tools_registry is not None else set()
    management_tool_names = (
        {
            ((tool.get("function") or {}).get("name") or tool.get("name") or "")
            for tool in (tool_manager.management_openai_tools() if tool_manager else [])
        }
    )

    for _round in range(max_rounds):
        response = await provider.create_chat_completion(request_payload, stream=False)
        response.raise_for_status()
        data = response.json()
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            _write_dev_log(
                config,
                "tool_execution_complete",
                {"round": _round, "reason": "no tool calls"},
            )
            return data

        # Separate management tools from bridge tools
        management_calls = []
        bridge_calls = []
        other_calls = []

        for call in tool_calls:
            func_name = (call.get("function") or {}).get("name") or ""
            if func_name in management_tool_names:
                management_calls.append(call)
            elif func_name in bridge_tool_names:
                bridge_calls.append(call)
            else:
                other_calls.append(call)

        if other_calls:
            _write_dev_log(
                config,
                "client_tool_calls_passthrough",
                {
                    "round": _round,
                    "tools": [
                        (call.get("function") or {}).get("name") or ""
                        for call in other_calls
                    ],
                },
            )
            return data

        # Handle management tools internally
        if management_calls:
            request_payload["messages"].append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": management_calls + bridge_calls + other_calls,
                }
            )

            for tool_call in management_calls:
                function = tool_call.get("function") or {}
                name = function.get("name") or ""
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                _write_dev_log(
                    config,
                    "management_tool_call",
                    {"tool": name, "call_id": tool_call.get("id"), "arguments": arguments},
                )

                # Call management tool via ToolManager
                if tool_manager:
                    result = await tool_manager.call_management_tool(
                        name, arguments, "openai", None
                    )
                else:
                    result = {"ok": False, "error": "ToolManager not available"}

                request_payload["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": json.dumps(result, ensure_ascii=True),
                    }
                )

            # If only management tools were called, continue the loop
            if not bridge_calls and not other_calls:
                continue

            # Remove management calls from tool_calls for further processing
            tool_calls = [c for c in tool_calls if c not in management_calls]

        if not tool_calls or (len(tool_calls) == len(management_calls) and not bridge_calls):
            return data

        request_payload["messages"].append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            }
        )

        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            name = function.get("name") or ""
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            _write_dev_log(
                config,
                "tool_call_start",
                {
                    "tool": name,
                    "call_id": tool_call.get("id"),
                    "arguments": arguments,
                },
            )

            tools_registry = getattr(app.state, "tools", None)
            if tools_registry is None:
                tool_result = {
                    "ok": False,
                    "tool": name,
                    "error": "Tools are not available on this server instance",
                    "retryable": False,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            else:
                try:
                    tool_result = await tools_registry.call_structured(name, arguments)
                except Exception as exc:
                    error_msg = str(exc)
                    _write_dev_log(
                        config,
                        "tool_call_error",
                        {
                            "tool": name,
                            "call_id": tool_call.get("id"),
                            "error": error_msg,
                            "error_type": type(exc).__name__,
                        },
                    )
                    tool_result = {
                        "ok": False,
                        "tool": name,
                        "error": error_msg,
                        "retryable": not isinstance(exc, (KeyError, ValueError)),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

            request_payload["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": json.dumps(tool_result, ensure_ascii=True),
                }
            )

    return data


def _resolve_bridge_model(requested_model: str, config: BridgeConfig) -> ResolvedModel:
    try:
        copilot_model, upstream_model = resolve_vs_copilot_model(config, requested_model)
        return ResolvedModel(
            alias=copilot_model.name,
            upstream_model=upstream_model,
            provider=config.providers[copilot_model.provider],
        )
    except KeyError:
        pass

    try:
        return resolve_model(requested_model, config.providers, config.anthropic_models)
    except KeyError:
        codex_model = resolve_codex_model(config)
        if codex_model and requested_model == codex_model:
            return ResolvedModel(
                alias=requested_model,
                upstream_model=codex_model,
                provider=config.providers[config.codex.provider],
            )

        available = set(config.anthropic_models)
        available.update(
            alias.model
            for alias in config.anthropic_models.values()
            if alias.model
        )
        if codex_model:
            available.add(codex_model)
        available.update(model.name for model in config.vs_copilot_models)
        available.update(model.model for model in config.vs_copilot_models if model.model)
        available_text = ", ".join(sorted(available))
        raise KeyError(
            f"Unknown model '{requested_model}'. Configure it as an anthropic_models "
            f"alias, use a configured upstream model, or set codex.model. "
            f"Available models: {available_text}"
        )


def _provider_for(app: FastAPI, resolved: ResolvedModel) -> OpenAICompatibleProvider:
    return app.state.providers[resolved.provider.name]


def _anthropic_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


async def _optional_json(request: Request) -> dict[str, Any]:
    try:
        value = await request.json()
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _anthropic_list_response(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": items,
        "has_more": False,
        "first_id": items[0]["id"] if items else None,
        "last_id": items[-1]["id"] if items else None,
    }


def _anthropic_complete_to_messages_request(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": body.get("model", ""),
        "max_tokens": body.get("max_tokens_to_sample", body.get("max_tokens", 2048)),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "stop_sequences": body.get("stop_sequences"),
        "stream": body.get("stream", False),
        "messages": [{"role": "user", "content": _string_content(body.get("prompt", ""))}],
    }


def _chat_completion_to_anthropic_complete_response(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    stop_reason = "stop_sequence" if finish_reason == "stop" else finish_reason
    if finish_reason == "length":
        stop_reason = "max_tokens"
    return {
        "id": data.get("id") or f"compl_{uuid.uuid4().hex}",
        "type": "completion",
        "completion": _string_content(message.get("content")),
        "stop_reason": stop_reason or "stop_sequence",
        "model": request_body.get("model") or data.get("model"),
    }


def _anthropic_batch_record(body: dict[str, Any]) -> dict[str, Any]:
    batch_id = f"msgbatch_{uuid.uuid4().hex}"
    now = _anthropic_timestamp()
    requests = body.get("requests") or []
    return {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": "ended",
        "request_counts": {
            "processing": 0,
            "succeeded": 0,
            "errored": len(requests),
            "canceled": 0,
            "expired": 0,
        },
        "created_at": now,
        "ended_at": now,
        "expires_at": now,
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": f"/v1/messages/batches/{batch_id}/results",
    }


async def _anthropic_batch_results(
    requests: list[dict[str, Any]],
) -> AsyncIterator[str]:
    for index, item in enumerate(requests):
        custom_id = item.get("custom_id") or f"request_{index}"
        result = {
            "custom_id": custom_id,
            "result": {
                "type": "errored",
                "error": {
                    "type": "not_supported_error",
                    "message": (
                        "llama bridge exposes the Message Batches route for client "
                        "compatibility, but does not process async batches."
                    ),
                },
            },
        }
        yield json.dumps(result, ensure_ascii=True) + "\n"


def _anthropic_file_record(content_type: str | None, body: bytes) -> dict[str, Any]:
    file_id = f"file_{uuid.uuid4().hex}"
    return {
        "id": file_id,
        "type": "file",
        "filename": "upload",
        "mime_type": content_type or "application/octet-stream",
        "size_bytes": len(body),
        "created_at": _anthropic_timestamp(),
        "downloadable": True,
        "_content": body,
    }


def _anthropic_public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _anthropic_skill_record(body: dict[str, Any]) -> dict[str, Any]:
    now = _anthropic_timestamp()
    return {
        "id": body.get("id") or f"skill_{uuid.uuid4().hex}",
        "type": "skill",
        "name": body.get("name") or "skill",
        "description": body.get("description", ""),
        "created_at": now,
        "updated_at": now,
        **{key: value for key, value in body.items() if key not in {"id", "type"}},
    }


def _anthropic_admin_response(
    method: str,
    path: str,
    body: dict[str, Any],
) -> JSONResponse:
    normalized_path = path.strip("/")
    if method == "GET":
        if "usage_report" in normalized_path or "cost_report" in normalized_path:
            return JSONResponse({"data": [], "has_more": False})
        if "/" not in normalized_path:
            return JSONResponse(_anthropic_list_response([]))
        resource_type = normalized_path.rsplit("/", 1)[-2].rstrip("s") or "organization"
        return JSONResponse(
            {
                "id": normalized_path.rsplit("/", 1)[-1],
                "type": resource_type,
                "created_at": _anthropic_timestamp(),
            }
        )
    if method == "DELETE":
        return JSONResponse(
            {"id": normalized_path.rsplit("/", 1)[-1], "deleted": True}
        )
    resource = normalized_path.rsplit("/", 1)[-1] if normalized_path else "organization"
    return JSONResponse(
        {
            "id": body.get("id") or f"{resource.rstrip('s')}_{uuid.uuid4().hex}",
            "type": resource.rstrip("s") or "organization",
            "created_at": _anthropic_timestamp(),
            **body,
        }
    )


def _openai_list_response(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": items,
        "first_id": items[0]["id"] if items else None,
        "last_id": items[-1]["id"] if items else None,
        "has_more": False,
    }


def _openai_assistant_record(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"asst_{uuid.uuid4().hex}",
        "object": "assistant",
        "created_at": int(time.time()),
        "name": body.get("name"),
        "description": body.get("description"),
        "model": body.get("model", ""),
        "instructions": body.get("instructions"),
        "tools": body.get("tools", []),
        "metadata": body.get("metadata", {}),
    }


def _openai_thread_record(body: dict[str, Any]) -> dict[str, Any]:
    thread_id = f"thread_{uuid.uuid4().hex}"
    return {
        "id": thread_id,
        "object": "thread",
        "created_at": int(time.time()),
        "metadata": body.get("metadata", {}),
        "messages": [],
        "runs": [],
    }


def _openai_thread_message_record(
    thread_id: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "object": "thread.message",
        "created_at": int(time.time()),
        "thread_id": thread_id,
        "role": body.get("role", "user"),
        "content": body.get("content", []),
        "assistant_id": body.get("assistant_id"),
        "run_id": body.get("run_id"),
        "metadata": body.get("metadata", {}),
    }


def _openai_thread_run_record(
    thread_id: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": f"run_{uuid.uuid4().hex}",
        "object": "thread.run",
        "created_at": now,
        "thread_id": thread_id,
        "assistant_id": body.get("assistant_id"),
        "model": body.get("model"),
        "instructions": body.get("instructions"),
        "tools": body.get("tools", []),
        "metadata": body.get("metadata", {}),
        "status": "completed",
        "started_at": now,
        "completed_at": now,
        "required_action": None,
        "last_error": None,
    }


def _find_thread_run(
    threads: dict[str, dict[str, Any]],
    thread_id: str,
    run_id: str,
) -> dict[str, Any] | None:
    thread = threads.get(thread_id)
    if thread is None:
        return None
    for run in thread["runs"]:
        if run["id"] == run_id:
            return run
    return None


def _openai_fine_tuning_job_record(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"ftjob_{uuid.uuid4().hex}",
        "object": "fine_tuning.job",
        "created_at": int(time.time()),
        "model": body.get("model"),
        "training_file": body.get("training_file"),
        "validation_file": body.get("validation_file"),
        "hyperparameters": body.get("hyperparameters", {}),
        "status": "queued",
        "fine_tuned_model": None,
        "result_files": [],
        "error": None,
    }


def _openai_moderation_response(body: dict[str, Any]) -> dict[str, Any]:
    categories = {
        "harassment": False,
        "harassment/threatening": False,
        "hate": False,
        "hate/threatening": False,
        "self-harm": False,
        "self-harm/intent": False,
        "self-harm/instructions": False,
        "sexual": False,
        "sexual/minors": False,
        "violence": False,
        "violence/graphic": False,
    }
    return {
        "id": f"modr_{uuid.uuid4().hex}",
        "model": body.get("model", "llama-bridge-moderation"),
        "results": [
            {
                "flagged": False,
                "categories": categories,
                "category_scores": {key: 0.0 for key in categories},
            }
        ],
    }


def _unsupported_endpoint(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"error": {"type": "not_supported_error", "message": message}},
    )


def _cohere_chat_request_to_chat_completion(
    body: dict[str, Any],
    upstream_model: str,
) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = []
        for item in body.get("chat_history") or []:
            role = str(item.get("role", "user")).lower()
            if role == "chatbot":
                role = "assistant"
            messages.append({"role": role, "content": _string_content(item.get("message", ""))})
        messages.append({"role": "user", "content": _string_content(body.get("message", ""))})
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages or [{"role": "user", "content": ""}],
    }
    for key in ("temperature", "top_p", "stream"):
        if key in body:
            payload[key] = body[key]
    if "max_tokens" in body:
        payload["max_tokens"] = body["max_tokens"]
    return payload


def _chat_completion_to_cohere_chat_response(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = _string_content(message.get("content"))
    return {
        "id": data.get("id") or f"chat_{uuid.uuid4().hex}",
        "text": text,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "finish_reason": choice.get("finish_reason") or "COMPLETE",
        "meta": {"api_version": {"version": "llama-bridge"}},
        "model": request_body.get("model") or data.get("model"),
    }


def _openai_embedding_to_cohere_embed(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    embeddings = [
        item.get("embedding", [])
        for item in sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
    ]
    return {
        "id": f"embed_{uuid.uuid4().hex}",
        "texts": request_body.get("texts", request_body.get("input", [])),
        "embeddings": embeddings,
        "meta": {"api_version": {"version": "llama-bridge"}},
    }


def _gemini_request_to_chat_completion(
    body: dict[str, Any],
    upstream_model: str,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for content in body.get("contents") or []:
        role = "assistant" if content.get("role") == "model" else "user"
        parts = content.get("parts") or []
        text = "\n".join(str(part.get("text", "")) for part in parts if "text" in part)
        messages.append({"role": role, "content": text})
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages or [{"role": "user", "content": ""}],
    }
    generation_config = body.get("generationConfig") or {}
    for source_key, target_key in (
        ("temperature", "temperature"),
        ("topP", "top_p"),
        ("maxOutputTokens", "max_tokens"),
    ):
        if source_key in generation_config:
            payload[target_key] = generation_config[source_key]
    if "stopSequences" in generation_config:
        payload["stop"] = generation_config["stopSequences"]
    return payload


def _chat_completion_to_gemini_response(data: dict[str, Any]) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": _string_content(message.get("content"))}],
                },
                "finishReason": str(choice.get("finish_reason") or "STOP").upper(),
                "index": choice.get("index", 0),
            }
        ],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
    }


def _available_model_ids(config: BridgeConfig) -> list[str]:
    pi_model = resolve_pi_model(config)
    codex_model = resolve_codex_model(config)
    model_ids = (
        set(config.anthropic_models)
        | {
            model_alias.model
            for model_alias in config.anthropic_models.values()
            if model_alias.model
        }
        | ({pi_model} if pi_model else set())
        | ({codex_model} if codex_model else set())
        | {model.name for model in config.vs_copilot_models}
        | {model.model for model in config.vs_copilot_models if model.model}
    )
    return sorted(model_ids)


def _poolside_agent_records(config: BridgeConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model_id in _available_model_ids(config):
        records.append(
            {
                "id": model_id,
                "name": model_id,
                "display_name": model_id,
                "description": f"llama bridge model {model_id}",
                "model": model_id,
                "model_ref": model_id,
            }
        )
    return records


def _ollama_model_ids(config: BridgeConfig) -> list[str]:
    return [model.name for model in config.vs_copilot_models]


def _ollama_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


OLLAMA_MODEL_METADATA: dict[str, dict[str, Any]] = {
    "gemma3:4b": {
        "modified_at": "2025-03-12T00:00:00Z",
        "size": 8600000000,
        "digest": "c1d2e3f40002",
    },
    "gemma4:31b": {
        "modified_at": "2026-04-02T09:00:00-08:00",
        "size": 62546177752,
        "digest": "221b330d11a8",
    },
}


def _ollama_model_details(model_id: str, tags_shape: bool = False) -> dict[str, Any]:
    if tags_shape:
        return {
            "parent_model": "",
            "format": "",
            "family": "",
            "families": None,
            "parameter_size": "",
            "quantization_level": "",
        }
    family = model_id.split(":", 1)[0].split("/", 1)[-1] or "llama"
    return {
        "parent_model": "",
        "format": "bridge",
        "family": family,
        "families": [family],
        "parameter_size": "unknown",
        "quantization_level": "unknown",
    }


def _ollama_model_record(model) -> dict[str, Any]:
    model_id = model.name
    metadata = OLLAMA_MODEL_METADATA.get(model_id, {})
    return {
        "name": model_id,
        "model": model_id,
        "modified_at": model.modified_at or metadata.get("modified_at") or "2025-01-01T00:00:00Z",
        "size": model.size if model.size is not None else metadata.get("size", _estimate_ollama_model_size(model_id)),
        "digest": model.digest or metadata.get("digest") or _stable_ollama_digest(model_id),
        "details": _ollama_model_details(model_id, tags_shape=True),
    }


def _vs_copilot_context_size(config: BridgeConfig, model_id: str) -> int:
    for model in config.vs_copilot_models:
        if model_id in {model.name, model.model}:
            return model.context_size
    return 65536


def _ollama_model_info(model_id: str, context_size: int) -> dict[str, Any]:
    architecture = model_id.split(":", 1)[0].split("/", 1)[-1] or "llama"
    return {
        "general.architecture": architecture,
        "general.parameter_count": _estimate_ollama_parameter_count(model_id),
        "general.context_length": context_size,
        f"{architecture}.context_length": context_size,
    }


def _estimate_ollama_model_size(model_id: str) -> int:
    match = re.search(r"(?i)(\d+(?:\.\d+)?)b", model_id)
    if not match:
        return 10000000000
    return int(float(match.group(1)) * 2_000_000_000)


def _estimate_ollama_parameter_count(model_id: str) -> int:
    match = re.search(r"(?i)(\d+(?:\.\d+)?)b", model_id)
    if not match:
        return 8_000_000_000
    return int(float(match.group(1)) * 1_000_000_000)


def _stable_ollama_digest(model_id: str) -> str:
    import hashlib

    return hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]


def _completion_request_to_chat_completion(
    body: dict[str, Any],
    upstream_model: str,
) -> dict[str, Any]:
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(_string_content(item) for item in prompt)
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": [{"role": "user", "content": _string_content(prompt)}],
    }
    for key in (
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
        "stream",
        "temperature",
        "top_p",
        "max_tokens",
        "logit_bias",
        "user",
        "n",
    ):
        if key in body:
            payload[key] = body[key]
    if body.get("format") == "json":
        payload["response_format"] = {"type": "json_object"}
    return payload


def _chat_completion_to_completion_response(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "id": data.get("id") or f"cmpl_{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": data.get("created") or int(time.time()),
        "model": request_body.get("model") or data.get("model"),
        "choices": [
            {
                "text": _string_content(message.get("content")),
                "index": choice.get("index", 0),
                "logprobs": None,
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        ],
        "usage": usage,
    }


def _ollama_chat_request_to_chat_completion(
    body: dict[str, Any],
    upstream_model: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": body.get("messages") or [],
        "stream": bool(body.get("stream", True)),
    }
    _copy_ollama_generation_options(body, payload)
    tools = body.get("tools")
    if isinstance(tools, list):
        payload["tools"] = [_ollama_tool_to_chat_tool(tool) for tool in tools]
    return payload


def _ollama_generate_request_to_chat_completion(
    body: dict[str, Any],
    upstream_model: str,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if body.get("system"):
        messages.append({"role": "system", "content": _string_content(body["system"])})
    messages.append({"role": "user", "content": _string_content(body.get("prompt", ""))})
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(body.get("stream", True)),
    }
    _copy_ollama_generation_options(body, payload)
    return payload


def _copy_ollama_generation_options(
    body: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    for source_key, target_key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("seed", "seed"),
        ("stop", "stop"),
        ("num_predict", "max_tokens"),
    ):
        if source_key in options:
            payload[target_key] = options[source_key]
    if "format" in body:
        payload["response_format"] = (
            {"type": "json_object"} if body.get("format") == "json" else body["format"]
        )
    think = body.get("think")
    if isinstance(think, str):
        payload["reasoning_effort"] = think
    elif think is True:
        payload["reasoning_effort"] = "medium"


def _ollama_tool_to_chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return {
        "type": "function",
        "function": {
            "name": function.get("name") or "tool",
            "description": function.get("description", ""),
            "parameters": function.get("parameters") or {},
        },
    }


def _chat_completion_to_ollama_chat_response(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "model": request_body.get("model") or data.get("model"),
        "created_at": _ollama_timestamp(),
        "message": {
            "role": "assistant",
            "content": _string_content(message.get("content")),
            **_ollama_tool_calls_field(message.get("tool_calls")),
        },
        "done": True,
        "done_reason": choice.get("finish_reason") or "stop",
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


def _chat_completion_to_ollama_generate_response(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "model": request_body.get("model") or data.get("model"),
        "created_at": _ollama_timestamp(),
        "response": _string_content(message.get("content")),
        "done": True,
        "done_reason": choice.get("finish_reason") or "stop",
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


def _ollama_management_response(body: dict[str, Any], final_status: str):
    stream = bool(body.get("stream", True))
    payloads = [{"status": final_status}]
    if stream:
        return StreamingResponse(
            (f"{json.dumps(payload, ensure_ascii=True)}\n" for payload in payloads),
            media_type="application/x-ndjson",
        )
    return JSONResponse(payloads[-1])


async def _ollama_embedding_request(
    app: FastAPI,
    config: BridgeConfig,
    body: dict[str, Any],
    legacy: bool,
) -> JSONResponse:
    try:
        resolved = _resolve_bridge_model(body["model"], config)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    openai_body = _ollama_embed_request_to_openai_embedding(body, resolved.upstream_model)
    provider = _provider_for(app, resolved)
    try:
        response = await provider.create_embedding(openai_body)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return _upstream_error(exc.response)
    except httpx.RequestError as exc:
        return _request_error(exc)
    data = response.json()
    if legacy:
        return JSONResponse(_openai_embedding_to_ollama_legacy_embedding(data))
    return JSONResponse(_openai_embedding_to_ollama_embed(data, body))


def _ollama_embed_request_to_openai_embedding(
    body: dict[str, Any],
    upstream_model: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": upstream_model,
        "input": body.get("input", body.get("prompt", "")),
    }
    if "dimensions" in body:
        payload["dimensions"] = body["dimensions"]
    return payload


def _openai_embedding_to_ollama_embed(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    embeddings = [
        item.get("embedding", [])
        for item in sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
    ]
    usage = data.get("usage") or {}
    return {
        "model": request_body.get("model") or data.get("model"),
        "embeddings": embeddings,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }


def _openai_embedding_to_ollama_legacy_embedding(data: dict[str, Any]) -> dict[str, Any]:
    embeddings = [
        item.get("embedding", [])
        for item in sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
    ]
    return {"embedding": embeddings[0] if embeddings else []}


def _ollama_embed_to_openai_embedding(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    embeddings = data.get("embeddings")
    if embeddings is None and "embedding" in data:
        embeddings = [data["embedding"]]
    if not isinstance(embeddings, list):
        embeddings = []
    if embeddings and all(isinstance(value, (int, float)) for value in embeddings):
        embeddings = [embeddings]
    return {
        "object": "list",
        "model": request_body.get("model") or data.get("model"),
        "data": [
            {"object": "embedding", "embedding": embedding, "index": index}
            for index, embedding in enumerate(embeddings)
        ],
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0),
        },
    }


def _ollama_tool_calls_field(tool_calls: Any) -> dict[str, Any]:
    converted = _openai_tool_calls_to_ollama(tool_calls)
    return {"tool_calls": converted} if converted else {}


def _openai_tool_calls_to_ollama(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                arguments = {"arguments": arguments}
        converted.append(
            {
                "function": {
                    "name": function.get("name") or "tool",
                    "arguments": arguments,
                }
            }
        )
    return converted


def _responses_request_to_chat_completion(
    body: dict[str, Any],
    upstream_model: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": _responses_input_to_messages(body.get("input")),
    }
    if body.get("instructions"):
        payload["messages"].insert(
            0,
            {"role": "system", "content": str(body["instructions"])},
        )
    for responses_key, chat_key in (
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("parallel_tool_calls", "parallel_tool_calls"),
        ("tool_choice", "tool_choice"),
    ):
        if responses_key in body:
            payload[chat_key] = body[responses_key]
    tools = _responses_tools_to_chat_tools(body.get("tools"))
    if tools:
        payload["tools"] = tools
    return payload


def _responses_input_to_messages(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        return [{"role": "user", "content": ""}]

    messages: list[dict[str, Any]] = []
    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or "call_0"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "tool",
                                "arguments": _string_content(item.get("arguments") or "{}"),
                            },
                        }
                    ],
                }
            )
            continue
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "call_0",
                    "content": _string_content(item.get("output")),
                }
            )
            continue

        role = item.get("role") or ("assistant" if item_type == "message" else "user")
        messages.append(
            {
                "role": _chat_role(role),
                "content": _responses_content_to_chat_content(item.get("content")),
            }
        )
    return messages or [{"role": "user", "content": ""}]


def _responses_content_to_chat_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _string_content(content)

    parts: list[Any] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            parts.append({"type": "text", "text": _string_content(part.get("text"))})
        elif part_type in {"input_image", "image_url"}:
            image_url = part.get("image_url") or part.get("url")
            if image_url:
                parts.append({"type": "image_url", "image_url": image_url})
    if all(part.get("type") == "text" for part in parts if isinstance(part, dict)):
        return "".join(part.get("text", "") for part in parts)
    return parts or ""


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    chat_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        if isinstance(tool.get("function"), dict):
            chat_tools.append(tool)
            continue
        name = tool.get("name")
        if not name:
            continue
        chat_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {},
                },
            }
        )
    return chat_tools


def _chat_completion_to_responses_response(
    data: dict[str, Any],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    response_id = data.get("id") or f"resp_{int(time.time() * 1000)}"
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = _responses_output_items_from_message(message, response_id)
    usage = data.get("usage") or {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": data.get("created") or int(time.time()),
        "status": "completed",
        "model": request_body.get("model") or data.get("model"),
        "output": output,
        "parallel_tool_calls": bool(request_body.get("parallel_tool_calls", True)),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _responses_output_items_from_message(
    message: dict[str, Any],
    response_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    content = message.get("content")
    if content:
        output.append(
            {
                "id": f"{response_id}_msg",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": _string_content(content),
                        "annotations": [],
                    }
                ],
            }
        )
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call.get("function") or {}
        output.append(
            {
                "id": tool_call.get("id") or f"{response_id}_fc_{index}",
                "type": "function_call",
                "status": "completed",
                "call_id": tool_call.get("id") or f"call_{index}",
                "name": function.get("name") or "tool",
                "arguments": function.get("arguments") or "{}",
            }
        )
    return output


async def _stream_responses_response(
    provider,
    payload: dict[str, Any],
    request_body: dict[str, Any],
    config: BridgeConfig,
) -> AsyncIterator[str]:
    response_id = f"resp_{int(time.time() * 1000)}"
    model = request_body.get("model") or payload.get("model")
    message_id = f"{response_id}_msg"
    text_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    text_started = False
    yield _responses_sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        },
    )
    try:
        async for line in provider.stream_chat_completion(payload):
            if not line or not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                yield _responses_sse("error", {"type": "error", "error": chunk["error"]})
                return
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
            text_delta = delta.get("content")
            if text_delta:
                if not text_started:
                    text_started = True
                    yield _responses_sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": message_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        },
                    )
                    yield _responses_sse(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "item_id": message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        },
                    )
                text_parts.append(text_delta)
                yield _responses_sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text_delta,
                    },
                )
            for tool_delta in delta.get("tool_calls") or []:
                index = int(tool_delta.get("index", 0))
                current = tool_calls.setdefault(index, {"arguments": ""})
                if tool_delta.get("id"):
                    current["id"] = tool_delta["id"]
                function = tool_delta.get("function") or {}
                if function.get("name"):
                    current["name"] = function["name"]
                if function.get("arguments"):
                    current["arguments"] += function["arguments"]
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        yield _responses_sse("error", {"type": "error", "error": {"message": message}})
        return
    except httpx.RequestError as exc:
        yield _responses_sse("error", {"type": "error", "error": {"message": str(exc)}})
        return

    output = []
    if text_started:
        text = "".join(text_parts)
        message_item = {
            "id": message_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        output.append(message_item)
        yield _responses_sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
        )
        yield _responses_sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "part": message_item["content"][0],
            },
        )
        yield _responses_sse(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": 0, "item": message_item},
        )
    for tool_index, tool_call in sorted(tool_calls.items()):
        item = {
            "id": tool_call.get("id") or f"{response_id}_fc_{tool_index}",
            "type": "function_call",
            "status": "completed",
            "call_id": tool_call.get("id") or f"call_{tool_index}",
            "name": tool_call.get("name") or "tool",
            "arguments": tool_call.get("arguments") or "{}",
        }
        output.append(item)
        yield _responses_sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": len(output) - 1,
                "item": item,
            },
        )
    yield _responses_sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": model,
                "output": output,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            },
        },
    )
    yield "data: [DONE]\n\n"


def _responses_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _chat_role(role: Any) -> str:
    if role == "developer":
        return "system"
    if role in {"system", "assistant", "tool"}:
        return str(role)
    return "user"


def _string_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _delta_text(delta: dict[str, Any]) -> str:
    return _string_content(delta.get("content"))


def _message_text(message: dict[str, Any]) -> str:
    return _string_content(message.get("content"))


def _message_summary(message: dict[str, Any]) -> dict[str, Any]:
    text = _message_text(message).strip()
    return {
        "role": message.get("role"),
        "text_preview": text[:500],
        "tool_calls": [
            {
                "id": call.get("id"),
                "name": ((call.get("function") or {}).get("name")),
            }
            for call in message.get("tool_calls") or []
            if isinstance(call, dict)
        ],
    }


async def _ollama_web_request(
    app: FastAPI,
    config: BridgeConfig,
    action: str,
    body: dict[str, Any],
) -> JSONResponse:
    provider = _resolve_ollama_web_provider(config)
    cached_provider = app.state.providers[provider.name]
    url = f"{_ollama_api_base_url(provider)}/{action}"
    headers = {"Content-Type": "application/json", **provider.headers}
    api_key = provider.api_key or ""
    if provider.type == "ollama_cloud" and api_key and not api_key.startswith("${"):
        headers["Authorization"] = f"Bearer {api_key}"

    _write_dev_log(
        config,
        f"pi_{action}_request",
        {"provider": provider.name, "url": url, "body": body},
    )
    try:
        response = await cached_provider._client.post(url, headers=headers, json=body)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _safe_response_text(exc.response)
        _write_dev_log(
            config,
            f"pi_{action}_error",
            {"status_code": exc.response.status_code, "body": detail},
        )
        return _upstream_error(exc.response)
    except httpx.RequestError as exc:
        _write_dev_log(config, f"pi_{action}_error", {"message": str(exc)})
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc)}},
        )

    data = response.json()
    _write_dev_log(config, f"pi_{action}_response", data)
    return JSONResponse(data)


def _resolve_ollama_web_provider(config: BridgeConfig) -> ProviderConfig:
    pi_provider = config.providers[config.pi.provider]
    if pi_provider.type in {"ollama", "ollama_local", "ollama_cloud"}:
        return pi_provider

    for provider in config.providers.values():
        if provider.type == "ollama_cloud":
            return provider

    for provider in config.providers.values():
        if provider.type in {"ollama", "ollama_local"}:
            return provider

    raise HTTPException(
        status_code=400,
        detail="No Ollama provider is configured for web_search/web_fetch.",
    )


def _ollama_api_base_url(provider: ProviderConfig) -> str:
    base_url = provider.base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    if base_url.endswith("/api"):
        return base_url
    return f"{base_url}/api"


async def _stream_anthropic_response(
    provider, body: dict[str, Any], upstream_model: str
) -> AsyncIterator[str]:
    try:
        async for chunk in provider.stream_anthropic_message(body, upstream_model):
            yield chunk
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        yield _stream_error_event(message)
        return
    except httpx.RequestError as exc:
        yield _stream_error_event(str(exc))
        return


async def _stream_openai_response(
    provider,
    payload: dict[str, Any],
    config: BridgeConfig,
) -> AsyncIterator[str]:
    available_tool_names = _tool_names(payload.get("tools"))
    tool_name_by_index: dict[int, str] = {}
    try:
        async for line in provider.stream_chat_completion(payload):
            if not line:
                continue
            line = _normalize_streaming_tool_delta(line, available_tool_names, tool_name_by_index)
            yield f"{line}\n\n"
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        _write_dev_log(
            config,
            "pi_response_error",
            {"status_code": exc.response.status_code, "message": message},
        )
        yield f"data: {json.dumps({'error': {'message': message}}, ensure_ascii=True)}\n\n"
        return
    except httpx.RequestError as exc:
        _write_dev_log(config, "pi_response_error", {"message": str(exc)})
        yield f"data: {json.dumps({'error': {'message': str(exc)}}, ensure_ascii=True)}\n\n"
        return


async def _stream_buffered_openai_completion(data: dict[str, Any]) -> AsyncIterator[str]:
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    content = _chat_completion_text(data)
    chunk_id = str(data.get("id") or f"chatcmpl-{uuid.uuid4().hex}")
    created = int(data.get("created") or time.time())
    model = str(data.get("model") or "")
    first_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=True)}\n\n"
    if content:
        content_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(content_chunk, ensure_ascii=True)}\n\n"
    finish_reason = ((data.get("choices") or [{}])[0].get("finish_reason") or "stop")
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if data.get("usage") is not None:
        final_chunk["usage"] = data.get("usage")
    if isinstance(message, dict) and message.get("tool_calls"):
        final_chunk["choices"][0]["delta"]["tool_calls"] = message.get("tool_calls")
    yield f"data: {json.dumps(final_chunk, ensure_ascii=True)}\n\n"
    yield "data: [DONE]\n\n"


def _normalize_streaming_tool_delta(
    line: str,
    available_tool_names: set[str],
    tool_name_by_index: dict[int, str],
) -> str:
    if not available_tool_names or not line.startswith("data:"):
        return line
    raw = line.removeprefix("data:").strip()
    if raw == "[DONE]":
        return line
    try:
        chunk = json.loads(raw)
    except json.JSONDecodeError:
        return line

    changed = False
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for tool_delta in delta.get("tool_calls") or []:
            if not isinstance(tool_delta, dict):
                continue
            try:
                index = int(tool_delta.get("index", 0))
            except (TypeError, ValueError):
                index = 0
            function = tool_delta.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                corrected = _streaming_tool_alias(name, available_tool_names)
                if corrected != name:
                    function["name"] = corrected
                    changed = True
                tool_name_by_index[index] = corrected

            target_name = tool_name_by_index.get(index) or function.get("name")
            arguments = function.get("arguments")
            if isinstance(arguments, str) and target_name:
                rewritten = _rewrite_streaming_tool_arguments(target_name, arguments)
                if rewritten != arguments:
                    function["arguments"] = rewritten
                    changed = True

    if not changed:
        return line
    return f"data: {json.dumps(chunk, ensure_ascii=True)}"


def _streaming_tool_alias(name: str, available_tool_names: set[str]) -> str:
    if name in available_tool_names:
        return name

    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    shell_aliases = {
        "shell_exec",
        "shell_execute",
        "shell",
        "run_cmd",
        "run_command",
        "cmd",
        "command",
        "run_cmd",
    }
    if normalized in shell_aliases:
        if "shell" in available_tool_names:
            return "shell"
        if "llama_bridge_tools__shell.execute" in available_tool_names:
            return "llama_bridge_tools__shell.execute"

    bridge_aliases = {
        "tool_catalog_search": "llama_bridge_tools__tool_catalog_search",
        "tool_schema_get": "llama_bridge_tools__tool_schema_get",
    }
    target = bridge_aliases.get(normalized)
    if target in available_tool_names:
        return target
    return name


def _rewrite_streaming_tool_arguments(tool_name: str, arguments: str) -> str:
    if tool_name == "shell":
        return re.sub(r'(["\'])command\1\s*:', r'\1cmd\1:', arguments)
    if tool_name == "llama_bridge_tools__shell.execute":
        return re.sub(r'(["\'])cmd\1\s*:', r'\1command\1:', arguments)
    return arguments


async def _stream_anthropic_complete_response(
    provider,
    payload: dict[str, Any],
    request_body: dict[str, Any],
) -> AsyncIterator[str]:
    try:
        async for line in provider.stream_chat_completion(payload):
            if not line or not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if raw == "[DONE]":
                yield "data: [DONE]\n\n"
                return
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            completion = {
                "type": "completion",
                "completion": _string_content(delta.get("content")),
                "stop_reason": choice.get("finish_reason"),
                "model": request_body.get("model") or chunk.get("model"),
            }
            yield f"data: {json.dumps(completion, ensure_ascii=True)}\n\n"
        yield "data: [DONE]\n\n"
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        yield f"data: {json.dumps({'error': {'message': message}}, ensure_ascii=True)}\n\n"
    except httpx.RequestError as exc:
        yield f"data: {json.dumps({'error': {'message': str(exc)}}, ensure_ascii=True)}\n\n"


async def _stream_completion_response(
    provider,
    payload: dict[str, Any],
    request_body: dict[str, Any],
    config: BridgeConfig,
) -> AsyncIterator[str]:
    try:
        async for line in provider.stream_chat_completion(payload):
            if not line or not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if raw == "[DONE]":
                yield "data: [DONE]\n\n"
                return
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            completion_chunk = {
                "id": chunk.get("id") or f"cmpl_{int(time.time() * 1000)}",
                "object": "text_completion",
                "created": chunk.get("created") or int(time.time()),
                "model": request_body.get("model") or chunk.get("model"),
                "choices": [
                    {
                        "text": _string_content(delta.get("content")),
                        "index": choice.get("index", 0),
                        "logprobs": None,
                        "finish_reason": choice.get("finish_reason"),
                    }
                ],
            }
            yield f"data: {json.dumps(completion_chunk, ensure_ascii=True)}\n\n"
        yield "data: [DONE]\n\n"
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        _write_dev_log(
            config,
            "completion_response_error",
            {"status_code": exc.response.status_code, "message": message},
        )
        yield f"data: {json.dumps({'error': {'message': message}}, ensure_ascii=True)}\n\n"
    except httpx.RequestError as exc:
        _write_dev_log(config, "completion_response_error", {"message": str(exc)})
        yield f"data: {json.dumps({'error': {'message': str(exc)}}, ensure_ascii=True)}\n\n"


async def _stream_cohere_chat_response(
    provider,
    payload: dict[str, Any],
    request_body: dict[str, Any],
    config: BridgeConfig,
) -> AsyncIterator[str]:
    try:
        async for line in provider.stream_chat_completion(payload):
            if not line or not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if raw == "[DONE]":
                yield "event: stream-end\ndata: {}\n\n"
                return
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = _string_content(delta.get("content"))
            if text:
                payload_data = {"event_type": "text-generation", "text": text}
                yield f"event: text-generation\ndata: {json.dumps(payload_data, ensure_ascii=True)}\n\n"
        yield "event: stream-end\ndata: {}\n\n"
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        _write_dev_log(config, "cohere_chat_error", {"message": message})
        yield f"event: error\ndata: {json.dumps({'message': message}, ensure_ascii=True)}\n\n"
    except httpx.RequestError as exc:
        _write_dev_log(config, "cohere_chat_error", {"message": str(exc)})
        yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=True)}\n\n"


async def _gemini_generate_content(
    app: FastAPI,
    config: BridgeConfig,
    model_id: str,
    body: dict[str, Any],
    stream: bool,
):
    requested_model = model_id.split("/", 1)[-1]
    try:
        resolved = _resolve_bridge_model(requested_model, config)
    except KeyError:
        try:
            resolved = _resolve_bridge_model(body.get("model", requested_model), config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = _with_bridge_tools(
        _gemini_request_to_chat_completion(body, resolved.upstream_model),
        app.state.tools,
        config,
        tool_manager=app.state.tool_manager,
        provider_config=resolved.provider,
    )
    provider = _provider_for(app, resolved)
    if stream:
        _log_streaming_tool_policy(config, "gemini_generate_content", payload)
        return StreamingResponse(
            _safe_stream(_stream_gemini_generate_content(provider, payload, config)),
            media_type="text/event-stream",
        )
    try:
        data = await _chat_completion_with_bridge_tools(app, provider, payload, config)
    except httpx.HTTPStatusError as exc:
        return _upstream_error(exc.response)
    except httpx.RequestError as exc:
        return _request_error(exc)
    return JSONResponse(_chat_completion_to_gemini_response(data))


async def _stream_gemini_generate_content(
    provider,
    payload: dict[str, Any],
    config: BridgeConfig,
) -> AsyncIterator[str]:
    try:
        async for line in provider.stream_chat_completion(payload):
            if not line or not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if raw == "[DONE]":
                return
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = _string_content(delta.get("content"))
            if text:
                event = {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": text}],
                            },
                            "index": choice.get("index", 0),
                        }
                    ]
                }
                yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        _write_dev_log(config, "gemini_generate_error", {"message": message})
        yield f"data: {json.dumps({'error': {'message': message}}, ensure_ascii=True)}\n\n"
    except httpx.RequestError as exc:
        _write_dev_log(config, "gemini_generate_error", {"message": str(exc)})
        yield f"data: {json.dumps({'error': {'message': str(exc)}}, ensure_ascii=True)}\n\n"


async def _stream_ollama_chat_response(
    provider,
    payload: dict[str, Any],
    request_body: dict[str, Any],
    config: BridgeConfig,
) -> AsyncIterator[str]:
    async for event in _stream_ollama_chat_events(provider, payload, request_body, config):
        yield f"{json.dumps(event, ensure_ascii=True)}\n"


async def _stream_ollama_generate_response(
    provider,
    payload: dict[str, Any],
    request_body: dict[str, Any],
    config: BridgeConfig,
) -> AsyncIterator[str]:
    async for event in _stream_ollama_chat_events(provider, payload, request_body, config):
        response_event = {
            "model": event.get("model"),
            "created_at": event.get("created_at"),
            "response": (event.get("message") or {}).get("content", ""),
            "done": event.get("done", False),
        }
        for key in ("done_reason", "prompt_eval_count", "eval_count"):
            if key in event:
                response_event[key] = event[key]
        yield f"{json.dumps(response_event, ensure_ascii=True)}\n"


async def _stream_ollama_chat_events(
    provider,
    payload: dict[str, Any],
    request_body: dict[str, Any],
    config: BridgeConfig,
) -> AsyncIterator[dict[str, Any]]:
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "stop"
    try:
        async for line in provider.stream_chat_completion(payload):
            if not line or not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)
            choice = (chunk.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            message = {
                "role": delta.get("role") or "assistant",
                "content": _string_content(delta.get("content")),
                **_ollama_tool_calls_field(delta.get("tool_calls")),
            }
            if message["content"] or message.get("tool_calls"):
                yield {
                    "model": request_body.get("model") or chunk.get("model"),
                    "created_at": _ollama_timestamp(),
                    "message": message,
                    "done": False,
                }
        yield {
            "model": request_body.get("model") or payload.get("model"),
            "created_at": _ollama_timestamp(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": finish_reason,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        _write_dev_log(
            config,
            "ollama_response_error",
            {"status_code": exc.response.status_code, "message": message},
        )
        yield {"error": message, "done": True}
    except httpx.RequestError as exc:
        _write_dev_log(config, "ollama_response_error", {"message": str(exc)})
        yield {"error": str(exc), "done": True}


async def _shutdown_after_idle(
    app: FastAPI,
    idle_timeout_seconds: int,
    idle_after_file: Path | None = None,
) -> None:
    if idle_after_file is not None:
        while not idle_after_file.exists():
            await asyncio.sleep(5)
        app.state.last_request_at = time.monotonic()

    while True:
        await asyncio.sleep(min(30, idle_timeout_seconds))
        idle_for = time.monotonic() - app.state.last_request_at
        if idle_for >= idle_timeout_seconds:
            # Signal uvicorn to shut down via app state
            app.state.shutdown_requested = True
            # Give current requests time to finish
            await asyncio.sleep(1)
            # Use uvicorn's shutdown mechanism
            import signal as signal_module
            if hasattr(signal_module, 'SIGTERM'):
                os.kill(os.getpid(), signal_module.SIGTERM)
            else:
                # Windows fallback - exit more gracefully
                print(f"\nIdle timeout reached ({idle_timeout_seconds}s). Shutting down...", flush=True)
                os._exit(0)  # Use hard exit on Windows if needed


def _check_auth(
    config: BridgeConfig, x_api_key: str | None, authorization: str | None
) -> None:
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    accepted_tokens = [str(config.server.auth_token or "")]
    if config.server.host in {"127.0.0.1", "localhost", "::1"}:
        accepted_tokens.append("ollama")
    if config.server.auth_token and not any(
        candidate and hmac.compare_digest(token or "", candidate)
        for candidate in accepted_tokens
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _check_ollama_auth(
    config: BridgeConfig, x_api_key: str | None, authorization: str | None
) -> None:
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    if token:
        _check_auth(config, x_api_key, authorization)


def _check_openai_compat_auth(
    config: BridgeConfig,
    body: dict[str, Any],
    x_api_key: str | None,
    authorization: str | None,
) -> None:
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    if token:
        _check_auth(config, x_api_key, authorization)
        return
    requested_model = body.get("model")
    if any(requested_model in {model.name, model.model} for model in config.vs_copilot_models):
        return
    _check_auth(config, x_api_key, authorization)


def _check_tools_auth(
    config: BridgeConfig, x_api_key: str | None, authorization: str | None
) -> None:
    if config.tools.require_auth:
        _check_auth(config, x_api_key, authorization)
    else:
        _check_ollama_auth(config, x_api_key, authorization)


async def _call_bridge_tool(
    app: FastAPI,
    name: str,
    arguments: dict[str, Any],
) -> JSONResponse:
    if name not in app.state.tools._tools:
        available = ", ".join(sorted(app.state.tools._tools)) or "none"
        unavailable_tools = app.state.tools.unavailable_tools()
        detail = f"Unknown tool '{name}'. Available tools: {available}"
        if unavailable_tools:
            unavailable = ", ".join(
                f"{tool_name} ({reason})"
                for tool_name, reason in sorted(unavailable_tools.items())
            )
            detail = f"{detail}. Unavailable tools: {unavailable}"
        _write_dev_log(
            app.state.bridge_config,
            "direct_tool_unknown",
            {
                "tool": name,
                "arguments": arguments,
                "available_tools": sorted(app.state.tools._tools),
                "unavailable_tools": unavailable_tools,
            },
        )
        raise HTTPException(
            status_code=404,
            detail=detail,
        )
    structured = await app.state.tools.call_structured(name, arguments)
    _write_dev_log(
        app.state.bridge_config,
        "direct_tool_call",
        {
            "tool": name,
            "ok": structured.get("ok"),
            "arguments": arguments,
            "error": structured.get("error"),
            "duration_ms": structured.get("duration_ms"),
        },
    )
    return JSONResponse(
        status_code=200,
        content={
            "tool": name,
            "result": structured,
            "data": structured.get("data"),
            "ok": structured.get("ok"),
        },
    )


def _upstream_error(response: httpx.Response) -> JSONResponse:
    try:
        detail = response.json()
    except Exception:
        detail = {"message": response.text}
    return JSONResponse(status_code=response.status_code, content={"error": detail})


def _request_error(exc: httpx.RequestError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": {"message": str(exc)}},
    )


def _write_dev_log(config: BridgeConfig | None, event: str, payload: Any) -> None:
    if not DEV_LOG_ENABLED:
        return
    if config is None:
        return
    try:
        log_path = config.source_path.parent / "llama.dev.log"
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "payload": _sanitize_dev_payload(config, payload),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except Exception:
        return


def _sanitize_dev_payload(config: BridgeConfig, payload: Any) -> Any:
    return _sanitize_dev_value(
        payload,
        log_outputs=bool(getattr(config.tools, "log_outputs", False)),
    )


def _sanitize_dev_value(value: Any, *, log_outputs: bool, key: str | None = None, depth: int = 0) -> Any:
    if key and key.lower() in {"api_key", "apikey", "authorization", "x-api-key", "auth_token", "token"}:
        return "<redacted>"
    if depth >= 8:
        return "<nested value omitted>"
    if isinstance(value, str):
        value = _redact_secret_text(value)
        limit = 300 if key != "content" else 700
        if len(value) <= limit:
            return value
        return f"{value[:limit]}... <truncated {len(value) - limit} chars>"
    if isinstance(value, list):
        limit = 20 if key in {"tools", "available_tools", "selected_tools"} else 8
        items = [
            _sanitize_dev_value(item, log_outputs=log_outputs, depth=depth + 1)
            for item in value[:limit]
        ]
        if len(value) > limit:
            items.append(f"<{len(value) - limit} more items>")
        return items
    if isinstance(value, dict):
        if key in {"result", "data"} and not log_outputs:
            return _tool_result_summary(value)
        sanitized: dict[str, Any] = {}
        for index, (item_key, item_value) in enumerate(value.items()):
            if index >= 50:
                sanitized["<more>"] = f"{len(value) - 50} more keys"
                break
            sanitized[str(item_key)] = _sanitize_dev_value(
                item_value,
                log_outputs=log_outputs,
                key=str(item_key),
                depth=depth + 1,
            )
        return sanitized
    return value


def _redact_secret_text(value: str) -> str:
    value = re.sub(r"(?i)(api[_-]?key|authorization|token)=([^&\s]+)", r"\1=<redacted>", value)
    value = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", value)
    return value


def _tool_result_summary(value: dict[str, Any]) -> dict[str, Any]:
    data = value.get("data") if isinstance(value.get("data"), dict) else value
    summary: dict[str, Any] = {
        "ok": value.get("ok"),
        "tool": value.get("tool"),
        "error": value.get("error"),
        "duration_ms": value.get("duration_ms"),
    }
    if isinstance(data, dict):
        for key in ("query", "source", "verdict", "timestamp"):
            if data.get(key) is not None:
                summary[key] = data.get(key)
        for key in ("results", "organic_results", "images", "verified_sources"):
            if isinstance(data.get(key), list):
                summary[f"{key}_count"] = len(data[key])
    return {key: item for key, item in summary.items() if item is not None}


async def _stream_response(
    provider,
    payload: dict[str, Any],
    requested_model: str,
    config: BridgeConfig | None = None,
) -> AsyncIterator[str]:
    for event in anthropic_stream_prefix(requested_model):
        yield event

    text_started = False
    text_index: int | None = None
    next_content_index = 0
    tool_indices: dict[int, int] = {}
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"

    try:
        async for line in provider.stream_chat_completion(payload):
            if not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if raw.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                yield _stream_error_event(json.dumps(chunk["error"], ensure_ascii=True))
                return

            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})

            text_delta = _delta_text(delta)
            if not text_started and text_delta:
                text_started = True
                text_index = next_content_index
                next_content_index += 1
                yield sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )

            if text_delta:
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": text_delta},
                    },
                )

            for tool_delta in delta.get("tool_calls") or []:
                index = tool_delta.get("index", 0)
                if index not in tool_indices:
                    tool_indices[index] = next_content_index
                    next_content_index += 1
                    function_name = (
                        (tool_delta.get("function") or {}).get("name") or "tool"
                    )
                    tool_id = tool_delta.get("id") or f"toolu_{index}"
                    yield sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": tool_indices[index],
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": function_name,
                                "input": {},
                            },
                        },
                    )
                arguments = (tool_delta.get("function") or {}).get("arguments")
                if arguments:
                    yield sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_indices[index],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        },
                    )
                    stop_reason = "tool_use"

            finish_reason = choice.get("finish_reason")
            if finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif finish_reason == "length":
                stop_reason = "max_tokens"

            usage = chunk.get("usage") or {}
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)
    except httpx.HTTPStatusError as exc:
        message = await _response_error_message(exc.response)
        if config is not None:
            _write_dev_log(
                config,
                "anthropic_stream_error",
                {
                    "status_code": exc.response.status_code,
                    "body": _safe_response_text(exc.response),
                    "requested_model": requested_model,
                    "upstream_model": payload.get("model"),
                },
            )
        yield _stream_error_event(message)
        return
    except httpx.RequestError as exc:
        if config is not None:
            _write_dev_log(
                config,
                "anthropic_stream_error",
                {
                    "message": str(exc),
                    "requested_model": requested_model,
                    "upstream_model": payload.get("model"),
                },
            )
        yield _stream_error_event(str(exc))
        return

    if text_started and text_index is not None:
        yield sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": text_index},
        )
    for index in tool_indices.values():
        yield sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )

    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield sse_event(
        "message_stop",
        {
            "type": "message_stop",
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )


async def _response_error_message(response: httpx.Response) -> str:
    try:
        await response.aread()
    except Exception:
        pass

    try:
        payload = response.json()
    except Exception:
        message = _safe_response_text(response)
        if message:
            return message
        reason = response.reason_phrase or "upstream error"
        return f"{response.status_code} {reason}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return json.dumps(error, ensure_ascii=True)
        return json.dumps(payload, ensure_ascii=True)
    return str(payload)


def _safe_response_text(response: httpx.Response) -> str:
    try:
        return response.text
    except httpx.ResponseNotRead:
        return ""


def _stream_error_event(message: str) -> str:
    return sse_event(
        "error",
        {
            "type": "error",
            "error": {
                "type": "upstream_error",
                "message": message,
            },
        },
    )


async def _run_embedded_telegram_bot(app: FastAPI) -> None:
    config = app.state.bridge_config
    try:
        from .teligram import TeligramBot
    except Exception:
        LOGGER.exception("Telegram bot skipped: could not load Teligram runtime")
        return

    try:
        bot = TeligramBot(
            config,
            provider=app.state.providers.get(config.telegram.provider),
            tools=app.state.tools,
        )
    except Exception:
        LOGGER.exception("Telegram bot skipped: Teligram initialization failed")
        return

    LOGGER.info("Telegram bot worker started with Teligram runtime")
    try:
        await bot.poll_forever()
    except asyncio.CancelledError:
        LOGGER.info("Telegram bot worker stopped")
        raise
    except Exception:
        LOGGER.exception("Telegram bot worker crashed")
    finally:
        await bot.aclose()


async def _handle_embedded_telegram_update(
    app: FastAPI,
    client: httpx.AsyncClient,
    telegram: Any,
    server_base_url: str,
    headers: dict[str, str],
    model: str,
    bridge_timeout: httpx.Timeout,
    update: dict[str, Any],
) -> None:
    message = update.get("message") or update.get("edited_message") or {}
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    chat_username = str(chat.get("username") or "").strip()
    text = str(message.get("text") or "").strip()
    if not chat_id or not text:
        return
    app.state.last_request_at = time.monotonic()

    allowed_chat_matches = {chat_id}
    if chat_username:
        allowed_chat_matches.add(chat_username)
        allowed_chat_matches.add(f"@{chat_username.lstrip('@')}")
    if telegram.allowed_chat_ids and not any(
        str(allowed).strip() in allowed_chat_matches for allowed in telegram.allowed_chat_ids
    ):
        await _telegram_send_message(client, telegram.bot_token or "", chat_id, "This chat is not allowed.")
        return
    _write_telegram_last_chat(app.state.bridge_config, chat_id, chat_username)

    canned_response = _telegram_canned_response(text)
    if canned_response is not None:
        await _telegram_send_message(client, telegram.bot_token or "", chat_id, canned_response)
        return

    pending_command = _telegram_pending_command(app.state.bridge_config, chat_id)
    command_text = text
    if pending_command is not None and not text.startswith("/"):
        command_text = f"/{pending_command} {text}"
        _clear_telegram_pending_command(app.state.bridge_config, chat_id)

    command_response, body, restart_requested = _telegram_command_response(
        app,
        telegram.system_prompt,
        command_text,
        model,
        chat_id,
    )
    if command_response is not None:
        await _telegram_send_message(client, telegram.bot_token or "", chat_id, command_response)
        if restart_requested:
            asyncio.create_task(_telegram_restart_server_process(app.state.bridge_config))
        return

    if body is None:
        body = {
            "model": model,
            "messages": _telegram_messages(telegram.system_prompt, text[: telegram.max_input_chars]),
            "max_tokens": telegram.max_output_tokens,
            "stream": False,
        }
    progress_steps = _telegram_progress_steps(body.get("_telegram_mode"))
    typing_task = asyncio.create_task(
        _telegram_typing_loop(client, telegram.bot_token or "", chat_id)
    )
    try:
        for step in progress_steps:
            await _telegram_send_message(client, telegram.bot_token or "", chat_id, step)
        response = await client.post(
            f"{server_base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=bridge_timeout,
        )
        response.raise_for_status()
        content = _chat_completion_text(response.json()) or "I couldn't produce a reply."
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Telegram bridge request failed")
        content = f"Bridge request failed: {exc}"
    finally:
        typing_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await typing_task

    await _telegram_send_message(
        client,
        telegram.bot_token or "",
        chat_id,
        _telegram_polish_text(content)[:3500],
    )


def _telegram_messages(system_prompt: str, user_text: str) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": user_text}]
    styled_system_prompt = _telegram_style_system_prompt(system_prompt)
    if styled_system_prompt.strip():
        messages.insert(0, {"role": "system", "content": styled_system_prompt})
    return messages


async def _telegram_send_message(
    client: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    text: str,
) -> None:
    response = await client.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
    )
    response.raise_for_status()


async def _telegram_send_chat_action(
    client: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    action: str = "typing",
) -> None:
    response = await client.post(
        f"https://api.telegram.org/bot{bot_token}/sendChatAction",
        json={"chat_id": chat_id, "action": action},
    )
    response.raise_for_status()


async def _telegram_typing_loop(
    client: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    interval_seconds: float = 4.0,
) -> None:
    while True:
        with contextlib.suppress(Exception):
            await _telegram_send_chat_action(client, bot_token, chat_id, "typing")
        await asyncio.sleep(interval_seconds)


async def _telegram_set_my_commands(client: httpx.AsyncClient, bot_token: str) -> None:
    if not bot_token:
        return
    response = await client.post(
        f"https://api.telegram.org/bot{bot_token}/setMyCommands",
        json={"commands": _telegram_command_definitions()},
    )
    response.raise_for_status()


def _telegram_command_definitions() -> list[dict[str, str]]:
    return [
        {"command": "help", "description": "Show available commands"},
        {"command": "status", "description": "Show bot and server status"},
        {"command": "restart", "description": "Restart the llama server"},
        {"command": "web", "description": "Web search a query"},
        {"command": "search", "description": "Search the web"},
        {"command": "deep", "description": "Run deeper research"},
        {"command": "deepresearch", "description": "Run deeper research"},
        {"command": "summarize", "description": "Summarize text or a topic"},
        {"command": "explain", "description": "Explain a topic clearly"},
        {"command": "clear", "description": "Reset context for the next message"},
    ]


def _telegram_canned_response(user_text: str) -> str | None:
    normalized = re.sub(r"[^\w\s]", "", user_text.lower()).strip()
    compact = " ".join(normalized.split())

    if user_text.strip() == "/":
        return _telegram_help_text()

    greeting_messages = {
        "hi",
        "hii",
        "hiii",
        "hello",
        "hey",
        "yo",
        "hola",
        "namaste",
        "good morning",
        "good afternoon",
        "good evening",
    }
    thanks_messages = {
        "thanks",
        "thank you",
        "thx",
        "ty",
    }
    bye_messages = {
        "bye",
        "goodbye",
        "see you",
        "see ya",
        "tc",
        "take care",
    }

    if compact in greeting_messages:
        return (
            "Hello! I'm Zara.\n\n"
            "I can help with research, summaries, explanations, coding questions, and quick fact checks. "
            "Send me what you need, and I'll keep it clear and concise."
        )
    if compact in thanks_messages:
        return "You're welcome. If you want, send the next question and I'll help with that too."
    if compact in bye_messages:
        return "Take care. I'm here whenever you need help again."
    return None


def _telegram_style_system_prompt(system_prompt: str) -> str:
    style_rules = (
        "For Telegram replies, sound like a modern assistant. "
        "Write in clean plain text with natural phrasing. "
        "Do not use Markdown markers like *, **, _, #, >, or backticks. "
        "Avoid raw bullet syntax like '* item'. "
        "Use short paragraphs, and only use the '•' bullet when it truly helps readability. "
        "Keep greetings and simple replies warm, natural, and brief."
    )
    base = system_prompt.strip()
    if not base:
        return style_rules
    return f"{base}\n\n{style_rules}"


def _telegram_polish_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*\n]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"_([^_\n]+)_", r"\1", cleaned)
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*]\s+", "• ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned or "I couldn't produce a reply."


def _telegram_command_response(
    app: FastAPI,
    system_prompt: str,
    text: str,
    model: str,
    chat_id: str,
) -> tuple[str | None, dict[str, Any] | None, bool]:
    parsed = _telegram_parse_command(text)
    if parsed is None:
        return None, None, False

    command, argument = parsed
    if command == "start":
        return _telegram_start_text(), None, False
    if command in {"help", "commands"}:
        return _telegram_help_text(), None, False
    if command == "status":
        idle_for = max(0, int(time.monotonic() - app.state.last_request_at))
        return (
            "Llama server is running.\n\n"
            f"Model: {model}\n"
            f"Idle timer: active\n"
            f"Last activity: {idle_for} seconds ago\n\n"
            "Use /help to see available commands."
        ), None, False
    if command == "restart":
        return "Restarting the llama server now. Give me a few seconds, then send your next message.", None, True
    if command == "clear":
        _clear_telegram_pending_command(app.state.bridge_config, chat_id)
        return (
            "Done. I cleared the pending command and reset the next request."
        ), None, False
    if command in {"web", "search"}:
        if not argument:
            _set_telegram_pending_command(app.state.bridge_config, chat_id, command)
            return (
                "Web search mode is ready.\n\n"
                "Send the query you want me to search for."
            ), None, False
        return None, _telegram_command_body(
            system_prompt,
            model,
            argument,
            (
                "Use web or search tools when needed. "
                "Return a crisp, well-structured answer in plain text for Telegram. "
                "Prefer current information and mention sources naturally when relevant."
            ),
            telegram_mode="web",
        ), False
    if command in {"deep", "deepresearch", "research"}:
        if not argument:
            _set_telegram_pending_command(app.state.bridge_config, chat_id, "deepresearch")
            return (
                "Deep research mode is ready.\n\n"
                "Send the topic, question, or problem you want me to research."
            ), None, False
        return None, _telegram_command_body(
            system_prompt,
            model,
            argument,
            (
                "Do a deeper research-style answer. "
                "Use tools when helpful, verify current facts, and organize the reply clearly with concise sections. "
                "Keep the writing clean and plain-text for Telegram."
            ),
            max_tokens= min(1200, max(700, 2 * 512)),
            telegram_mode="deepresearch",
        ), False
    if command == "summarize":
        if not argument:
            _set_telegram_pending_command(app.state.bridge_config, chat_id, command)
            return "Summarize mode is ready.\n\nSend the text or topic you want summarized.", None, False
        return None, _telegram_command_body(
            system_prompt,
            model,
            argument,
            "Summarize this clearly in plain text for Telegram. Keep only the important points.",
            telegram_mode="summarize",
        ), False
    if command == "explain":
        if not argument:
            _set_telegram_pending_command(app.state.bridge_config, chat_id, command)
            return "Explain mode is ready.\n\nSend the topic you want me to explain.", None, False
        return None, _telegram_command_body(
            system_prompt,
            model,
            argument,
            "Explain this clearly like a modern assistant. Use simple wording and practical examples when helpful.",
            telegram_mode="explain",
        ), False

    return (
        "Unknown command.\n\nUse /help to see the available Telegram commands.",
        None,
        False,
    )


def _telegram_command_body(
    system_prompt: str,
    model: str,
    user_text: str,
    task_instruction: str,
    max_tokens: int = 512,
    telegram_mode: str | None = None,
) -> dict[str, Any]:
    command_prompt = (
        f"{task_instruction}\n\n"
        f"User request:\n{user_text.strip()}"
    )
    body = {
        "model": model,
        "messages": _telegram_messages(system_prompt, command_prompt),
        "max_tokens": max_tokens,
        "stream": False,
    }
    if telegram_mode:
        body["_telegram_mode"] = telegram_mode
    return body


def _telegram_parse_command(text: str) -> tuple[str, str] | None:
    match = re.match(r"^\s*/([a-zA-Z0-9_]+)(?:@[a-zA-Z0-9_]+)?(?:\s+(.*))?\s*$", text)
    if not match:
        return None
    command = match.group(1).lower()
    argument = (match.group(2) or "").strip()
    return command, argument


def _telegram_help_text() -> str:
    return (
        "Available commands\n\n"
        "/help - Show this command list\n"
        "/status - Show current bot/server status\n"
        "/restart - Restart the llama server\n"
        "/web <query> - Do a web-backed search style answer\n"
        "/search <query> - Same as /web\n"
        "/deep <topic> - Run a deeper research-style answer\n"
        "/summarize <text> - Summarize text or a topic\n"
        "/explain <topic> - Explain something clearly\n"
        "/clear - Reset context for the next message\n\n"
        "You can also just type normally without any command."
    )


def _telegram_start_text() -> str:
    return (
        "Hello! I'm Zara.\n\n"
        "I can help with web search, deep research, summaries, explanations, coding questions, and quick fact checks.\n\n"
        "Try one of these:\n"
        "/web latest AI news\n"
        "/deep compare OpenAI and Anthropic models\n"
        "/summarize paste text here\n"
        "/explain how RAG works\n\n"
        "Use /help to see the full command list."
    )


def _telegram_state_path(config: BridgeConfig) -> Path:
    return config.source_path.parent / "llama.telegram.json"


def _read_telegram_state(config: BridgeConfig) -> dict[str, Any]:
    state_path = _telegram_state_path(config)
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_telegram_state(config: BridgeConfig, payload: dict[str, Any]) -> None:
    _telegram_state_path(config).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _write_telegram_last_chat(config: BridgeConfig, chat_id: str, chat_username: str) -> None:
    payload = _read_telegram_state(config)
    chats = payload.setdefault("chats", {})
    chat_entry = chats.setdefault(chat_id, {})
    chat_entry["chat_id"] = chat_id
    chat_entry["chat_username"] = chat_username
    chat_entry["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload["chat_id"] = chat_id
    payload["chat_username"] = chat_username
    payload["updated_at"] = chat_entry["updated_at"]
    _write_telegram_state(config, payload)


def _set_telegram_pending_command(config: BridgeConfig, chat_id: str, command: str) -> None:
    payload = _read_telegram_state(config)
    chats = payload.setdefault("chats", {})
    chat_entry = chats.setdefault(chat_id, {"chat_id": chat_id})
    chat_entry["pending_command"] = command
    chat_entry["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    _write_telegram_state(config, payload)


def _clear_telegram_pending_command(config: BridgeConfig, chat_id: str) -> None:
    payload = _read_telegram_state(config)
    chats = payload.get("chats") or {}
    chat_entry = chats.get(chat_id)
    if not isinstance(chat_entry, dict):
        return
    chat_entry.pop("pending_command", None)
    chat_entry["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    _write_telegram_state(config, payload)


def _telegram_pending_command(config: BridgeConfig, chat_id: str) -> str | None:
    payload = _read_telegram_state(config)
    chats = payload.get("chats") or {}
    chat_entry = chats.get(chat_id)
    if not isinstance(chat_entry, dict):
        return None
    command = str(chat_entry.get("pending_command") or "").strip().lower()
    return command or None


def _telegram_progress_steps(mode: Any) -> list[str]:
    # Keep Telegram chats quiet: typing indicators are enough while work runs.
    return []


async def _telegram_restart_server_process(config: BridgeConfig) -> None:
    await asyncio.sleep(1.0)
    LOGGER.warning("Restarting llama server process on Telegram command")
    config_path = config.source_path
    pid_path = config_path.parent / DEFAULT_PID_PATH.name
    log_path = config_path.parent / DEFAULT_LOG_PATH.name
    helper_code = (
        "import subprocess, sys, time;"
        "python_exe, config_path, pid_path, log_path, cwd = sys.argv[1:];"
        "subprocess.run([python_exe, '-m', 'llama_bridge', 'stop', '--pid-file', pid_path], cwd=cwd, check=False);"
        "deadline=time.time()+20.0;"
        "while time.time()<deadline and __import__('pathlib').Path(pid_path).exists(): time.sleep(0.25);"
        "time.sleep(1.0);"
        "subprocess.run([python_exe, '-m', 'llama_bridge', 'start', '--config', config_path, '--pid-file', pid_path, '--log-file', log_path], cwd=cwd, check=False)"
    )
    popen_kwargs: dict[str, Any] = {
        "cwd": str(config_path.parent),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        )
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            helper_code,
            sys.executable,
            str(config_path),
            str(pid_path),
            str(log_path),
            str(config_path.parent),
        ],
        **popen_kwargs,
    )


def _chat_completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _local_server_url(host: str, port: int) -> str:
    local_host = host
    if host in {"0.0.0.0", "::"}:
        local_host = "127.0.0.1"
    return f"http://{local_host}:{port}"
