from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import DEFAULT_CONFIG_PATH, ToolConfig, load_config


PROTOCOL_VERSION = "2025-06-18"
class SubagentManager:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def spawn(self, topic: str, agent_names: list[str]) -> dict[str, Any]:
        session_id = f"sa_{uuid.uuid4().hex[:12]}"
        assignments = self._select_assignments(agent_names)
        session = {
            "session_id": session_id,
            "topic": topic,
            "status": "active",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "agent_count": len(agent_names),
            "agents": assignments,
            "selection_strategy": "dynamic_limit_aware_from_env_yml",
        }
        self._sessions[session_id] = session
        return dict(session)

    def mark(self, session_id: str, *, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown subagent session: {session_id}")
        session["status"] = status
        session["updated_at"] = int(time.time())
        if extra:
            session.update(extra)
        return dict(session)

    def ensure(self, session_id: str, topic: str, agent_names: list[str]) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is not None:
            return dict(session)
        assignments = self._select_assignments(agent_names)
        session = {
            "session_id": session_id,
            "topic": topic,
            "status": "active",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "agent_count": len(agent_names),
            "agents": assignments,
            "selection_strategy": "dynamic_limit_aware_from_env_yml",
        }
        self._sessions[session_id] = session
        return dict(session)

    def kill(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown subagent session: {session_id}")
        session["status"] = "killed"
        session["updated_at"] = int(time.time())
        return dict(session)

    def status(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown subagent session: {session_id}")
            return {"sessions": [dict(session)]}
        sessions = sorted(self._sessions.values(), key=lambda item: int(item.get("created_at", 0)), reverse=True)
        return {"sessions": [dict(item) for item in sessions]}

    def get(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown subagent session: {session_id}")
        return dict(session)

    def _select_assignments(self, agent_names: list[str]) -> list[dict[str, Any]]:
        try:
            config = load_config(DEFAULT_CONFIG_PATH)
        except Exception as exc:  # noqa: BLE001
            return [
                {
                    "agent": name,
                    "provider": None,
                    "model": None,
                    "alias": None,
                    "endpoint": None,
                    "status": f"env.yml unavailable: {exc}",
                }
                for name in agent_names
            ]

        pool = _provider_model_pool(config)
        if not pool:
            return [
                {
                    "agent": name,
                    "provider": None,
                    "model": None,
                    "alias": None,
                    "endpoint": None,
                    "status": "no eligible provider/model entries found in env.yml",
                }
                for name in agent_names
            ]

        used_counts: dict[str, int] = {}
        assignments: list[dict[str, Any]] = []
        for name in agent_names:
            role = _subagent_role(name)
            candidates = _rank_provider_candidates(pool, role=role, used_counts=used_counts)
            selected = dict(candidates[0] if candidates else pool[0])
            key = f"{selected.get('provider')}::{selected.get('model')}"
            used_counts[key] = used_counts.get(key, 0) + 1
            selected.update(
                {
                    "agent": name,
                    "role": role,
                    "endpoint": selected.get("base_url"),
                    "selection_strategy": "dynamic_limit_aware_from_env_yml",
                }
            )
            assignments.append(selected)
        return assignments


_SUBAGENT_MANAGER = SubagentManager()


def main() -> None:
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
        tools_config = config.tools
    except Exception:
        tools_config = ToolConfig()
    server = BridgeMcpServer(
        base_url=os.environ.get("LLAMA_BRIDGE_BASE_URL", "http://127.0.0.1:8089").rstrip("/"),
        api_key=os.environ.get("LLAMA_BRIDGE_API_KEY", ""),
        tools_config=tools_config,
    )
    server.run()


class BridgeMcpServer:
    def __init__(self, base_url: str, api_key: str, tools_config: ToolConfig | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.tools_config = tools_config or ToolConfig()

    def _timeout_for_tool(self, name: str | None = None) -> float:
        deep = self.tools_config.deep_research
        if name in {"deep_tavily_agent", "deep_serp_agent", "deep_wiki_agent", "tavily_search", "serpapi_search", "wikipedia_search"}:
            return deep.search_agent_timeout_seconds
        if name in {"deep_verify_agent", "verify_sources"}:
            return deep.verify_agent_timeout_seconds
        if name == "source_research":
            return deep.source_research_timeout_seconds
        if name in {"image_research", "deep_image_agent"}:
            return deep.image_research_timeout_seconds
        if name == "manim_render":
            return deep.manim_timeout_seconds
        return deep.tool_timeout_seconds

    def run(self) -> None:
        while True:
            message = _read_message(sys.stdin.buffer)
            if message is None:
                return
            response = self._handle(message)
            if response is not None:
                _write_message(sys.stdout.buffer, response)

    def _handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        message_id = message.get("id")
        try:
            if method == "initialize":
                params = message.get("params") or {}
                protocol_version = (
                    params.get("protocolVersion")
                    if isinstance(params, dict) and isinstance(params.get("protocolVersion"), str)
                    else PROTOCOL_VERSION
                )
                return _result(
                    message_id,
                    {
                        "protocolVersion": protocol_version,
                        "capabilities": {"tools": {}, "prompts": {}},
                        "serverInfo": {"name": "llama-bridge-tools", "version": "0.1.0"},
                    },
                )
            if method == "tools/list":
                return _result(message_id, {"tools": self._list_tools()})
            if method == "tools/call":
                params = message.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("tools/call params must be an object")
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if not isinstance(name, str) or not name:
                    raise ValueError("tools/call requires a tool name")
                if not isinstance(arguments, dict):
                    raise ValueError("tools/call arguments must be an object")
                return _result(message_id, self._call_tool(name, arguments))
            if method == "prompts/list":
                return _result(message_id, {"prompts": _prompt_definitions()})
            if method == "prompts/get":
                params = message.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("prompts/get params must be an object")
                return _result(message_id, _prompt_response(str(params.get("name") or ""), params))
            if method and method.startswith("notifications/"):
                return None
            if message_id is None:
                return None
            return _error(message_id, -32601, f"Unsupported MCP method: {method}")
        except Exception as exc:  # noqa: BLE001 - MCP clients need structured tool errors.
            if message_id is None:
                return None
            return _error(message_id, -32000, str(exc))

    def _list_tools(self) -> list[dict[str, Any]]:
        # First try to get tools with full schemas
        try:
            data = self._request("GET", "/api/tools?full_schema=true")
            tools = data.get("tools") or data.get("data") or []

            # If that didn't work, fall back to regular endpoint
            if not tools:
                data = self._request("GET", "/api/tools")
                tools = data.get("tools") or data.get("data") or []
        except Exception:
            tools = []

        if not isinstance(tools, list):
            tools = []

        result = []
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue

            # Get parameters - if missing, try to get from individual endpoint
            params = tool.get("parameters")
            if not params and tool.get("name"):
                try:
                    detail = self._request("GET", f"/api/tools/{tool['name']}/schema")
                    if detail.get("ok") and detail.get("tool"):
                        params = detail["tool"].get("parameters")
                except Exception:
                    params = {"type": "object"}

            result.append({
                "name": str(tool.get("name", "")),
                "description": str(tool.get("description", "")),
                "inputSchema": params or {"type": "object"},
            })
        known_names = {tool["name"] for tool in result}
        for tool in _virtual_tools():
            if tool["name"] not in known_names:
                result.append(tool)
        return result

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "deep":
            return _deep_research_handoff(arguments)
        if name == "manim_render":
            return _call_manim_render(self, arguments)
        if name == "subagent_spawn":
            return _call_subagent_spawn(arguments)
        if name == "subagent_kill":
            return _call_subagent_kill(arguments)
        if name == "subagent_status":
            return _call_subagent_status(arguments)
        if name == "deep_lead_agent":
            return _call_deep_lead_agent(self, arguments)
        if name == "deep_plan_agent":
            return _call_deep_plan_agent(arguments)
        if name == "deep_collect_agent":
            return _call_deep_collect_agent(self, arguments)
        if name == "deep_review_agent":
            return _call_deep_review_agent(self, arguments)
        if name == "deep_master_review_agent":
            return _call_deep_master_review_agent(arguments)
        if name in _DEEP_MASTER_REVIEW_TOOL_NAMES:
            return _call_deep_master_review_subagent(name, arguments)
        if name == "deep_image_agent":
            return _call_deep_image_agent(self, arguments)
        if name in {"deep_tavily_agent", "deep_serp_agent", "deep_wiki_agent", "deep_verify_agent"}:
            return self._call_deep_agent(name, arguments)
        name, arguments = _normalize_tool_call(name, arguments)
        if name == "source_research":
            arguments = _poolside_source_research_arguments(arguments)
        try:
            data = self._request("POST", f"/api/tools/{name}", arguments, timeout_seconds=self._timeout_for_tool(name))
        except Exception as exc:  # noqa: BLE001 - MCP tools should fail fast with a JSON answer.
            data = {"ok": False, "tool": name, "error": str(exc), "timeout_budget_seconds": self._timeout_for_tool(name)}
        is_error = bool(data.get("ok") is False)
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    def _call_deep_agent(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "deep_verify_agent":
            try:
                data = self._request(
                    "POST",
                    "/api/tools/verify_sources",
                    _deep_verify_arguments(arguments),
                    timeout_seconds=self._timeout_for_tool(name),
                )
            except Exception as exc:  # noqa: BLE001 - return timeout/error details as the agent brief.
                data = {"ok": False, "tool": name, "error": str(exc), "timeout_budget_seconds": self._timeout_for_tool(name)}
            result = _compact_verify_agent_result(data)
            agent_name = str(arguments.get("agent_name") or "").strip()
            if agent_name:
                result["assigned_agent_name"] = agent_name
            text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
            return {"content": [{"type": "text", "text": text}], "isError": bool(data.get("ok") is False)}

        topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
        focus = str(arguments.get("focus") or "").strip()
        count = 1
        queries = _deep_agent_queries(topic, focus, count=count, agent=name)
        provider = {
            "deep_tavily_agent": "tavily_search",
            "deep_serp_agent": "serpapi_search",
            "deep_wiki_agent": "wikipedia_search",
        }[name]
        collected: list[dict[str, Any]] = []
        errors: list[str] = []
        for query in queries:
            try:
                payload = _deep_agent_payload(provider, query)
                data = self._request("POST", f"/api/tools/{provider}", payload, timeout_seconds=self._timeout_for_tool(name))
                if data.get("ok") is False:
                    errors.append(f"{query}: {data.get('error') or data}")
                    continue
                collected.extend(_compact_search_sources(provider, query, data))
            except Exception as exc:  # noqa: BLE001 - return agent errors as evidence metadata.
                errors.append(f"{query}: {exc}")

        result = _compact_deep_agent_result(
            agent=name,
            provider=provider,
            topic=topic,
            focus=focus,
            queries=queries,
            sources=collected,
            errors=errors,
        )
        agent_name = str(arguments.get("agent_name") or "").strip()
        if agent_name:
            result["assigned_agent_name"] = agent_name
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        payload = None
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=payload, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout_seconds or self._timeout_for_tool()) as response:  # noqa: S310 - user-configured local bridge URL.
                text = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama bridge {path} failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"could not reach llama bridge at {self.base_url}: {exc}") from exc
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"llama bridge returned non-JSON data from {path}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"llama bridge returned unexpected data from {path}")
        return data


def _read_message(stream) -> dict[str, Any] | None:
    line = stream.readline()
    if not line:
        return None
    line_text = line.decode("utf-8", errors="replace").strip()
    if line_text.startswith("{"):
        message = json.loads(line_text)
        if isinstance(message, dict):
            return message
        return None

    headers: dict[str, str] = {}
    while line:
        header_text = line.decode("ascii", errors="replace").strip()
        if not header_text:
            break
        name, _, value = header_text.partition(":")
        headers[name.lower()] = value.strip()
        line = stream.readline()

    length_text = headers.get("content-length")
    if not length_text:
        return None
    body = stream.read(int(length_text))
    message = json.loads(body.decode("utf-8")) if body else None
    if isinstance(message, dict):
        return message
    return None


def _write_message(stream, message: dict[str, Any]) -> None:
    body = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    stream.write(body + b"\n")
    stream.flush()


def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _virtual_tools() -> list[dict[str, Any]]:
    deep_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Research topic or question."},
            "max_results": {
                "type": "integer",
                "description": "Maximum search results to collect per provider.",
                "minimum": 1,
                "maximum": 10,
                "default": 8,
            },
            "required_verified_sources": {
                "type": "integer",
                "description": "Minimum verified sources for the evidence verdict.",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
            },
        },
    }
    return [
        {
            "name": "deep",
            "description": (
                "MANDATORY entrypoint for /deep requests. Use this instead of source_research, tavily_search, "
                "or serpapi_search when the user asks for /deep or deep research; it returns the staged deep "
                "workflow using deep_plan_agent, deep_collect_agent, specialist agents, and deep_review_agent."
            ),
            "inputSchema": deep_schema,
        },
        {
            "name": "subagent_spawn",
            "description": "Spawn a managed llama-bridge subagent session with automatic provider/model selection from env.yml.",
            "inputSchema": {
                "type": "object",
                "required": ["topic"],
                "properties": {
                    "topic": {"type": "string", "description": "Research topic or task for the subagent team."},
                    "agent_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit agent names. Defaults to the standard deep research team.",
                    },
                },
            },
        },
        {
            "name": "subagent_kill",
            "description": "Kill a managed llama-bridge subagent session.",
            "inputSchema": {
                "type": "object",
                "required": ["session_id"],
                "properties": {
                    "session_id": {"type": "string", "description": "Subagent session id to terminate."},
                },
            },
        },
        {
            "name": "subagent_status",
            "description": "Get status for one managed subagent session or list all sessions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Optional subagent session id."},
                },
            },
        },
        {
            "name": "deep_lead_agent",
            "description": (
                "Native llama-bridge lead research controller for /deep. Use stage=plan by default. "
                "Do not replace this workflow with source_research or raw Tavily searches."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Main research topic."},
                    "query": {"type": "string", "description": "Alias for topic."},
                    "stage": {
                        "type": "string",
                        "enum": ["plan", "collect", "review", "full"],
                        "default": "plan",
                        "description": "Deep workflow stage. Use full only for the legacy one-shot run.",
                    },
                    "session_id": {"type": "string", "description": "Existing deep research session id."},
                    "required_verified_sources": {"type": "integer", "default": 4, "minimum": 1, "maximum": 6},
                    "include_images": {"type": "boolean", "default": True},
                    "query_count": {"type": "integer", "default": 2, "minimum": 1, "maximum": 3},
                    "include_official_hunt": {
                        "type": "boolean",
                        "default": True,
                        "description": "Run an extra official-source pass inside the lead agent.",
                    },
                },
            },
        },
        {
            "name": "deep_plan_agent",
            "description": "Stage 1 REQUIRED for /deep: create a session, assign the fixed deep research team, and return the small-step workflow. Use this before Tavily/SerpAPI/Wikipedia specialist calls.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Main research topic."},
                    "query": {"type": "string", "description": "Alias for topic."},
                    "required_verified_sources": {"type": "integer", "default": 4, "minimum": 1, "maximum": 6},
                    "include_images": {"type": "boolean", "default": True},
                    "query_count": {"type": "integer", "default": 2, "minimum": 1, "maximum": 3},
                    "include_official_hunt": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "deep_collect_agent",
            "description": "Stage 2 for /deep: coordinate collection as separate MCP calls. First call returns the single-agent calls to run; second call with subagent_briefs assembles temp/ad.md markdown.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Main research topic."},
                    "query": {"type": "string", "description": "Alias for topic."},
                    "session_id": {"type": "string", "description": "Existing deep research session id from deep_plan_agent."},
                    "query_count": {"type": "integer", "default": 2, "minimum": 1, "maximum": 3},
                    "include_official_hunt": {"type": "boolean", "default": True},
                    "subagent_briefs": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Results from the individual deep_tavily_agent/deep_serp_agent/deep_wiki_agent calls. Pass this only after running those tools one by one.",
                    },
                },
                "required": ["topic"],
            },
        },
        {
            "name": "deep_review_agent",
            "description": "Stage 3 for /deep: coordinate verification as separate MCP calls. First call returns deep_verify_agent calls; second call with verification_briefs returns reviewed markdown and final handoff.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Main research topic."},
                    "query": {"type": "string", "description": "Alias for topic."},
                    "session_id": {"type": "string", "description": "Existing deep research session id from deep_plan_agent/deep_collect_agent."},
                    "required_verified_sources": {"type": "integer", "default": 4, "minimum": 1, "maximum": 6},
                    "include_images": {"type": "boolean", "default": True},
                    "selected_urls": {"type": "array", "items": {"type": "string"}, "description": "Optional explicit URL shortlist to review first."},
                    "verification_briefs": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Results from individual deep_verify_agent calls. Pass this only after running verification calls one by one.",
                    },
                    "verify_timeout_seconds": {"type": "integer", "default": 10, "minimum": 1, "maximum": 120},
                },
                "required": ["topic"],
            },
        },
        {
            "name": "deep_master_review_agent",
            "description": "Stage 4 for /deep: coordinate the 8 configured master-review MCP calls before final submission to the main LLM.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Main research topic."},
                    "query": {"type": "string", "description": "Alias for topic."},
                    "session_id": {"type": "string", "description": "Existing deep research session id."},
                    "reviewed_markdown": {"type": "string", "description": "Markdown returned by deep_review_agent."},
                    "final_handoff": {"type": "object", "description": "Final handoff returned by deep_review_agent."},
                    "master_review_briefs": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Results from the 8 individual deep_master_* review tools.",
                    },
                },
                "required": ["topic"],
            },
        },
        *_deep_master_review_tool_schemas(),
        {
            "name": "manim_render",
            "description": "Generate a short Manim Community animation video from text and return the scene/video paths.",
            "inputSchema": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string", "description": "Animation request or explanation text."},
                    "title": {"type": "string", "description": "Optional title shown in the video."},
                    "quality": {"type": "string", "enum": ["low", "medium", "high"], "default": "low"},
                    "output_dir": {"type": "string", "description": "Optional output directory. Defaults to ./manim_outputs."},
                    "render": {"type": "boolean", "default": True},
                    "timeout_seconds": {"type": "integer", "default": 180, "minimum": 30, "maximum": 600},
                },
            },
        },
        {
            "name": "deep_tavily_agent",
            "description": (
                "Specialist sub-agent for /deep: run several Tavily searches and return only a compact evidence brief."
            ),
            "inputSchema": _deep_search_agent_schema(),
        },
        {
            "name": "deep_serp_agent",
            "description": (
                "Specialist sub-agent for /deep: run several SerpAPI searches and return only a compact evidence brief."
            ),
            "inputSchema": _deep_search_agent_schema(),
        },
        {
            "name": "deep_wiki_agent",
            "description": (
                "Specialist sub-agent for /deep: search Wikipedia for background context and return only compact notes."
            ),
            "inputSchema": _deep_search_agent_schema(),
        },
        {
            "name": "deep_verify_agent",
            "description": (
                "Specialist sub-agent for /deep: verify selected URLs and return a compact claim/source verdict."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "Claim or topic to verify against the URLs."},
                    "urls": {"type": "array", "items": {"type": "string"}, "description": "URLs selected by search sub-agents."},
                    "agent_name": {"type": "string", "description": "Optional assigned name from deep_review_agent agent_calls."},
                    "required_verified_sources": {"type": "integer", "default": 3, "minimum": 1, "maximum": 6},
                    "verify_timeout_seconds": {"type": "integer", "default": 10, "minimum": 1, "maximum": 120},
                },
                "required": ["urls"],
            },
        },
        {
            "name": "deep_image_agent",
            "description": "Deep image sub-agent for /deep: find sourced image candidates, download usable image files into the repo, and return markdown-ready attachments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Main research topic."},
                    "query": {"type": "string", "description": "Alias for topic."},
                    "max_results": {"type": "integer", "default": 3, "minimum": 1, "maximum": 6},
                    "output_dir": {"type": "string", "default": "report_assets/images", "description": "Repo-relative image output directory."},
                },
                "required": ["topic"],
            },
        },
    ]


def _deep_search_agent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Main research topic."},
            "focus": {"type": "string", "description": "Optional subtopic assigned by the lead agent."},
            "agent_name": {"type": "string", "description": "Optional assigned name from deep_collect_agent agent_calls."},
            "query_count": {"type": "integer", "default": 2, "minimum": 1, "maximum": 3},
        },
        "required": ["topic"],
    }


_DEEP_MASTER_REVIEW_AGENTS: list[tuple[str, str, str]] = [
    ("spelling_grammar", "deep_master_spelling_grammar_agent", "Check spelling, grammar, typos, repeated words, and awkward sentences."),
    ("evidence_validity", "deep_master_evidence_validity_agent", "Check whether major claims are supported by verified evidence."),
    ("evidence_reliability", "deep_master_evidence_reliability_agent", "Check source quality, independence, authority, and weak-source risks."),
    ("citation_coverage", "deep_master_citation_coverage_agent", "Check whether factual paragraphs and key claims have nearby citations."),
    ("neutrality_bias", "deep_master_neutrality_bias_agent", "Check neutrality, loaded language, one-sided framing, and missing caveats."),
    ("logic_consistency", "deep_master_logic_consistency_agent", "Check contradictions, timeline issues, unsupported leaps, and overclaims."),
    ("format_quality", "deep_master_format_quality_agent", "Check markdown structure, readability, headings, lists, and broken formatting."),
    ("final_synthesis", "deep_master_final_synthesis_agent", "Produce final synthesis instructions from the other master-review findings."),
]
_DEEP_MASTER_REVIEW_TOOL_NAMES = {tool_name for _key, tool_name, _description in _DEEP_MASTER_REVIEW_AGENTS}


def _deep_master_review_tool_schemas() -> list[dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Main research topic."},
            "reviewed_markdown": {"type": "string", "description": "Markdown returned by deep_review_agent."},
            "final_handoff": {"type": "object", "description": "Final handoff returned by deep_review_agent."},
            "master_review_briefs": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Optional earlier master review briefs, mainly for final synthesis.",
            },
        },
        "required": ["reviewed_markdown"],
    }
    return [
        {
            "name": tool_name,
            "description": f"Deep master-review sub-agent for /deep: {description}",
            "inputSchema": schema,
        }
        for _key, tool_name, description in _DEEP_MASTER_REVIEW_AGENTS
    ]


def _normalize_tool_call(name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if name != "deep":
        return name, arguments
    query = str(arguments.get("query") or arguments.get("topic") or "").strip()
    normalized = {
        "query": query,
        "max_results": min(_int_argument(arguments.get("max_results"), 4), 4),
        "required_verified_sources": min(_int_argument(arguments.get("required_verified_sources"), 2), 2),
        "include_images": False,
    }
    for key in ("include_domains", "exclude_domains"):
        if arguments.get(key) is not None:
            normalized[key] = arguments[key]
    return "source_research", normalized


def _poolside_source_research_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    normalized["max_results"] = min(_int_argument(normalized.get("max_results"), 4), 4)
    normalized["required_verified_sources"] = min(
        _int_argument(normalized.get("required_verified_sources"), 2),
        2,
    )
    normalized["include_images"] = False
    normalized["skip_master_review"] = True
    normalized["max_verify_urls"] = min(_int_argument(normalized.get("max_verify_urls"), 4), 4)
    normalized["verify_timeout_seconds"] = min(
        _int_argument(normalized.get("verify_timeout_seconds"), 6),
        6,
    )
    return normalized


def _call_manim_render(server: BridgeMcpServer, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        data = server._request(
            "POST",
            "/api/tools/manim_render",
            arguments,
            timeout_seconds=server._timeout_for_tool("manim_render"),
        )
    except Exception as exc:  # noqa: BLE001 - keep MCP tool responses under the timeout budget.
        data = {"ok": False, "tool": "manim_render", "error": str(exc), "timeout_budget_seconds": server._timeout_for_tool("manim_render")}
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    is_error = bool(data.get("ok") is False or payload.get("ok") is False)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _default_subagent_names() -> list[str]:
    return [
        "web-realtime-1",
        "web-realtime-2",
        "web-realtime-3",
        "wiki-context-1",
        "wiki-context-2",
        "wiki-context-3",
        "verify-pass-1",
        "verify-pass-2",
        "verify-pass-3",
        "final-fixer",
    ]


def _call_subagent_spawn(arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    agent_names = arguments.get("agent_names")
    if not isinstance(agent_names, list) or not agent_names:
        agent_names = _default_subagent_names()
    else:
        agent_names = [str(item).strip() for item in agent_names if str(item).strip()]
    result = _SUBAGENT_MANAGER.spawn(topic, agent_names)
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _call_subagent_kill(arguments: dict[str, Any]) -> dict[str, Any]:
    session_id = str(arguments.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("session_id is required")
    result = _SUBAGENT_MANAGER.kill(session_id)
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _call_subagent_status(arguments: dict[str, Any]) -> dict[str, Any]:
    session_id = str(arguments.get("session_id") or "").strip() or None
    result = _SUBAGENT_MANAGER.status(session_id)
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _call_deep_lead_agent(server: BridgeMcpServer, arguments: dict[str, Any]) -> dict[str, Any]:
    stage = str(arguments.get("stage") or "plan").strip().lower()
    if stage == "plan":
        return _call_deep_plan_agent(arguments)
    if stage == "collect":
        return _call_deep_collect_agent(server, arguments)
    if stage == "review":
        return _call_deep_review_agent(server, arguments)
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    query_count = min(max(_int_argument(arguments.get("query_count"), 2), 1), 3)
    required_verified_sources = min(max(_int_argument(arguments.get("required_verified_sources"), 4), 1), 6)
    include_images = bool(arguments.get("include_images", True))
    include_official_hunt = bool(arguments.get("include_official_hunt", True))
    try:
        result = _run_native_deep_lead_agent(
            server,
            topic=topic,
            query_count=query_count,
            required_verified_sources=required_verified_sources,
            include_images=include_images,
            include_official_hunt=include_official_hunt,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "agent": "deep_lead_agent",
            "topic": topic,
            "error": str(exc),
            "remaining_tasks": [
                "Make sure the llama bridge server is running and reachable.",
                "Retry deep_lead_agent or fall back to the compact specialist tools.",
            ],
        }
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _call_deep_plan_agent(arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    query_count = min(max(_int_argument(arguments.get("query_count"), 2), 1), 3)
    required_verified_sources = min(max(_int_argument(arguments.get("required_verified_sources"), 4), 1), 6)
    include_images = bool(arguments.get("include_images", True))
    include_official_hunt = bool(arguments.get("include_official_hunt", True))
    session = _SUBAGENT_MANAGER.spawn(
        topic,
        [
            "tavily-realtime-1",
            "tavily-realtime-2",
            "serp-realtime-1",
            "serp-realtime-2",
            "wiki-context-1",
            "wiki-context-2",
            "wiki-context-3",
            "review-verifier",
            "markdown-reviewer",
            "final-handoff",
        ],
    )
    plan = _native_query_plan(topic)
    _SUBAGENT_MANAGER.mark(
        str(session.get("session_id")),
        status="planned",
        extra={
            "topic": topic,
            "plan": plan,
            "query_count": query_count,
            "required_verified_sources": required_verified_sources,
            "include_images": include_images,
            "include_official_hunt": include_official_hunt,
            "temp_markdown_path": "temp/ad.md",
            "final_markdown_path": "report.md",
        },
    )
    result = {
        "ok": True,
        "agent": "deep_plan_agent",
        "topic": topic,
        "session_id": session.get("session_id"),
        "session_status": "planned",
        "query_plan": plan,
        "team_layout": {
            "tavily_agents": 2,
            "serpapi_agents": 2,
            "wikipedia_agents": 3,
            "review_agents": 2,
            "final_handoff_agents": 1,
        },
        "brain_assignments": session.get("agents", []),
        "paths": {
            "temp_markdown_path": "temp/ad.md",
            "final_markdown_path": "report.md",
        },
        "next_steps": [
            "Call deep_collect_agent with the same topic and this session_id to get the single-agent MCP call list.",
            "Call each returned deep_tavily_agent, deep_serp_agent, and deep_wiki_agent item separately, one by one.",
            "Call deep_collect_agent again with subagent_briefs to assemble collect_markdown, then write it to temp/ad.md.",
            "Call deep_review_agent with the same session_id to get separate deep_verify_agent calls.",
            "Call each returned deep_verify_agent separately, then call deep_review_agent again with verification_briefs.",
            "Call deep_image_agent to download report-ready image files when images are useful.",
            "Call deep_master_review_agent with reviewed_markdown and final_handoff to get 8 master-review MCP calls.",
            "Run each returned deep_master_* review agent separately, then call deep_master_review_agent again with master_review_briefs.",
            "Overwrite temp/ad.md with the reviewed markdown and apply the master-review final_llm_instructions before writing report.md.",
        ],
    }
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _call_deep_collect_agent(server: BridgeMcpServer, arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    session_id = _deep_session_id(arguments, topic)
    query_count = min(max(_int_argument(arguments.get("query_count"), 2), 1), 3)
    include_official_hunt = bool(arguments.get("include_official_hunt", True))
    subagent_briefs = _coerce_collect_subagent_briefs(arguments.get("subagent_briefs"))
    if not subagent_briefs:
        result = _deep_collect_agent_call_plan(
            session_id=session_id,
            topic=topic,
            query_count=query_count,
            include_official_hunt=include_official_hunt,
        )
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}
    try:
        result = _run_native_deep_collect_agent(
            server,
            session_id=session_id,
            topic=topic,
            query_count=query_count,
            include_official_hunt=include_official_hunt,
            subagent_briefs=subagent_briefs,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "agent": "deep_collect_agent",
            "topic": topic,
            "session_id": session_id,
            "error": str(exc),
            "next_steps": [
                "Retry deep_collect_agent for the same session.",
                "If one provider is flaky, continue with the successful briefs and move to deep_review_agent.",
            ],
        }
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _deep_collect_agent_calls(topic: str, query_count: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "tavily-realtime-1",
            "tool": "deep_tavily_agent",
            "arguments": {"topic": topic, "agent_name": "tavily-realtime-1", "focus": "latest developments, breaking updates, current status", "query_count": query_count},
        },
        {
            "name": "tavily-realtime-2",
            "tool": "deep_tavily_agent",
            "arguments": {"topic": topic, "agent_name": "tavily-realtime-2", "focus": "official schedule, current data, established reporting", "query_count": query_count},
        },
        {
            "name": "serp-realtime-1",
            "tool": "deep_serp_agent",
            "arguments": {"topic": topic, "agent_name": "serp-realtime-1", "focus": "official sources, current coverage, top-tier reporting", "query_count": query_count},
        },
        {
            "name": "serp-realtime-2",
            "tool": "deep_serp_agent",
            "arguments": {"topic": topic, "agent_name": "serp-realtime-2", "focus": "alternative coverage, disputed claims, conflicting summaries", "query_count": query_count},
        },
        {
            "name": "wiki-context-1",
            "tool": "deep_wiki_agent",
            "arguments": {"topic": topic, "agent_name": "wiki-context-1", "focus": "background, definitions, overview", "query_count": min(query_count, 4)},
        },
        {
            "name": "wiki-context-2",
            "tool": "deep_wiki_agent",
            "arguments": {"topic": topic, "agent_name": "wiki-context-2", "focus": "history, timeline anchors, key entities", "query_count": min(query_count, 4)},
        },
        {
            "name": "wiki-context-3",
            "tool": "deep_wiki_agent",
            "arguments": {"topic": topic, "agent_name": "wiki-context-3", "focus": "institutions, terminology, key people", "query_count": min(query_count, 4)},
        },
    ]


def _deep_collect_agent_call_plan(
    *,
    session_id: str,
    topic: str,
    query_count: int,
    include_official_hunt: bool,
) -> dict[str, Any]:
    agent_calls = _deep_collect_agent_calls(topic, query_count)
    _SUBAGENT_MANAGER.ensure(session_id, topic, [str(item["name"]) for item in agent_calls] + ["review-verifier", "markdown-reviewer", "final-handoff"])
    _SUBAGENT_MANAGER.mark(
        session_id,
        status="collection_waiting_for_single_agent_calls",
        extra={
            "topic": topic,
            "query_count": query_count,
            "include_official_hunt": include_official_hunt,
            "collection_agent_calls": agent_calls,
        },
    )
    return {
        "ok": True,
        "agent": "deep_collect_agent",
        "topic": topic,
        "session_id": session_id,
        "session_status": "collection_waiting_for_single_agent_calls",
        "mode": "single_agent_calls_required",
        "team_layout": {"tavily_agents": 2, "serpapi_agents": 2, "wikipedia_agents": 3},
        "agent_calls": agent_calls,
        "next_steps": [
            "Call each item in agent_calls as its own separate MCP tool call, one by one.",
            "Collect each returned JSON object into subagent_briefs. Keep the assigned name with each result.",
            "Call deep_collect_agent again with topic, session_id, include_official_hunt, and subagent_briefs.",
            "Only the second deep_collect_agent call returns collect_markdown for temp/ad.md.",
        ],
    }


def _coerce_collect_subagent_briefs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    inferred_counts = {"deep_tavily_agent": 0, "deep_serp_agent": 0, "deep_wiki_agent": 0}
    default_names = {
        "deep_tavily_agent": ["tavily-realtime-1", "tavily-realtime-2"],
        "deep_serp_agent": ["serp-realtime-1", "serp-realtime-2"],
        "deep_wiki_agent": ["wiki-context-1", "wiki-context-2", "wiki-context-3"],
    }
    for item in value:
        if not isinstance(item, dict):
            continue
        brief = item.get("brief") if isinstance(item.get("brief"), dict) else item
        agent = str(brief.get("agent") or "").strip()
        name = str(item.get("name") or brief.get("assigned_agent_name") or "").strip()
        if not name and agent in inferred_counts:
            index = inferred_counts[agent]
            names = default_names.get(agent, [])
            name = names[index] if index < len(names) else f"{agent}-{index + 1}"
            inferred_counts[agent] = index + 1
        if not name:
            name = f"collector-{len(result) + 1}"
        result.append({"name": name, "brief": brief})
    return result


def _call_deep_review_agent(server: BridgeMcpServer, arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    session_id = _deep_session_id(arguments, topic)
    required_verified_sources = min(max(_int_argument(arguments.get("required_verified_sources"), 4), 1), 6)
    include_images = bool(arguments.get("include_images", True))
    default_verify_timeout = int(server._timeout_for_tool("deep_verify_agent"))
    verify_timeout_seconds = min(max(_int_argument(arguments.get("verify_timeout_seconds"), default_verify_timeout), 1), 120)
    selected_urls = arguments.get("selected_urls")
    if not isinstance(selected_urls, list):
        selected_urls = []
    selected_urls = [str(url).strip() for url in selected_urls if str(url).strip()]
    verification_briefs = _coerce_review_verification_briefs(arguments.get("verification_briefs"))
    if not verification_briefs:
        try:
            result = _deep_review_agent_call_plan(
                session_id=session_id,
                topic=topic,
                required_verified_sources=required_verified_sources,
                selected_urls=selected_urls,
                verify_timeout_seconds=verify_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "ok": False,
                "agent": "deep_review_agent",
                "topic": topic,
                "session_id": session_id,
                "error": str(exc),
                "next_steps": ["Retry deep_review_agent after collection is complete."],
            }
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}
    try:
        result = _run_native_deep_review_agent(
            server,
            session_id=session_id,
            topic=topic,
            required_verified_sources=required_verified_sources,
            include_images=include_images,
            selected_urls=selected_urls,
            verify_timeout_seconds=verify_timeout_seconds,
            verification_briefs=verification_briefs,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "agent": "deep_review_agent",
            "topic": topic,
            "session_id": session_id,
            "error": str(exc),
            "next_steps": [
                "Retry deep_review_agent with a smaller selected_urls list.",
                "Lower required_verified_sources if the topic has weak coverage.",
            ],
        }
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _call_deep_master_review_agent(arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    session_id = str(arguments.get("session_id") or _deep_session_id(arguments, topic)).strip()
    reviewed_markdown = _deep_review_text(arguments)
    master_review_briefs = _coerce_named_briefs(arguments.get("master_review_briefs"))
    if not master_review_briefs:
        agent_calls = [
            {
                "name": key,
                "tool": tool_name,
                "arguments": {
                    "topic": topic,
                    "reviewed_markdown": reviewed_markdown,
                    "final_handoff": arguments.get("final_handoff") if isinstance(arguments.get("final_handoff"), dict) else {},
                },
            }
            for key, tool_name, _description in _DEEP_MASTER_REVIEW_AGENTS
        ]
        result = {
            "ok": True,
            "agent": "deep_master_review_agent",
            "topic": topic,
            "session_id": session_id,
            "session_status": "master_review_waiting_for_8_calls",
            "mode": "eight_master_review_calls_required",
            "agent_calls": agent_calls,
            "next_steps": [
                "Call each item in agent_calls as its own separate MCP call, one by one.",
                "Collect each returned JSON object into master_review_briefs. Keep the assigned name with each result.",
                "Call deep_master_review_agent again with topic, session_id, reviewed_markdown, final_handoff, and master_review_briefs.",
                "Only after the second deep_master_review_agent call should the main LLM write report.md or send the final answer.",
            ],
        }
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}

    result = _assemble_deep_master_review(
        topic=topic,
        session_id=session_id,
        reviewed_markdown=reviewed_markdown,
        final_handoff=arguments.get("final_handoff") if isinstance(arguments.get("final_handoff"), dict) else {},
        master_review_briefs=master_review_briefs,
    )
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _call_deep_master_review_subagent(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    reviewed_markdown = _deep_review_text(arguments)
    master_review_briefs = _coerce_named_briefs(arguments.get("master_review_briefs"))
    result = _run_deep_master_review_check(
        name=name,
        topic=topic,
        reviewed_markdown=reviewed_markdown,
        final_handoff=arguments.get("final_handoff") if isinstance(arguments.get("final_handoff"), dict) else {},
        master_review_briefs=master_review_briefs,
    )
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}


def _call_deep_image_agent(server: BridgeMcpServer, arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    max_results = min(max(_int_argument(arguments.get("max_results"), 3), 1), 6)
    output_dir = str(arguments.get("output_dir") or "report_assets/images").strip() or "report_assets/images"
    result = _run_image_brief(server, topic, max_results=max_results, output_dir=output_dir, download=True)
    if isinstance(result, dict):
        result["agent"] = "deep_image_agent"
        result["topic"] = topic
        result["temp_file"] = _write_deep_temp_file("temp/deep_image_assets.json", json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": bool(isinstance(result, dict) and result.get("error"))}


def _deep_review_agent_call_plan(
    *,
    session_id: str,
    topic: str,
    required_verified_sources: int,
    selected_urls: list[str],
    verify_timeout_seconds: int,
) -> dict[str, Any]:
    session = _SUBAGENT_MANAGER.ensure(session_id, topic, ["review-verifier", "markdown-reviewer", "final-handoff"])
    all_sources = session.get("all_sources") if isinstance(session.get("all_sources"), list) else []
    shortlisted_urls = selected_urls or session.get("selected_urls") or _pick_best_urls(all_sources, target=max(required_verified_sources + 1, 4))
    shortlisted_urls = [str(url).strip() for url in shortlisted_urls if str(url).strip()][:10]
    verification_agents = _build_verification_agents(
        topic,
        shortlisted_urls,
        required_verified_sources,
        verify_timeout_seconds=verify_timeout_seconds,
    )
    agent_calls = [
        {
            "name": agent["name"],
            "tool": "deep_verify_agent",
            "arguments": {"agent_name": agent["name"], **dict(agent["arguments"])},
        }
        for agent in verification_agents
    ]
    _SUBAGENT_MANAGER.mark(
        session_id,
        status="review_waiting_for_single_verify_calls",
        extra={"review_agent_calls": agent_calls, "selected_urls": shortlisted_urls},
    )
    return {
        "ok": True,
        "agent": "deep_review_agent",
        "topic": topic,
        "session_id": session_id,
        "session_status": "review_waiting_for_single_verify_calls",
        "mode": "single_agent_calls_required",
        "selected_urls": shortlisted_urls,
        "agent_calls": agent_calls,
        "next_steps": [
            "Call each item in agent_calls as its own separate deep_verify_agent MCP call, one by one.",
            "Collect each returned JSON object into verification_briefs. Keep the assigned name with each result.",
            "Call deep_review_agent again with topic, session_id, include_images, and verification_briefs.",
            "Only the second deep_review_agent call returns reviewed_markdown and final_handoff.",
        ],
    }


def _coerce_review_verification_briefs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    count = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        brief = item.get("brief") if isinstance(item.get("brief"), dict) else item
        name = str(item.get("name") or brief.get("assigned_agent_name") or "").strip()
        if not name:
            count += 1
            name = f"verify-pass-{count}"
        result.append({"name": name, "brief": brief})
    return result


def _coerce_named_briefs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        brief = item.get("brief") if isinstance(item.get("brief"), dict) else item
        name = str(item.get("name") or brief.get("agent") or f"master-review-{index}").strip()
        result.append({"name": name, "brief": brief})
    return result


def _deep_review_text(arguments: dict[str, Any]) -> str:
    text = str(arguments.get("reviewed_markdown") or arguments.get("markdown") or arguments.get("draft") or "").strip()
    if text:
        return text
    handoff = arguments.get("final_handoff")
    if isinstance(handoff, dict):
        return json.dumps(handoff, ensure_ascii=False, default=str)
    return ""


def _run_deep_master_review_check(
    *,
    name: str,
    topic: str,
    reviewed_markdown: str,
    final_handoff: dict[str, Any],
    master_review_briefs: list[dict[str, Any]],
) -> dict[str, Any]:
    key = next((key for key, tool_name, _description in _DEEP_MASTER_REVIEW_AGENTS if tool_name == name), name)
    urls = sorted(set(re.findall(r"https?://[^\s)\\]>\"']+", reviewed_markdown)))
    findings: list[dict[str, str]] = []
    must_fix: list[str] = []
    should_fix: list[str] = []
    text_lower = reviewed_markdown.lower()

    if key == "spelling_grammar":
        repeated = re.findall(r"\b(\w+)\s+\1\b", reviewed_markdown, flags=re.IGNORECASE)
        if repeated:
            must_fix.append("Remove repeated words: " + ", ".join(sorted(set(repeated))[:6]))
        if reviewed_markdown.count("```") % 2:
            must_fix.append("Close or remove the unbalanced markdown code fence.")
        findings.extend({"issue": item, "severity": "must_fix"} for item in must_fix)
    elif key == "evidence_validity":
        if not urls:
            must_fix.append("No source URLs are visible in the reviewed markdown.")
        if any(term in text_lower for term in ("uncertain", "unverified", "insufficient")):
            should_fix.append("Keep uncertainty labels in the final report; do not upgrade them to facts.")
        findings.append({"issue": f"Detected {len(urls)} visible source URLs.", "severity": "info"})
    elif key == "evidence_reliability":
        weak_hosts = [url for url in urls if any(host in url.lower() for host in ("blogspot.", "facebook.", "x.com/", "twitter.com/", "reddit.com/"))]
        if weak_hosts:
            should_fix.append("Use weak/social sources only as leads, not as primary citations.")
        findings.append({"issue": f"Weak/social source candidates: {len(weak_hosts)}.", "severity": "review"})
    elif key == "citation_coverage":
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", reviewed_markdown) if len(part.strip()) > 100]
        uncited = [p for p in paragraphs if not re.search(r"https?://|\[\d+\]", p)]
        if uncited:
            should_fix.append(f"Add citations or soften {min(len(uncited), 5)} long uncited factual paragraph(s).")
        findings.append({"issue": f"Long uncited paragraphs: {len(uncited)}.", "severity": "review"})
    elif key == "neutrality_bias":
        loaded_terms = [term for term in ("obviously", "clearly", "disaster", "propaganda", "corrupt", "shocking") if term in text_lower]
        if loaded_terms:
            should_fix.append("Replace loaded language with attributed, neutral wording: " + ", ".join(loaded_terms))
        findings.append({"issue": f"Loaded-language markers: {len(loaded_terms)}.", "severity": "review"})
    elif key == "logic_consistency":
        if "however" in text_lower and "therefore" in text_lower:
            should_fix.append("Check that contrast and conclusion paragraphs do not contradict each other.")
        if "safe_to_write_final_report" in json.dumps(final_handoff, ensure_ascii=False) and not final_handoff:
            should_fix.append("Final handoff is empty; carry forward the deep_review_agent handoff before final writing.")
        findings.append({"issue": "Logic pass completed; preserve caveats around disputed or weak claims.", "severity": "review"})
    elif key == "format_quality":
        if not re.search(r"(?m)^#\s+", reviewed_markdown):
            should_fix.append("Add a clear top-level markdown heading.")
        if len(reviewed_markdown) > 1200 and not re.search(r"(?m)^##\s+", reviewed_markdown):
            should_fix.append("Add section headings so the report is scannable.")
        findings.append({"issue": "Format pass completed for headings and markdown balance.", "severity": "review"})
    elif key == "final_synthesis":
        prior_must = [
            str(issue)
            for item in master_review_briefs
            for issue in (item.get("brief", {}).get("must_fix") or [])
        ][:8]
        prior_should = [
            str(issue)
            for item in master_review_briefs
            for issue in (item.get("brief", {}).get("should_fix") or [])
        ][:8]
        must_fix.extend(prior_must)
        should_fix.extend(prior_should)
        findings.append({"issue": "Final synthesis collected earlier master-review findings for the main LLM.", "severity": "handoff"})

    return {
        "ok": True,
        "agent": name,
        "review_key": key,
        "topic": topic,
        "findings": findings[:12],
        "must_fix": must_fix[:8],
        "should_fix": should_fix[:8],
        "source_url_count": len(urls),
        "handoff": _master_review_subagent_handoff(key, must_fix, should_fix),
    }


def _master_review_subagent_handoff(key: str, must_fix: list[str], should_fix: list[str]) -> str:
    if must_fix:
        return f"{key}: fix required issues before final submission."
    if should_fix:
        return f"{key}: review suggested improvements before final submission."
    return f"{key}: no blocking issue found in compact deterministic review."


def _assemble_deep_master_review(
    *,
    topic: str,
    session_id: str,
    reviewed_markdown: str,
    final_handoff: dict[str, Any],
    master_review_briefs: list[dict[str, Any]],
) -> dict[str, Any]:
    must_fix = [
        str(issue)
        for item in master_review_briefs
        for issue in (item.get("brief", {}).get("must_fix") or [])
    ]
    should_fix = [
        str(issue)
        for item in master_review_briefs
        for issue in (item.get("brief", {}).get("should_fix") or [])
    ]
    completed = [str(item.get("name") or item.get("brief", {}).get("agent") or "") for item in master_review_briefs]
    final_instructions = [
        "Apply the deep master-review findings before writing report.md or sending the final answer.",
        "Preserve verified source URLs and uncertainty labels from deep_review_agent.",
        "Do not cite sources that the verification stage rejected or marked weak.",
    ]
    final_instructions.extend(f"Must fix: {item}" for item in must_fix[:8])
    final_instructions.extend(f"Should fix: {item}" for item in should_fix[:8])
    final_handoff_payload = {
        **final_handoff,
        "deep_master_review": {
            "completed": completed,
            "must_fix": must_fix[:12],
            "should_fix": should_fix[:12],
            "final_llm_instructions": final_instructions,
        },
    }
    temp_files = {
        "master_review_json": _write_deep_temp_file("temp/deep_master_review.json", json.dumps(master_review_briefs, indent=2, ensure_ascii=False, default=str)),
        "master_final_handoff_json": _write_deep_temp_file("temp/deep_master_final_handoff.json", json.dumps(final_handoff_payload, indent=2, ensure_ascii=False, default=str)),
        "final_instructions_md": _write_deep_temp_file("temp/deep_final_instructions.md", "\n".join(f"- {item}" for item in final_instructions)),
    }
    return {
        "ok": len(completed) >= len(_DEEP_MASTER_REVIEW_AGENTS),
        "agent": "deep_master_review_agent",
        "topic": topic,
        "session_id": session_id,
        "session_status": "master_review_completed",
        "completed_master_review_calls": completed,
        "expected_master_review_calls": [key for key, _tool, _description in _DEEP_MASTER_REVIEW_AGENTS],
        "must_fix": must_fix[:12],
        "should_fix": should_fix[:12],
        "final_llm_instructions": final_instructions,
        "final_handoff": final_handoff_payload,
        "reviewed_markdown": reviewed_markdown,
        "temp_files": temp_files,
        "next_steps": [
            "Apply final_llm_instructions to reviewed_markdown.",
            "Write report.md in the current working directory.",
            "Then send the final response to the user.",
        ],
    }


def _run_native_deep_lead_agent(
    server: BridgeMcpServer,
    *,
    topic: str,
    query_count: int,
    required_verified_sources: int,
    include_images: bool,
    include_official_hunt: bool,
) -> dict[str, Any]:
    plan = _native_query_plan(topic)
    session = _SUBAGENT_MANAGER.spawn(topic, _default_subagent_names())
    brain_assignments = list(session.get("agents", []))
    web_agents = [
        {
            "name": "web-realtime-1",
            "tool": "deep_tavily_agent",
            "arguments": {"topic": topic, "focus": "latest developments, breaking updates, current status", "query_count": query_count},
        },
        {
            "name": "web-realtime-2",
            "tool": "deep_serp_agent",
            "arguments": {"topic": topic, "focus": "official sources, current coverage, top-tier reporting", "query_count": query_count},
        },
        {
            "name": "web-realtime-3",
            "tool": "deep_tavily_agent",
            "arguments": {"topic": topic, "focus": "conflicts, controversies, alternative current coverage", "query_count": query_count},
        },
    ]
    wiki_agents = [
        {
            "name": "wiki-context-1",
            "tool": "deep_wiki_agent",
            "arguments": {"topic": topic, "focus": "background, definitions, overview", "query_count": min(query_count, 4)},
        },
        {
            "name": "wiki-context-2",
            "tool": "deep_wiki_agent",
            "arguments": {"topic": topic, "focus": "history, timeline anchors, prior developments", "query_count": min(query_count, 4)},
        },
        {
            "name": "wiki-context-3",
            "tool": "deep_wiki_agent",
            "arguments": {"topic": topic, "focus": "key entities, institutions, terminology", "query_count": min(query_count, 4)},
        },
    ]
    web_briefs = [
        {"name": agent["name"], "brief": _run_deep_agent_brief(server, str(agent["tool"]), dict(agent["arguments"]))}
        for agent in web_agents
    ]
    wiki_briefs = [
        {"name": agent["name"], "brief": _run_deep_agent_brief(server, str(agent["tool"]), dict(agent["arguments"]))}
        for agent in wiki_agents
    ]

    official = (
        _build_official_source_brief(
            topic,
            *[item["brief"] for item in web_briefs],
            *[item["brief"] for item in wiki_briefs],
        )
        if include_official_hunt
        else {"official_sources": [], "warnings": []}
    )

    all_sources = _merge_candidate_sources(
        *[item["brief"] for item in web_briefs],
        *[item["brief"] for item in wiki_briefs],
    )
    selected_urls = _pick_best_urls(all_sources, target=max(required_verified_sources + 2, 4))
    verification_agents = _build_verification_agents(topic, selected_urls, required_verified_sources)
    verification_briefs = [
        {"name": agent["name"], "brief": _run_verify_or_stub(server, agent["arguments"])}
        for agent in verification_agents
    ]
    combined_verify = _combine_verification_briefs(verification_briefs)
    source_review = _build_source_quality_review(all_sources, combined_verify, official)
    audit = _audit_native_research(plan, all_sources, combined_verify, official)
    fixer = _apply_final_fixer(
        topic=topic,
        plan=plan,
        claims=_build_compact_claims(*[item["brief"] for item in web_briefs], *[item["brief"] for item in wiki_briefs], combined_verify),
        verify=combined_verify,
        audit=audit,
    )
    images = _run_image_brief(server, topic) if include_images else None
    subagent_errors = {
        "web": {item["name"]: item["brief"].get("errors", []) for item in web_briefs},
        "wiki": {item["name"]: item["brief"].get("errors", []) for item in wiki_briefs},
        "verify": {item["name"]: [] if item["brief"].get("ok", True) else [item["brief"].get("notes") or item["brief"].get("verdict")] for item in verification_briefs},
    }

    final_payload = {
        "ok": bool(fixer.get("safe_to_write_final_report")),
        "agent": "deep_lead_agent",
        "session_id": session.get("session_id"),
        "session_status": "completed" if fixer.get("safe_to_write_final_report") else "needs_revision",
        "topic": topic,
        "public_reasoning_summary": _build_public_reasoning_summary(plan),
        "subagents_used": [
            "web-realtime-1",
            "web-realtime-2",
            "web-realtime-3",
            "wiki-context-1",
            "wiki-context-2",
            "wiki-context-3",
            "verify-pass-1",
            "verify-pass-2",
            "verify-pass-3",
            "final-fixer",
        ] + (["image-media-researcher"] if include_images else []),
        "brain_assignments": brain_assignments,
        "team_layout": {
            "web_scraping_agents": 3,
            "wiki_scraping_agents": 3,
            "verification_agents": 3,
            "final_fixer_agents": 1,
        },
        "compact_claims": fixer.get("compact_claims", []),
        "verified_sources": combined_verify.get("verified_sources", []),
        "rejected_or_uncertain_sources": combined_verify.get("rejected_sources", []),
        "source_quality_review": source_review,
        "suggested_report_outline": _suggested_report_outline(plan),
        "checkpoint_markdown": _build_checkpoint_markdown(
            topic=topic,
            plan=plan,
            web_briefs=web_briefs,
            wiki_briefs=wiki_briefs,
            verification_briefs=verification_briefs,
            official=official,
            verify=combined_verify,
            audit=fixer,
            images=images,
        ),
        "remaining_tasks": fixer.get("remaining_tasks", []),
        "query_plan": plan,
        "official_sources": official.get("official_sources", []),
        "background_notes": _combine_background_notes(wiki_briefs),
        "audit": fixer,
        "images": images,
        "subagent_errors": subagent_errors,
        "web_briefs": web_briefs,
        "wiki_briefs": wiki_briefs,
        "verification_briefs": verification_briefs,
        "final_fixer": fixer,
    }
    _SUBAGENT_MANAGER.mark(
        str(session.get("session_id")),
        status=str(final_payload.get("session_status")),
        extra={
            "topic": topic,
            "final_confidence": fixer.get("final_confidence"),
            "remaining_tasks": fixer.get("remaining_tasks", []),
        },
    )
    return final_payload


def _run_native_deep_collect_agent(
    server: BridgeMcpServer,
    *,
    session_id: str,
    topic: str,
    query_count: int,
    include_official_hunt: bool,
    subagent_briefs: list[dict[str, Any]],
) -> dict[str, Any]:
    session = _SUBAGENT_MANAGER.ensure(
        session_id,
        topic,
        [str(item["name"]) for item in _deep_collect_agent_calls(topic, query_count)] + ["review-verifier", "markdown-reviewer", "final-handoff"],
    )
    plan = session.get("plan") if isinstance(session.get("plan"), dict) else _native_query_plan(topic)
    del server, query_count
    search_briefs = [
        item
        for item in subagent_briefs
        if str(item.get("brief", {}).get("agent") or "") in {"deep_tavily_agent", "deep_serp_agent"}
    ]
    wiki_briefs = [
        item
        for item in subagent_briefs
        if str(item.get("brief", {}).get("agent") or "") == "deep_wiki_agent"
    ]
    official = (
        _build_official_source_brief(
            topic,
            *[item["brief"] for item in search_briefs],
            *[item["brief"] for item in wiki_briefs],
        )
        if include_official_hunt
        else {"official_sources": [], "warnings": []}
    )
    all_sources = _merge_candidate_sources(
        *[item["brief"] for item in search_briefs],
        *[item["brief"] for item in wiki_briefs],
    )
    selected_urls = _pick_best_urls(all_sources, target=10)
    collect_markdown = _build_collection_markdown(
        topic=topic,
        plan=plan,
        search_briefs=search_briefs,
        wiki_briefs=wiki_briefs,
        official=official,
        selected_urls=selected_urls,
    )
    temp_files = {
        "collection_markdown": _write_deep_temp_file("temp/deep_collection.md", collect_markdown),
        "search_briefs_json": _write_deep_temp_file("temp/deep_search_briefs.json", json.dumps(search_briefs, indent=2, ensure_ascii=False, default=str)),
        "wiki_briefs_json": _write_deep_temp_file("temp/deep_wiki_briefs.json", json.dumps(wiki_briefs, indent=2, ensure_ascii=False, default=str)),
        "source_shortlist_json": _write_deep_temp_file("temp/deep_source_shortlist.json", json.dumps({"selected_urls": selected_urls, "official": official, "all_sources": all_sources}, indent=2, ensure_ascii=False, default=str)),
    }
    _SUBAGENT_MANAGER.mark(
        session_id,
        status="collected",
        extra={
            "topic": topic,
            "plan": plan,
            "search_briefs": search_briefs,
            "wiki_briefs": wiki_briefs,
            "official": official,
            "all_sources": all_sources,
            "selected_urls": selected_urls,
            "collect_markdown": collect_markdown,
            "temp_markdown_path": "temp/ad.md",
            "temp_files": temp_files,
        },
    )
    return {
        "ok": True,
        "agent": "deep_collect_agent",
        "topic": topic,
        "session_id": session_id,
        "session_status": "collected",
        "team_layout": {"tavily_agents": 2, "serpapi_agents": 2, "wikipedia_agents": 3},
        "selected_urls": selected_urls,
        "official_sources": official.get("official_sources", []),
        "search_briefs": search_briefs,
        "wiki_briefs": wiki_briefs,
        "collect_markdown": collect_markdown,
        "temp_markdown_path": "temp/ad.md",
        "temp_files": temp_files,
        "next_steps": [
            "Use temp_files for detailed checkpoints; temp/deep_collection.md contains the collection draft.",
            "Write collect_markdown to temp/ad.md if the client needs the legacy checkpoint path.",
            "Call deep_review_agent with the same session_id to get separate deep_verify_agent calls.",
        ],
    }


def _run_native_deep_review_agent(
    server: BridgeMcpServer,
    *,
    session_id: str,
    topic: str,
    required_verified_sources: int,
    include_images: bool,
    selected_urls: list[str],
    verify_timeout_seconds: int,
    verification_briefs: list[dict[str, Any]],
) -> dict[str, Any]:
    del verify_timeout_seconds
    session = _SUBAGENT_MANAGER.ensure(session_id, topic, ["review-verifier", "markdown-reviewer", "final-handoff"])
    plan = session.get("plan") if isinstance(session.get("plan"), dict) else _native_query_plan(topic)
    search_briefs = session.get("search_briefs") if isinstance(session.get("search_briefs"), list) else []
    wiki_briefs = session.get("wiki_briefs") if isinstance(session.get("wiki_briefs"), list) else []
    official = session.get("official") if isinstance(session.get("official"), dict) else {"official_sources": [], "warnings": []}
    all_sources = session.get("all_sources") if isinstance(session.get("all_sources"), list) else []
    if not all_sources:
        all_sources = _merge_candidate_sources(
            *[item.get("brief", {}) for item in search_briefs if isinstance(item, dict)],
            *[item.get("brief", {}) for item in wiki_briefs if isinstance(item, dict)],
        )
    shortlisted_urls = selected_urls or session.get("selected_urls") or _pick_best_urls(all_sources, target=max(required_verified_sources + 1, 4))
    shortlisted_urls = [str(url).strip() for url in shortlisted_urls if str(url).strip()][:10]
    combined_verify = _combine_verification_briefs(verification_briefs)
    source_review = _build_source_quality_review(all_sources, combined_verify, official)
    audit = _audit_native_research(plan, all_sources, combined_verify, official)
    fixer = _apply_final_fixer(
        topic=topic,
        plan=plan,
        claims=_build_compact_claims(
            *[item.get("brief", {}) for item in search_briefs if isinstance(item, dict)],
            *[item.get("brief", {}) for item in wiki_briefs if isinstance(item, dict)],
            combined_verify,
        ),
        verify=combined_verify,
        audit=audit,
    )
    images = _run_image_brief(server, topic) if include_images else None
    reviewed_markdown = _build_checkpoint_markdown(
        topic=topic,
        plan=plan,
        web_briefs=search_briefs,
        wiki_briefs=wiki_briefs,
        verification_briefs=verification_briefs,
        official=official,
        verify=combined_verify,
        audit=fixer,
        images=images,
    )
    final_handoff = _build_final_handoff(topic, fixer, combined_verify, source_review)
    temp_files = {
        "reviewed_markdown": _write_deep_temp_file("temp/deep_reviewed.md", reviewed_markdown),
        "verification_briefs_json": _write_deep_temp_file("temp/deep_verification_briefs.json", json.dumps(verification_briefs, indent=2, ensure_ascii=False, default=str)),
        "verified_sources_json": _write_deep_temp_file("temp/deep_verified_sources.json", json.dumps(combined_verify, indent=2, ensure_ascii=False, default=str)),
        "final_handoff_json": _write_deep_temp_file("temp/deep_final_handoff.json", json.dumps(final_handoff, indent=2, ensure_ascii=False, default=str)),
    }
    if images:
        temp_files["image_assets_json"] = _write_deep_temp_file("temp/deep_image_assets.json", json.dumps(images, indent=2, ensure_ascii=False, default=str))
    session_status = "completed" if fixer.get("safe_to_write_final_report") else "needs_revision"
    _SUBAGENT_MANAGER.mark(
        session_id,
        status=session_status,
        extra={
            "verified_sources": combined_verify.get("verified_sources", []),
            "rejected_or_uncertain_sources": combined_verify.get("rejected_sources", []),
            "reviewed_markdown": reviewed_markdown,
            "final_handoff": final_handoff,
            "final_confidence": fixer.get("final_confidence"),
            "remaining_tasks": fixer.get("remaining_tasks", []),
            "images": images,
            "temp_files": temp_files,
        },
    )
    return {
        "ok": bool(fixer.get("safe_to_write_final_report")),
        "agent": "deep_review_agent",
        "topic": topic,
        "session_id": session_id,
        "session_status": session_status,
        "verified_sources": combined_verify.get("verified_sources", []),
        "rejected_or_uncertain_sources": combined_verify.get("rejected_sources", []),
        "source_quality_review": source_review,
        "compact_claims": fixer.get("compact_claims", []),
        "reviewed_markdown": reviewed_markdown,
        "temp_markdown_path": "temp/ad.md",
        "final_handoff": final_handoff,
        "remaining_tasks": fixer.get("remaining_tasks", []),
        "images": images,
        "verification_briefs": verification_briefs,
        "temp_files": temp_files,
        "next_steps": [
            "Call deep_master_review_agent with reviewed_markdown and final_handoff to get 8 master-review MCP calls.",
            "Run each returned deep_master_* review tool separately.",
            "Call deep_master_review_agent again with master_review_briefs before report.md or the final user response.",
        ],
    }


def _run_deep_agent_brief(server: BridgeMcpServer, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "deep_verify_agent":
        data = server._request("POST", "/api/tools/verify_sources", _deep_verify_arguments(arguments))
        return _compact_verify_agent_result(data)

    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    focus = str(arguments.get("focus") or "").strip()
    count = 1
    queries = _deep_agent_queries(topic, focus, count=count, agent=name)
    provider = {
        "deep_tavily_agent": "tavily_search",
        "deep_serp_agent": "serpapi_search",
        "deep_wiki_agent": "wikipedia_search",
    }[name]
    collected: list[dict[str, Any]] = []
    errors: list[str] = []
    for query in queries:
        try:
            payload = _deep_agent_payload(provider, query)
            data = server._request("POST", f"/api/tools/{provider}", payload)
            if data.get("ok") is False:
                errors.append(f"{query}: {data.get('error') or data}")
                continue
            collected.extend(_compact_search_sources(provider, query, data))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{query}: {exc}")
    return _compact_deep_agent_result(
        agent=name,
        provider=provider,
        topic=topic,
        focus=focus,
        queries=queries,
        sources=collected,
        errors=errors,
    )


def _run_verify_brief(server: BridgeMcpServer, arguments: dict[str, Any]) -> dict[str, Any]:
    data = server._request("POST", "/api/tools/verify_sources", _deep_verify_arguments(arguments))
    return _compact_verify_agent_result(data)


def _run_verify_or_stub(server: BridgeMcpServer, arguments: dict[str, Any]) -> dict[str, Any]:
    urls = arguments.get("urls") or []
    if not isinstance(urls, list) or not urls:
        return {
            "ok": False,
            "agent": "deep_verify_agent",
            "verdict": "not verified",
            "verified_sources": [],
            "rejected_sources": [],
            "notes": "No URLs were assigned to this verification pass.",
        }
    return _run_verify_brief(server, arguments)


def _build_verification_agents(
    topic: str,
    urls: list[str],
    required_verified_sources: int,
    *,
    verify_timeout_seconds: int = 10,
) -> list[dict[str, Any]]:
    batches = _split_urls_for_review(urls, 3)
    focuses = [
        "strongest official and top-tier URLs",
        "cross-check URLs for conflicting summaries",
        "remaining URLs for weak evidence or stale claims",
    ]
    agents: list[dict[str, Any]] = []
    for index in range(3):
        agents.append(
            {
                "name": f"verify-pass-{index + 1}",
                "arguments": {
                    "claim": f"{topic} ({focuses[index]})",
                    "urls": batches[index],
                    "required_verified_sources": max(1, min(required_verified_sources, len(batches[index]) or 1)),
                    "verify_timeout_seconds": verify_timeout_seconds,
                },
            }
        )
    return agents


def _split_urls_for_review(urls: list[str], groups: int) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(groups)]
    for index, url in enumerate(urls):
        buckets[index % groups].append(url)
    return buckets


def _combine_verification_briefs(briefs: list[dict[str, Any]]) -> dict[str, Any]:
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    notes: list[str] = []
    seen_verified: set[str] = set()
    seen_rejected: set[str] = set()
    for item in briefs:
        brief = item.get("brief", {})
        for source in brief.get("verified_sources", []):
            url = str(source.get("url") or "").strip()
            if url and url not in seen_verified:
                seen_verified.add(url)
                verified.append(source)
        for source in brief.get("rejected_sources", []):
            url = str(source.get("url") or "").strip()
            key = url or str(source)
            if key not in seen_rejected:
                seen_rejected.add(key)
                rejected.append(source)
        note = str(brief.get("notes") or brief.get("verdict") or "").strip()
        if note:
            notes.append(f"{item.get('name')}: {note}")
    return {
        "ok": bool(verified),
        "agent": "deep_verify_agent",
        "verdict": "verified" if verified else "not verified",
        "verified_sources": verified,
        "rejected_sources": rejected,
        "notes": " | ".join(notes[:6]),
    }


def _combine_background_notes(wiki_briefs: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    seen: set[str] = set()
    for item in wiki_briefs:
        for note in _background_notes_from_brief(item.get("brief", {})):
            key = note.lower()
            if key in seen:
                continue
            seen.add(key)
            notes.append(note)
            if len(notes) >= 9:
                return notes
    return notes


def _apply_final_fixer(
    *,
    topic: str,
    plan: dict[str, Any],
    claims: list[dict[str, Any]],
    verify: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    verified_urls = {str(item.get("url") or "").strip() for item in verify.get("verified_sources", [])}
    fixed_claims = [
        claim
        for claim in claims
        if str(claim.get("supporting_url") or "").strip() in verified_urls or not verified_urls
    ]
    final_confidence = "High" if audit.get("safe_to_write_final_report") else ("Medium" if verify.get("verified_sources") else "Low")
    return {
        "audit_result": audit.get("audit_result"),
        "issues_found": audit.get("issues_found", []),
        "safe_to_write_final_report": audit.get("safe_to_write_final_report", False),
        "remaining_tasks": audit.get("remaining_tasks", []),
        "topic_type": plan.get("topic_type"),
        "final_confidence": final_confidence,
        "compact_claims": fixed_claims,
        "fixer_note": f"Final fixer reviewed 3 verification passes and removed unsupported or weakly backed claims for {topic}.",
    }


def _run_image_brief(
    server: BridgeMcpServer,
    topic: str,
    *,
    max_results: int = 3,
    output_dir: str = "report_assets/images",
    download: bool = True,
) -> dict[str, Any] | None:
    try:
        data = server._request("POST", "/api/tools/image_research", {"query": topic, "max_results": max_results})
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    payload = _unwrap_tool_payload(data)
    results = payload.get("results") if isinstance(payload.get("results"), list) else payload.get("images")
    if not isinstance(results, list):
        results = []
    images: list[dict[str, Any]] = []
    for index, item in enumerate(results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        image_url = str(item.get("url") or item.get("image_url") or item.get("thumbnail") or "").strip()
        local_path = str(item.get("local_path") or "").strip()
        if download and not local_path and image_url:
            local_path = _download_deep_image(image_url, topic=topic, index=index, output_dir=output_dir) or ""
        markdown_path = local_path.replace("\\", "/") if local_path else image_url
        images.append(
            {
                "title": _compact_text(item.get("title") or item.get("caption") or "image", 120),
                "url": image_url,
                "source_url": str(item.get("source_url") or item.get("source") or ""),
                "source": _compact_text(item.get("source_name") or item.get("source") or item.get("domain") or "", 80),
                "local_path": local_path,
                "markdown": f"![{_markdown_safe_alt(item.get('title') or 'image')}]({markdown_path})" if markdown_path else "",
            }
        )
    return {
        "results": images,
        "downloaded_count": len([item for item in images if item.get("local_path")]),
        "output_dir": output_dir,
        "report_markdown_hint": "Use local_path in report.md image embeds when present, and cite source_url below each image.",
    }


def _download_deep_image(url: str, *, topic: str, index: int, output_dir: str) -> str | None:
    if not url.startswith(("http://", "https://")):
        return None
    try:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; llama-bridge-deep/0.1)",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
        with urlopen(request, timeout=20) as response:  # noqa: S310 - user-configured image URL from image tool.
            content_type = str(response.headers.get("content-type") or "").split(";", maxsplit=1)[0].lower()
            if content_type and not content_type.startswith("image/"):
                return None
            data = response.read(12 * 1024 * 1024 + 1)
            if len(data) > 12 * 1024 * 1024:
                return None
    except Exception:
        return None
    suffix = _image_suffix_from_url_or_type(url, content_type)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", topic).strip("-._")[:60] or "deep-image"
    path = directory / f"{stem}-{index}{suffix}"
    counter = 2
    while path.exists():
        path = directory / f"{stem}-{index}-{counter}{suffix}"
        counter += 1
    try:
        path.write_bytes(data)
    except OSError:
        return None
    return path.as_posix()


def _image_suffix_from_url_or_type(url: str, content_type: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return suffix
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    return ".jpg"


def _markdown_safe_alt(value: Any) -> str:
    text = _compact_text(value or "image", 80)
    return text.replace("[", "(").replace("]", ")")


def _native_query_plan(topic: str) -> dict[str, Any]:
    lowered = topic.lower()
    topic_type = "general background topic"
    if any(word in lowered for word in ("election", "vote", "party", "assembly", "minister", "parliament")):
        topic_type = "political/election topic"
    elif any(word in lowered for word in ("law", "court", "regulation", "policy", "bill", "legal")):
        topic_type = "legal/regulatory topic"
    elif any(word in lowered for word in ("market", "stock", "revenue", "profit", "economy", "financial")):
        topic_type = "financial/economic topic"
    elif any(word in lowered for word in ("battery", "ai", "model", "software", "technical", "science")):
        topic_type = "technical/scientific topic"
    elif any(word in lowered for word in ("health", "medical", "coffee", "disease", "drug")):
        topic_type = "health/medical topic"
    freshness = (
        ["latest status", "recent changes", "current official numbers"]
        if any(word in lowered for word in ("latest", "current", "today", "2025", "2026", "recent", "now"))
        or topic_type in {"political/election topic", "financial/economic topic"}
        else []
    )
    return {
        "topic": topic,
        "topic_type": topic_type,
        "main_questions": [
            f"What are the key verified facts about {topic}?",
            f"What official or primary sources exist for {topic}?",
            f"What current developments or disputed points matter for {topic}?",
        ],
        "official_source_targets": [
            "government or regulator pages",
            "primary organization announcements or filings",
            "official datasets, papers, or dashboards",
        ],
        "freshness_needs": freshness,
        "data_needs": ["exact figures", "dates", "units", "final vs provisional status"],
        "background_terms": [topic, f"{topic} background", f"{topic} timeline"],
        "possible_conflicts": [
            "secondary sources may summarize numbers differently from official pages",
            "background pages may lag current developments",
        ],
        "verification_questions": [
            "Does the cited page directly support the claim?",
            "Is the source official, primary, or top-tier?",
            "Is the information current enough for the topic?",
        ],
    }


def _merge_candidate_sources(*briefs: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for brief in briefs:
        for source in brief.get("candidate_sources", []):
            if not isinstance(source, dict):
                continue
            key = str(source.get("url") or source.get("title") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(source)
    merged.sort(key=lambda item: _source_quality_rank(str(item.get("source_quality") or "")))
    return merged


def _pick_best_urls(sources: list[dict[str, Any]], *, target: int) -> list[str]:
    urls: list[str] = []
    for quality in ("primary/official", "scholarly/domain-authority", "top-tier-news", "established-news", "needs-review"):
        for source in sources:
            if str(source.get("source_quality") or "") != quality:
                continue
            url = str(source.get("url") or "").strip()
            if url and url not in urls:
                urls.append(url)
            if len(urls) >= target:
                return urls
    return urls


def _official_source_expected(plan: dict[str, Any]) -> bool:
    topic_type = str(plan.get("topic_type") or "")
    if topic_type in {
        "political/election topic",
        "legal/regulatory topic",
        "financial/economic topic",
        "health/medical topic",
    }:
        return True
    freshness_needs = plan.get("freshness_needs")
    return isinstance(freshness_needs, list) and bool(freshness_needs)


def _build_official_source_brief(topic: str, *briefs: dict[str, Any]) -> dict[str, Any]:
    official_sources: list[dict[str, Any]] = []
    for source in _merge_candidate_sources(*briefs):
        if str(source.get("source_quality") or "") != "primary/official":
            continue
        official_sources.append(
            {
                "title": source.get("title"),
                "url": source.get("url"),
                "what_it_proves": _compact_text(source.get("evidence"), 180),
                "date_checked": "current session",
                "source_strength": "Strong",
            }
        )
    return {
        "official_sources": official_sources[:6],
        "missing_official_sources": [] if official_sources else [f"No clear official source found yet for {topic}"],
        "warnings": [] if official_sources else ["Core claims may rely on top-tier secondary reporting until official pages are found."],
    }


def _build_source_quality_review(
    sources: list[dict[str, Any]],
    verify: dict[str, Any],
    official: dict[str, Any],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for source in sources:
        quality = str(source.get("source_quality") or "needs-review")
        counts[quality] = counts.get(quality, 0) + 1
    return {
        "counts": counts,
        "official_source_count": len(official.get("official_sources", [])),
        "verified_source_count": len(verify.get("verified_sources", [])),
        "rejected_source_count": len(verify.get("rejected_sources", [])),
    }


def _audit_native_research(
    plan: dict[str, Any],
    sources: list[dict[str, Any]],
    verify: dict[str, Any],
    official: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    blocking_issues = 0
    if not official.get("official_sources") and _official_source_expected(plan):
        issues.append(
            {
                "issue": "No official or primary source identified for a topic that likely needs one.",
                "severity": "High",
                "fix": "Run another official-source hunt or clearly lower confidence in the final report.",
            }
        )
        blocking_issues += 1
    elif not official.get("official_sources"):
        issues.append(
            {
                "issue": "No official or primary source was found, so the report should lean on clearly attributed secondary reporting.",
                "severity": "Medium",
                "fix": "Prefer stronger domain-authority or top-tier reporting and label uncertainty where direct primary evidence is unavailable.",
            }
        )
    if len(verify.get("verified_sources", [])) < 2:
        issues.append(
            {
                "issue": "Too few verified sources for a strong final report.",
                "severity": "High",
                "fix": "Select more strong URLs and verify them before synthesis.",
            }
        )
        blocking_issues += 1
    if not sources:
        issues.append(
            {
                "issue": "No candidate sources were collected.",
                "severity": "High",
                "fix": "Re-run the search subagents with broader queries.",
            }
        )
        blocking_issues += 1
    return {
        "audit_result": "Pass" if blocking_issues == 0 else "Needs revision",
        "issues_found": issues,
        "safe_to_write_final_report": blocking_issues == 0,
        "remaining_tasks": [item["fix"] for item in issues],
        "topic_type": plan.get("topic_type"),
    }


def _build_public_reasoning_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": [
            "Classify the topic and identify freshness risk.",
            "Search official or primary sources first.",
            "Compare reputable secondary sources.",
            "Verify key URLs, then audit contradictions.",
        ],
        "evidence_standard": [
            "Official or primary sources for key facts whenever available.",
            "Top-tier news for developments and reactions.",
            "Background sources for context only.",
        ],
        "verification_result": {
            "topic_type": plan.get("topic_type"),
            "freshness_needs": plan.get("freshness_needs", []),
        },
    }


def _subagent_brain_assignments(agent_names: list[str]) -> list[dict[str, Any]]:
    return _SUBAGENT_MANAGER._select_assignments(agent_names)


def _subagent_role(agent_name: str) -> str:
    lowered = agent_name.lower()
    if lowered.startswith("web-"):
        return "web"
    if lowered.startswith("wiki-"):
        return "wiki"
    if lowered.startswith("verify-"):
        return "verify"
    if "fixer" in lowered:
        return "fixer"
    return "general"


def _provider_model_pool(config: Any) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for alias_name, alias in config.anthropic_models.items():
        provider = config.providers.get(alias.provider)
        if provider is None:
            continue
        model = alias.model or provider.default_model or ""
        if not _provider_is_usable(provider, model):
            continue
        key = (provider.name, model, alias_name)
        if key in seen:
            continue
        seen.add(key)
        pool.append(_provider_pool_entry(provider, model, alias_name))
    for provider in config.providers.values():
        model = provider.default_model or ""
        if not _provider_is_usable(provider, model):
            continue
        key = (provider.name, model, None)
        if key in seen:
            continue
        seen.add(key)
        pool.append(_provider_pool_entry(provider, model, None))
    return pool


def _provider_is_usable(provider: Any, model: str) -> bool:
    if not model:
        return False
    api_key = str(getattr(provider, "api_key", "") or "")
    is_local = str(getattr(provider, "base_url", "") or "").startswith("http://127.0.0.1") or str(getattr(provider, "base_url", "") or "").startswith("http://localhost")
    if api_key and api_key.startswith("${"):
        return False
    if not api_key and not is_local:
        return False
    return _provider_limit_headroom(provider) > 0.0


def _provider_limit_headroom(provider: Any) -> float:
    limits = getattr(provider, "usage_limits", {}) or {}
    scores: list[float] = []
    for entry in limits.values():
        if not isinstance(entry, dict):
            continue
        limit = entry.get("limit")
        used = entry.get("used", 0)
        try:
            limit_value = float(limit)
            used_value = float(used)
        except (TypeError, ValueError):
            continue
        if limit_value <= 0:
            continue
        scores.append(max(0.0, (limit_value - used_value) / limit_value))
    return min(scores) if scores else 1.0


def _provider_pool_entry(provider: Any, model: str, alias_name: str | None) -> dict[str, Any]:
    return {
        "provider": provider.name,
        "model": model,
        "alias": alias_name,
        "provider_type": provider.type,
        "base_url": provider.base_url,
        "supports_tools": bool(getattr(provider, "supports_tools", True)),
        "headroom": _provider_limit_headroom(provider),
        "local": str(provider.base_url).startswith("http://127.0.0.1") or str(provider.base_url).startswith("http://localhost"),
    }


def _rank_provider_candidates(pool: list[dict[str, Any]], *, role: str, used_counts: dict[str, int]) -> list[dict[str, Any]]:
    preferences = {
        "web": {"haiku", "sonnet"},
        "wiki": {"haiku", "sonnet"},
        "verify": {"sonnet", "opus", "haiku"},
        "fixer": {"opus", "sonnet", "haiku"},
        "general": {"sonnet", "haiku", "opus"},
    }
    preferred_aliases = preferences.get(role, preferences["general"])

    def rank(item: dict[str, Any]) -> tuple[int, float, int, int]:
        alias = str(item.get("alias") or "")
        alias_penalty = 0 if alias in preferred_aliases else 1
        key = f"{item.get('provider')}::{item.get('model')}"
        used_penalty = used_counts.get(key, 0)
        headroom_score = -float(item.get("headroom") or 0.0)
        local_penalty = 0 if item.get("local") else 1
        return (alias_penalty, headroom_score, used_penalty, local_penalty)

    return sorted(pool, key=rank)


def _build_compact_claims(*briefs: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for brief in briefs:
        for item in brief.get("compact_claims", []):
            if not isinstance(item, dict):
                continue
            claim = _compact_text(item.get("claim"), 220)
            if not claim or claim.lower() in seen:
                continue
            seen.add(claim.lower())
            claims.append(
                {
                    "claim": claim,
                    "supporting_url": item.get("supporting_url"),
                    "source": item.get("source"),
                }
            )
            if len(claims) >= 30:
                return claims
    return claims


def _background_notes_from_brief(brief: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for item in brief.get("compact_claims", []):
        if not isinstance(item, dict):
            continue
        claim = _compact_text(item.get("claim"), 220)
        if claim:
            notes.append(claim)
        if len(notes) >= 12:
            break
    return notes


def _suggested_report_outline(plan: dict[str, Any]) -> list[str]:
    return [
        "Information Last Checked",
        "Executive Summary",
        "Background",
        "Key Verified Facts",
        "Data and Statistics",
        "Detailed Analysis",
        "Stakeholders",
        "Timeline",
        "Controversies and Uncertainties",
        "Source Quality Review",
        "Conclusion",
        f"Confidence Rating ({plan.get('topic_type', 'general topic')})",
    ]


def _build_checkpoint_markdown(
    *,
    topic: str,
    plan: dict[str, Any],
    web_briefs: list[dict[str, Any]],
    wiki_briefs: list[dict[str, Any]],
    verification_briefs: list[dict[str, Any]],
    official: dict[str, Any],
    verify: dict[str, Any],
    audit: dict[str, Any],
    images: dict[str, Any] | None,
) -> str:
    web_count = len(web_briefs)
    wiki_count = len(wiki_briefs)
    verify_count = len(verification_briefs)
    verified_count = len(verify.get("verified_sources", []))
    rejected_count = len(verify.get("rejected_sources", []))
    final_confidence = str(audit.get("final_confidence") or "Unknown")
    lines = [
        f"# Deep Research Checkpoint: {topic}",
        "",
        "## Status",
        f"- Final confidence: {final_confidence}",
        f"- Verified sources: {verified_count}",
        f"- Rejected or uncertain sources: {rejected_count}",
        "",
        "## Completed",
        "- Query plan created",
        f"- {web_count} web research briefs collected",
        f"- {wiki_count} wiki context briefs collected",
        f"- {verify_count} verification passes completed",
        "- Final fixer pass completed",
        "",
        "## Web Subagents",
    ]
    for item in web_briefs:
        lines.append(f"- {item.get('name')}: {len((item.get('brief') or {}).get('compact_claims', []))} claims")
    lines.extend([
        "",
        "## Wiki Subagents",
    ])
    for item in wiki_briefs:
        lines.append(f"- {item.get('name')}: {len((item.get('brief') or {}).get('compact_claims', []))} notes")
    lines.extend([
        "",
        "## Verification Subagents",
    ])
    for item in verification_briefs:
        brief = item.get("brief") or {}
        lines.append(f"- {item.get('name')}: {brief.get('verdict')}")
    lines.extend([
        "",
        "## Selected URLs",
    ])
    for item in verify.get("verified_sources", [])[:10]:
        lines.append(f"- {item.get('url')}")
    lines.extend(["", "## Official Sources"])
    for item in official.get("official_sources", [])[:8]:
        lines.append(f"- {item.get('title')}: {item.get('url')}")
    lines.extend(["", "## Remaining Tasks"])
    tasks = audit.get("remaining_tasks", [])
    if tasks:
        lines.extend([f"- {task}" for task in tasks])
    else:
        lines.append("- None")
    if images and images.get("results"):
        lines.extend(["", "## Image Candidates"])
        for item in images.get("results", []):
            lines.append(f"- {item.get('title')}: {item.get('url')}")
    return "\n".join(lines)


def _write_deep_temp_file(path: str, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.rstrip() + "\n", encoding="utf-8")
    return target.as_posix()


def _build_collection_markdown(
    *,
    topic: str,
    plan: dict[str, Any],
    search_briefs: list[dict[str, Any]],
    wiki_briefs: list[dict[str, Any]],
    official: dict[str, Any],
    selected_urls: list[str],
) -> str:
    lines = [
        f"# Research Draft: {topic}",
        "",
        "## Status",
        "- Collection complete",
        "- Review pending",
        "",
        "## Main Questions",
    ]
    lines.extend(f"- {item}" for item in plan.get("main_questions", []))
    freshness_needs = plan.get("freshness_needs", [])
    if freshness_needs:
        lines.extend(["", "## Freshness Checks"])
        lines.extend(f"- {item}" for item in freshness_needs)
    lines.extend(["", "## Search Subagents"])
    for item in search_briefs:
        brief = item.get("brief") or {}
        lines.append(f"- {item.get('name')}: {len(brief.get('compact_claims', []))} compact claims")
    lines.extend(["", "## Wiki Subagents"])
    for item in wiki_briefs:
        brief = item.get("brief") or {}
        lines.append(f"- {item.get('name')}: {len(brief.get('compact_claims', []))} background notes")
    lines.extend(["", "## Official Sources"])
    official_sources = official.get("official_sources", [])
    if official_sources:
        for source in official_sources[:8]:
            lines.append(f"- {source.get('title')}: {source.get('url')}")
    else:
        lines.append("- None clearly identified yet")
    lines.extend(["", "## Review Shortlist"])
    for url in selected_urls[:10]:
        lines.append(f"- {url}")
    lines.extend([
        "",
        "## Next Step",
        "- Run deep_review_agent, then replace this file with the reviewed markdown.",
    ])
    return "\n".join(lines)


def _build_final_handoff(
    topic: str,
    fixer: dict[str, Any],
    verify: dict[str, Any],
    source_review: dict[str, Any],
) -> dict[str, Any]:
    return {
        "topic": topic,
        "ready_for_main_ai": bool(fixer.get("safe_to_write_final_report")),
        "final_confidence": fixer.get("final_confidence"),
        "compact_claims": fixer.get("compact_claims", []),
        "verified_sources": verify.get("verified_sources", []),
        "source_quality_review": source_review,
        "remaining_tasks": fixer.get("remaining_tasks", []),
        "instruction": (
            "Use the reviewed markdown and verified sources to write the final report.md. "
            "Clearly label uncertain claims and keep citations close to the claim they support."
        ),
    }


def _deep_agent_queries(topic: str, focus: str, *, count: int, agent: str) -> list[str]:
    focus_text = f" {focus}" if focus else ""
    if agent == "deep_wiki_agent":
        candidates = [
            topic,
            f"{topic} background",
            f"{topic} history key people parties institutions",
            f"{topic} political context",
        ]
    elif agent == "deep_serp_agent":
        candidates = [
            f"{topic}{focus_text}",
            f"{topic} official primary source results schedule data",
            f"{topic} Reuters AP BBC Bloomberg latest news analysis",
            f"{topic} major news channel newspaper key actors candidates parties",
            f"{topic} forecasts polling predictions official source top news",
            f"{topic} source verification primary sources",
        ]
    else:
        candidates = [
            f"{topic}{focus_text}",
            f"{topic} latest current landscape Reuters AP BBC top news",
            f"{topic} key candidates parties issues major news official",
            f"{topic} polling predictions forecasts official primary source",
            f"{topic} official schedule election commission government source",
            f"{topic} analysis background context established news",
        ]
    queries: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            queries.append(normalized)
        if len(queries) >= count:
            break
    return queries


def _deep_agent_payload(provider: str, query: str) -> dict[str, Any]:
    if provider == "tavily_search":
        return {"query": query, "search_depth": "advanced", "max_results": 8, "include_answer": True}
    if provider == "serpapi_search":
        return {"query": query, "num": 8}
    return {"query": query, "limit": 8, "language": "en"}


def _deep_verify_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    urls = arguments.get("urls") or []
    if not isinstance(urls, list):
        urls = []
    return {
        "urls": [str(url) for url in urls[:6] if str(url).strip()],
        "claim": str(arguments.get("claim") or arguments.get("topic") or "the researched claim"),
        "required_verified_sources": min(max(_int_argument(arguments.get("required_verified_sources"), 3), 1), 6),
        "verify_timeout_seconds": min(max(_int_argument(arguments.get("verify_timeout_seconds"), 10), 1), 120),
    }


def _deep_session_id(arguments: dict[str, Any], topic: str) -> str:
    session_id = str(arguments.get("session_id") or "").strip()
    if session_id:
        return session_id
    session = _SUBAGENT_MANAGER.spawn(
        topic,
        [
            "tavily-realtime-1",
            "tavily-realtime-2",
            "serp-realtime-1",
            "serp-realtime-2",
            "wiki-context-1",
            "wiki-context-2",
            "wiki-context-3",
            "review-verifier",
            "markdown-reviewer",
            "final-handoff",
        ],
    )
    return str(session.get("session_id"))


def _compact_search_sources(provider: str, query: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _unwrap_tool_payload(data)
    if provider == "serpapi_search":
        items = payload.get("organic_results") or payload.get("results") or []
    else:
        items = payload.get("results") or []
    if not isinstance(items, list):
        items = []
    sources: list[dict[str, Any]] = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("name") or item.get("page_title") or "Untitled source"
        url = item.get("url") or item.get("link") or item.get("source_url")
        snippet = (
            item.get("content")
            or item.get("snippet")
            or item.get("summary")
            or item.get("extract")
            or item.get("description")
            or ""
        )
        sources.append(
            {
                "query": query,
                "title": _compact_text(title, 140),
                "url": str(url or ""),
                "evidence": _compact_text(snippet, 800),
                "source_quality": _source_quality(str(url or "")),
            }
        )
    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        sources.insert(
            0,
            {
                "query": query,
                "title": f"{provider} answer summary",
                "url": "",
                "evidence": _compact_text(answer, 900),
                "source_quality": "provider-summary",
            },
        )
    return sources


def _compact_deep_agent_result(
    *,
    agent: str,
    provider: str,
    topic: str,
    focus: str,
    queries: list[str],
    sources: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        key = str(source.get("url") or source.get("title") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    deduped.sort(key=lambda source: _source_quality_rank(str(source.get("source_quality") or "")))
    deduped = deduped[:24]
    claims = [
        {
            "claim": _compact_text(source.get("evidence"), 420),
            "supporting_url": source.get("url"),
            "source": source.get("title"),
        }
        for source in deduped[:16]
        if source.get("evidence")
    ]
    return {
        "ok": True,
        "agent": agent,
        "provider": provider,
        "topic": topic,
        "focus": focus,
        "queries_run": queries,
        "compact_claims": claims,
        "candidate_sources": deduped,
        "errors": errors[:5],
        "handoff": "Use this compact brief only; do not ask for raw search dumps unless verification fails.",
    }


def _compact_verify_agent_result(data: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_tool_payload(data)
    verified = payload.get("verified_sources") or payload.get("verified") or []
    rejected = payload.get("rejected_sources") or payload.get("rejected") or []
    return {
        "ok": data.get("ok", True),
        "agent": "deep_verify_agent",
        "verdict": payload.get("verdict") or payload.get("status") or "verification complete",
        "verified_sources": _compact_source_list(verified, limit=8),
        "rejected_sources": _compact_source_list(rejected, limit=5),
        "notes": _compact_text(payload.get("summary") or payload.get("notes") or payload.get("message") or "", 700),
        "handoff": "Use verified_sources for citations; treat rejected_sources as leads or uncertainty only.",
    }


def _compact_source_list(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            result.append(
                {
                    "title": _compact_text(item.get("title") or item.get("url") or "source", 140),
                    "url": str(item.get("url") or ""),
                    "evidence": _compact_text(item.get("evidence") or item.get("summary") or item.get("snippet") or "", 350),
                    "status": item.get("status") or item.get("verdict") or item.get("supports"),
                }
            )
    return result


def _unwrap_tool_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("data")
    if isinstance(payload, dict):
        return payload
    result = data.get("result")
    if isinstance(result, dict):
        nested = result.get("data")
        return nested if isinstance(nested, dict) else result
    return data


def _compact_text(value: Any, max_length: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _source_quality(url: str) -> str:
    lowered = url.lower()
    official_markers = (
        ".gov",
        ".gov.",
        ".edu",
        ".ac.",
        ".mil",
        ".int",
        "eci.gov.in",
        "pib.gov.in",
        "parliament",
        "data.gov",
        "who.int",
        "un.org",
        "worldbank.org",
        "imf.org",
        "oecd.org",
        "sec.gov",
        "rbi.org.in",
        "isro.gov.in",
        "nasa.gov",
    )
    top_news_markers = (
        "reuters.com",
        "apnews.com",
        "bbc.",
        "bloomberg.com",
        "ft.com",
        "financialtimes.com",
        "wsj.com",
        "nytimes.com",
        "washingtonpost.com",
        "theguardian.com",
        "cnn.com",
        "cnbc.com",
        "aljazeera.com",
        "economist.com",
        "npr.org",
        "pbs.org",
        "abcnews.go.com",
        "cbsnews.com",
        "nbcnews.com",
        "thehindu.com",
        "indianexpress.com",
        "hindustantimes.com",
        "livemint.com",
        "frontline.thehindu.com",
        "ndtv.com",
        "thewire.in",
    )
    scholarly_markers = (
        "nature.com",
        "science.org",
        "nejm.org",
        "thelancet.com",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "arxiv.org",
        "doi.org",
        "jstor.org",
        "ieee.org",
        "acm.org",
        "springer.com",
        "wiley.com",
        "sciencedirect.com",
    )
    weak_markers = (
        "blogspot.",
        "medium.com",
        "substack.com",
        "wordpress.",
        "quora.com",
        "reddit.com",
        "pinterest.",
        "fandom.com",
    )
    if any(domain in lowered for domain in official_markers):
        return "primary/official"
    if any(domain in lowered for domain in scholarly_markers):
        return "scholarly/domain-authority"
    if any(domain in lowered for domain in top_news_markers):
        return "top-tier-news"
    if "wikipedia.org" in lowered:
        return "encyclopedia-background"
    if any(domain in lowered for domain in weak_markers):
        return "weak-lead/random-web"
    return "needs-review"


def _source_quality_rank(source_quality: str) -> int:
    order = {
        "primary/official": 0,
        "scholarly/domain-authority": 1,
        "top-tier-news": 2,
        "established-news": 3,
        "encyclopedia-background": 4,
        "provider-summary": 5,
        "needs-review": 6,
        "weak-lead/random-web": 7,
    }
    return order.get(source_quality, 6)


def _int_argument(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _deep_research_handoff(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query") or arguments.get("topic") or "").strip()
    if not query:
        query = "the requested research topic"
    text = json.dumps(
        {
            "ok": True,
            "tool": "deep",
            "workflow": "lead_agent_with_specialist_research_subagents",
            "topic": query,
            "run_mode": {
                "poolside": "Run in normal mode and keep the workflow simple.",
                "instruction": "Do not switch to Plan mode. The main brain should call one small MCP tool at a time and continue from compact results.",
            },
            "minimum_work": {
                "subagent_calls": [
                    "deep_plan_agent",
                    "deep_collect_agent",
                    "deep_tavily_agent",
                    "deep_tavily_agent",
                    "deep_serp_agent",
                    "deep_serp_agent",
                    "deep_wiki_agent",
                    "deep_wiki_agent",
                    "deep_wiki_agent",
                    "deep_collect_agent",
                    "deep_review_agent",
                    "deep_verify_agent",
                    "deep_verify_agent",
                    "deep_verify_agent",
                    "deep_review_agent",
                    "deep_image_agent",
                    "deep_master_review_agent",
                    "deep_master_spelling_grammar_agent",
                    "deep_master_evidence_validity_agent",
                    "deep_master_evidence_reliability_agent",
                    "deep_master_citation_coverage_agent",
                    "deep_master_neutrality_bias_agent",
                    "deep_master_logic_consistency_agent",
                    "deep_master_format_quality_agent",
                    "deep_master_final_synthesis_agent",
                    "deep_master_review_agent",
                ],
                "tavily_collection_agents": 2,
                "serpapi_collection_agents": 2,
                "wikipedia_collection_agents": 3,
                "verified_urls": 4,
                "required_file": "report.md",
                "rule": "Do not paste raw search result dumps into the main context. Use compact sub-agent briefs and write checkpoints before synthesis.",
            },
            "steps": [
                {
                    "name": "start",
                    "action": "Start the normal-mode deep research workflow, collect evidence with separate MCP calls, verify the strongest claims with separate MCP calls, and write report.md.",
                },
                {
                    "name": "native_plan_controller",
                    "tool": "llama_bridge_tools__deep_plan_agent",
                    "arguments": {
                        "topic": query,
                        "query_count": 2,
                        "required_verified_sources": 4,
                        "include_official_hunt": True,
                    },
                    "returns": "Session id and a compact research plan.",
                },
                {
                    "name": "collection_call_list",
                    "tool": "llama_bridge_tools__deep_collect_agent",
                    "arguments": {"topic": query, "query_count": 2, "include_official_hunt": True},
                    "returns": "agent_calls: exact single-agent MCP calls to run separately.",
                },
                {
                    "name": "single_collection_agents",
                    "tool": "llama_bridge_tools__deep_tavily_agent / deep_serp_agent / deep_wiki_agent",
                    "arguments": "Use each item from agent_calls exactly, including agent_name, as its own separate MCP call.",
                    "returns": "One compact JSON brief per single collection agent.",
                },
                {
                    "name": "collection_finalize",
                    "tool": "llama_bridge_tools__deep_collect_agent",
                    "arguments_template": {
                        "session_id": "Use the session_id from deep_plan_agent.",
                        "topic": query,
                        "include_official_hunt": True,
                        "subagent_briefs": "Array of compact JSON results returned by the single collection agents.",
                    },
                    "returns": "Compact collected evidence and a draft markdown summary.",
                },
                {
                    "name": "review_call_list",
                    "tool": "llama_bridge_tools__deep_review_agent",
                    "arguments_template": {
                        "session_id": "Use the session_id from deep_plan_agent/deep_collect_agent.",
                        "topic": query,
                        "required_verified_sources": 4,
                        "verify_timeout_seconds": 6,
                    },
                    "returns": "agent_calls: exact deep_verify_agent MCP calls to run separately.",
                },
                {
                    "name": "single_verify_agents",
                    "tool": "llama_bridge_tools__deep_verify_agent",
                    "arguments": "Use each item from agent_calls exactly, including agent_name, as its own separate MCP call.",
                    "returns": "One compact verification brief per single verify agent.",
                },
                {
                    "name": "review_finalize",
                    "tool": "llama_bridge_tools__deep_review_agent",
                    "arguments_template": {
                        "session_id": "Use the same session_id.",
                        "topic": query,
                        "required_verified_sources": 4,
                        "verification_briefs": "Array of compact JSON results returned by the single deep_verify_agent calls.",
                    },
                    "returns": "Verified findings and the final handoff for the main AI.",
                },
                {
                    "name": "deep_images",
                    "tool": "llama_bridge_tools__deep_image_agent",
                    "arguments": {
                        "topic": query,
                        "max_results": 3,
                        "output_dir": "report_assets/images",
                    },
                    "returns": "Downloaded image assets with local_path and markdown snippets for report.md.",
                },
                {
                    "name": "master_review_call_list",
                    "tool": "llama_bridge_tools__deep_master_review_agent",
                    "arguments_template": {
                        "session_id": "Use the same session_id.",
                        "topic": query,
                        "reviewed_markdown": "Use reviewed_markdown from deep_review_agent.",
                        "final_handoff": "Use final_handoff from deep_review_agent.",
                    },
                    "returns": "agent_calls: exactly 8 deep master-review MCP calls to run separately.",
                },
                {
                    "name": "eight_master_review_agents",
                    "tool": "llama_bridge_tools__deep_master_spelling_grammar_agent / deep_master_evidence_validity_agent / deep_master_evidence_reliability_agent / deep_master_citation_coverage_agent / deep_master_neutrality_bias_agent / deep_master_logic_consistency_agent / deep_master_format_quality_agent / deep_master_final_synthesis_agent",
                    "arguments": "Use each item from agent_calls exactly as its own separate MCP call.",
                    "returns": "One compact master-review brief per configured review area.",
                },
                {
                    "name": "master_review_finalize",
                    "tool": "llama_bridge_tools__deep_master_review_agent",
                    "arguments_template": {
                        "session_id": "Use the same session_id.",
                        "topic": query,
                        "reviewed_markdown": "Use reviewed_markdown from deep_review_agent.",
                        "final_handoff": "Use final_handoff from deep_review_agent.",
                        "master_review_briefs": "Array of compact JSON results returned by the 8 deep master-review calls.",
                    },
                    "returns": "Final LLM instructions and final handoff after master review.",
                },
                {
                    "name": "checkpoint",
                    "action": "If context gets tight, write a compact checkpoint so the report can be finished smoothly.",
                },
                {
                    "name": "write_report",
                    "tool": "write",
                    "arguments_template": {
                        "path": "report.md",
                        "content": "<full markdown report with inline source URLs/citations, verified findings, uncertainty, and final warning>",
                    },
                    "action": "Create report.md in the current working directory. The file write is required before the final answer.",
                },
                {
                    "name": "complete_plan",
                    "action": "Finish the report and close out the short plan.",
                },
            ],
            "instructions": [
                "Run in normal mode; do not switch to Plan mode or create todos.",
                "Use llama_bridge_tools__deep_plan_agent first when the deep tools are available.",
                "Use llama_bridge_tools__deep_collect_agent first only to get agent_calls.",
                "Run each returned collection agent as its own separate MCP call, one by one.",
                "Call llama_bridge_tools__deep_collect_agent again with subagent_briefs to assemble compact evidence.",
                "Use llama_bridge_tools__deep_review_agent first only to get deep_verify_agent calls.",
                "Run each returned verification agent as its own separate MCP call, one by one.",
                "Call llama_bridge_tools__deep_review_agent again with verification_briefs before writing.",
                "Call llama_bridge_tools__deep_image_agent if images can improve report.md; use returned local_path values in markdown embeds.",
                "Call llama_bridge_tools__deep_master_review_agent to get the 8 master-review MCP calls.",
                "Run all 8 deep master-review calls separately: spelling_grammar, evidence_validity, evidence_reliability, citation_coverage, neutrality_bias, logic_consistency, format_quality, and final_synthesis.",
                "Call llama_bridge_tools__deep_master_review_agent again with master_review_briefs before report.md or final submission to the main LLM.",
                "Use extra specialist tools only when they clearly improve coverage or verification.",
                "Keep only compact findings, URLs, source quality, and caveats in the main context.",
                "Write report.md in the current working directory before producing the final response.",
                "If context is compacted or nearly full, continue from a checkpoint instead of restarting.",
            ],
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _prompt_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "serp",
            "description": "Search the web with SerpAPI through llama bridge.",
            "arguments": [{"name": "query", "description": "Search query", "required": True}],
        },
        {
            "name": "tavily",
            "description": "Search the web with Tavily through llama bridge.",
            "arguments": [{"name": "query", "description": "Search query", "required": True}],
        },
        {
            "name": "web",
            "description": "Search the web through llama bridge.",
            "arguments": [{"name": "query", "description": "Search query", "required": True}],
        },
        {
            "name": "image",
            "description": "Find sourced image candidates through llama bridge.",
            "arguments": [{"name": "query", "description": "Image query", "required": True}],
        },
        {
            "name": "wiki",
            "description": "Search Wikipedia through llama bridge.",
            "arguments": [{"name": "query", "description": "Wikipedia search query", "required": True}],
        },
        {
            "name": "deep",
            "description": "Run a sourced research workflow with llama bridge tools.",
            "arguments": [{"name": "topic", "description": "Research topic", "required": True}],
        },
        {
            "name": "manim",
            "description": "Generate a Manim animation video from text.",
            "arguments": [{"name": "prompt", "description": "Animation prompt", "required": True}],
        },
    ]


def _prompt_response(name: str, params: dict[str, Any]) -> dict[str, Any]:
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    text = str(arguments.get("query") or arguments.get("topic") or arguments.get("prompt") or "").strip()
    prompt_name = name
    missing_inputs = {
        "serp": "Ask the user for the search query, then use the llama bridge SerpAPI MCP tool.",
        "tavily": "Ask the user for the search query, then use the llama bridge Tavily MCP tool.",
        "web": "Ask the user for the web search query, then use the best available llama bridge search MCP tool.",
        "image": "Ask the user for the image search topic, then use the llama bridge image_research MCP tool.",
        "wiki": "Ask the user for the Wikipedia search query, then use the llama bridge wikipedia_search MCP tool.",
        "deep": "Ask the user for the research topic, then run the llama bridge deep research workflow.",
        "manim": "Ask the user what animation to create, then use the llama bridge manim_render MCP tool.",
    }
    if not text:
        if prompt_name not in missing_inputs:
            raise ValueError(f"Unknown prompt: {name}")
        prompt_text = missing_inputs[prompt_name]
        return {
            "description": next((item["description"] for item in _prompt_definitions() if item["name"] == name), name),
            "messages": [{"role": "user", "content": {"type": "text", "text": prompt_text}}],
        }
    prompts = {
        "serp": f"Use the llama bridge SerpAPI MCP tool for this search query: {text}",
        "tavily": f"Use the llama bridge Tavily MCP tool for this search query: {text}",
        "web": f"Use the llama bridge web or Tavily MCP search tool for this query: {text}",
        "image": f"Use the llama bridge image_research MCP tool and return compact sourced image candidates for: {text}",
        "wiki": f"Use the llama bridge wikipedia_search MCP tool for this query: {text}",
        "manim": (
            "Use the llama bridge manim_render MCP tool to create a short Manim Community animation video. "
            "Pass the user's text as prompt, use quality='low' unless the user asks otherwise, and return "
            f"the scene_path and video_path. Animation prompt: {text}"
        ),
        "deep": (
            "Run deep research in normal mode. Do not switch to Plan mode and do not create todos. Use small "
            "steps: call llama_bridge_tools__deep or deep_plan_agent first. For /deep, only call tools whose names contain deep; "
            "do not call normal search, source, image, Wikipedia, weather, or time tools. "
            "Then call deep_collect_agent to get agent_calls, then run every listed "
            "deep_tavily_agent/deep_serp_agent/deep_wiki_agent as its own separate MCP call, one by one, then call "
            "deep_collect_agent again with subagent_briefs, then write temp/ad.md, then call deep_review_agent to get verify agent_calls, "
            "then run every listed deep_verify_agent as its own separate MCP call, one by one, then call deep_review_agent again with verification_briefs, then call deep_image_agent to download report-ready image assets when useful, then call deep_master_review_agent to get 8 master-review agent_calls, "
            "then run every listed deep_master_* agent separately, then call deep_master_review_agent again with master_review_briefs, then overwrite temp/ad.md with the reviewed markdown; "
            "choose at least 4 strong URLs from the deep briefs, write "
            "report.md in the current working directory, and only then "
            f"give the final response. If the model thinks deep tools are unavailable, call llama_bridge_tools__deep or deep_plan_agent anyway and report the deep-tool failure instead of switching to normal tools. Topic: {text}"
        ),
    }
    if prompt_name not in prompts:
        raise ValueError(f"Unknown prompt: {name}")
    return {
        "description": next((item["description"] for item in _prompt_definitions() if item["name"] == name), name),
        "messages": [{"role": "user", "content": {"type": "text", "text": prompts[prompt_name]}}],
    }


if __name__ == "__main__":
    main()
