from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROTOCOL_VERSION = "2025-06-18"


def main() -> None:
    server = BridgeMcpServer(
        base_url=os.environ.get("LLAMA_BRIDGE_BASE_URL", "http://127.0.0.1:8089").rstrip("/"),
        api_key=os.environ.get("LLAMA_BRIDGE_API_KEY", ""),
    )
    server.run()


class BridgeMcpServer:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key

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
            return []

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
        if name == "deep_claude_agent":
            return _call_claude_deep_agent(self.base_url, self.api_key, arguments)
        if name in {"deep_tavily_agent", "deep_serp_agent", "deep_wiki_agent", "deep_verify_agent"}:
            return self._call_deep_agent(name, arguments)
        name, arguments = _normalize_tool_call(name, arguments)
        if name == "source_research":
            arguments = _poolside_source_research_arguments(arguments)
        data = self._request("POST", f"/api/tools/{name}", arguments)
        is_error = bool(data.get("ok") is False)
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    def _call_deep_agent(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "deep_verify_agent":
            data = self._request("POST", "/api/tools/verify_sources", _deep_verify_arguments(arguments))
            text = json.dumps(_compact_verify_agent_result(data), indent=2, ensure_ascii=False, default=str)
            return {"content": [{"type": "text", "text": text}], "isError": bool(data.get("ok") is False)}

        topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
        focus = str(arguments.get("focus") or "").strip()
        count = min(max(_int_argument(arguments.get("query_count"), 4), 1), 6)
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
                data = self._request("POST", f"/api/tools/{provider}", payload)
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
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False, default=str)}], "isError": False}

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        payload = None
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=payload, headers=headers, method=method)
        try:
            with urlopen(request, timeout=300) as response:  # noqa: S310 - user-configured local bridge URL.
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
                "Start a Poolside-friendly planned deep research workflow for /deep. This quick "
                "router splits the work into small search, verification, image, and synthesis steps."
            ),
            "inputSchema": deep_schema,
        },
        {
            "name": "deep_claude_agent",
            "description": (
                "Use Claude Agent SDK subagents for /deep research. A lead Claude agent delegates to "
                "specialized Tavily, SerpAPI, Wikipedia, and verification subagents and returns a compact handoff."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Main research topic."},
                    "query": {"type": "string", "description": "Alias for topic."},
                    "max_turns": {"type": "integer", "default": 12, "minimum": 4, "maximum": 24},
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
                    "required_verified_sources": {"type": "integer", "default": 3, "minimum": 1, "maximum": 6},
                },
                "required": ["urls"],
            },
        },
    ]


def _deep_search_agent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Main research topic."},
            "focus": {"type": "string", "description": "Optional subtopic assigned by the lead agent."},
            "query_count": {"type": "integer", "default": 4, "minimum": 1, "maximum": 6},
        },
        "required": ["topic"],
    }


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


def _call_claude_deep_agent(base_url: str, api_key: str, arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or arguments.get("query") or "").strip() or "the research topic"
    max_turns = min(max(_int_argument(arguments.get("max_turns"), 12), 4), 24)
    try:
        text = asyncio.run(_run_claude_deep_agent(base_url, api_key, topic=topic, max_turns=max_turns))
    except ImportError:
        text = json.dumps(
            {
                "ok": False,
                "agent": "deep_claude_agent",
                "error": "claude-agent-sdk is not installed. Install with: pip install claude-agent-sdk",
                "fallback": "Use deep_tavily_agent, deep_serp_agent, deep_wiki_agent, and deep_verify_agent.",
            },
            indent=2,
        )
        return {"content": [{"type": "text", "text": text}], "isError": False}
    except Exception as exc:  # noqa: BLE001 - report SDK failures as fallback instructions.
        text = json.dumps(
            {
                "ok": False,
                "agent": "deep_claude_agent",
                "error": str(exc),
                "fallback": "Use deep_tavily_agent, deep_serp_agent, deep_wiki_agent, and deep_verify_agent.",
            },
            indent=2,
            ensure_ascii=False,
        )
        return {"content": [{"type": "text", "text": text}], "isError": False}
    return {"content": [{"type": "text", "text": text}], "isError": False}


async def _run_claude_deep_agent(base_url: str, api_key: str, *, topic: str, max_turns: int) -> str:
    try:
        import claude_agent_sdk
        from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, query
    except ImportError:
        raise

    mcp_servers = {
        "llama_bridge_tools": {
            "command": sys.executable,
            "args": ["-m", "llama_bridge.mcp_tools"],
            "env": {
                "LLAMA_BRIDGE_BASE_URL": base_url,
                "LLAMA_BRIDGE_API_KEY": api_key,
            },
        }
    }
    agents = {
        "tavily-researcher": AgentDefinition(
            description="Tavily current-web research specialist. Use for current news, current web evidence, and source discovery.",
            prompt=(
                "You are the Tavily research subagent. Run focused Tavily searches through the llama bridge MCP tool. "
                "Return only a compact brief: 5-8 claims, best URLs, source quality, conflicts, and what still needs verification. "
                "Do not write files. Do not include raw search dumps."
            ),
            tools=["mcp__llama_bridge_tools__tavily_search", "WebFetch"],
            model="sonnet",
        ),
        "serpapi-researcher": AgentDefinition(
            description="SerpAPI search specialist. Use for broad Google-style result discovery, official pages, and source diversity.",
            prompt=(
                "You are the SerpAPI research subagent. Run focused SerpAPI searches through the llama bridge MCP tool. "
                "Return only compact evidence: claims, best URLs, source quality, conflicts, and verification priorities. "
                "Do not write files. Do not include raw search dumps."
            ),
            tools=["mcp__llama_bridge_tools__serpapi_search", "WebFetch"],
            model="sonnet",
        ),
        "wiki-backgrounder": AgentDefinition(
            description="Wikipedia/background specialist. Use for entity background, historical context, terminology, and timelines.",
            prompt=(
                "You are the Wikipedia/background subagent. Use Wikipedia search/page tools for context only. "
                "Return compact background notes, entity names, dates, and caveats. Do not treat Wikipedia as final proof."
            ),
            tools=[
                "mcp__llama_bridge_tools__wikipedia_search",
                "mcp__llama_bridge_tools__wikipedia_page",
                "WebFetch",
            ],
            model="haiku",
        ),
        "source-verifier": AgentDefinition(
            description="Source verification specialist. Use after candidate URLs are selected to verify claims and citation quality.",
            prompt=(
                "You are the verification subagent. Use verify_sources and WebFetch to test whether selected URLs support the claim. "
                "Return a compact verdict with verified URLs, rejected/weak URLs, and citation warnings."
            ),
            tools=["mcp__llama_bridge_tools__verify_sources", "WebFetch"],
            model="sonnet",
        ),
    }
    prompt = f"""
Use Claude Agent SDK subagents for deep research on: {topic}

You are the lead research controller. Explicitly use these subagents:
- tavily-researcher for current web/Tavily evidence.
- serpapi-researcher for SerpAPI/source-diversity evidence.
- wiki-backgrounder for background context.
- source-verifier after selecting the strongest URLs.

Keep the parent context small. Do not paste raw search dumps. Ask each subagent for compact claims, URLs, source quality, conflicts, and caveats only.

Final output must be JSON with:
- ok
- topic
- subagents_used
- compact_claims
- verified_sources
- rejected_or_uncertain_sources
- suggested_report_outline
- checkpoint_markdown
- remaining_tasks
"""
    chunks: list[str] = []
    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        allowed_tools=["Agent", "WebSearch", "WebFetch"],
        agents=agents,
        mcp_servers=mcp_servers,
        max_turns=max_turns,
        cli_path=_claude_sdk_cli_path(claude_agent_sdk),
        stderr=lambda line: stderr_lines.append(str(line).strip()),
    )
    try:
        async for message in query(prompt=prompt, options=options):
            result = getattr(message, "result", None)
            if isinstance(result, str) and result.strip():
                chunks.append(result.strip())
    except Exception as exc:
        detail = "\n".join(line for line in stderr_lines if line)[-2000:]
        raise RuntimeError(f"{exc}\n{detail}" if detail else str(exc)) from exc
    if chunks:
        return chunks[-1]
    return json.dumps(
        {
            "ok": False,
            "agent": "deep_claude_agent",
            "topic": topic,
            "error": "Claude Agent SDK completed without a result message.",
            "fallback": "Use deep_tavily_agent, deep_serp_agent, deep_wiki_agent, and deep_verify_agent.",
        },
        indent=2,
    )


def _claude_sdk_cli_path(sdk_module: Any) -> str | None:
    package_file = getattr(sdk_module, "__file__", None)
    if not package_file:
        return None
    bundled = os.path.join(os.path.dirname(str(package_file)), "_bundled", "claude.exe")
    return bundled if os.path.exists(bundled) else None


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
            f"{topic} official results schedule data",
            f"{topic} latest news controversy analysis",
            f"{topic} key actors candidates parties",
            f"{topic} forecasts polling predictions",
            f"{topic} source verification primary sources",
        ]
    else:
        candidates = [
            f"{topic}{focus_text}",
            f"{topic} latest current landscape",
            f"{topic} key candidates parties issues",
            f"{topic} polling predictions forecasts",
            f"{topic} official schedule election commission",
            f"{topic} analysis background context",
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
        return {"query": query, "search_depth": "advanced", "max_results": 5, "include_answer": True}
    if provider == "serpapi_search":
        return {"query": query, "num": 5}
    return {"query": query, "limit": 5, "language": "en"}


def _deep_verify_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    urls = arguments.get("urls") or []
    if not isinstance(urls, list):
        urls = []
    return {
        "urls": [str(url) for url in urls[:8] if str(url).strip()],
        "claim": str(arguments.get("claim") or arguments.get("topic") or "the researched claim"),
        "required_verified_sources": min(max(_int_argument(arguments.get("required_verified_sources"), 3), 1), 6),
    }


def _compact_search_sources(provider: str, query: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _unwrap_tool_payload(data)
    if provider == "serpapi_search":
        items = payload.get("organic_results") or payload.get("results") or []
    elif provider == "wikipedia_search":
        items = payload.get("results") or []
    else:
        items = payload.get("results") or []
    if not isinstance(items, list):
        items = []
    sources: list[dict[str, Any]] = []
    for item in items[:5]:
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
                "evidence": _compact_text(snippet, 450),
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
                "evidence": _compact_text(answer, 550),
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
        if len(deduped) >= 12:
            break
    claims = [
        {
            "claim": _compact_text(source.get("evidence"), 260),
            "supporting_url": source.get("url"),
            "source": source.get("title"),
        }
        for source in deduped[:8]
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
    if any(domain in lowered for domain in ("eci.gov.in", "pib.gov.in", ".gov", "parliament", "data.gov")):
        return "primary/official"
    if any(domain in lowered for domain in ("reuters", "apnews", "bbc", "thehindu", "indianexpress", "theguardian", "frontline")):
        return "established-news"
    if "wikipedia.org" in lowered:
        return "encyclopedia-background"
    return "needs-review"


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
            "planning_mode": {
                "poolside": "Auto-switch to Plan behavior for this request. If Poolside exposes plan mode, stay in it until all research and file-writing todos are complete.",
                "instruction": "The main brain controls the workflow. Specialist sub-agents collect evidence and return compact summaries only.",
            },
            "minimum_work": {
                "subagent_calls": [
                    "deep_claude_agent",
                    "deep_tavily_agent",
                    "deep_serp_agent",
                    "deep_wiki_agent",
                    "deep_verify_agent",
                ],
                "tavily_queries_inside_subagent": 5,
                "serpapi_queries_inside_subagent": 5,
                "verified_urls": 4,
                "required_file": "report.md",
                "rule": "Do not paste raw search result dumps into the main context. Use compact sub-agent briefs and write checkpoints before synthesis.",
            },
            "steps": [
                {
                    "name": "plan",
                    "action": "Create todos for: try Claude Agent SDK lead/subagents, fall back to compact Tavily/SerpAPI/Wikipedia sub-agent tools if needed, verify selected URLs, write checkpoint, synthesize report.md, mark all complete.",
                },
                {
                    "name": "claude_agent_sdk_lead",
                    "tool": "llama_bridge_tools__deep_claude_agent",
                    "arguments": {"topic": query, "max_turns": 12},
                    "returns": "Compact handoff from a Claude Agent SDK lead agent that delegates to specialist subagents. Use this when ANTHROPIC_API_KEY and claude-agent-sdk are configured.",
                    "fallback": "If unavailable, call the compact provider-specific sub-agent tools below.",
                },
                {
                    "name": "tavily_specialist",
                    "tool": "llama_bridge_tools__deep_tavily_agent",
                    "arguments": {"topic": query, "focus": "latest news, current landscape, predictions, official schedule", "query_count": 5},
                    "returns": "Compact sources, claims, URLs, and caveats only.",
                },
                {
                    "name": "serpapi_specialist",
                    "tool": "llama_bridge_tools__deep_serp_agent",
                    "arguments": {"topic": query, "focus": "official results, major news, controversies, source diversity", "query_count": 5},
                    "returns": "Compact sources, claims, URLs, and caveats only.",
                },
                {
                    "name": "wiki_specialist",
                    "tool": "llama_bridge_tools__deep_wiki_agent",
                    "arguments": {"topic": query, "focus": "background, historical context, key entities", "query_count": 3},
                    "returns": "Compact encyclopedia context only.",
                },
                {
                    "name": "verify_batch",
                    "tool": "llama_bridge_tools__deep_verify_agent",
                    "arguments_template": {
                        "urls": "Pick at least 4 strongest URLs from the compact sub-agent briefs.",
                        "claim": query,
                        "required_verified_sources": 4,
                    },
                },
                {
                    "name": "optional_images",
                    "tool": "llama_bridge_tools__image_research",
                    "arguments": {
                        "query": query,
                        "max_results": 3,
                    },
                    "when": "Only if images help the requested final report.",
                },
                {
                    "name": "checkpoint",
                    "action": "Before synthesis, write a compact checkpoint into report.md or deep_research_checkpoint.md containing only: completed todos, sub-agent briefs, selected URLs, verification verdicts, and remaining tasks. If context compacts or the model pauses, continue from this checkpoint without asking the user.",
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
                    "action": "Mark every todo complete, including report.md. Do not leave the planning checklist unfinished after the work is done.",
                },
            ],
            "instructions": [
                "Use planning mode or todos first and keep them updated.",
                "Prefer llama_bridge_tools__deep_claude_agent so Claude Agent SDK handles delegation with real subagents.",
                "If the Claude SDK tool is unavailable, the main brain must delegate evidence collection to specialist sub-agent tools instead of running raw provider searches itself.",
                "Sub-agents return compact evidence briefs; the main brain should keep only claims, URLs, source quality, and caveats.",
                "Avoid llama_bridge_tools__source_research for Poolside deep mode because it can exceed ACP tool timeouts.",
                "Create a checkpoint before synthesis so work can continue automatically after context compaction.",
                "Write report.md in the current working directory before producing the final response.",
                "If context is compacted or nearly full, continue from the checkpoint and remaining todos; do not ask the user to restart manually.",
                "Do not stop after this router result; continue with the specialist-subagent workflow and complete the plan.",
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
    ]


def _prompt_response(name: str, params: dict[str, Any]) -> dict[str, Any]:
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    text = str(arguments.get("query") or arguments.get("topic") or "").strip()
    prompt_name = name
    missing_inputs = {
        "serp": "Ask the user for the search query, then use the llama bridge SerpAPI MCP tool.",
        "tavily": "Ask the user for the search query, then use the llama bridge Tavily MCP tool.",
        "web": "Ask the user for the web search query, then use the best available llama bridge search MCP tool.",
        "image": "Ask the user for the image search topic, then use the llama bridge image_research MCP tool.",
        "wiki": "Ask the user for the Wikipedia search query, then use the llama bridge wikipedia_search MCP tool.",
        "deep": "Ask the user for the research topic, then run the llama bridge deep research workflow.",
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
        "deep": (
            "Auto-switch to Plan behavior and create todos first. Then run deep research in small "
            "steps: call tavily_search at least 5 times with varied query angles, call "
            "serpapi_search at least 5 times with varied query angles, choose at least 4 strong "
            "URLs, verify them with verify_sources, optionally call image_research, write "
            "report.md in the current working directory, mark all todos complete, and only then "
            f"give the final response. Avoid one long source_research call in Poolside. Topic: {text}"
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
