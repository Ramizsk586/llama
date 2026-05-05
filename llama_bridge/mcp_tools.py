from __future__ import annotations

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
        data = self._request("GET", "/api/tools?full_schema=true")
        tools = data.get("tools") or data.get("data") or []

        # If that didn't work, fall back to regular endpoint
        if not tools:
            data = self._request("GET", "/api/tools")
            tools = data.get("tools") or data.get("data") or []

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
        return result

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        data = self._request("POST", f"/api/tools/{name}", arguments)
        is_error = bool(data.get("ok") is False)
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

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


def _prompt_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "serp",
            "description": "Search the web with SerpAPI through llama bridge.",
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
    missing_inputs = {
        "serp": "Ask the user for the search query, then use the llama bridge SerpAPI MCP tool.",
        "web": "Ask the user for the web search query, then use the best available llama bridge search MCP tool.",
        "image": "Ask the user for the image search topic, then use the llama bridge image_research MCP tool.",
        "deep": "Ask the user for the research topic, then run the llama bridge deep research workflow.",
    }
    if not text:
        if name not in missing_inputs:
            raise ValueError(f"Unknown prompt: {name}")
        prompt_text = missing_inputs[name]
        return {
            "description": next((item["description"] for item in _prompt_definitions() if item["name"] == name), name),
            "messages": [{"role": "user", "content": {"type": "text", "text": prompt_text}}],
        }
    prompts = {
        "serp": f"Use the llama bridge SerpAPI MCP tool for this search query: {text}",
        "web": f"Use the llama bridge web or Tavily MCP search tool for this query: {text}",
        "image": f"Use the llama bridge image_research MCP tool and return compact sourced image candidates for: {text}",
        "deep": (
            "Use the llama bridge source_research MCP tool first, then verify important claims with "
            f"available search tools and use image_research for 2-3 useful sourced images. Topic: {text}"
        ),
    }
    if name not in prompts:
        raise ValueError(f"Unknown prompt: {name}")
    return {
        "description": next((item["description"] for item in _prompt_definitions() if item["name"] == name), name),
        "messages": [{"role": "user", "content": {"type": "text", "text": prompts[name]}}],
    }


if __name__ == "__main__":
    main()
