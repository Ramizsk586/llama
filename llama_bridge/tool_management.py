"""Tool management system for compact-first tool access."""

import time
from dataclasses import dataclass, field
from typing import Any

from .tools import select_relevant_tools


MANAGEMENT_TOOL_CATALOG_SEARCH = "tool_catalog_search"
MANAGEMENT_TOOL_SCHEMA_GET = "tool_schema_get"


TOOL_CATALOG_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": MANAGEMENT_TOOL_CATALOG_SEARCH,
        "description": "Search tools by query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}

TOOL_SCHEMA_GET_SCHEMA = {
    "type": "function",
    "function": {
        "name": MANAGEMENT_TOOL_SCHEMA_GET,
        "description": "Get full tool schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
    },
}


@dataclass
class ToolKnowledgeState:
    known_schemas: dict = field(default_factory=dict)
    expires_at: float = field(default_factory=lambda: time.time() + 86400)

    def is_known(self, tool_name: str, schema_id: str) -> bool:
        return self.known_schemas.get(tool_name) == schema_id

    def mark_known(self, tool_name: str, schema_id: str):
        self.known_schemas[tool_name] = schema_id
        self.expires_at = time.time() + 86400


class ToolManager:
    """Manages compact tool manifests and on-demand schema fetching."""

    def __init__(self, registry, config):
        self.registry = registry
        self.config = config
        self._states: dict = {}

    def _get_state(self, session_key: str) -> ToolKnowledgeState:
        if session_key not in self._states:
            self._states[session_key] = ToolKnowledgeState()
        return self._states[session_key]

    def compact_manifest(
        self,
        query: str,
        session_key: str = "default",
        client_capabilities: dict[str, Any] | None = None,
    ) -> list:
        """Build compact tool manifest."""
        if not self.config.tools.management_enabled or not self.config.tools.compact_manifest_enabled:
            return []
        if self._simple_chat_query(query):
            return []

        tools = []
        state = self._get_state(session_key)
        options = self._request_options(client_capabilities)
        items = self._relevant_registry_items(
            query,
            options["max_manifest_tools"],
            disable_fallback=options["disable_fallback"],
        )
        for name, tool in items:
            summary = tool.as_summary()
            summary["schema_known"] = state.is_known(name, summary["schema_id"])
            tools.append(summary)

        return tools

    def schemas_for_request(
        self,
        query: str,
        session_key: str = "default",
        client_capabilities: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return full tool schemas to attach to a request."""
        if not self.config.tools.management_enabled:
            return self.registry.openai_tools()
        if self._simple_chat_query(query):
            return []

        options = self._request_options(client_capabilities)
        tools = self._select_full_schemas(
            query,
            max_tools=options["max_tools"],
            disable_fallback=options["disable_fallback"],
        )
        management_tools = self.management_openai_tools() if options["include_management_tools"] and tools else []
        return self._dedupe_tool_schemas([*tools, *management_tools])

    def management_tools(self) -> list:
        """Return management tool schemas."""
        if not self.config.tools.always_expose_management_tools:
            return []
        return [TOOL_CATALOG_SEARCH_SCHEMA, TOOL_SCHEMA_GET_SCHEMA]

    def management_openai_tools(self) -> list[dict[str, Any]]:
        """Compatibility wrapper used by server logging and request assembly."""
        return self.management_tools()

    def is_management_tool(self, name: str) -> bool:
        return name in {
            MANAGEMENT_TOOL_CATALOG_SEARCH,
            MANAGEMENT_TOOL_SCHEMA_GET,
            "tool_usage_help",
        }

    async def call_management_tool(
        self,
        name: str,
        arguments: dict,
        session_key: str = "default",
        client_capabilities: dict[str, Any] | None = None,
    ) -> dict:
        """Handle management tool calls."""
        if name == MANAGEMENT_TOOL_CATALOG_SEARCH:
            return await self._catalog_search(arguments)
        if name == MANAGEMENT_TOOL_SCHEMA_GET:
            return await self._schema_get(arguments, session_key)
        if name == "tool_usage_help":
            return await self._usage_help(arguments)
        return {"ok": False, "error": "Unknown management tool"}

    async def _catalog_search(self, arguments: dict) -> dict:
        """Search tools by query."""
        query = arguments.get("query", "")
        limit = int(arguments.get("limit", 5))

        trimmed = []
        for _name, tool in self._relevant_registry_items(query, limit):
            trimmed.append(tool.as_summary())
        if not trimmed:
            trimmed = [tool.as_summary() for tool in list(self.registry._tools.values())[:limit]]

        return {"ok": True, "tools": trimmed, "query": query}

    async def _schema_get(self, arguments: dict, session_key: str) -> dict:
        """Get full schema for a tool."""
        name = arguments.get("name", "")
        tool = self.registry._tools.get(name)

        if not tool:
            return {"ok": False, "error": f"Tool '{name}' not found"}

        schema_id = tool.schema_id()
        state = self._get_state(session_key)
        state.mark_known(name, schema_id)

        return {
            "ok": True,
            "tool": tool.as_openai_tool(),
            "schema_id": schema_id,
        }

    async def _usage_help(self, arguments: dict) -> dict:
        """Return a compact usage description for a tool."""
        name = arguments.get("name", "")
        tool = self.registry._tools.get(name)
        if not tool:
            return {"ok": False, "error": f"Tool '{name}' not found"}

        summary = tool.as_summary()
        return {
            "ok": True,
            "name": tool.name,
            "description": tool.description,
            "summary": summary["summary"],
            "use_when": summary.get("use_when", []),
            "args_hint": summary.get("args_hint"),
            "schema_id": tool.schema_id(),
        }

    def compact_instruction(self, client_capabilities: dict[str, Any] | None = None) -> str:
        """Instruction text for compact tool usage."""
        options = self._request_options(client_capabilities)
        if not options["attach_system_instructions"]:
            return ""
        if options["minimal"]:
            return "Use a bridge tool only when it is clearly needed for current facts or external data."
        if self.config.tools.always_expose_management_tools:
            return (
                "Use bridge tools only when they clearly help. "
                "For exact arguments call tool_schema_get. "
                "To find the right tool call tool_catalog_search."
            )
        return "Use bridge tools only when they clearly help."

    def compact_instruction_text(self, client_capabilities: dict[str, Any] | None = None) -> str:
        """Compatibility wrapper used by the server."""
        return self.compact_instruction(client_capabilities)

    def _select_full_schemas(
        self,
        query: str,
        max_tools: int | None = None,
        disable_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        policy = (self.config.tools.expose_full_schema_policy or "relevant").strip().lower()
        all_tools = self.registry.openai_tools()

        if policy == "always":
            return all_tools
        if policy in {"never", "on_demand"}:
            return []

        max_tools = max(1, int(max_tools or self.config.tools.max_exposed or 5))
        selected, _scores = select_relevant_tools(
            all_tools,
            query,
            max_tools=max_tools,
            min_score=self.config.tools.confidence_threshold,
            force_for_keywords=self.config.tools.force_for_keywords,
            default_search_provider=self.config.tools.default_search_provider,
        )

        if selected:
            return selected
        if disable_fallback or self._simple_chat_query(query):
            return []
        if self.config.tools.fallback_to_full_schemas_for_unsupported_clients:
            return all_tools[: min(max_tools, 2)]
        return []

    def _relevant_registry_items(
        self,
        query: str,
        limit: int,
        disable_fallback: bool = False,
    ) -> list[tuple[str, Any]]:
        limit = max(1, limit)
        items = list(self.registry._tools.items())
        if not items:
            return []
        if not query.strip() or self._simple_chat_query(query):
            return items[:limit]

        tool_schemas = [tool.as_openai_tool() for _, tool in items]
        selected, _scores = select_relevant_tools(
            tool_schemas,
            query,
            max_tools=limit,
            min_score=self.config.tools.confidence_threshold,
            force_for_keywords=self.config.tools.force_for_keywords,
            default_search_provider=self.config.tools.default_search_provider,
        )
        selected_names = {
            (tool.get("function") or {}).get("name", "")
            for tool in selected
        }
        ranked = [(name, tool) for name, tool in items if name in selected_names]
        if ranked:
            return ranked[:limit]
        if disable_fallback:
            return []
        return items[:limit]

    def _dedupe_tool_schemas(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tool in tools:
            name = ((tool.get("function") or {}).get("name") or tool.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            deduped.append(tool)
        return deduped

    def _request_options(self, client_capabilities: dict[str, Any] | None) -> dict[str, Any]:
        options = {
            "minimal": False,
            "max_tools": max(1, int(self.config.tools.max_exposed or 5)),
            "max_manifest_tools": max(0, int(self.config.tools.compact_manifest_max_tools or 0)),
            "include_management_tools": self.config.tools.always_expose_management_tools,
            "disable_fallback": False,
            "attach_system_instructions": True,
        }
        if client_capabilities:
            options.update(
                {key: value for key, value in client_capabilities.items() if key in options}
            )
        return options

    def _simple_chat_query(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return True
        words = text.split()
        if len(words) > 4:
            return False
        toolish_markers = (
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
            "stock",
            "source",
            "verify",
            "citation",
            "wiki",
            "wikipedia",
            "fetch",
            "image",
            "photo",
            "research",
        )
        return not any(marker in text for marker in toolish_markers)
