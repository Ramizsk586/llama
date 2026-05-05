"""Tool management system for compact-first tool access."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


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

    def compact_manifest(self, query: str, session_key: str = "default") -> list:
        """Build compact tool manifest."""
        if not self.config.tools.management_enabled:
            return []

        tools = []
        state = self._get_state(session_key)

        items = list(self.registry._tools.items())[:self.config.tools.compact_manifest_max_tools]
        for name, tool in items:
            summary = tool.as_summary()
            summary["schema_known"] = state.is_known(name, summary["schema_id"])
            tools.append(summary)

        return tools

    def management_tools(self) -> list:
        """Return management tool schemas."""
        if not self.config.tools.always_expose_management_tools:
            return []
        return [TOOL_CATALOG_SEARCH_SCHEMA, TOOL_SCHEMA_GET_SCHEMA]

    def is_management_tool(self, name: str) -> bool:
        return name in {MANAGEMENT_TOOL_CATALOG_SEARCH, MANAGEMENT_TOOL_SCHEMA_GET}

    async def call_management_tool(self, name: str, arguments: dict, session_key: str = "default") -> dict:
        """Handle management tool calls."""
        if name == MANAGEMENT_TOOL_CATALOG_SEARCH:
            return await self._catalog_search(arguments)
        elif name == MANAGEMENT_TOOL_SCHEMA_GET:
            return await self._schema_get(arguments, session_key)
        return {"ok": False, "error": "Unknown management tool"}

    async def _catalog_search(self, arguments: dict) -> dict:
        """Search tools by query."""
        query = arguments.get("query", "")
        limit = int(arguments.get("limit", 5))

        results = []
        for name, tool in list(self.registry._tools.items())[:limit]:
            results.append(tool.as_summary())

        return {"ok": True, "tools": results, "query": query}

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

    def compact_instruction(self) -> str:
        """Instruction text for compact tool usage."""
        return (
            "Tool access is compact-first. "
            "Call tool_schema_get to get full schema before using a tool. "
            "Call tool_catalog_search when unsure which tool to use."
        )
