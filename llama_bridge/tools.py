from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .config import BridgeConfig, ExternalToolProviderConfig
from .master import MasterReviewer


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
WIKIMEDIA_USER_AGENT = "llama-bridge/0.1 (local personal use; contact: user@example.com)"
GEOCODING_USER_AGENT = "llama-bridge/0.1 (local personal use)"
MOZILLA_IMAGE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
COUNTRY_TIMEZONES = {
    "india": "Asia/Kolkata",
    "in": "Asia/Kolkata",
    "united states": "America/New_York",
    "usa": "America/New_York",
    "us": "America/New_York",
    "united kingdom": "Europe/London",
    "uk": "Europe/London",
    "great britain": "Europe/London",
    "australia": "Australia/Sydney",
    "canada": "America/Toronto",
    "germany": "Europe/Berlin",
    "france": "Europe/Paris",
    "japan": "Asia/Tokyo",
    "china": "Asia/Shanghai",
    "singapore": "Asia/Singapore",
}
TIMEZONE_OFFSETS = {
    "Asia/Kolkata": timezone(timedelta(hours=5, minutes=30), "IST"),
    "Asia/Calcutta": timezone(timedelta(hours=5, minutes=30), "IST"),
}


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def as_summary(self, max_chars: int = 160) -> dict[str, Any]:
        """Create a compact summary of the tool for compact manifest."""
        summary = _extract_first_sentence(self.description)
        if len(summary) > max_chars:
            summary = summary[:max_chars - 3] + "..."

        use_when = _extract_use_when(self.description)
        if not use_when:
            use_when = _infer_use_when_from_name(self.name)

        args_hint = _build_args_hint(self.parameters)
        category = _infer_category(self.name, self.description)

        return {
            "name": self.name,
            "summary": summary,
            "use_when": use_when[:5],
            "args_hint": args_hint,
            "category": category,
            "schema_id": self.schema_id(),
            "requires_full_schema": False,
        }

    def schema_id(self) -> str:
        """Generate a stable hash of the full tool schema."""
        schema = self.as_openai_tool()
        schema_str = json.dumps(schema, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[:16]


def _extract_first_sentence(text: str) -> str:
    """Extract the first sentence from a tool description."""
    # Look for "USE WHEN:" section and extract text before it
    use_when_match = re.search(r"USE WHEN:", text, re.IGNORECASE)
    if use_when_match:
        text = text[:use_when_match.start()].strip()

    # Extract first sentence (ending with ., !, ?)
    match = re.match(r"^(.*?[.!?])\s", text)
    if match:
        return match.group(1).strip()

    # If no sentence ending found, take first line or up to 160 chars
    first_line = text.split("\n")[0].strip()
    return first_line


def _extract_use_when(description: str) -> list[str]:
    """Extract USE WHEN keywords from tool description."""
    match = re.search(r"USE WHEN:\s*(.+?)(?:\n|$)", description, re.IGNORECASE)
    if not match:
        return []
    use_when_text = match.group(1).strip()
    # Extract keywords - words that are meaningful
    keywords = re.findall(r"\b\w{3,}\b", use_when_text.lower())
    # Filter out common words
    stop_words = {"the", "and", "for", "with", "this", "that", "when", "user", "asks", "about"}
    return [kw for kw in keywords if kw not in stop_words][:10]


def _infer_use_when_from_name(tool_name: str) -> list[str]:
    """Infer use_when keywords from tool name."""
    name_lower = tool_name.lower()
    keywords = []

    # Common patterns
    if "weather" in name_lower:
        keywords = ["weather", "temperature", "rain", "wind"]
    elif "wiki" in name_lower:
        keywords = ["wikipedia", "information", "topic", "article"]
    elif "search" in name_lower:
        keywords = ["search", "find", "lookup", "research"]
    elif "time" in name_lower or "date" in name_lower or "datetime" in name_lower:
        keywords = ["time", "date", "timezone", "current"]
    elif "image" in name_lower:
        keywords = ["image", "picture", "photo", "visual"]
    elif "source" in name_lower or "verify" in name_lower:
        keywords = ["source", "verify", "evidence", "research"]
    elif "master" in name_lower or "review" in name_lower:
        keywords = ["review", "quality", "check", "analyze"]

    return keywords[:5]


def _build_args_hint(parameters: dict[str, Any]) -> str:
    """Build a short hint about required arguments."""
    if not parameters or "properties" not in parameters:
        return "no arguments"

    properties = parameters.get("properties", {})
    required = parameters.get("required", [])

    if required:
        required_hints = [f"{arg} required" for arg in required[:3]]
        return ", ".join(required_hints)

    # If no required args, show optional ones
    optional = list(properties.keys())[:3]
    if optional:
        return f"{', '.join(optional)} optional"

    return "see schema for details"


def _infer_category(tool_name: str, description: str) -> str:
    """Infer tool category from name and description."""
    name_lower = tool_name.lower()
    desc_lower = description.lower()

    if "weather" in name_lower or "weather" in desc_lower:
        return "weather"
    if "wiki" in name_lower or "wikipedia" in desc_lower:
        return "knowledge"
    if "search" in name_lower or "search" in desc_lower:
        return "search"
    if "time" in name_lower or "date" in name_lower or "datetime" in name_lower:
        return "time"
    if "image" in name_lower or "image" in desc_lower:
        return "image"
    if "source" in name_lower or "verify" in name_lower or "research" in desc_lower:
        return "research"
    if "master" in name_lower or "review" in name_lower:
        return "review"

    return "general"


class ToolValidationError(ValueError):
    """Raised when a model supplies invalid tool arguments."""


class UnknownToolError(KeyError):
    """Raised when a requested tool is not registered."""


class ToolCache:
    def __init__(self) -> None:
        self._items: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        expires_at, value = self._items.get(key, (0.0, {}))
        if expires_at <= datetime.now(UTC).timestamp():
            self._items.pop(key, None)
            return None
        return value

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        expires_at = datetime.now(UTC).timestamp() + ttl_seconds
        cached = json.loads(json.dumps(value, ensure_ascii=True, default=str))
        metadata = dict(cached.get("metadata") or {})
        metadata["cache_hit"] = False
        cached["metadata"] = metadata
        self._items[key] = (expires_at, cached)


class ToolRegistry:
    def __init__(self, config: BridgeConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=30.0)
        self._tools: dict[str, ToolDefinition] = {}
        self._unavailable_tools: dict[str, str] = {}
        self._cache = ToolCache()
        self._master_reviewer = (
            MasterReviewer(config.master_review)
            if getattr(config, "master_review", None) is not None and config.master_review.enabled
            else None
        )
        if config.tools.enabled:
            self._register_default_tools()

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._master_reviewer is not None:
            await self._master_reviewer.aclose()

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.as_dict() for tool in self._tools.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.as_openai_tool() for tool in self._tools.values()]

    def unavailable_tools(self) -> dict[str, str]:
        return dict(self._unavailable_tools)

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(self._unknown_tool_message(name))
        return await tool.handler(arguments or {})

    async def call_structured(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool and always return model-readable structured JSON."""
        started_at = datetime.now(UTC)
        try:
            if name not in self._tools:
                raise UnknownToolError(self._unknown_tool_message(name))
            validated_args = validate_tool_arguments(name, arguments or {})
            cache_key = self._cache_key(name, validated_args)
            if cache_key:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    cached_result = dict(cached)
                    metadata = dict(cached_result.get("metadata") or {})
                    metadata["cache_hit"] = True
                    cached_result["metadata"] = metadata
                    cached_result["timestamp"] = metadata.get("finished_at")
                    return cached_result
            result = await self.call(name, validated_args)
            result = _validate_tool_output(name, result)
            structured = _structured_tool_success(name, result, started_at)
            if cache_key and structured.get("ok"):
                self._cache.set(cache_key, structured, int(self.config.tools.cache_ttl_seconds))
            return structured
        except Exception as exc:
            return _structured_tool_error(name, exc, started_at)

    def _cache_key(self, name: str, arguments: dict[str, Any]) -> str | None:
        if not self.config.tools.cache_enabled:
            return None
        if name not in {
            "wikipedia_search",
            "wikipedia_page",
            "weather_current",
            "serpapi_search",
            "tavily_search",
        }:
            return None
        body = json.dumps(arguments, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(f"{name}:{body}".encode("utf-8")).hexdigest()

    def _unknown_tool_message(self, name: str) -> str:
        available = ", ".join(sorted(self._tools)) or "none"
        message = f"Unknown tool '{name}'. Available tools: {available}"
        if self._unavailable_tools:
            unavailable = ", ".join(
                f"{tool_name} ({reason})"
                for tool_name, reason in sorted(self._unavailable_tools.items())
            )
            message = f"{message}. Unavailable tools: {unavailable}"
        return message

    async def _request_with_retries(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError) as exc:
                last_exc = exc
                if attempt == 2:
                    raise
            await asyncio.sleep(0.25 * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    def _register(self, tool: ToolDefinition, *, provider: ExternalToolProviderConfig | None = None) -> None:
        include = self.config.tools.include
        if include and tool.name not in include:
            self._unavailable_tools[tool.name] = "not included by tools.include"
            return
        if provider is not None and not provider.enabled:
            self._unavailable_tools[tool.name] = "provider disabled"
            return
        self._tools[tool.name] = tool
        self._unavailable_tools.pop(tool.name, None)

    def _register_default_tools(self) -> None:
        self._register(
            ToolDefinition(
                name="shell.execute",
                description=(
                    "Run a local shell command and return stdout, stderr, exit code, and timeout status.\n"
                    "USE WHEN: User asks to create files, inspect folders, run tests, or execute local commands in the workspace.\n"
                    "DO NOT USE: For web search or external factual lookup when a bridge research tool fits better.\n"
                    "RESULT FORMAT: Command, working directory, exit code, stdout, stderr, and timed_out."
                ),
                parameters={
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to run.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory.",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Command timeout in seconds.",
                            "default": 60,
                            "minimum": 1,
                            "maximum": 600,
                        },
                    },
                },
                handler=self._shell_execute,
            )
        )
        self._register(
            ToolDefinition(
                name="datetime_now",
                description=(
                    "Get the current date and time for a specific timezone or country.\n"
                    "USE WHEN: User asks for current time, date, timezone info, or needs to know what time it is.\n"
                    "DO NOT USE: For historical dates, scheduling future events, or time conversions (only retrieves current time).\n"
                    "RESULT FORMAT: ISO timestamp, Unix timestamp, date string, time string, and UTC offset."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone (e.g., UTC, Asia/Kolkata, America/New_York). Defaults to UTC if not specified.",
                            "default": "UTC",
                        },
                        "country": {
                            "type": "string",
                            "description": "Country name (e.g., India, USA, UK) to auto-detect timezone. Overridden by explicit timezone parameter.",
                        },
                    },
                },
                handler=self._datetime_now,
            )
        )
        self._register(
            ToolDefinition(
                name="manim_render",
                description=(
                    "Generate a short educational animation video with the Python Manim Community library.\n"
                    "USE WHEN: User types /manim or asks to turn text, concepts, math, diagrams, or explanations into an animated video.\n"
                    "DO NOT USE: For ordinary image generation, web search, or long cinematic video. Requires the local `manim` Python package and its runtime dependencies.\n"
                    "RESULT FORMAT: Generated Python scene path, rendered MP4 path, stdout/stderr excerpts, and install guidance if Manim is missing."
                ),
                parameters={
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the animation to create, e.g. 'explain Pythagorean theorem with moving squares'.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional title shown at the top of the animation.",
                        },
                        "quality": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Render quality. Low is fastest and maps to Manim -ql.",
                            "default": "low",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Directory for generated scene files and videos. Defaults to ./manim_outputs.",
                        },
                        "render": {
                            "type": "boolean",
                            "description": "Whether to render the video after writing the scene file.",
                            "default": True,
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Maximum render time in seconds.",
                            "default": 180,
                            "minimum": 30,
                            "maximum": 600,
                        },
                    },
                },
                handler=self._manim_render,
            )
        )
        self._register(
            ToolDefinition(
                name="wikipedia_search",
                description=(
                    "Search Wikipedia for articles matching a query and return page summaries.\n"
                    "USE WHEN: User asks for information about a topic, person, place, concept, or historical event.\n"
                    "DO NOT USE: For current events, latest news, real-time data, or opinions.\n"
                    "RESULT FORMAT: List of page titles, snippets, and URLs. Follow up with wikipedia_page for full article."
                ),
                parameters={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term or phrase (e.g., 'Albert Einstein', 'French Revolution').",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of search results (1-20).",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "language": {
                            "type": "string",
                            "description": "Wikipedia language code (en, fr, de, es, etc.).",
                            "default": "en",
                        },
                    },
                },
                handler=self._wikipedia_search,
            ),
            provider=self.config.tools.wikipedia,
        )
        self._register(
            ToolDefinition(
                name="wikipedia_page",
                description=(
                    "Fetch the full summary of a Wikipedia page by its exact title.\n"
                    "USE WHEN: User wants detailed information from a specific Wikipedia article, or after wikipedia_search returns a promising result.\n"
                    "DO NOT USE: If unsure about the exact page title; use wikipedia_search first.\n"
                    "RESULT FORMAT: Full article summary (extract), description, and page URL."
                ),
                parameters={
                    "type": "object",
                    "required": ["title"],
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Exact Wikipedia page title (case-insensitive). Examples: 'Albert Einstein', 'Python (programming language)'.",
                        },
                        "language": {
                            "type": "string",
                            "description": "Wikipedia language code (en, fr, de, etc.).",
                            "default": "en",
                        },
                    },
                },
                handler=self._wikipedia_page,
            ),
            provider=self.config.tools.wikipedia,
        )
        self._register(
            ToolDefinition(
                name="weather_current",
                description=(
                    "Get current weather conditions (temperature, humidity, wind, rain) from Open-Meteo API.\n"
                    "USE WHEN: User asks about current weather, temperature, wind, rain, or weather conditions for a location.\n"
                    "DO NOT USE: For historical weather, forecasts, or climate data.\n"
                    "RESULT FORMAT: Temperature, humidity, wind speed, weather code, and location info."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Place name to geocode (e.g., 'Delhi', 'New York', 'Tokyo'). Resolved to lat/lon.",
                        },
                        "latitude": {
                            "type": "number",
                            "description": "WGS84 latitude (-90 to 90). Required if location not provided.",
                        },
                        "longitude": {
                            "type": "number",
                            "description": "WGS84 longitude (-180 to 180). Required if location not provided.",
                        },
                        "temperature_unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit for results.",
                            "default": "celsius",
                        },
                        "wind_speed_unit": {
                            "type": "string",
                            "enum": ["kmh", "ms", "mph", "kn"],
                            "description": "Wind speed unit.",
                            "default": "kmh",
                        },
                    },
                    "anyOf": [
                        {"required": ["location"]},
                        {"required": ["latitude", "longitude"]},
                    ],
                },
                handler=self._weather_current,
            ),
            provider=self.config.tools.weather,
        )
        self._register(
            ToolDefinition(
                name="serpapi_search",
                description=(
                    "Search the web using Google/SerpAPI to find current information, news, and web results.\n"
                    "USE WHEN: User asks for latest news, current prices, product reviews, or web search results.\n"
                    "DO NOT USE: For historical facts (use Wikipedia), or if Tavily is available and configured.\n"
                    "RESULT FORMAT: Organic search results with titles, snippets, and links. May include answer box and knowledge graph."
                ),
                parameters={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'Python 3.12 release date', 'latest iPhone price').",
                        },
                        "engine": {
                            "type": "string",
                            "enum": ["google", "bing", "baidu"],
                            "description": "Search engine to use.",
                            "default": "google",
                        },
                        "location": {
                            "type": "string",
                            "description": "Location for localized search results (optional).",
                        },
                        "hl": {
                            "type": "string",
                            "description": "Language code for results (e.g., en, fr, es).",
                        },
                        "gl": {
                            "type": "string",
                            "description": "Country code for results (e.g., US, UK, IN).",
                        },
                        "num": {
                            "type": "integer",
                            "description": "Number of results to return (1-20).",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                },
                handler=self._serpapi_search,
            ),
            provider=self.config.tools.serpapi,
        )
        self._register(
            ToolDefinition(
                name="tavily_search",
                description=(
                    "Search the web using Tavily API for current, factual, and research-based information.\n"
                    "USE WHEN: User asks for latest news, research, current facts, or web search with sources.\n"
                    "DO NOT USE: For historical facts (use Wikipedia), or user preferences/opinions.\n"
                    "RESULT FORMAT: List of web results with content, URLs, and relevance scores. Includes answer if requested."
                ),
                parameters={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'latest AI news 2024', 'Bitcoin price today').",
                        },
                        "search_depth": {
                            "type": "string",
                            "enum": ["basic", "advanced"],
                            "description": "Search depth: basic for quick results, advanced for thorough research.",
                            "default": "basic",
                        },
                        "topic": {
                            "type": "string",
                            "enum": ["general", "news", "finance"],
                            "description": "Topic type to optimize results.",
                            "default": "general",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results (0-20).",
                            "default": 5,
                            "minimum": 0,
                            "maximum": 20,
                        },
                        "include_answer": {
                            "type": "boolean",
                            "description": "Include direct answer if available.",
                            "default": False,
                        },
                        "include_raw_content": {
                            "type": "boolean",
                            "description": "Include raw HTML content from pages.",
                            "default": False,
                        },
                    },
                },
                handler=self._tavily_search,
            ),
            provider=self.config.tools.tavily,
        )
        self._register(
            ToolDefinition(
                name="source_research",
                description=(
                    "Advanced source-backed research using SerpAPI and Tavily together, followed by parallel source verification workers.\n"
                    "USE WHEN: User asks for current, high-stakes, source-backed, markdown/report, buying, travel, technical, or factual answers that need citations.\n"
                    "GUARDRAILS: Treat results as evidence, not truth. Prefer primary/official sources. If fewer than the requested number of independent sources verify the topic, say evidence is insufficient.\n"
                    "RESULT FORMAT: Verified source records, rejected/weak records, optional image candidates, and a conservative verdict for use in final answers."
                ),
                parameters={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Research question or search query to investigate.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to gather from each provider (1-10).",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "required_verified_sources": {
                            "type": "integer",
                            "description": "Minimum independent reachable sources required before the result is considered well-supported.",
                            "default": 2,
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "include_images": {
                            "type": "boolean",
                            "description": "Also collect image candidates with source/provenance metadata.",
                            "default": False,
                        },
                        "skip_master_review": {
                            "type": "boolean",
                            "description": "Skip the optional master-review pass for latency-sensitive clients.",
                            "default": False,
                        },
                        "max_verify_urls": {
                            "type": "integer",
                            "description": "Maximum discovered URLs to fetch during verification.",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "verify_timeout_seconds": {
                            "type": "integer",
                            "description": "Per-source fetch timeout during verification.",
                            "default": 20,
                            "minimum": 3,
                            "maximum": 20,
                        },
                        "include_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional domains to prefer/include when Tavily is configured.",
                        },
                        "exclude_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional domains to exclude when Tavily is configured.",
                        },
                    },
                },
                handler=self._source_research,
            )
        )
        self._register(
            ToolDefinition(
                name="image_research",
                description=(
                    "Find images for markdown/report generation using Wikipedia/Wikimedia, Tavily image results, and SerpAPI image search.\n"
                    "USE WHEN: The model needs image URLs or local downloadable Wikipedia images with provenance for a .md file, article, report, product/place/person page, or visual comparison.\n"
                    "GUARDRAILS: Do not invent image URLs. Use returned image_url/thumbnail/source_url only. Prefer images with a source page and cite that page near the image in markdown.\n"
                    "RESULT FORMAT: Image candidates with title, image URL, optional local_path for downloaded Wikipedia images, thumbnail, source page, provider, and markdown embed examples."
                ),
                parameters={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Image search query, ideally including the subject and needed visual style/context.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum image candidates to return (1-3).",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 3,
                        },
                        "include_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional domains to prefer/include when Tavily is configured.",
                        },
                        "exclude_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional domains to exclude when Tavily is configured.",
                        },
                    },
                },
                handler=self._image_research,
            )
        )
        self._register(
            ToolDefinition(
                name="image_download",
                description=(
                    "Download a public HTTP image URL to a local file and return its local_path.\n"
                    "USE WHEN: image_research returns an image_url or thumbnail that must be attached, uploaded, or embedded as a local file.\n"
                    "GUARDRAILS: Only download public http/https image URLs. Do not use this for private, localhost, or non-image URLs.\n"
                    "RESULT FORMAT: local_path, absolute_path, media_type, bytes, source_url, and title."
                ),
                parameters={
                    "type": "object",
                    "required": ["image_url"],
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "Public direct image URL returned by image_research.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional image title used to name the downloaded file.",
                        },
                        "source_url": {
                            "type": "string",
                            "description": "Optional source page URL for provenance and request headers.",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Repo-relative output directory. Defaults to downloaded_images.",
                            "default": "downloaded_images",
                        },
                    },
                },
                handler=self._image_download,
            )
        )
        self._register(
            ToolDefinition(
                name="verify_sources",
                description=(
                    "Launch parallel source-verifier workers for specific URLs and return reachable evidence snippets.\n"
                    "USE WHEN: The model already has links/sources and must check whether they are reachable and relevant before citing them.\n"
                    "GUARDRAILS: Only cite URLs marked reachable and relevant. If the claim is not supported by enough sources, say so rather than asserting it.\n"
                    "RESULT FORMAT: Per-link verifier results with status, title, evidence snippets, relevance score, and overall verdict."
                ),
                parameters={
                    "type": "object",
                    "required": ["urls"],
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs to verify in parallel.",
                            "minItems": 1,
                            "maxItems": 10,
                        },
                        "claim": {
                            "type": "string",
                            "description": "Optional claim/question the sources should support.",
                        },
                        "required_verified_sources": {
                            "type": "integer",
                            "description": "Minimum relevant reachable sources required for a supported verdict.",
                            "default": 2,
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                },
                handler=self._verify_sources,
            )
        )
        self._register(
            ToolDefinition(
                name="master_review",
                description=(
                    "Review a deep/source research result with specialist sub-agents and deterministic fallback checks.\n"
                    "USE WHEN: You have a draft report or source_research/deep_research result and need quality, citation, neutrality, and evidence review.\n"
                    "RESULT FORMAT: Structured master_review JSON with final LLM instructions, risk score, critiques, and optional revised draft."
                ),
                parameters={
                    "type": "object",
                    "required": ["research_result"],
                    "properties": {
                        "research_result": {
                            "type": "object",
                            "description": "Deep research/source research result to review.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["fast", "balanced", "strict"],
                            "default": "balanced",
                        },
                        "return_revised_draft": {
                            "type": "boolean",
                            "default": True,
                        },
                    },
                },
                handler=self._master_review,
            )
        )

    async def _shell_execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = _required_string(arguments, "command")
        cwd_value = str(arguments.get("cwd") or "").strip()
        cwd = cwd_value or None
        if cwd and not os.path.isdir(cwd):
            raise ValueError(f"Working directory does not exist: {cwd}")
        timeout_seconds = _bounded_int(
            arguments.get("timeout_seconds"),
            default=60,
            minimum=1,
            maximum=600,
        )
        process_args = (
            ["powershell", "-NoProfile", "-Command", command]
            if os.name == "nt"
            else ["sh", "-lc", command]
        )
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                process_args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                cwd=cwd,
                check=False,
            )
            return {
                "command": command,
                "cwd": cwd or os.getcwd(),
                "exit_code": completed.returncode,
                "stdout": _truncate_tool_output(completed.stdout),
                "stderr": _truncate_tool_output(completed.stderr),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "command": command,
                "cwd": cwd or os.getcwd(),
                "exit_code": None,
                "stdout": _truncate_tool_output(exc.stdout or ""),
                "stderr": _truncate_tool_output(exc.stderr or ""),
                "timed_out": True,
            }

    async def _datetime_now(self, arguments: dict[str, Any]) -> dict[str, Any]:
        country = str(arguments.get("country") or self.config.tools.country or "").strip()
        timezone = str(
            arguments.get("timezone")
            or _timezone_for_country(country)
            or "UTC"
        )
        timezone, tzinfo = _timezone_info(timezone)
        now = datetime.now(tzinfo)
        utc_now = datetime.now(UTC)
        return {
            "country": country or None,
            "timezone": timezone,
            "iso": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "utc_offset": now.strftime("%z"),
            "unix": int(now.timestamp()),
            "local": _time_payload(now, timezone),
            "utc": _time_payload(utc_now, "UTC"),
        }

    async def _manim_render(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(render_manim_video, arguments)

    async def _wikipedia_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _required_string(arguments, "query")
        limit = _bounded_int(arguments.get("limit"), default=5, minimum=1, maximum=20)
        language = _language(arguments, self.config.tools.wikipedia)
        url = f"https://{language}.wikipedia.org/w/api.php"
        response = await self._request_with_retries(
            "GET",
            url,
            headers=_wikipedia_headers(self.config.tools.wikipedia),
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
                "utf8": 1,
            },
        )
        data = response.json()
        return {
            "query": query,
            "language": language,
            "results": [
                {
                    "title": item.get("title"),
                    "pageid": item.get("pageid"),
                    "snippet": _strip_html(item.get("snippet", "")),
                    "url": f"https://{language}.wikipedia.org/wiki/{_wiki_title(item.get('title', ''))}",
                }
                for item in data.get("query", {}).get("search", [])
            ],
        }

    async def _wikipedia_page(self, arguments: dict[str, Any]) -> dict[str, Any]:
        title = _required_string(arguments, "title")
        language = _language(arguments, self.config.tools.wikipedia)
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{_wiki_title(title)}"
        response = await self._request_with_retries(
            "GET",
            url,
            headers=_wikipedia_headers(self.config.tools.wikipedia),
        )
        data = response.json()
        return {
            "title": data.get("title"),
            "description": data.get("description"),
            "extract": data.get("extract"),
            "url": (data.get("content_urls") or {}).get("desktop", {}).get("page"),
            "language": language,
            "image_url": (data.get("originalimage") or {}).get("source"),
            "thumbnail": (data.get("thumbnail") or {}).get("source"),
        }

    async def _weather_current(self, arguments: dict[str, Any]) -> dict[str, Any]:
        latitude = arguments.get("latitude")
        longitude = arguments.get("longitude")
        location = arguments.get("location")
        resolved_location: dict[str, Any] | None = None
        if latitude is None or longitude is None:
            if not location:
                raise ValueError("Provide either location or latitude and longitude.")
            resolved_location = await self._geocode(str(location))
            latitude = resolved_location["latitude"]
            longitude = resolved_location["longitude"]
        provider = self.config.tools.weather
        response = await self._request_with_retries(
            "GET",
            provider.base_url or "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
                "temperature_unit": arguments.get("temperature_unit") or "celsius",
                "wind_speed_unit": arguments.get("wind_speed_unit") or "kmh",
                "timezone": "auto",
            },
        )
        data = response.json()
        return {
            "location": resolved_location or {"latitude": latitude, "longitude": longitude},
            "timezone": data.get("timezone"),
            "current": data.get("current"),
            "units": data.get("current_units"),
        }

    async def _serpapi_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        provider = self.config.tools.serpapi
        api_key = _api_key(provider, "SerpAPI")
        query = _required_string(arguments, "query")
        params = {
            **provider.defaults,
            "q": query,
            "api_key": api_key,
            "output": "json",
        }
        for key in ("engine", "location", "hl", "gl", "num"):
            if arguments.get(key) is not None:
                params[key] = arguments[key]
        response = await self._request_with_retries(
            "GET",
            provider.base_url or "https://serpapi.com/search",
            params=params,
        )
        data = response.json()
        return {
            "query": query,
            "organic_results": data.get("organic_results", []),
            "answer_box": data.get("answer_box"),
            "knowledge_graph": data.get("knowledge_graph"),
            "search_metadata": data.get("search_metadata"),
        }

    async def _tavily_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        provider = self.config.tools.tavily
        api_key = _api_key(provider, "Tavily")
        query = _required_string(arguments, "query")
        payload = {**provider.defaults, "query": query}
        for key in (
            "search_depth",
            "topic",
            "max_results",
            "include_answer",
            "include_raw_content",
            "include_images",
            "include_domains",
            "exclude_domains",
        ):
            if arguments.get(key) is not None:
                payload[key] = arguments[key]
        response = await self._request_with_retries(
            "POST",
            provider.base_url or "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        return response.json()

    async def _source_research(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _required_string(arguments, "query")
        max_results = _bounded_int(arguments.get("max_results"), default=5, minimum=1, maximum=10)
        required_sources = _bounded_int(
            arguments.get("required_verified_sources"),
            default=2,
            minimum=1,
            maximum=5,
        )
        include_images = bool(arguments.get("include_images", False))
        image_limit = min(max_results, 3)
        max_verify_urls = _bounded_int(arguments.get("max_verify_urls"), default=10, minimum=1, maximum=10)
        verify_timeout_seconds = _bounded_int(
            arguments.get("verify_timeout_seconds"),
            default=20,
            minimum=3,
            maximum=20,
        )

        search_tasks = []
        if self.config.tools.tavily.enabled:
            search_tasks.append(
                self._research_tavily(
                    query,
                    image_limit if include_images else max_results,
                    include_images=include_images,
                    include_domains=arguments.get("include_domains"),
                    exclude_domains=arguments.get("exclude_domains"),
                )
            )
        if self.config.tools.serpapi.enabled:
            search_tasks.append(self._research_serpapi(query, max_results))

        if not search_tasks:
            raise ValueError("Configure SerpAPI or Tavily to use source_research.")

        gathered = await asyncio.gather(*search_tasks, return_exceptions=True)
        search_errors = [
            {"error": str(item), "type": type(item).__name__}
            for item in gathered
            if isinstance(item, Exception)
        ]
        results = [
            result
            for item in gathered
            if isinstance(item, dict)
            for result in item.get("results", [])
        ]
        images = [
            image
            for item in gathered
            if isinstance(item, dict)
            for image in item.get("images", [])
        ]
        unique_results = _dedupe_source_results(results)[: max_results * 2]
        verification = await self._verify_urls_parallel(
            [result["url"] for result in unique_results if result.get("url")],
            claim=query,
            required_sources=required_sources,
            max_urls=max_verify_urls,
            timeout_seconds=verify_timeout_seconds,
        )
        fallback_verified = _verified_from_search_excerpts(
            unique_results,
            verification["verified_sources"],
            query,
        )
        if fallback_verified:
            verification["verified_sources"].extend(fallback_verified)
            verification["verified_count"] = len(verification["verified_sources"])
            verified_keys = {
                str(source.get("url") or "").rstrip("/").lower()
                for source in verification["verified_sources"]
            }
            verification["weak_or_unreachable_sources"] = [
                source
                for source in verification["weak_or_unreachable_sources"]
                if str(source.get("url") or "").rstrip("/").lower() not in verified_keys
            ]
            verification["verdict"] = (
                "supported"
                if verification["verified_count"] >= required_sources
                else "insufficient_evidence"
            )
        verified_urls = {item["url"] for item in verification["verified_sources"]}
        enriched_sources = [
            {
                **result,
                "verified": result.get("url") in verified_urls,
            }
            for result in unique_results
        ]
        compact_images = _dedupe_images(images)[:image_limit] if include_images else []
        result = {
            "query": query,
            "guardrails": [
                "Use only verified_sources for citation-worthy claims.",
                "Prefer official or primary sources when multiple sources agree: government agencies, regulators, courts, election commissions, statistics offices, company filings, official datasets, and original reports.",
                "For current events, prefer established wire/news sources when relevant: Reuters, Associated Press, AFP, BBC, Bloomberg, Financial Times, The Hindu, Indian Express, NDTV, Hindustan Times, LiveMint, Business Standard, Economic Times, Al Jazeera, DW, France24, and similar reputable outlets.",
                "For technical, health, science, economy, or policy topics, prefer academic papers, universities, official institutions, WHO, UN, World Bank, IMF, OECD, IEA, IPCC, PubMed/NCBI, Nature, Science, Lancet, JAMA, and official datasets.",
                "Treat SEO pages, anonymous blogs, copied press releases, social posts, YouTube commentary, forums, and content farms as weak leads, not main evidence.",
                "If verdict is insufficient_evidence, answer with uncertainty and cite the gap.",
                "For markdown images, cite the image source page near each image.",
                "Use only 2-3 compact images in markdown reports.",
                "Use clear, readable images only; skip blurry thumbnails, tiny previews, cropped maps/charts, or weakly sourced images.",
                "Prefer full image_url values over thumbnails, and include a source_url for every embedded image when possible.",
            ],
            "verdict": verification["verdict"],
            "required_verified_sources": required_sources,
            "verified_sources": verification["verified_sources"],
            "weak_or_unreachable_sources": verification["weak_or_unreachable_sources"],
            "search_results": enriched_sources,
            "markdown_css": _compact_image_markdown_css() if include_images else None,
            "images": compact_images,
            "markdown_examples": [
                {
                    "title": image.get("title"),
                    "markdown": _compact_image_markdown(image),
                    "source_note": f"Source: {image.get('source_url')}",
                }
                for image in compact_images
                if image.get("image_url") or image.get("thumbnail")
            ],
            "search_errors": search_errors,
        }
        if bool(arguments.get("skip_master_review", False)):
            return result
        return await self._attach_master_review(result)

    async def _attach_master_review(self, result: dict[str, Any]) -> dict[str, Any]:
        if (
            self._master_reviewer is None
            or not self.config.master_review.run_after_deep_research
        ):
            return result
        review = await self._master_reviewer.review_deep_research({"data": result})
        result["master_review"] = review
        reviewed_answer = review.get("data", {}).get("revised_draft") if review.get("ok") else None
        if reviewed_answer:
            result["reviewed_answer"] = reviewed_answer
        final_instructions = review.get("data", {}).get("final_llm_instructions") if review.get("ok") else None
        if final_instructions:
            result["final_llm_instructions"] = final_instructions
        return result

    async def _master_review(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._master_reviewer is None:
            reviewer = MasterReviewer(self.config.master_review)
            try:
                return await reviewer.review_deep_research(
                    arguments.get("research_result") or {},
                    mode=arguments.get("mode"),
                )
            finally:
                await reviewer.aclose()
        config = self.config.master_review
        original_return_revised = config.output.return_revised_draft
        if arguments.get("return_revised_draft") is not None:
            config.output.return_revised_draft = bool(arguments.get("return_revised_draft"))
        try:
            return await self._master_reviewer.review_deep_research(
                arguments.get("research_result") or {},
                mode=arguments.get("mode"),
            )
        finally:
            config.output.return_revised_draft = original_return_revised

    async def _image_research(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _required_string(arguments, "query")
        max_results = _bounded_int(arguments.get("max_results"), default=3, minimum=1, maximum=3)
        gathered: list[dict[str, Any] | Exception] = []
        fallback_used = False
        wikipedia_used = False
        ddg_result = await asyncio.gather(
            self._research_duckduckgo_images(query, max_results),
            return_exceptions=True,
        )
        gathered.extend(ddg_result)
        if self.config.tools.wikipedia.enabled:
            wiki_result = await asyncio.gather(
                self._research_wikipedia_images(query, max_results),
                return_exceptions=True,
            )
            gathered.extend(wiki_result)
            wiki_images = _dedupe_images(
                [
                    image
                    for item in wiki_result
                    if isinstance(item, dict)
                    for image in item.get("images", [])
                ]
            )
            wikipedia_used = bool(wiki_images)
        if self.config.tools.tavily.enabled:
            tavily_result = await asyncio.gather(
                self._research_tavily(
                    query,
                    max_results,
                    include_images=True,
                    include_domains=arguments.get("include_domains"),
                    exclude_domains=arguments.get("exclude_domains"),
                ),
                return_exceptions=True,
            )
            gathered.extend(tavily_result)
            tavily_images = _dedupe_images(
                [
                    image
                    for item in tavily_result
                    if isinstance(item, dict)
                    for image in item.get("images", [])
                ]
            )
            fallback_used = not tavily_images and not wikipedia_used

        if self.config.tools.serpapi.enabled:
            if fallback_used or not self.config.tools.tavily.enabled:
                serpapi_result = await asyncio.gather(
                    self._research_serpapi_images(query, max_results),
                    return_exceptions=True,
                )
                gathered.extend(serpapi_result)

        if not gathered:
            raise ValueError("Configure SerpAPI or Tavily to use image_research.")

        errors = [
            {"error": str(item), "type": type(item).__name__}
            for item in gathered
            if isinstance(item, Exception)
        ]
        images = _dedupe_images(
            [
                image
                for item in gathered
                if isinstance(item, dict)
                for image in item.get("images", [])
            ]
        )[:max_results]
        return {
            "query": query,
            "guardrails": [
                "Do not invent image URLs.",
                "Use image_url or thumbnail exactly as returned.",
                "When a Wikipedia image has local_path, prefer that local file in markdown embeds.",
                "Prefer candidates with source_url, and cite source_url near the image in markdown.",
                "If no image has a source_url, say image provenance is weak.",
                "Use only 2-3 compact images in markdown reports.",
                "Use clear, readable images only; skip blurry thumbnails, tiny previews, cropped maps/charts, or weakly sourced images.",
                "Prefer full image_url values over thumbnails, and include a source_url for every embedded image when possible.",
                "Use regular search/source data for the article text, then place images near the relevant section.",
                "DuckDuckGo image search uses browser-style Mozilla headers and is preferred for Telegram sendable image requests.",
                "Use SerpAPI image search as a fallback when the preferred image provider fails or returns no images.",
            ],
            "provider_policy": {
                "preferred": "duckduckgo_images",
                "fallback": "serpapi_images",
                "fallback_used": fallback_used,
                "wikipedia_used": wikipedia_used,
            },
            "markdown_css": _compact_image_markdown_css(),
            "images": images,
            "markdown_examples": [
                {
                    "title": image.get("title"),
                    "markdown": _compact_image_markdown(image),
                    "source_note": f"Source: {image.get('source_url')}",
                }
                for image in images
                if image.get("image_url") or image.get("thumbnail")
            ],
            "errors": errors,
        }

    async def _research_duckduckgo_images(self, query: str, max_results: int) -> dict[str, Any]:
        headers = {
            "User-Agent": MOZILLA_IMAGE_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        page = await self._request_with_retries(
            "GET",
            "https://duckduckgo.com/",
            params={"q": query, "iax": "images", "ia": "images"},
            headers=headers,
            follow_redirects=True,
        )
        match = re.search(r"vqd=['\"]?([^'\"&<>\\]+)", page.text)
        if not match:
            raise ValueError("DuckDuckGo image token was not found.")
        vqd = match.group(1)
        response = await self._request_with_retries(
            "GET",
            "https://duckduckgo.com/i.js",
            params={
                "l": "wt-wt",
                "o": "json",
                "q": query,
                "vqd": vqd,
                "f": ",,,,,",
                "p": "1",
                "s": "0",
            },
            headers={
                **headers,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Referer": str(page.url),
                "X-Requested-With": "XMLHttpRequest",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            follow_redirects=True,
        )
        data = response.json()
        images = [
            _image_candidate("duckduckgo", item, title=item.get("title") if isinstance(item, dict) else query)
            for item in (data.get("results") or [])[:max_results]
        ]
        return {"provider": "duckduckgo", "results": [], "images": images}

    async def _research_wikipedia_images(self, query: str, max_results: int) -> dict[str, Any]:
        provider = self.config.tools.wikipedia
        language = str(provider.defaults.get("language") or "en").lower()
        response = await self._request_with_retries(
            "GET",
            f"https://{language}.wikipedia.org/w/api.php",
            headers=_wikipedia_headers(provider),
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max(max_results * 2, 5),
                "format": "json",
                "utf8": 1,
            },
        )
        data = response.json()
        titles = [
            str(item.get("title") or "").strip()
            for item in data.get("query", {}).get("search", [])
            if str(item.get("title") or "").strip()
        ][: max(max_results * 2, 5)]
        summaries = await asyncio.gather(
            *(self._wikipedia_page({"title": title, "language": language}) for title in titles),
            return_exceptions=True,
        )
        images: list[dict[str, Any]] = []
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            image_url = str(summary.get("image_url") or "").strip()
            if not image_url:
                continue
            local_path = await self._download_wikipedia_image(
                title=str(summary.get("title") or query),
                image_url=image_url,
            )
            images.append(
                {
                    "provider": "wikipedia",
                    "title": summary.get("title"),
                    "image_url": image_url,
                    "thumbnail": summary.get("thumbnail"),
                    "source_url": summary.get("url"),
                    "source_name": "Wikipedia",
                    "local_path": local_path,
                }
            )
            if len(images) >= max_results:
                break
        return {"provider": "wikipedia", "results": [], "images": images}

    async def _download_wikipedia_image(self, *, title: str, image_url: str) -> str | None:
        try:
            _validate_public_http_url(image_url)
            response = await self._request_with_retries(
                "GET",
                image_url,
                headers={"User-Agent": WIKIMEDIA_USER_AGENT},
                follow_redirects=True,
            )
        except Exception:
            return None
        suffix = Path(urlparse(str(response.url)).path).suffix.lower()
        if not suffix or len(suffix) > 6:
            content_type = response.headers.get("content-type", "").lower()
            if "png" in content_type:
                suffix = ".png"
            elif "jpeg" in content_type or "jpg" in content_type:
                suffix = ".jpg"
            elif "webp" in content_type:
                suffix = ".webp"
            elif "gif" in content_type:
                suffix = ".gif"
            else:
                suffix = ".img"
        image_dir = Path.cwd() / "wiki_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify_filename(title)[:80] or "wikipedia-image"
        file_path = image_dir / f"{slug}{suffix}"
        counter = 2
        while file_path.exists():
            file_path = image_dir / f"{slug[:76]}-{counter}{suffix}"
            counter += 1
        file_path.write_bytes(response.content)
        return file_path.relative_to(Path.cwd()).as_posix()

    async def _image_download(self, arguments: dict[str, Any]) -> dict[str, Any]:
        image_url = _validate_public_http_url(_required_string(arguments, "image_url"))
        source_url = str(arguments.get("source_url") or "").strip()
        if source_url:
            _validate_public_http_url(source_url)
        title = str(arguments.get("title") or "image").strip()
        output_dir = safe_tool_output_dir(arguments.get("output_dir") or "downloaded_images")
        headers = {
            "User-Agent": MOZILLA_IMAGE_USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if source_url:
            headers["Referer"] = source_url
        try:
            response = await self._request_with_retries(
                "GET",
                image_url,
                headers=headers,
                follow_redirects=True,
            )
        except Exception:
            proxy_url = f"https://external-content.duckduckgo.com/iu/?u={quote(image_url, safe='')}&f=1&nofb=1"
            response = await self._request_with_retries(
                "GET",
                proxy_url,
                headers={
                    **headers,
                    "Referer": "https://duckduckgo.com/",
                },
                follow_redirects=True,
            )
        content_type = response.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
        if content_type and not (content_type.startswith("image/") or content_type == "application/octet-stream"):
            raise ValueError(f"URL did not return an image content type: {content_type}")
        content = response.content
        if len(content) > 25 * 1024 * 1024:
            raise ValueError("Image is too large to download safely.")
        suffix = image_suffix_from_response(str(response.url), content_type)
        if suffix not in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}:
            raise ValueError(f"Unsupported image type: {content_type or suffix}")
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify_filename(title)[:80] or "image"
        path = unique_tool_path(output_dir / f"{slug}{suffix}")
        path.write_bytes(content)
        return {
            "title": title,
            "image_url": image_url,
            "source_url": source_url or None,
            "local_path": path.relative_to(Path.cwd()).as_posix(),
            "absolute_path": str(path.resolve()),
            "media_type": content_type or None,
            "bytes": len(content),
        }

    async def _verify_sources(self, arguments: dict[str, Any]) -> dict[str, Any]:
        urls = arguments.get("urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError("urls must be a non-empty list.")
        clean_urls = [_clean_url(str(url)) for url in urls if str(url).strip()]
        if not clean_urls:
            raise ValueError("urls must contain at least one valid URL.")
        required_sources = _bounded_int(
            arguments.get("required_verified_sources"),
            default=2,
            minimum=1,
            maximum=5,
        )
        return await self._verify_urls_parallel(
            clean_urls[:10],
            claim=str(arguments.get("claim") or ""),
            required_sources=required_sources,
        )

    async def _research_tavily(
        self,
        query: str,
        max_results: int,
        *,
        include_images: bool,
        include_domains: Any = None,
        exclude_domains: Any = None,
    ) -> dict[str, Any]:
        provider = self.config.tools.tavily
        api_key = _api_key(provider, "Tavily")
        payload = {
            **provider.defaults,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": True,
            "include_images": include_images,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        response = await self._request_with_retries(
            "POST",
            provider.base_url or "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        data = response.json()
        return {
            "provider": "tavily",
            "answer": data.get("answer"),
            "results": [
                {
                    "provider": "tavily",
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content") or item.get("raw_content"),
                    "score": item.get("score"),
                }
                for item in data.get("results") or []
                if item.get("url")
            ],
            "images": [
                _image_candidate("tavily", image, title=query)
                for image in data.get("images") or []
            ],
        }

    async def _research_serpapi(self, query: str, max_results: int) -> dict[str, Any]:
        provider = self.config.tools.serpapi
        api_key = _api_key(provider, "SerpAPI")
        params = {
            **provider.defaults,
            "q": query,
            "api_key": api_key,
            "output": "json",
            "num": max_results,
        }
        response = await self._request_with_retries(
            "GET",
            provider.base_url or "https://serpapi.com/search",
            params=params,
        )
        data = response.json()
        results = []
        for item in data.get("organic_results") or []:
            if item.get("link"):
                results.append(
                    {
                        "provider": "serpapi",
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "snippet": item.get("snippet"),
                        "position": item.get("position"),
                    }
                )
        return {
            "provider": "serpapi",
            "answer": data.get("answer_box"),
            "results": results,
            "images": [],
        }

    async def _research_serpapi_images(self, query: str, max_results: int) -> dict[str, Any]:
        provider = self.config.tools.serpapi
        api_key = _api_key(provider, "SerpAPI")
        params = {
            **provider.defaults,
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
            "output": "json",
        }
        params.pop("num", None)
        response = await self._request_with_retries(
            "GET",
            provider.base_url or "https://serpapi.com/search",
            params=params,
        )
        data = response.json()
        return {
            "provider": "serpapi",
            "images": [
                _image_candidate("serpapi", item, title=item.get("title"))
                for item in (data.get("images_results") or [])[:max_results]
            ],
        }

    async def _verify_urls_parallel(
        self,
        urls: list[str],
        *,
        claim: str,
        required_sources: int,
        max_urls: int = 10,
        timeout_seconds: float = 20.0,
    ) -> dict[str, Any]:
        unique_urls = _dedupe_urls(urls)
        tasks = [
            self._verify_one_source(index + 1, url, claim, timeout_seconds=timeout_seconds)
            for index, url in enumerate(unique_urls[:max_urls])
        ]
        results = await asyncio.gather(*tasks)
        verified = [
            result
            for result in results
            if result.get("reachable") and result.get("relevance_score", 0.0) >= 0.2
        ]
        weak = [result for result in results if result not in verified]
        verdict = "supported" if len(verified) >= required_sources else "insufficient_evidence"
        return {
            "claim": claim or None,
            "verdict": verdict,
            "required_verified_sources": required_sources,
            "verified_count": len(verified),
            "verified_sources": verified,
            "weak_or_unreachable_sources": weak,
            "source_policy": (
                "Use verified_sources only for factual citations. "
                "If verdict is insufficient_evidence, do not present the claim as certain."
            ),
        }

    async def _verify_one_source(
        self,
        agent_id: int,
        url: str,
        claim: str,
        *,
        timeout_seconds: float = 20.0,
    ) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        try:
            _validate_public_http_url(url)
            response = await self._request_with_retries(
                "GET",
                url,
                headers={"User-Agent": WIKIMEDIA_USER_AGENT},
                follow_redirects=True,
                timeout=timeout_seconds,
            )
            _validate_public_http_url(str(response.url))
            html = response.text[:200_000]
            text = _html_to_text(html)
            title = _html_title(html) or urlparse(str(response.url)).netloc
            relevance = _text_relevance(text, claim or title)
            return {
                "agent_id": f"source-verifier-{agent_id}",
                "url": str(response.url),
                "original_url": url,
                "domain": urlparse(str(response.url)).netloc,
                "reachable": True,
                "status_code": response.status_code,
                "title": title,
                "evidence_snippet": _evidence_snippet(text, claim),
                "relevance_score": relevance,
                "duration_ms": _duration_ms(started_at),
            }
        except Exception as exc:
            return {
                "agent_id": f"source-verifier-{agent_id}",
                "url": url,
                "domain": urlparse(url).netloc,
                "reachable": False,
                "error": str(exc) or type(exc).__name__,
                "relevance_score": 0.0,
                "duration_ms": _duration_ms(started_at),
            }

    async def _geocode(self, location: str) -> dict[str, Any]:
        geocoding_url = self.config.tools.weather.defaults.get(
            "geocoding_url",
            "https://geocoding-api.open-meteo.com/v1/search",
        )
        results: list[dict[str, Any]] = []
        for query in _location_queries(location):
            response = await self._request_with_retries(
                "GET",
                geocoding_url,
                params={"name": query, "count": 10, "format": "json"},
            )
            results = _rank_open_meteo_results(
                response.json().get("results") or [],
                location,
            )
            if results:
                break
        if results:
            first = results[0]
            return {
                "name": first.get("name"),
                "country": first.get("country"),
                "admin1": first.get("admin1"),
                "latitude": first["latitude"],
                "longitude": first["longitude"],
            }

        fallback = await self._geocode_with_osm(location)
        if fallback is None:
            raise ValueError(f"Could not geocode location: {location}")
        return fallback

    async def _geocode_with_osm(self, location: str) -> dict[str, Any] | None:
        url = self.config.tools.weather.defaults.get(
            "osm_geocoding_url",
            "https://nominatim.openstreetmap.org/search",
        )
        response = await self._request_with_retries(
            "GET",
            url,
            headers={"User-Agent": GEOCODING_USER_AGENT},
            params={"q": location, "format": "jsonv2", "limit": 1},
        )
        results = response.json() or []
        if not results:
            return None
        first = results[0]
        address = first.get("address") or {}
        return {
            "name": first.get("name") or first.get("display_name"),
            "country": address.get("country"),
            "admin1": address.get("state"),
            "latitude": float(first["lat"]),
            "longitude": float(first["lon"]),
        }


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def render_manim_video(arguments: dict[str, Any], *, cwd: Path | None = None) -> dict[str, Any]:
    prompt = _required_string(arguments, "prompt")
    title = str(arguments.get("title") or _manim_title(prompt)).strip()
    quality = str(arguments.get("quality") or "low").lower()
    quality_flag = {"low": "-ql", "medium": "-qm", "high": "-qh"}.get(quality, "-ql")
    should_render = bool(arguments.get("render", True))
    timeout_seconds = _bounded_int(arguments.get("timeout_seconds"), default=180, minimum=30, maximum=600)
    base_dir = Path(cwd or Path.cwd())
    output_dir = Path(str(arguments.get("output_dir") or base_dir / "manim_outputs")).expanduser()
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(title or prompt, default="manim_scene")
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    scene_name = "GeneratedManimScene"
    scene_path = output_dir / f"{slug}_{stamp}.py"
    media_dir = output_dir / "media"
    output_name = f"{slug}_{stamp}"
    script = _manim_scene_script(prompt=prompt, title=title, scene_name=scene_name)
    scene_path.write_text(script, encoding="utf-8")

    result: dict[str, Any] = {
        "ok": True,
        "prompt": prompt,
        "title": title,
        "scene": scene_name,
        "scene_path": str(scene_path),
        "script": script,
        "rendered": False,
        "video_path": None,
        "quality": quality,
    }
    if not should_render:
        result["message"] = "Scene file created; render=false so no video was rendered."
        return result

    command = [
        sys.executable,
        "-m",
        "manim",
        quality_flag,
        "--media_dir",
        str(media_dir),
        "--output_file",
        output_name,
        str(scene_path),
        scene_name,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(output_dir),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            **result,
            "ok": False,
            "rendered": False,
            "command": command,
            "error": f"Manim render timed out after {timeout_seconds} seconds.",
            "stdout": _truncate_tool_output(exc.stdout or ""),
            "stderr": _truncate_tool_output(exc.stderr or ""),
        }

    video_path = _find_manim_video(media_dir, output_name)
    result.update(
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout": _truncate_tool_output(completed.stdout),
            "stderr": _truncate_tool_output(completed.stderr),
            "video_path": str(video_path) if video_path else None,
            "rendered": completed.returncode == 0 and video_path is not None,
        }
    )
    if completed.returncode != 0:
        result["ok"] = False
        result["error"] = (
            "Manim render failed. Install Manim Community and runtime dependencies, then retry. "
            "Typical command: python -m pip install manim"
        )
    elif video_path is None:
        result["ok"] = False
        result["error"] = "Manim reported success but no MP4 output was found."
    return result


def _manim_scene_script(*, prompt: str, title: str, scene_name: str) -> str:
    bullets = _manim_bullets(prompt)
    bullet_lines = ",\n            ".join(json.dumps(item, ensure_ascii=False) for item in bullets)
    return f'''from manim import *


class {scene_name}(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"
        title = Text({json.dumps(title, ensure_ascii=False)}, font_size=38, weight=BOLD, color=WHITE)
        title.to_edge(UP)
        underline = Line(LEFT * 3.2, RIGHT * 3.2, color=TEAL).next_to(title, DOWN, buff=0.18)
        self.play(Write(title), Create(underline), run_time=1.2)

        bullets = [
            {bullet_lines}
        ]
        rows = VGroup()
        palette = [BLUE, GREEN, YELLOW, ORANGE, PURPLE, TEAL]
        for index, text in enumerate(bullets):
            dot = Dot(color=palette[index % len(palette)])
            label = Text(text, font_size=25, color=WHITE).scale_to_fit_width(9.8)
            row = VGroup(dot, label).arrange(RIGHT, buff=0.22)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.36)
        rows.next_to(underline, DOWN, buff=0.55)
        rows.to_edge(LEFT, buff=1.0)

        for row in rows:
            self.play(FadeIn(row[0], scale=1.6), Write(row[1]), run_time=0.75)

        frame = SurroundingRectangle(rows, color=BLUE_E, buff=0.35, corner_radius=0.12)
        accent = Circle(radius=0.5, color=TEAL, fill_opacity=0.25).to_corner(DR)
        self.play(Create(frame), GrowFromCenter(accent), run_time=1.0)
        self.play(Rotate(accent, angle=TAU), run_time=2.0)
        self.wait(1.0)
'''


def _manim_bullets(prompt: str) -> list[str]:
    parts = [part.strip(" -:\t\r\n") for part in re.split(r"[.\n;]+", prompt) if part.strip(" -:\t\r\n")]
    if len(parts) <= 1:
        words = prompt.split()
        parts = [" ".join(words[index:index + 9]) for index in range(0, min(len(words), 54), 9)]
    bullets = [_compact_for_manim(part, 82) for part in parts[:6]]
    return bullets or ["Visualize the idea", "Break it into clear steps", "Animate each step simply"]


def _compact_for_manim(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _manim_title(prompt: str) -> str:
    first = _manim_bullets(prompt)[0]
    return _compact_for_manim(first, 54)


def _find_manim_video(media_dir: Path, output_name: str) -> Path | None:
    candidates = list(media_dir.rglob(f"{output_name}.mp4"))
    if not candidates:
        candidates = list(media_dir.rglob("*.mp4"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _slugify(value: str, *, default: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return (slug or default)[:60]


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ToolValidationError("Tool arguments must be a JSON object.")
    cleaned = _strip_string_values(arguments)

    if name in {"wikipedia_search", "serpapi_search", "tavily_search", "source_research", "image_research"}:
        _ensure_non_empty_string(cleaned, "query")
    if name == "image_download":
        _ensure_non_empty_string(cleaned, "image_url")
        cleaned["image_url"] = _validate_public_http_url(_clean_url(str(cleaned["image_url"])))
        if cleaned.get("source_url"):
            cleaned["source_url"] = _validate_public_http_url(_clean_url(str(cleaned["source_url"])))
        if cleaned.get("output_dir") is not None and not isinstance(cleaned["output_dir"], str):
            cleaned["output_dir"] = str(cleaned["output_dir"])
    if name == "manim_render":
        _ensure_non_empty_string(cleaned, "prompt")
        if cleaned.get("quality") is not None and cleaned["quality"] not in {"low", "medium", "high"}:
            raise ToolValidationError("quality must be one of: low, medium, high.")
        if "timeout_seconds" in cleaned and cleaned["timeout_seconds"] is not None:
            cleaned["timeout_seconds"] = _bounded_int(
                cleaned["timeout_seconds"],
                default=180,
                minimum=30,
                maximum=600,
            )
    if name == "wikipedia_page":
        _ensure_non_empty_string(cleaned, "title")
    if name == "weather_current":
        if cleaned.get("location") is not None:
            _ensure_non_empty_string(cleaned, "location")
        has_location = bool(cleaned.get("location"))
        has_lat_lon = cleaned.get("latitude") is not None and cleaned.get("longitude") is not None
        if not has_location and not has_lat_lon:
            raise ToolValidationError("Provide either location or latitude and longitude.")
        if cleaned.get("latitude") is not None:
            cleaned["latitude"] = _bounded_float(cleaned["latitude"], key="latitude", minimum=-90.0, maximum=90.0)
        if cleaned.get("longitude") is not None:
            cleaned["longitude"] = _bounded_float(cleaned["longitude"], key="longitude", minimum=-180.0, maximum=180.0)
    if name == "verify_sources":
        urls = cleaned.get("urls")
        if not isinstance(urls, list) or not urls:
            raise ToolValidationError("urls must be a non-empty list.")
        cleaned["urls"] = [_validate_public_http_url(_clean_url(str(url))) for url in urls if str(url).strip()]
        if not cleaned["urls"]:
            raise ToolValidationError("urls must contain at least one valid public HTTP URL.")
        if cleaned.get("claim") is not None and not isinstance(cleaned["claim"], str):
            cleaned["claim"] = str(cleaned["claim"])
    if name == "master_review":
        if not isinstance(cleaned.get("research_result"), dict):
            raise ToolValidationError("research_result must be a JSON object.")
        if cleaned.get("mode") is not None and cleaned["mode"] not in {"fast", "balanced", "strict"}:
            raise ToolValidationError("mode must be one of: fast, balanced, strict.")

    for key, maximum in {"limit": 20, "num": 20, "max_results": 20}.items():
        if key in cleaned and cleaned[key] is not None:
            cleaned[key] = _bounded_int(cleaned[key], default=5, minimum=1, maximum=maximum)
    if "required_verified_sources" in cleaned and cleaned["required_verified_sources"] is not None:
        cleaned["required_verified_sources"] = _bounded_int(
            cleaned["required_verified_sources"],
            default=2,
            minimum=1,
            maximum=5,
        )
    return cleaned


def _strip_string_values(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_strip_string_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_string_values(item) for key, item in value.items()}
    return value


def _ensure_non_empty_string(arguments: dict[str, Any], key: str) -> None:
    if not isinstance(arguments.get(key), str) or not arguments[key].strip():
        raise ToolValidationError(f"{key} must be a non-empty string.")


def _bounded_float(value: Any, *, key: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"{key} must be a number.") from exc
    if number < minimum or number > maximum:
        raise ToolValidationError(f"{key} must be between {minimum:g} and {maximum:g}.")
    return number


def _validate_public_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolValidationError("URL must be an absolute http or https URL.")
    host = parsed.hostname or ""
    if host.lower() in {"localhost", "127.0.0.1", "::1"}:
        raise ToolValidationError("Private or localhost URLs are not allowed.")
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ToolValidationError(f"Could not resolve URL host: {host}") from exc
    for entry in addresses:
        ip_text = entry[4][0]
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ToolValidationError("Private or localhost URLs are not allowed.")
    return value


def _clean_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value
    parsed = urlparse(value)
    if not parsed.scheme:
        return f"https://{value}"
    return value


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        clean = _clean_url(url)
        if not clean:
            continue
        key = clean.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(clean)
    return unique


def _dedupe_source_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for result in results:
        url = _clean_url(str(result.get("url") or ""))
        if not url:
            continue
        key = url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append({**result, "url": url, "domain": urlparse(url).netloc})
    return unique


def _verified_from_search_excerpts(
    results: list[dict[str, Any]],
    verified_sources: list[dict[str, Any]],
    claim: str,
) -> list[dict[str, Any]]:
    verified_urls = {str(source.get("url") or "").rstrip("/").lower() for source in verified_sources}
    fallback: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        url = str(result.get("url") or "")
        snippet = _collapse_whitespace(str(result.get("snippet") or ""))
        if not url or not snippet:
            continue
        key = url.rstrip("/").lower()
        if key in verified_urls:
            continue
        relevance = _text_relevance(snippet, claim)
        if relevance < 0.2:
            continue
        fallback.append(
            {
                "agent_id": f"search-excerpt-verifier-{index}",
                "url": url,
                "domain": urlparse(url).netloc,
                "reachable": None,
                "provider_excerpt_verified": True,
                "verification_method": "search_provider_excerpt",
                "title": result.get("title"),
                "evidence_snippet": snippet[:600],
                "relevance_score": relevance,
                "provider": result.get("provider"),
            }
        )
    return fallback


def _image_candidate(provider: str, item: Any, *, title: str | None = None) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "provider": provider,
            "title": title,
            "image_url": item,
            "thumbnail": None,
            "source_url": None,
        }
    if not isinstance(item, dict):
        return {
            "provider": provider,
            "title": title,
            "image_url": None,
            "thumbnail": None,
            "source_url": None,
        }
    image_url = (
        item.get("original")
        or item.get("image")
        or item.get("image_url")
        or item.get("url")
    )
    source_url = (
        item.get("source")
        or item.get("source_url")
        or item.get("link")
        or item.get("page_url")
    )
    thumbnail = item.get("thumbnail") or item.get("thumbnail_url")
    return {
        "provider": provider,
        "title": item.get("title") or title,
        "image_url": image_url,
        "thumbnail": thumbnail,
        "source_url": source_url,
        "source_name": item.get("source_name") or item.get("source"),
        "local_path": item.get("local_path"),
    }


def _dedupe_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for image in images:
        url = image.get("local_path") or image.get("image_url") or image.get("thumbnail")
        if not url:
            continue
        key = str(url).rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(image)
    return unique


def _markdown_alt(image: dict[str, Any]) -> str:
    title = str(image.get("title") or "image").strip()
    return title.replace("[", "(").replace("]", ")")


def _compact_image_markdown_css() -> str:
    return (
        "<style>\n"
        ".image-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));"
        "gap:14px;margin:16px 0;align-items:start;}\n"
        ".image-card{margin:0;font-size:.9em;line-height:1.4;}\n"
        ".image-card img{display:block;width:100%;height:auto;max-height:240px;object-fit:contain;"
        "background:#f6f6f6;border:1px solid #ddd;border-radius:6px;padding:4px;}\n"
        ".image-card figcaption{margin-top:6px;color:#555;}\n"
        "</style>"
    )


def _compact_image_markdown(image: dict[str, Any]) -> str:
    url = image.get("local_path") or image.get("image_url") or image.get("thumbnail")
    alt = _markdown_alt(image)
    source_url = image.get("source_url")
    source = f'<a href="{source_url}">source</a>' if source_url else "source unavailable"
    return (
        '<figure class="image-card">\n'
        f'  <img src="{url}" alt="{alt}" loading="lazy">\n'
        f"  <figcaption>{alt} - {source}</figcaption>\n"
        "</figure>"
    )


def _slugify_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    return text.strip(".-").lower()


def safe_tool_output_dir(value: Any) -> Path:
    raw = str(value or "downloaded_images").strip() or "downloaded_images"
    path = Path(raw)
    if path.is_absolute():
        raise ToolValidationError("output_dir must be relative to the workspace.")
    root = Path.cwd().resolve()
    resolved = (root / path).resolve()
    if root != resolved and root not in resolved.parents:
        raise ToolValidationError("output_dir must stay inside the workspace.")
    return resolved


def image_suffix_from_response(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}:
        return suffix
    if "png" in content_type:
        return ".png"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    if "svg" in content_type:
        return ".svg"
    return ".img"


def unique_tool_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for counter in range(2, 10_000):
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError("Could not choose a unique output path.")


def _html_title(html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _collapse_whitespace(_strip_html(match.group(1)))[:200] or None


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p\s*>", "\n", html)
    text = _strip_html(html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return _collapse_whitespace(text)


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _text_relevance(text: str, claim: str) -> float:
    if not claim:
        return 0.5 if text else 0.0
    terms = _meaningful_terms(claim)
    if not terms:
        return 0.5 if text else 0.0
    haystack = text.lower()
    matched = sum(1 for term in terms if term in haystack)
    return round(matched / len(terms), 3)


def _meaningful_terms(value: str) -> list[str]:
    stop = {
        "about", "after", "again", "also", "and", "are", "can", "could",
        "does", "for", "from", "have", "how", "into", "latest", "more",
        "news", "not", "the", "this", "today", "was", "what", "when",
        "where", "which", "with", "would", "you",
    }
    return [
        term
        for term in re.findall(r"[a-z0-9][a-z0-9-]{2,}", value.lower())
        if term not in stop
    ][:12]


def _evidence_snippet(text: str, claim: str) -> str:
    if not text:
        return ""
    terms = _meaningful_terms(claim)
    lower = text.lower()
    first_index = min(
        (lower.find(term) for term in terms if term in lower),
        default=0,
    )
    start = max(0, first_index - 180)
    end = min(len(text), first_index + 420)
    return text[start:end].strip()


def _timezone_for_country(country: str) -> str | None:
    if not country:
        return None
    return COUNTRY_TIMEZONES.get(country.strip().lower())


def _timezone_info(timezone: str) -> tuple[str, Any]:
    if timezone.upper() in {"UTC", "UST", "Z"}:
        return "UTC", UTC
    if timezone == "Asia/Calcutta":
        timezone = "Asia/Kolkata"
    if timezone in TIMEZONE_OFFSETS:
        return timezone, TIMEZONE_OFFSETS[timezone]
    try:
        return timezone, ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone}") from exc


def _time_payload(value: datetime, timezone: str) -> dict[str, Any]:
    return {
        "timezone": timezone,
        "iso": value.isoformat(),
        "date": value.date().isoformat(),
        "time": value.strftime("%H:%M:%S"),
        "utc_offset": value.strftime("%z"),
        "unix": int(value.timestamp()),
    }


def _location_queries(location: str) -> list[str]:
    parts = [part.strip() for part in location.split(",") if part.strip()]
    queries = [location.strip()]
    if parts:
        queries.append(parts[0])
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)
    return unique


def _rank_open_meteo_results(
    results: list[dict[str, Any]],
    location: str,
) -> list[dict[str, Any]]:
    if not results:
        return []
    terms = {part.strip().lower() for part in location.split(",") if part.strip()}
    if not terms:
        return results

    def score(item: dict[str, Any]) -> int:
        values = {
            str(item.get(key) or "").lower()
            for key in ("name", "country", "country_code", "admin1", "admin2", "admin3")
        }
        return sum(1 for term in terms if any(term in value or value in term for value in values))

    return sorted(results, key=score, reverse=True)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _truncate_tool_output(value: Any, limit: int = 12000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... <truncated {omitted} chars>"


def _api_key(provider: ExternalToolProviderConfig, label: str) -> str:
    api_key = provider.api_key or ""
    if not api_key or api_key.startswith("${"):
        raise ValueError(f"{label} API key is not configured.")
    return api_key


def _language(arguments: dict[str, Any], provider: ExternalToolProviderConfig) -> str:
    return str(arguments.get("language") or provider.defaults.get("language") or "en").lower()


def _wikipedia_headers(provider: ExternalToolProviderConfig) -> dict[str, str]:
    user_agent = str(provider.defaults.get("user_agent") or WIKIMEDIA_USER_AGENT)
    return {"User-Agent": user_agent}


def _wiki_title(title: str) -> str:
    return quote(str(title).strip().replace(" ", "_"))


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _validate_tool_output(name: str, result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError(f"{name} returned non-object output")

    if name == "datetime_now":
        _require_output_keys(name, result, {"timezone", "iso", "date", "time"})
    elif name == "weather_current":
        _require_output_keys(name, result, {"location", "current"})
        if result.get("current") is not None and not isinstance(result.get("current"), dict):
            raise ValueError("weather_current returned invalid current weather object")
    elif name == "wikipedia_search":
        _require_list_output(name, result, "results")
    elif name == "wikipedia_page":
        _require_output_keys(name, result, {"title", "extract", "url"})
    elif name == "serpapi_search":
        _require_list_output(name, result, "organic_results")
    elif name == "tavily_search":
        _require_list_output(name, result, "results")
    elif name in {"source_research", "verify_sources"}:
        if "verdict" not in result:
            raise ValueError(f"{name} returned no verdict")
    elif name == "image_research":
        _require_list_output(name, result, "images")
    elif name == "image_download":
        _require_output_keys(name, result, {"local_path", "absolute_path", "bytes"})
    elif name == "master_review":
        _require_output_keys(name, result, {"ok", "tool", "data", "metadata"})
    elif name == "manim_render":
        _require_output_keys(name, result, {"scene_path", "rendered", "video_path"})

    return result


def _require_output_keys(name: str, result: dict[str, Any], keys: set[str]) -> None:
    missing = sorted(key for key in keys if key not in result)
    if missing:
        raise ValueError(f"{name} output missing keys: {', '.join(missing)}")


def _require_list_output(name: str, result: dict[str, Any], key: str) -> None:
    if key not in result:
        raise ValueError(f"{name} output missing key: {key}")
    if not isinstance(result.get(key), list):
        raise ValueError(f"{name} output key '{key}' must be a list")


def _structured_tool_success(
    name: str,
    result: dict[str, Any],
    started_at: datetime,
) -> dict[str, Any]:
    data = result if isinstance(result, dict) else {"value": result}
    finished_at = datetime.now(UTC)
    latency_ms = _duration_ms(started_at, finished_at)
    source = _tool_source(name)
    return {
        "ok": True,
        "tool": name,
        "data": data,
        "metadata": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "latency_ms": latency_ms,
            "source": source,
            "cache_hit": False,
        },
        "source": source,
        "timestamp": finished_at.isoformat(),
        "duration_ms": latency_ms,
    }


def _structured_tool_error(
    name: str,
    exc: Exception,
    started_at: datetime,
) -> dict[str, Any]:
    finished_at = datetime.now(UTC)
    latency_ms = _duration_ms(started_at, finished_at)
    return {
        "ok": False,
        "tool": name,
        "error": {
            "type": _tool_error_type(exc),
            "message": _redact_secret_text(str(exc) or type(exc).__name__),
            "retryable": _tool_error_retryable(exc),
        },
        "metadata": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "latency_ms": latency_ms,
        },
        "retryable": _tool_error_retryable(exc),
        "timestamp": finished_at.isoformat(),
        "duration_ms": latency_ms,
    }


def _duration_ms(started_at: datetime, finished_at: datetime | None = None) -> int:
    finished = finished_at or datetime.now(UTC)
    return max(0, int((finished - started_at).total_seconds() * 1000))


def _tool_error_type(exc: Exception) -> str:
    if isinstance(exc, UnknownToolError):
        return "UnknownToolError"
    if isinstance(exc, ToolValidationError):
        return "ValidationError"
    if isinstance(exc, httpx.TimeoutException):
        return "TimeoutError"
    if isinstance(exc, httpx.HTTPStatusError):
        return "HTTPStatusError"
    if isinstance(exc, httpx.RequestError):
        return "ToolExecutionError"
    return type(exc).__name__ or "ToolExecutionError"


def _tool_error_retryable(exc: Exception) -> bool:
    if isinstance(exc, (KeyError, ValueError, UnknownToolError, ToolValidationError)):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError))


def _redact_secret_text(value: str) -> str:
    value = re.sub(r"(?i)(api[_-]?key|authorization|token)=([^&\s]+)", r"\1=<redacted>", value)
    value = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", value)
    return value


def _tool_source(name: str) -> str:
    if name.startswith("wikipedia_"):
        return "wikipedia"
    if name == "weather_current":
        return "open-meteo"
    if name == "serpapi_search":
        return "serpapi"
    if name == "tavily_search":
        return "tavily"
    if name == "source_research":
        return "serpapi+tavily+parallel-verifiers"
    if name == "image_research":
        return "serpapi+tavily-images"
    if name == "image_download":
        return "http-image-download"
    if name == "verify_sources":
        return "parallel-source-verifiers"
    if name == "master_review":
        return "master-review"
    if name == "manim_render":
        return "manim"
    if name == "datetime_now":
        return "system-clock"
    return "llama-bridge"


# Tool Relevance Scoring and Selection

TOOL_KEYWORDS = {
    "datetime_now": {
        "keywords": ["time", "date", "now", "current time", "timezone", "what time"],
        "weight": 3.0,
    },
    "weather_current": {
        "keywords": ["weather", "temperature", "rain", "wind", "forecast", "cold", "hot", "humid", "snowing"],
        "weight": 3.0,
    },
    "wikipedia_search": {
        "keywords": ["who is", "what is", "history", "explain", "definition", "biography", "concept", "term"],
        "weight": 2.5,
    },
    "wikipedia_page": {
        "keywords": ["summary", "about", "wikipedia", "article", "learn about"],
        "weight": 2.0,
    },
    "tavily_search": {
        "keywords": ["latest", "news", "current", "recent", "today", "price", "search", "find", "research"],
        "weight": 2.5,
    },
    "serpapi_search": {
        "keywords": ["search", "find", "look for", "what about", "price", "latest", "news"],
        "weight": 2.5,
    },
    "source_research": {
        "keywords": [
            "verify", "source", "sources", "citation", "cite", "accurate", "solid",
            "research", "deep research", "latest", "current", "evidence", "fact check", "markdown",
        ],
        "weight": 3.5,
    },
    "image_research": {
        "keywords": ["image", "images", "photo", "picture", "visual", "markdown", "illustrate", "screenshot"],
        "weight": 3.5,
    },
    "image_download": {
        "keywords": ["download image", "send image", "attach image", "upload photo", "local image"],
        "weight": 3.2,
    },
    "verify_sources": {
        "keywords": ["verify", "check source", "validate", "link", "links", "url", "citation", "fact check"],
        "weight": 3.0,
    },
    "master_review": {
        "keywords": [
            "review report", "rate report", "audit report",
            "check evidence", "citation audit", "source quality",
            "improve research", "make it 9.5", "final reviewer",
            "quality check", "master review", "fact check report",
        ],
        "weight": 4.0,
    },
    "manim_render": {
        "keywords": ["manim", "animation", "animated video", "video", "math animation", "visualize", "diagram"],
        "weight": 4.0,
    },
}


def classify_query_intent(query: str) -> dict[str, float]:
    text = f" {query.lower()} "
    if not text.strip():
        return {}
    no_tool = _phrase_score(
        text,
        ("write a poem", "write a story", "rewrite", "proofread", "summarize this", "refactor", "debug", "implement"),
    )
    intents = {
        "weather": _phrase_score(
            text,
            ("weather", "temperature", "rain", "raining", "will it rain", "humidity", "wind", "conditions in"),
        ),
        "time": _phrase_score(
            text,
            ("current time", "what time", "time in", "today's date", "date today", "today", "tomorrow", "this week", "timezone"),
        ),
        "web_search": _phrase_score(
            text,
            ("latest", "news", "current", "recent", "today", "price", "released", "schedule", "look up", "search web", "find sources"),
        ),
        "encyclopedia": _phrase_score(
            text,
            ("who was", "who is", "what is", "history of", "biography", "definition", "explain", "origin of"),
        ),
        "verify": _phrase_score(
            text,
            ("verify", "is this true", "fact check", "sources", "citation", "citations", "reliable", "evidence", "check citations"),
        ),
        "review": _phrase_score(
            text,
            ("review report", "rate report", "audit report", "citation audit", "source quality", "improve research", "make it 9.5", "final reviewer", "quality check", "master review", "fact check report"),
        ),
        "image": _phrase_score(
            text,
            ("image", "images", "photo", "picture", "visual", "screenshot", "illustration"),
        ),
        "animation": _phrase_score(
            text,
            ("manim", "/manim", "animation", "animated video", "math animation", "make a video", "generate video", "visualize"),
        ),
        "no_tool": no_tool,
    }
    if intents["weather"]:
        intents["web_search"] *= 0.4
    if intents["web_search"]:
        intents["encyclopedia"] *= 0.5
    if no_tool:
        for key in ("weather", "time", "web_search", "encyclopedia", "verify", "review", "image", "animation"):
            intents[key] *= 0.25
    return {key: round(value, 3) for key, value in intents.items() if value > 0}


def score_tool_for_query(
    tool: str | dict[str, Any],
    query: str,
    config: BridgeConfig | None = None,
    *,
    force_for_keywords: bool = True,
    default_search_provider: str = "tavily",
) -> float:
    """
    Score how relevant a tool is for a given query.
    Returns a score 0.0-10.0 where higher is more relevant.
    """
    description = ""
    if isinstance(tool, dict):
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        tool_name = str(function.get("name") or "")
        description = str(function.get("description") or "")
    else:
        tool_name = tool

    if tool_name not in TOOL_KEYWORDS:
        return 0.0
    
    tool_info = TOOL_KEYWORDS[tool_name]
    keywords = tool_info["keywords"]
    weight = tool_info["weight"]
    
    query_lower = query.lower()
    description_lower = description.lower()
    intents = classify_query_intent(query)
    score = 0.0

    intent_weights = {
        "datetime_now": {"time": 7.0},
        "weather_current": {"weather": 7.0},
        "wikipedia_search": {"encyclopedia": 5.5},
        "wikipedia_page": {"encyclopedia": 3.5},
        "tavily_search": {"web_search": 5.0, "verify": 2.0},
        "serpapi_search": {"web_search": 4.7, "verify": 1.5},
        "source_research": {"verify": 5.5, "web_search": 3.0},
        "verify_sources": {"verify": 6.0},
        "master_review": {"review": 7.0, "verify": 2.5},
        "image_research": {"image": 7.0},
        "manim_render": {"animation": 8.0, "image": 1.0},
    }
    for intent, weight in intent_weights.get(tool_name, {}).items():
        score += intents.get(intent, 0.0) * weight
    
    # Exact phrase matches get highest score
    for keyword in keywords:
        if keyword in query_lower:
            score += weight * 1.5
    
    # Individual word matches
    query_words = set(query_lower.split())
    for keyword in keywords:
        keyword_words = set(keyword.split())
        matching_words = len(query_words & keyword_words)
        if matching_words > 0:
            score += weight * (matching_words / len(keyword_words))

    # Description matches add a small nudge so richer schemas improve routing.
    for token in query_words:
        if len(token) >= 4 and token in description_lower:
            score += 0.5

    if force_for_keywords:
        score += _forced_tool_bonus(tool_name, query_lower)

    if tool_name == f"{default_search_provider}_search":
        score += 0.8
    if tool_name in {"serpapi_search", "tavily_search"} and intents.get("encyclopedia") and not intents.get("web_search"):
        score -= 1.5
    if tool_name.startswith("wikipedia_") and intents.get("web_search"):
        score -= 2.0
    if intents.get("no_tool"):
        score -= intents["no_tool"] * 6.0
    
    return max(0.0, min(score, 10.0))


def _phrase_score(text: str, phrases: tuple[str, ...]) -> float:
    score = 0.0
    for phrase in phrases:
        if " " in phrase.strip():
            if phrase in text:
                score += 1.0
        elif re.search(rf"\b{re.escape(phrase)}\b", text):
            score += 0.8
    return min(score, 1.5)


def _forced_tool_bonus(tool_name: str, query_lower: str) -> float:
    creative_or_code = (
        "write a poem",
        "story",
        "creative",
        "code",
        "debug",
        "refactor",
        "implement",
    )
    if any(phrase in query_lower for phrase in creative_or_code):
        return 0.0
    if tool_name == "manim_render" and any(
        phrase in query_lower
        for phrase in ("manim", "/manim", "animation", "animated video", "make a video", "generate video")
    ):
        return 5.0
    if tool_name == "weather_current" and any(
        word in query_lower for word in ("weather", "temperature", "rain", "wind", "humid")
    ):
        return 4.0
    if tool_name == "datetime_now" and any(
        phrase in query_lower
        for phrase in ("current time", "what time", "today's date", "date today", "current date", "timezone")
    ):
        return 4.0
    if tool_name in {"tavily_search", "serpapi_search"} and any(
        word in query_lower for word in ("latest", "news", "current", "recent", "price", "web")
    ):
        return 3.0
    if tool_name == "source_research" and any(
        word in query_lower
        for word in ("verify", "source", "sources", "citation", "accurate", "research", "deep research", "current", "latest")
    ):
        return 4.0
    if tool_name == "image_research" and any(
        word in query_lower for word in ("image", "images", "photo", "picture", "visual", "markdown")
    ):
        return 4.0
    if tool_name == "verify_sources" and any(
        word in query_lower for word in ("verify", "validate", "link", "links", "url", "citation")
    ):
        return 4.0
    if tool_name == "master_review" and any(
        phrase in query_lower
        for phrase in (
            "review report",
            "rate report",
            "audit report",
            "check evidence",
            "citation audit",
            "source quality",
            "improve research",
            "make it 9.5",
            "final reviewer",
            "quality check",
            "master review",
            "fact check report",
        )
    ):
        return 4.5
    if tool_name == "wikipedia_search" and any(
        phrase in query_lower for phrase in ("who is", "what is", "history", "definition")
    ):
        return 3.0
    return 0.0


def select_relevant_tools(
    tools: list[dict[str, Any]],
    query: str,
    max_tools: int = 5,
    min_score: float = 0.5,
    force_for_keywords: bool = True,
    default_search_provider: str = "tavily",
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Select the most relevant tools for a query.
    
    Returns:
        Tuple of (filtered_tools, scores_dict)
    """
    if not tools:
        return [], {}
    
    # Score all tools
    scores = {}
    for tool in tools:
        function = tool.get("function", {})
        tool_name = function.get("name", "")
        score = score_tool_for_query(
            tool,
            query,
            force_for_keywords=force_for_keywords,
            default_search_provider=default_search_provider,
        )
        scores[tool_name] = score
    
    # Filter and sort
    relevant = [
        tool for tool in tools
        if scores.get(tool.get("function", {}).get("name", ""), 0.0) >= min_score
    ]

    forced_names = _forced_tool_names(query, default_search_provider) if force_for_keywords else set()
    if forced_names:
        forced_tools = [
            tool
            for tool in tools
            if (tool.get("function", {}).get("name", "") in forced_names)
        ]
        if forced_tools:
            forced_tools.sort(
                key=lambda t: _tool_sort_key(t, scores, default_search_provider),
                reverse=True,
            )
            return forced_tools[:max_tools], scores
    
    # Sort by score descending and limit
    relevant.sort(key=lambda t: _tool_sort_key(t, scores, default_search_provider), reverse=True)
    
    if not relevant and min_score <= 0:
        relevant = sorted(
            tools,
            key=lambda t: _tool_sort_key(t, scores, default_search_provider),
            reverse=True,
        )[: min(max_tools, 2)]

    return relevant[:max_tools], scores


def _tool_sort_key(tool: dict[str, Any], scores: dict[str, float], default_search_provider: str) -> tuple[float, int]:
    name = (tool.get("function") or {}).get("name", "")
    priority = {
        "weather_current": 90,
        "datetime_now": 85,
        "verify_sources": 80,
        "master_review": 78,
        "source_research": 75,
        f"{default_search_provider}_search": 70,
        "tavily_search": 65,
        "serpapi_search": 60,
        "image_download": 59,
        "image_research": 58,
        "wikipedia_search": 55,
        "wikipedia_page": 45,
    }.get(name, 0)
    return (scores.get(name, 0.0), priority)


def _forced_tool_names(query: str, default_search_provider: str = "tavily") -> set[str]:
    query_lower = query.lower()
    creative_or_code = (
        "write a poem",
        "story",
        "creative",
        "code",
        "debug",
        "refactor",
        "implement",
    )
    if any(phrase in query_lower for phrase in creative_or_code):
        return set()
    if any(word in query_lower for word in ("weather", "temperature", "rain", "wind", "humid")):
        return {"weather_current"}
    if any(word in query_lower for word in ("latest", "news", "current", "recent", "price", "web")):
        preferred = f"{default_search_provider}_search"
        return {preferred, "source_research"}
    if any(
        phrase in query_lower
        for phrase in (
            "deep research",
            "deep_research",
            "deep-research",
            "/deep",
            "source research",
            "research report",
            "source verification",
        )
    ):
        return {"source_research", "verify_sources"}
    if any(
        phrase in query_lower
        for phrase in ("current time", "what time", "time in", "today's date", "date today", "current date", "timezone")
    ):
        return {"datetime_now"}
    if any(word in query_lower for word in ("image", "images", "photo", "picture", "visual")):
        return {"image_research", "image_download"}
    if any(
        phrase in query_lower
        for phrase in (
            "review report",
            "rate report",
            "audit report",
            "check evidence",
            "citation audit",
            "source quality",
            "improve research",
            "make it 9.5",
            "final reviewer",
            "quality check",
            "master review",
            "fact check report",
        )
    ):
        return {"master_review"}
    if any(phrase in query_lower for phrase in ("verify", "is this true", "fact check", "citation", "sources")):
        return {"verify_sources", "source_research"}
    if any(phrase in query_lower for phrase in ("who is", "who was", "what is", "history", "definition")):
        return {"wikipedia_search"}
    return set()
