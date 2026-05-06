---
name: llama-bridge-tools
description: Use when Claude Code needs llama bridge MCP tools for staged deep research, current web search, source verification, image research, weather, Wikipedia, or date/time lookups.
---

# Llama Bridge Tools

Use the `llama_bridge_tools` MCP server when a task needs current information,
source verification, image candidates, weather, Wikipedia, or local date/time
lookups.

Prefer the highest-level bridge tool that fits the task:

- For deep research, use the staged flow: `deep_plan_agent`, then
  `deep_collect_agent`, then save `temp/ad.md`, then `deep_review_agent`,
  then write the final `report.md`.
  Keep the collection stage fixed at 2 Tavily agents, 2 SerpAPI agents, and 3 Wikipedia agents.
- `source_research` for cited factual research and evidence gathering.
- `image_research` for compact sourced image candidates.
- `tavily_search` or `serpapi_search` for current web results.
- `wikipedia_search` and `wikipedia_page` for encyclopedia context.
- `weather_current` for live weather.
- `datetime_now` for current time or timezone questions.

The MCP server calls the local llama bridge HTTP tool endpoints, so the llama
server must be running for tool calls to succeed.
