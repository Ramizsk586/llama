---
name: llama-bridge-tools
description: Use when Claude Code needs llama bridge MCP tools for deep research, current web search, source verification, image research, weather, Wikipedia, or date/time lookups.
---

# Llama Bridge Tools

Use the `llama_bridge_tools` MCP server when a task needs current information,
source verification, image candidates, weather, Wikipedia, or local date/time
lookups.

Prefer the highest-level bridge tool that fits the task:

- For `/deep` and deep research, only call tools whose names contain `deep`.
  Start with `deep` or `deep_plan_agent`; do not use normal search, source,
  image, Wikipedia, weather, or time tools for `/deep`. Use the staged flow:
  `deep_plan_agent`, then
  `deep_collect_agent` for single-agent collection calls, then each listed
  collection agent separately, then `deep_collect_agent` with
  `subagent_briefs`, then use the returned `temp_files`, then `deep_review_agent` for
  single-agent verification calls, then each listed verifier separately, then
  `deep_review_agent` with `verification_briefs`, then `deep_master_review_agent`
  to get 8 master-review calls, optionally `deep_image_agent` to download
  report-ready image files, then each listed `deep_master_*` call separately,
  then `deep_master_review_agent` with `master_review_briefs`, then write the
  final `report.md` using any returned image `local_path` values.
- `source_research` for cited factual research and evidence gathering.
- `image_research` for compact sourced image candidates.
- `tavily_search` or `serpapi_search` for current web results.
- `wikipedia_search` and `wikipedia_page` for encyclopedia context.
- `weather_current` for live weather.
- `datetime_now` for current time or timezone questions.

The MCP server calls the local llama bridge HTTP tool endpoints, so the llama
server must be running for tool calls to succeed.
