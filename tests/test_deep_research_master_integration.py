from __future__ import annotations

import pytest

from llama_bridge.config import BridgeConfig, MasterReviewConfig
from llama_bridge.tools import ToolRegistry


@pytest.mark.asyncio
async def test_master_review_direct_tool_works(config: BridgeConfig) -> None:
    config.tools.include = ["master_review"]
    registry = ToolRegistry(config)
    try:
        result = await registry.call_structured(
            "master_review",
            {"research_result": {"data": {"answer": "The the draft alleges manipulation.", "sources": []}}},
        )
    finally:
        await registry.aclose()
    assert result["ok"] is True
    assert result["data"]["ok"] is True
    assert result["data"]["data"]["final_llm_instructions"]


@pytest.mark.asyncio
async def test_source_research_attaches_master_review(monkeypatch: pytest.MonkeyPatch, config: BridgeConfig) -> None:
    config.tools.include = ["source_research"]
    config.master_review = MasterReviewConfig()
    registry = ToolRegistry(config)

    async def fake_tavily(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"results": [{"url": "https://example.gov/report", "title": "Official report", "snippet": "Evidence"}], "images": []}

    async def fake_verify(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"verdict": "supported", "verified_sources": [], "weak_or_unreachable_sources": [], "verified_count": 0}

    monkeypatch.setattr(registry, "_research_tavily", fake_tavily)
    monkeypatch.setattr(registry, "_verify_urls_parallel", fake_verify)
    config.tools.tavily.enabled = True
    config.tools.tavily.api_key = "test-key"
    config.tools.serpapi.enabled = False
    try:
        result = await registry.call_structured("source_research", {"query": "test topic"})
    finally:
        await registry.aclose()
    assert result["ok"] is True
    assert "master_review" in result["data"]
    assert "final_llm_instructions" in result["data"]


@pytest.mark.asyncio
async def test_groq_unavailable_uses_fallback(config: BridgeConfig) -> None:
    config.master_review.groq.api_keys = []
    registry = ToolRegistry(config)
    try:
        result = await registry.call_structured(
            "master_review",
            {"research_result": {"data": {"answer": "The the draft alleges manipulation.", "sources": []}}},
        )
    finally:
        await registry.aclose()
    assert result["ok"] is True
    assert result["data"]["metadata"]["fallback_used"] is True

