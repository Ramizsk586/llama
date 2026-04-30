from __future__ import annotations

import asyncio

import pytest

from llama_bridge.config import MasterReviewConfig
from llama_bridge.master import (
    GroqKeyRotator,
    MasterReviewer,
    extract_json_object,
    local_citation_checks,
    local_loaded_language_checks,
    local_source_quality_checks,
    normalize_deep_research_result,
)


def test_normalize_deep_research_result_full_shape() -> None:
    result = {
        "data": {
            "query": "question",
            "answer": "Draft answer",
            "sources": [{"url": "https://example.gov/report", "title": "Report"}],
            "key_findings": [{"claim": "Claim"}],
            "contradictions": [{"a": "x", "b": "y"}],
            "uncertainties": ["gap"],
            "research_trace": [{"step": "search"}],
        }
    }
    normalized = normalize_deep_research_result(result)
    assert normalized.query == "question"
    assert normalized.draft_answer == "Draft answer"
    assert normalized.sources[0]["url"] == "https://example.gov/report"
    assert normalized.key_findings[0]["claim"] == "Claim"


def test_normalize_deep_research_result_missing_fields() -> None:
    normalized = normalize_deep_research_result({"data": {}})
    assert normalized.query == ""
    assert normalized.sources == []
    assert normalized.key_findings == []


@pytest.mark.asyncio
async def test_groq_key_rotator_rotates_across_three_keys() -> None:
    rotator = GroqKeyRotator(["k1", "k2", "k3"], 20)
    labels = [(await rotator.acquire_key())[1] for _ in range(3)]
    assert labels == ["groq_key_1", "groq_key_2", "groq_key_3"]


@pytest.mark.asyncio
async def test_groq_key_rotator_cools_down_429_key() -> None:
    rotator = GroqKeyRotator(["k1", "k2"], 20, cooldown_seconds_after_429=60)
    key, label = await rotator.acquire_key()
    assert label == "groq_key_1"
    rotator.mark_rate_limited(key)
    _, next_label = await rotator.acquire_key()
    assert next_label == "groq_key_2"


@pytest.mark.asyncio
async def test_groq_key_rotator_disables_auth_error_key() -> None:
    rotator = GroqKeyRotator(["k1", "k2"], 20)
    key, _ = await rotator.acquire_key()
    rotator.mark_auth_error(key)
    _, label = await rotator.acquire_key()
    assert label == "groq_key_2"


@pytest.mark.asyncio
async def test_groq_key_rotator_respects_rpm() -> None:
    rotator = GroqKeyRotator(["k1"], 1)
    await rotator.acquire_key()
    with pytest.raises(RuntimeError):
        await rotator.acquire_key()


def test_extract_json_object_clean_json() -> None:
    assert extract_json_object('{"ok": true, "value": 1}')["value"] == 1


def test_extract_json_object_embedded_json() -> None:
    assert extract_json_object('prefix {"ok": true, "value": 2} suffix')["value"] == 2


def test_extract_json_object_unrecoverable_text() -> None:
    parsed = extract_json_object("no object here")
    assert parsed["ok"] is False
    assert parsed["error"]["type"] == "JSONParseError"


def test_loaded_language_checker_catches_terms() -> None:
    findings = local_loaded_language_checks("The report alleges manipulation and a crackdown.")
    assert {item["term"].lower() for item in findings} >= {"manipulation", "crackdown"}


def test_source_quality_checker_marks_social_media_low() -> None:
    assessments = local_source_quality_checks([{"url": "https://twitter.com/a/status/1", "title": "post"}])
    assert assessments[0]["reliability_score"] <= 2
    assert assessments[0]["source_type"] == "social"


def test_citation_checker_catches_uncited_claim() -> None:
    text = "In 2025, the agency reported a 42% increase in cases across three states without enough context."
    issues = local_citation_checks(text, [])
    assert issues


@pytest.mark.asyncio
async def test_failed_sub_agent_does_not_stop_master_review(monkeypatch: pytest.MonkeyPatch) -> None:
    reviewer = MasterReviewer(MasterReviewConfig())

    async def broken_run_agent(spec, review_input, mode):  # type: ignore[no-untyped-def]
        if spec["name"] == "spelling_grammar_agent":
            raise RuntimeError("boom")
        return spec["fallback"](review_input)

    monkeypatch.setattr(reviewer, "_run_agent", broken_run_agent)
    try:
        review = await reviewer.review_deep_research({"data": {"answer": "The the draft alleges manipulation.", "sources": []}})
    finally:
        await reviewer.aclose()
    assert review["ok"] is True
    assert review["data"]["final_llm_instructions"]
    assert any("fallback" in warning.lower() for warning in review["warnings"])


@pytest.mark.asyncio
async def test_debate_round_merges_disagreements() -> None:
    reviewer = MasterReviewer(MasterReviewConfig())
    try:
        review = await reviewer.review_deep_research(
            {"data": {"answer": "The the report alleges manipulation in 2025 without citation.", "sources": []}},
            mode="fast",
        )
    finally:
        await reviewer.aclose()
    assert review["ok"] is True
    assert review["data"]["debate_results"]
    assert review["data"]["final_llm_instructions"]

