from __future__ import annotations

from pathlib import Path

import pytest

from llama_bridge.config import (
    BridgeConfig,
    CodexConfig,
    CopilotCliConfig,
    MasterReviewConfig,
    ModelAlias,
    PiConfig,
    ProviderConfig,
    ToolConfig,
)


@pytest.fixture
def config(tmp_path: Path) -> BridgeConfig:
    return BridgeConfig(
        server=None,  # type: ignore[arg-type]
        providers={
            "local": ProviderConfig(
                name="local",
                type="openai_compatible",
                base_url="http://127.0.0.1:1/v1",
                api_key="test",
                default_model="test-model",
            )
        },
        anthropic_models={"sonnet": ModelAlias(alias="sonnet", provider="local", model="test-model")},
        pi=PiConfig(provider="local", model="test-model"),
        codex=CodexConfig(provider="local", model="test-model"),
        copilot_cli=CopilotCliConfig(provider="local", model="test-model"),
        vs_copilot_models=[],
        tools=ToolConfig(include=["master_review"], cache_enabled=False),
        master_review=MasterReviewConfig(),
        source_path=tmp_path / "env.yml",
    )

