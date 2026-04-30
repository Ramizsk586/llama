from __future__ import annotations

import json
from pathlib import Path

from llama_bridge.cli import _cmd_master_review


def _write_config(path: Path) -> None:
    path.write_text(
        """
server:
  host: 127.0.0.1
  port: 8089
  auth_token: change-me
providers:
  local:
    type: openai_compatible
    base_url: http://127.0.0.1:1/v1
    api_key: test
    default_model: test-model
anthropic_models:
  sonnet:
    provider: local
    model: test-model
tools:
  enabled: true
  include:
    - master_review
master_review:
  enabled: true
  groq:
    enabled: true
    api_keys:
      - test-groq-key-one
      - ${GROQ_API_KEY_2}
""",
        encoding="utf-8",
    )


def test_cli_check_keys_masks_values(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    config_path = tmp_path / "env.yml"
    _write_config(config_path)
    _cmd_master_review(
        config_path,
        report_path=None,
        mode=None,
        use_stdin=False,
        check_keys=True,
    )
    output = capsys.readouterr().out
    assert "groq_key_1" in output
    assert "test-groq-key-one" not in output


def test_cli_stdin_accepts_json(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    config_path = tmp_path / "env.yml"
    _write_config(config_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        type("Stdin", (), {"read": lambda self: json.dumps({"data": {"answer": "The the draft alleges manipulation.", "sources": []}})})(),
    )
    _cmd_master_review(
        config_path,
        report_path=None,
        mode="fast",
        use_stdin=True,
        check_keys=False,
    )
    output = capsys.readouterr().out
    assert "Master Review" in output
    assert (tmp_path / "master_review_instructions.txt").exists()

