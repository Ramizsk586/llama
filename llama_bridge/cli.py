from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Any
from pathlib import Path

import httpx

from .config import (
    DEFAULT_API_SETTINGS_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXAMPLE_CONFIG_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_NGROK_LOG_PATH,
    DEFAULT_NGROK_PID_PATH,
    DEFAULT_PID_PATH,
    ensure_default_dirs,
    merge_missing_config_fields,
    load_config,
    codex_model_error,
    copilot_cli_model_error,
    opencode_model_error,
    pi_model_error,
    resolve_codex_model,
    resolve_copilot_cli_model,
    resolve_opencode_model,
    resolve_pi_model,
    write_claude_api_settings,
    write_config_data,
    write_default_config,
)
from .llamafetch import print_llamafetch
from .master import MasterReviewer
from .mcp_tools import main as mcp_tools_main
from .tools import ToolRegistry, classify_query_intent, select_relevant_tools


PYTHON_REQUIREMENTS = {
    "fastapi": "fastapi>=0.115.0",
    "httpx": "httpx>=0.27.0",
    "yaml": "pyyaml>=6.0.2",
    "ruamel.yaml": "ruamel.yaml>=0.18.0",
    "uvicorn": "uvicorn>=0.32.0",
}


@dataclass(slots=True)
class SetupResult:
    config_path: Path
    api_settings_path: Path
    installed_python: list[str]
    missing_python: list[str]
    notes: list[str]


@dataclass(slots=True)
class ApiStatusResult:
    alias: str
    provider: str
    model: str
    ok: bool
    status: str
    detail: str


@dataclass(slots=True)
class CliTarget:
    name: str
    display_name: str
    launcher_command: str
    finder: Callable[[], str | None]
    install_method: str
    package: str | None = None
    uninstall_hint: str | None = None


class SetupCanceled(SystemExit):
    def __init__(self, message: str = "Setup canceled.") -> None:
        super().__init__(0)
        self.message = message


DEFAULT_START_IDLE_TIMEOUT_SECONDS = 180


def _configured_idle_timeout_seconds(config) -> int:
    server = getattr(config, "server", None)
    value = getattr(server, "idle_timeout_seconds", DEFAULT_START_IDLE_TIMEOUT_SECONDS)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return DEFAULT_START_IDLE_TIMEOUT_SECONDS


def _idle_timeout_note(client_name: str, idle_timeout_seconds: int) -> str:
    if idle_timeout_seconds <= 0:
        return f"llama server will stay running after {client_name} closes; run `llama stop` to stop it"
    duration = _format_idle_duration(idle_timeout_seconds)
    return f"llama server will stop {duration} after {client_name} closes with no requests"


def _format_idle_duration(idle_timeout_seconds: int) -> str:
    if idle_timeout_seconds % 60 == 0:
        minutes = idle_timeout_seconds // 60
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit}"
    return f"{idle_timeout_seconds} seconds"


ENDPOINT_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Health",
        [
            ("GET/HEAD", "/", "basic service probe"),
            ("GET", "/health", "bridge health status"),
        ],
    ),
    (
        "OpenAI and LM Studio compatible",
        [
            ("GET", "/v1/models", "list configured aliases and upstream model ids"),
            ("GET", "/v1/models/{model}", "show one configured model"),
            ("POST", "/v1/chat/completions", "chat completions, tools, streaming"),
            ("POST", "/v1/completions", "legacy text completions"),
            ("POST", "/v1/responses", "OpenAI Responses API for Codex/Copilot"),
            ("POST", "/v1/embeddings", "OpenAI-compatible embeddings"),
            ("POST", "/v1/moderations", "safe default moderation response"),
        ],
    ),
    (
        "OpenAI stateful and media",
        [
            ("GET/POST", "/v1/assistants", "Assistants compatibility"),
            ("GET/POST/DELETE", "/v1/assistants/{id}", "retrieve, update, or delete assistant"),
            ("GET/POST", "/v1/threads", "Threads compatibility"),
            ("GET/POST/DELETE", "/v1/threads/{id}", "retrieve, update, or delete thread"),
            ("GET/POST", "/v1/threads/{id}/messages", "thread messages"),
            ("GET", "/v1/threads/{id}/messages/{message_id}", "retrieve thread message"),
            ("GET/POST", "/v1/threads/{id}/runs", "thread runs"),
            ("GET/POST", "/v1/threads/{id}/runs/{run_id}", "retrieve or update run"),
            ("POST", "/v1/threads/{id}/runs/{run_id}/cancel", "cancel run"),
            ("POST", "/v1/threads/{id}/runs/{run_id}/submit_tool_outputs", "submit tool outputs"),
            ("GET", "/v1/threads/{id}/runs/{run_id}/steps", "list run steps"),
            ("GET/POST", "/v1/fine_tuning/jobs", "fine-tuning job compatibility"),
            ("GET", "/v1/fine_tuning/jobs/{id}", "retrieve fine-tuning job"),
            ("POST", "/v1/fine_tuning/jobs/{id}/cancel", "cancel fine-tuning job"),
            ("GET", "/v1/fine_tuning/jobs/{id}/events", "list fine-tuning events"),
            ("POST", "/v1/images/generations", "unsupported image generation probe"),
            ("POST", "/v1/audio/transcriptions", "unsupported audio transcription probe"),
            ("POST", "/v1/audio/speech", "unsupported text-to-speech probe"),
        ],
    ),
    (
        "Anthropic compatible",
        [
            ("POST", "/v1/messages", "Claude Code style messages"),
            ("POST", "/v1/messages/batches", "Message Batches compatibility"),
            ("GET", "/v1/messages/batches", "list in-memory message batches"),
            ("GET", "/v1/messages/batches/{id}", "retrieve a message batch"),
            ("POST", "/v1/messages/batches/{id}/cancel", "cancel a message batch"),
            ("GET", "/v1/messages/batches/{id}/results", "stream batch result JSONL"),
            ("POST", "/v1/messages/count_tokens", "token estimate for Claude traffic"),
            ("POST", "/v1/complete", "legacy Text Completions compatibility"),
            ("GET/POST", "/v1/files", "beta Files compatibility"),
            ("GET/DELETE", "/v1/files/{id}", "retrieve or delete a file record"),
            ("GET", "/v1/files/{id}/content", "download uploaded file bytes"),
            ("GET/POST", "/v1/skills", "beta Skills compatibility"),
            ("GET/POST/DELETE", "/v1/skills/{id}", "retrieve, update, or delete a skill"),
            ("GET/POST/DELETE", "/v1/organizations/{path}", "Admin API compatibility"),
        ],
    ),
    (
        "Ollama generation",
        [
            ("POST", "/api/generate", "prompt completion, streaming by default"),
            ("POST", "/api/chat", "chat messages, tools, streaming by default"),
            ("POST", "/api/embed", "current Ollama embedding response shape"),
            ("POST", "/api/embeddings", "legacy Ollama embedding response shape"),
        ],
    ),
    (
        "Cohere and Gemini compatible",
        [
            ("POST", "/v1/chat", "Cohere-style chat"),
            ("POST", "/v1/embed", "Cohere-style embeddings"),
            ("POST", "/v1beta/models/{model}:generateContent", "Gemini content generation"),
            ("POST", "/v1beta/models/{model}:streamGenerateContent", "Gemini streaming generation"),
            ("POST", "/v1/models/{model}:generateContent", "Gemini v1 content generation"),
            ("POST", "/v1/models/{model}:streamGenerateContent", "Gemini v1 streaming generation"),
        ],
    ),
    (
        "Ollama model management",
        [
            ("GET", "/api/tags", "list available bridge models"),
            ("GET", "/api/ps", "list running models"),
            ("POST", "/api/show", "show model metadata"),
            ("POST", "/v1/api/chat/api/show", "show model metadata for nested Ollama probes"),
            ("POST", "/api/create", "compatibility shim"),
            ("POST", "/api/pull", "compatibility shim"),
            ("POST", "/api/push", "compatibility shim"),
            ("POST", "/api/copy", "compatibility shim"),
            ("DELETE", "/api/delete", "compatibility shim"),
        ],
    ),
    (
        "Ollama system",
        [
            ("GET", "/api/version", "bridge version in Ollama shape"),
            ("HEAD", "/api/blobs/{digest}", "blob probe compatibility"),
            ("POST", "/api/blobs/{digest}", "blob upload compatibility"),
            ("POST", "/api/web_search", "proxy to configured Ollama web search"),
            ("POST", "/api/web_fetch", "proxy to configured Ollama web fetch"),
            ("POST", "/api/experimental/web_search", "legacy web search path"),
            ("POST", "/api/experimental/web_fetch", "legacy web fetch path"),
        ],
    ),
    (
        "Bridge tools",
        [
            ("GET", "/v1/tools", "list bridge-provided tool schemas"),
            ("POST", "/v1/tools/call", "call a bridge-provided tool by name"),
            ("POST", "/v1/tools/{name}", "call one bridge-provided tool"),
            ("GET", "/api/tools", "Ollama-style bridge tool listing"),
            ("POST", "/api/tools/{name}", "Ollama-style bridge tool call"),
        ],
    ),
]


def _enable_terminal_style() -> None:
    if os.name != "nt" or not sys.stdout.isatty():
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _color_enabled() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _style(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _title(text: str) -> None:
    print(_style(f"\n== {text} ==", "1;36"))


def _print_note(message: str) -> None:
    print(f"{_style('>', '36')} {message}")


def _print_state(label: str, message: str, color: str = "36") -> None:
    print(f"{_style(f'[{label}]', color)} {message}")


def _kv_rows(rows: list[tuple[str, str | int]]) -> None:
    width = max((len(label) for label, _ in rows), default=0)
    for label, value in rows:
        print(f"  {_style(label.ljust(width), '2')}: {value}")


def _status_label(running: bool) -> str:
    if running:
        return _style("running", "32")
    return _style("stopped", "33")


def _format_log_line(line: str) -> str:
    if not _color_enabled():
        return line

    if line.startswith("ERROR:"):
        return _style(line, "31")
    if line.startswith("WARNING:"):
        return _style(line, "33")
    if line.startswith("INFO:"):
        return _color_info_log_line(line)
    if "Traceback" in line or "Exception" in line:
        return _style(line, "31")
    if line.lstrip().startswith("File "):
        return _style(line, "90")
    return line


def _format_dev_log_line(line: str) -> str:
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return line

    event = str(entry.get("event", "event"))
    timestamp = str(entry.get("time", ""))
    payload = entry.get("payload")
    color = "36"
    if "error" in event:
        color = "31"
    elif "response" in event:
        color = "32"
    elif "request" in event:
        color = "34"

    header = f"{_style(timestamp, '2')} {_style(event, color)}"
    pretty = json.dumps(payload, indent=2, ensure_ascii=True)
    return f"{header}\n{pretty}\n"


def _compact_dev_log_payload(value, *, depth: int = 0, key: str | None = None):
    if depth >= 8:
        return "<nested value omitted>"
    if isinstance(value, str):
        limit = 180 if key == "description" else 350
        if len(value) <= limit:
            return value
        return f"{value[:limit]}... <truncated {len(value) - limit} chars>"
    if isinstance(value, list):
        if key == "tools":
            compacted = [_compact_dev_log_tool(item) for item in value[:20]]
        else:
            compacted = [_compact_dev_log_payload(item, depth=depth + 1) for item in value[:8]]
        if len(value) > len(compacted):
            compacted.append(f"<{len(value) - len(compacted)} more items>")
        return compacted
    if isinstance(value, dict):
        compacted = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 40:
                compacted["<more>"] = f"{len(value) - 40} more keys"
                break
            compacted[key] = _compact_dev_log_payload(item, depth=depth + 1, key=str(key))
        return compacted
    return value


def _compact_dev_log_tool(tool) -> dict[str, str]:
    if not isinstance(tool, dict):
        return {"tool": _compact_dev_log_payload(tool)}
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return {
        "type": str(tool.get("type") or "function"),
        "name": str(function.get("name") or tool.get("name") or "tool"),
        "description": _compact_dev_log_payload(
            function.get("description") or tool.get("description") or "",
            key="description",
        ),
    }


def _color_info_log_line(line: str) -> str:
    line = line.replace("INFO:", _style("INFO:", "34"), 1)
    line = re.sub(r'"(GET|POST|HEAD|PUT|PATCH|DELETE) ([^"]+)"', _color_http_request, line)
    line = re.sub(r"(?<=\s)([1-5][0-9][0-9])(?=\s+[A-Z])", _color_http_status, line)
    return line


def _color_http_request(match: re.Match[str]) -> str:
    method = _style(match.group(1), "36")
    path = _style(match.group(2), "37")
    return f'"{method} {path}"'


def _color_http_status(match: re.Match[str]) -> str:
    status = match.group(1)
    if status.startswith("2") or status.startswith("3"):
        return _style(status, "32")
    if status.startswith("4"):
        return _style(status, "33")
    return _style(status, "31")



def _pause_with_spinner(message: str, seconds: float) -> None:
    if not sys.stdout.isatty():
        time.sleep(seconds)
        return
    frames = "-\\|/"
    end_at = time.monotonic() + seconds
    index = 0
    while time.monotonic() < end_at:
        sys.stdout.write(f"\r{_style(frames[index % len(frames)], '36')} {message}")
        sys.stdout.flush()
        index += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
    sys.stdout.flush()


def _run_with_spinner(message: str, action: Callable[[], int]) -> int:
    if not sys.stdout.isatty():
        return action()
    sys.stdout.write(f"{_style('-', '36')} {message}")
    sys.stdout.flush()
    result = action()
    sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
    sys.stdout.flush()
    return result


def _default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def _default_peer_path(default_path: Path, config_path: Path | None = None) -> Path:
    if config_path is not None:
        return config_path.parent / default_path.name
    return default_path


def _arg_path(
    value: str | Path | None,
    default_path: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    if value is None:
        if default_path is None:
            return _default_config_path()
        return _default_peer_path(default_path, config_path)
    path = Path(value)
    if default_path is not None and path == default_path:
        return _default_peer_path(default_path, config_path)
    if default_path is None and path == DEFAULT_CONFIG_PATH:
        return _default_config_path()
    return path


def main() -> None:
    _enable_terminal_style()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.endpoints or args.command == "endpoints":
            _cmd_endpoints()
            return

        if args.tui:
            from . import tui as tui_module
            config_path = getattr(args, 'config', None)
            tui_module.run_main_tui(_arg_path(config_path) if config_path else None)
            return

        if args.command is None:
            _cmd_info(_default_config_path())
            return

        if args.command == "info":
            _cmd_info(_arg_path(args.config))
            return
        if args.command == "init":
            _cmd_init(_arg_path(args.config), args.force)
            return
        if args.command == "configure":
            _cmd_configure(_arg_path(args.config))
            return
        if args.command == "setup":
            _cmd_setup(_arg_path(args.config), install_system=not args.no_install_system)
            return
        if args.command == "serve":
            config_path = _arg_path(args.config)
            _ensure_setup(config_path)
            if args.foreground:
                _cmd_serve_foreground(
                    config_path,
                    args.log_file,
                    args.idle_timeout,
                    Path(args.idle_after_file) if args.idle_after_file else None,
                )
                return
            _cmd_start(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                args.idle_timeout,
            )
            return
        if args.command == "start":
            config_path = _arg_path(args.config)
            _ensure_setup(config_path)
            config = load_config(config_path)
            pid_path = _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path)
            log_path = _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path)
            _cmd_start(
                config_path,
                pid_path,
                log_path,
                0 if args.forever else _configured_idle_timeout_seconds(config),
            )
            if args.online:
                _cmd_ngrok_start(config_path)
            return
        if args.command == "stop":
            config_path = _default_config_path()
            _cmd_stop(_arg_path(args.pid_file, DEFAULT_PID_PATH, config_path))
            return
        if args.command == "restart":
            config_path = _arg_path(args.config) if args.config else _default_config_path()
            _ensure_setup(config_path)
            config = load_config(config_path)
            _cmd_restart(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                0 if args.forever else _configured_idle_timeout_seconds(config),
            )
            return
        if args.command == "logs":
            if args.tui:
                from . import tui as tui_module
                config_path = _arg_path(args.config) if args.config else _default_config_path()
                log_path = _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path)
                tui_module.run_logs_tui(log_path=log_path, dev=args.dev, config_path=config_path)
                return
            config_path = _arg_path(args.config) if args.config else _default_config_path()
            _cmd_logs(
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                args.follow,
                args.clear,
                args.dev,
                config_path,
                args.tail,
                use_active=not args.config and not args.log_file and not args.pid_file,
            )
            return
        if args.command == "status":
            config_path = _arg_path(args.config)
            _cmd_status(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
            )
            return
        if args.command == "usage":
            _cmd_usage(_arg_path(args.config), args.watch, args.json)
            return
        if args.command == "api":
            if args.limits or args.limets or args.api_command == "limits":
                _cmd_api_limits(_arg_path(args.config))
                return
            if args.api_command in {None, "status"}:
                _cmd_api_status(_arg_path(args.config), args.timeout)
                return
            parser.print_help()
            return
        if args.command == "tools":
            config_path = _arg_path(args.config)
            if args.tools_command in {None, "list"}:
                _cmd_tools_list(config_path)
                return
            if args.tools_command == "score":
                _cmd_tools_score(config_path, args.query)
                return
            if args.tools_command == "test":
                _cmd_tools_test(config_path, args.name, args.arguments)
                return
            if args.tools_command == "diagnose":
                _cmd_tools_diagnose(config_path, args.query)
                return
            parser.print_help()
            return
        if args.command == "mcp-tools":
            mcp_tools_main()
            return
        if args.command == "pi":
            config_path = _arg_path(args.config)
            _cmd_pi(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                provider_override=args.provider,
                model_override=args.model,
                pi_args=args.pi_args,
                install_pi=not args.no_install_pi,
                dev=args.dev,
            )
            return
        if args.command == "claude":
            config_path = _arg_path(args.config)
            _cmd_claude(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                args.claude_args,
                install_claude=not args.no_install_claude,
                dev=args.dev,
            )
            return
        if args.command == "codex":
            config_path = _arg_path(args.config)
            _cmd_codex(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                provider_override=args.provider,
                model_override=args.model,
                codex_args=args.codex_args,
                install_codex=not args.no_install_codex,
                dev=args.dev,
            )
            return
        if args.command == "copilot":
            config_path = _arg_path(args.config)
            _cmd_copilot(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                provider_override=args.provider,
                model_override=args.model,
                copilot_args=args.copilot_args,
                install_copilot=not args.no_install_copilot,
                dev=args.dev,
            )
            return
        if args.command == "opencode":
            config_path = _arg_path(args.config)
            _cmd_opencode(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                provider_override=args.provider,
                model_override=args.model,
                opencode_args=args.opencode_args,
                install_opencode=not args.no_install_opencode,
                project_config=args.project_config,
                dev=args.dev,
            )
            return
        if args.command == "poolside":
            config_path = _arg_path(args.config)
            _cmd_poolside(
                config_path,
                _arg_path(None, DEFAULT_PID_PATH, config_path),
                _arg_path(None, DEFAULT_LOG_PATH, config_path),
                args.poolside_args,
                provider_override=args.provider,
                model_override=args.model,
                install_poolside=not args.no_install_poolside,
                dev=args.dev,
            )
            return
        if args.command == "poolside-acp-proxy":
            _cmd_poolside_acp_proxy(args.poolside_acp_args)
            return
        if args.command == "agent":
            _cmd_agent(args)
            return
        if args.command == "cli":
            config_path = _arg_path(args.config)
            _cmd_cli(
                config_path,
                list_only=args.list,
                support_only=args.support,
                remove_target=args.rm,
            )
            return
        if args.command == "bot":
            config_path = _arg_path(getattr(args, "bot_run_config", None) or args.config)
            if args.bot_command in {None, "setup"}:
                _cmd_bot_setup(config_path)
                return
            if args.bot_command == "run":
                from .teligram import run_teligram

                run_teligram(str(config_path), args.workspace)
                return
            if args.bot_command == "status":
                _cmd_telegram_status(config_path)
                return
            if args.bot_command == "send":
                _cmd_bot_send(
                    config_path,
                    " ".join(args.message).strip(),
                    args.chat_id,
                )
                return
            if args.bot_command == "send-ai":
                _cmd_bot_send_ai(config_path, " ".join(args.message).strip(), args.chat_id)
                return
            if args.bot_command == "test-token":
                _cmd_bot_test_token(config_path)
                return
            if args.bot_command == "logs":
                _cmd_bot_logs(config_path)
                return
            parser.print_help()
            return

        if args.command in ("openapi", "api-spec"):
            config_path = _arg_path(args.config)
            if args.openapi_command in {None, "export"}:
                _cmd_openapi_export(config_path, args.output)
                return
            if args.openapi_command == "validate":
                _cmd_openapi_validate(config_path)
                return
            parser.print_help()
            return

        parser.print_help()
    except SetupCanceled as exc:
        _print_state("stop", exc.message, "33")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama",
        description="llama bridge CLI",
        usage="llama [-h] [--endpoints] <command> ...",
    )
    parser.add_argument(
        "--endpoints",
        action="store_true",
        help="show all HTTP endpoints supported by the bridge",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="open the Textual-based Terminal UI (main dashboard)",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    init_cmd = subparsers.add_parser("init", help="write a default config file")
    init_cmd.add_argument("--config")
    init_cmd.add_argument("--force", action="store_true")

    configure_cmd = subparsers.add_parser(
        "configure",
        help="interactive provider and Telegram setup wizard",
    )
    configure_cmd.add_argument("--config")

    setup_cmd = subparsers.add_parser("setup", help="create config and install missing requirements")
    setup_cmd.add_argument("--config")
    setup_cmd.add_argument(
        "--no-install-system",
        action="store_true",
        help="skip installing missing system tools such as Ollama",
    )

    serve_cmd = subparsers.add_parser("serve", help="start the bridge in background")
    serve_cmd.add_argument("--config")
    serve_cmd.add_argument("--pid-file")
    serve_cmd.add_argument("--log-file")
    serve_cmd.add_argument(
        "--foreground",
        action="store_true",
        help="run in the current terminal instead of the background",
    )
    serve_cmd.add_argument("--idle-timeout", type=int, default=0, help=argparse.SUPPRESS)
    serve_cmd.add_argument("--idle-after-file", default=None, help=argparse.SUPPRESS)

    start_cmd = subparsers.add_parser("start", help="run the bridge in background")
    start_cmd.add_argument("--config")
    start_cmd.add_argument("--pid-file")
    start_cmd.add_argument("--log-file")
    start_cmd.add_argument(
        "--forever",
        action="store_true",
        help="disable the configured idle auto-stop",
    )
    start_cmd.add_argument(
        "--online",
        action="store_true",
        help="publish the bridge with ngrok and print the public URL",
    )

    stop_cmd = subparsers.add_parser("stop", help="stop the background bridge")
    stop_cmd.add_argument("--pid-file")

    restart_cmd = subparsers.add_parser("restart", help="restart the background bridge")
    restart_cmd.add_argument("--config")
    restart_cmd.add_argument("--pid-file")
    restart_cmd.add_argument("--log-file")
    restart_cmd.add_argument(
        "--forever",
        action="store_true",
        help="disable the configured idle auto-stop",
    )

    logs_cmd = subparsers.add_parser("logs", help="show bridge logs")
    logs_cmd.add_argument("--config")
    logs_cmd.add_argument("--log-file")
    logs_cmd.add_argument("--pid-file")
    logs_cmd.add_argument(
        "--tui",
        action="store_true",
        help="open the Textual-based Terminal UI for logs",
    )
    logs_cmd.add_argument(
        "--clear",
        action="store_true",
        help="clear the bridge log before showing it",
    )
    logs_cmd.add_argument(
        "--dev",
        action="store_true",
        help="show Pi request and model response developer logs",
    )
    logs_cmd.add_argument(
        "-f",
        "--follow",
        action="store_true",
        default=True,
        help="keep printing new log lines (default)",
    )
    logs_cmd.add_argument(
        "--no-follow",
        dest="follow",
        action="store_false",
        help="show the current log and exit",
    )
    logs_cmd.add_argument(
        "--tail",
        type=int,
        default=200,
        help="number of existing log lines to show before following (default: 200, use 0 for all)",
    )

    status_cmd = subparsers.add_parser("status", help="show bridge status")
    status_cmd.add_argument("--config")
    status_cmd.add_argument("--pid-file")
    status_cmd.add_argument("--log-file")

    info_cmd = subparsers.add_parser("info", help="show system info (like neofetch)")
    info_cmd.add_argument("--config")

    api_cmd = subparsers.add_parser("api", help="inspect configured model APIs")
    api_cmd.add_argument("--config")
    api_cmd.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="seconds to wait for each API check",
    )
    api_cmd.add_argument(
        "--limits",
        action="store_true",
        help="open the provider usage and limits terminal view",
    )
    api_cmd.add_argument(
        "--limets",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    api_subparsers = api_cmd.add_subparsers(dest="api_command")
    api_status_cmd = api_subparsers.add_parser(
        "status",
        help="check saved model API connectivity",
    )
    api_status_cmd.add_argument("--config")
    api_status_cmd.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="seconds to wait for each API check",
    )
    api_subparsers.add_parser(
        "limits",
        help="open the provider usage and limits terminal view",
    )

    openapi_cmd = subparsers.add_parser(
        "openapi",
        help="export or validate OpenAPI specification for the bridge",
        aliases=["api-spec"],
    )
    openapi_cmd.add_argument("--config")
    openapi_subparsers = openapi_cmd.add_subparsers(dest="openapi_command")
    openapi_export_cmd = openapi_subparsers.add_parser(
        "export",
        help="export OpenAPI spec to a file or stdout",
    )
    openapi_export_cmd.add_argument(
        "--output", "-o",
        default=None,
        help="output file path (default: print to stdout)",
    )
    openapi_subparsers.add_parser(
        "validate",
        help="validate the OpenAPI spec structure",
    )

    tools_cmd = subparsers.add_parser("tools", help="inspect and test bridge tools")
    tools_cmd.add_argument("--config")
    tools_subparsers = tools_cmd.add_subparsers(dest="tools_command")
    tools_subparsers.add_parser("list", help="show enabled bridge tools")
    tools_score_cmd = tools_subparsers.add_parser(
        "score",
        help="score tool relevance for a query",
    )
    tools_score_cmd.add_argument("query")
    tools_test_cmd = tools_subparsers.add_parser(
        "test",
        help="call one tool with JSON arguments",
    )
    tools_test_cmd.add_argument("name")
    tools_test_cmd.add_argument("arguments", nargs="?", default="{}")
    tools_diagnose_cmd = tools_subparsers.add_parser(
        "diagnose",
        help="show intent, selected tools, rejected tools, and provider availability",
    )
    tools_diagnose_cmd.add_argument("query")

    usage_cmd = subparsers.add_parser("usage", help="show token usage per model")
    usage_cmd.add_argument("--config")
    usage_cmd.add_argument("--watch", action="store_true", help="refresh every 2 seconds")
    usage_cmd.add_argument("--json", action="store_true", help="print raw JSON and exit")

    subparsers.add_parser("mcp-tools", help="run the bridge tools MCP adapter")

    subparsers.add_parser(
        "endpoints",
        help="show all HTTP endpoints supported by the bridge",
    )

    pi_cmd = subparsers.add_parser(
        "pi",
        help="configure and launch the Pi coding agent",
    )
    pi_cmd.add_argument("--config")
    pi_cmd.add_argument("--pid-file")
    pi_cmd.add_argument("--log-file")
    pi_cmd.add_argument("--provider", help="provider name from env.yml to use for Pi")
    pi_cmd.add_argument("--model", help="model name to use for Pi")
    pi_cmd.add_argument(
        "--dev",
        action="store_true",
        help="show llama bridge launcher setup and server startup details",
    )
    pi_cmd.add_argument(
        "--no-install-pi",
        action="store_true",
        help="do not install Pi automatically if it is missing",
    )
    pi_cmd.add_argument(
        "pi_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to pi",
    )

    claude_cmd = subparsers.add_parser(
        "claude",
        help="launch Claude Code with the generated Api.json",
    )
    claude_cmd.add_argument("--config")
    claude_cmd.add_argument("--pid-file")
    claude_cmd.add_argument("--log-file")
    claude_cmd.add_argument(
        "--dev",
        action="store_true",
        help="show llama bridge launcher setup and server startup details",
    )
    claude_cmd.add_argument(
        "--no-install-claude",
        action="store_true",
        help="do not install Claude Code automatically if it is missing",
    )
    claude_cmd.add_argument(
        "claude_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to claude",
    )

    codex_cmd = subparsers.add_parser(
        "codex",
        help="configure and launch OpenAI Codex CLI with the bridge",
    )
    codex_cmd.add_argument("--config")
    codex_cmd.add_argument("--pid-file")
    codex_cmd.add_argument("--log-file")
    codex_cmd.add_argument("--provider", help="provider name from env.yml to use for Codex")
    codex_cmd.add_argument("--model", help="model name to use for Codex")
    codex_cmd.add_argument(
        "--dev",
        action="store_true",
        help="show llama bridge launcher setup and server startup details",
    )
    codex_cmd.add_argument(
        "--no-install-codex",
        action="store_true",
        help="do not install Codex automatically if it is missing",
    )
    codex_cmd.add_argument(
        "codex_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to codex",
    )

    copilot_cmd = subparsers.add_parser(
        "copilot",
        help="configure and launch GitHub Copilot CLI with the bridge",
    )
    copilot_cmd.add_argument("--config")
    copilot_cmd.add_argument("--pid-file")
    copilot_cmd.add_argument("--log-file")
    copilot_cmd.add_argument("--provider", help="provider name from env.yml to use for Copilot CLI")
    copilot_cmd.add_argument("--model", help="model name to use for Copilot CLI")
    copilot_cmd.add_argument(
        "--dev",
        action="store_true",
        help="show llama bridge launcher setup and server startup details",
    )
    copilot_cmd.add_argument(
        "--no-install-copilot",
        action="store_true",
        help="do not install Copilot CLI automatically if it is missing",
    )
    copilot_cmd.add_argument(
        "copilot_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to copilot",
    )

    opencode_cmd = subparsers.add_parser(
        "opencode",
        help="configure and launch OpenCode with llama bridge models",
    )
    opencode_cmd.add_argument("--config")
    opencode_cmd.add_argument("--pid-file")
    opencode_cmd.add_argument("--log-file")
    opencode_cmd.add_argument("--provider", help="provider name from env.yml to use for OpenCode")
    opencode_cmd.add_argument("--model", help="model name to use for OpenCode")
    opencode_cmd.add_argument(
        "--dev",
        action="store_true",
        help="show llama bridge launcher setup and server startup details",
    )
    opencode_cmd.add_argument(
        "--no-install-opencode",
        action="store_true",
        help="do not install OpenCode automatically if it is missing",
    )
    opencode_cmd.add_argument(
        "--project-config",
        action="store_true",
        help="write project-local .opencode/opencode.json instead of global config",
    )
    opencode_cmd.add_argument(
        "opencode_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to opencode",
    )

    poolside_cmd = subparsers.add_parser(
        "poolside",
        help="install and launch Poolside Agent CLI",
    )
    poolside_cmd.add_argument("--config")
    poolside_cmd.add_argument("--provider", help="provider name from env.yml to use for Poolside")
    poolside_cmd.add_argument("--model", help="model name to use for Poolside")
    poolside_cmd.add_argument(
        "--dev",
        action="store_true",
        help="show llama bridge launcher setup and server startup details",
    )
    poolside_cmd.add_argument(
        "--no-install-poolside",
        action="store_true",
        help="do not install Poolside automatically if it is missing",
    )
    poolside_cmd.add_argument(
        "poolside_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to pool",
    )

    poolside_acp_proxy_cmd = subparsers.add_parser(
        "poolside-acp-proxy",
        help=argparse.SUPPRESS,
    )
    poolside_acp_proxy_cmd.add_argument(
        "poolside_acp_args",
        nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS,
    )

    agent_cmd = subparsers.add_parser(
        "agent",
        help="start, stop, and inspect the installed llama_agent",
    )
    agent_cmd.add_argument("agent_action", nargs="?", choices=["start", "status", "stop", "logs"], default="start")
    agent_cmd.add_argument("--home", help="llama_agent install directory")
    agent_cmd.add_argument("--pid-file")
    agent_cmd.add_argument("--log-file")
    agent_cmd.add_argument("--foreground", action="store_true", help="run llama_agent in the current terminal")
    agent_cmd.add_argument("--status", action="store_true", help="show llama_agent status")
    agent_cmd.add_argument("--stop", action="store_true", help="stop llama_agent")
    agent_cmd.add_argument("--logs", action="store_true", help="show llama_agent logs")
    agent_cmd.add_argument("-f", "--follow", action="store_true", default=True, help="keep printing new log lines (default)")
    agent_cmd.add_argument("--no-follow", dest="follow", action="store_false", help="show the current log and exit")
    agent_cmd.add_argument("--tail", type=int, default=200, help="number of existing log lines to show before following")
    agent_cmd.add_argument("--no-start-bridge", action="store_true", help="do not start Llama Bridge before the agent")

    cli_cmd = subparsers.add_parser(
        "cli",
        help="list and remove CLI tools managed by llama",
    )
    cli_cmd.add_argument("--config")
    cli_cmd.add_argument(
        "--list",
        action="store_true",
        help="show which CLI tools are currently installed and usable",
    )
    cli_cmd.add_argument(
        "--support",
        action="store_true",
        help="show which CLI tools are supported by llama bridge",
    )
    cli_cmd.add_argument(
        "--rm",
        nargs="?",
        const="__prompt__",
        metavar="NAME",
        help="remove one installed CLI by name, or prompt to choose when no name is given",
    )

    bot_cmd = subparsers.add_parser(
        "bot",
        help="Telegram bot setup",
    )
    bot_cmd.add_argument("--config")
    bot_subparsers = bot_cmd.add_subparsers(dest="bot_command")
    bot_subparsers.add_parser("setup", help="run the Telegram bot setup workflow")
    bot_run_cmd = bot_subparsers.add_parser("run", help="run the Teligram polling bot")
    bot_run_cmd.add_argument("--config", dest="bot_run_config", help="path to env.yml")
    bot_run_cmd.add_argument("--workspace", help="workspace directory for Teligram Markdown files")
    bot_subparsers.add_parser("status", help="show Telegram bot configuration status")
    bot_send_cmd = bot_subparsers.add_parser("send", help="send a Telegram message from the bot")
    bot_send_cmd.add_argument("--chat-id", help="explicit Telegram chat ID to send to")
    bot_send_cmd.add_argument("message", nargs="+", help="message text to send")
    bot_send_ai = bot_subparsers.add_parser("send-ai", help="send an AI-generated message via Telegram")
    bot_send_ai.add_argument("--chat-id", help="explicit Telegram chat ID to send to")
    bot_send_ai.add_argument("message", nargs="+", help="prompt for the AI message")
    bot_subparsers.add_parser("test-token", help="test Telegram bot token via getMe")
    bot_subparsers.add_parser("logs", help="show last 50 lines of Telegram bot log")

    return parser


def _cmd_init(config_path: Path, force: bool) -> None:
    _cmd_init_hardcoded(config_path, force)


def _cmd_init_hardcoded(config_path: Path, force: bool) -> None:
    """Write env.yml directly with template defaults (no interactive prompts)."""
    created = write_default_config(config_path, force=force)
    merge_missing_config_fields(config_path)
    _title("llama init")
    _print_state("ok", "config file created", "32")
    _kv_rows(
        [
            ("config", str(created)),
            ("next", "edit API keys/models in the file, then run llama serve"),
        ]
    )


def _cmd_setup(config_path: Path, install_system: bool = False) -> None:
    result = _run_setup(config_path, install_system=install_system)

    _title("llama setup")
    rows: list[tuple[str, str | int]] = [("config", str(result.config_path))]
    if result.config_path.name == DEFAULT_EXAMPLE_CONFIG_PATH.name:
        rows.append(("next", f"edit and rename to {config_path.name}"))
    else:
        rows.extend(
            [
                ("claude settings", str(result.api_settings_path)),
                ("connect", f"claude --settings {result.api_settings_path}"),
            ]
        )
    _kv_rows(rows)
    if result.installed_python:
        _print_state("ok", f"installed python packages: {', '.join(result.installed_python)}", "32")
    if result.missing_python:
        _print_state("warn", f"missing python packages: {', '.join(result.missing_python)}", "33")
    if not result.installed_python and not result.missing_python:
        _print_state("ok", "python packages: ok", "32")
    for note in result.notes:
        _print_note(note)


def _cmd_configure(config_path: Path) -> None:
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    _title("llama configure")
    _print_note(f"Updating {config_path}")

    providers = raw.setdefault("providers", {})
    provider_names = list(providers.keys())
    default_provider = str(
        raw.get("codex", {}).get("provider")
        or raw.get("pi", {}).get("provider")
        or next(iter(provider_names), "ollama_cloud")
    )
    selected_provider = _prompt_choice(
        "Model/auth provider",
        provider_names,
        default_provider,
    )
    provider_entry = providers.setdefault(selected_provider, {})

    current_key = str(provider_entry.get("api_key") or "")
    api_key = _prompt_text(
        f"{selected_provider} API key",
        current_key,
        secret=True,
        allow_blank=True,
    )
    if api_key != "":
        provider_entry["api_key"] = api_key

    current_model = str(provider_entry.get("default_model") or "")
    model_value = _prompt_text(
        f"{selected_provider} default model",
        current_model,
        secret=False,
        allow_blank=False,
    )
    provider_entry["default_model"] = model_value

    if _prompt_yes_no("Point Claude-style aliases to this provider/model?", default=True):
        aliases = raw.setdefault("anthropic_models", {})
        for alias_name in ("haiku", "sonnet", "opus"):
            aliases[alias_name] = {"provider": selected_provider, "model": model_value}

    for section_name in ("pi", "codex", "copilot_cli", "opencode", "poolside", "telegram"):
        section = raw.setdefault(section_name, {})
        if _prompt_yes_no(f"Use {selected_provider} for {section_name}?", default=False):
            section["provider"] = selected_provider
            section["model"] = model_value

    telegram = raw.setdefault("telegram", {})
    configure_telegram = _prompt_yes_no(
        "Configure Telegram bot now?",
        default=bool(telegram.get("enabled", False)),
    )
    if configure_telegram:
        telegram["enabled"] = True
        token_value = _prompt_text(
            "Telegram bot token",
            str(telegram.get("bot_token") or ""),
            secret=True,
            allow_blank=False,
        )
        telegram["bot_token"] = token_value
        telegram["provider"] = _prompt_choice(
            "Telegram provider",
            provider_names,
            str(telegram.get("provider") or selected_provider),
        )
        telegram["model"] = _prompt_text(
            "Telegram model",
            str(telegram.get("model") or model_value),
            allow_blank=False,
        )
        chat_ids = _prompt_text(
            "Allowed chat IDs (comma separated, blank keeps current)",
            ",".join(str(item) for item in (telegram.get("allowed_chat_ids") or [])),
            allow_blank=True,
        )
        if chat_ids.strip():
            telegram["allowed_chat_ids"] = [item.strip() for item in chat_ids.split(",") if item.strip()]
        telegram["max_input_chars"] = int(
            _prompt_text(
                "Telegram max input chars",
                str(telegram.get("max_input_chars") or 4000),
                allow_blank=False,
            )
        )
        telegram["max_output_tokens"] = int(
            _prompt_text(
                "Telegram max output tokens",
                str(telegram.get("max_output_tokens") or 512),
                allow_blank=False,
            )
        )
    else:
        telegram["enabled"] = bool(telegram.get("enabled", False))

    write_config_data(config_path, raw)

    _print_state("ok", "configuration updated", "32")
    _kv_rows(
        [
            ("config", str(config_path)),
            ("provider", selected_provider),
            ("model", model_value),
            ("telegram", "enabled" if telegram.get("enabled") else "disabled"),
        ]
    )


def _cmd_bot_setup(config_path: Path) -> None:
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    providers = raw.setdefault("providers", {})
    provider_names = list(providers.keys())
    telegram = raw.setdefault("telegram", {})

    _title("llama bot")
    _print_note(f"Updating {config_path}")

    telegram["enabled"] = _prompt_yes_no(
        "Enable Telegram bot?",
        default=bool(telegram.get("enabled", True)),
    )
    if not telegram["enabled"]:
        write_config_data(config_path, raw)
        _print_state("ok", "Telegram bot remains disabled", "32")
        return

    telegram["bot_token"] = _prompt_text(
        "Telegram bot token",
        str(telegram.get("bot_token") or ""),
        secret=True,
        allow_blank=False,
    )
    default_provider = str(telegram.get("provider") or next(iter(provider_names), "ollama_cloud"))
    telegram["provider"] = _prompt_choice(
        "Telegram provider",
        provider_names,
        default_provider,
    )
    provider_entry = providers.get(telegram["provider"], {})
    default_model = str(
        telegram.get("model")
        or provider_entry.get("default_model")
        or ""
    )
    telegram["model"] = _prompt_text(
        "Telegram model",
        default_model,
        allow_blank=False,
    )
    chat_ids = _prompt_text(
        "Allowed chat IDs (comma separated, blank means all chats allowed)",
        ",".join(str(item) for item in (telegram.get("allowed_chat_ids") or [])),
        allow_blank=True,
    )
    telegram["allowed_chat_ids"] = [item.strip() for item in chat_ids.split(",") if item.strip()]
    telegram["max_input_chars"] = int(
        _prompt_text(
            "Telegram max input chars",
            str(telegram.get("max_input_chars") or 4000),
            allow_blank=False,
        )
    )
    telegram["max_output_tokens"] = int(
        _prompt_text(
            "Telegram max output tokens",
            str(telegram.get("max_output_tokens") or 512),
            allow_blank=False,
        )
    )
    telegram["system_prompt"] = _prompt_text(
        "Telegram bot system prompt",
        str(telegram.get("system_prompt") or ""),
        allow_blank=False,
    )

    write_config_data(config_path, raw)
    workspace_path = Path(__file__).resolve().parent / "bot_docs"
    _ensure_teligram_workspace_files(workspace_path)
    _print_state("ok", "Telegram bot configuration updated", "32")
    _kv_rows(
        [
            ("config", str(config_path)),
            ("workspace", str(workspace_path)),
            ("provider", telegram["provider"]),
            ("model", telegram["model"]),
            ("allowed chats", ", ".join(telegram["allowed_chat_ids"]) or "all"),
            ("run", f"llama bot run --config {config_path}"),
        ]
    )


def _ensure_teligram_workspace_files(workspace: Path) -> None:
    try:
        from .teligram import ensure_required_workspace_files
    except Exception as exc:  # noqa: BLE001 - setup should continue even if templates fail.
        _print_state("warn", f"could not load Teligram templates: {exc}", "33")
        return
    ensure_required_workspace_files(workspace)


def _cmd_endpoints() -> None:
    _title("llama endpoints")
    port = 8089
    host = "127.0.0.1"
    owui_port = None
    try:
        cfg = load_config()
        port = cfg.server.port
        host = cfg.server.host
        owui_port = cfg.server.openwebui_port
    except Exception:
        pass
    _print_note(f"Use http://{host}:{port} for Ollama-style clients.")
    _print_note(f"Use http://{host}:{port}/v1 for OpenAI, LM Studio, Copilot, and Codex clients.")
    if owui_port is not None:
        _print_note(f"Use http://{host}:{owui_port} for Open Web UI (LLM-only, no tools).")
    print()
    for group, rows in ENDPOINT_GROUPS:
        _print_endpoint_group(group, rows)
    _print_state(
        "note",
        "model-management and blob routes are compatibility shims; generation and embeddings call your configured provider",
        "33",
    )


def _print_endpoint_group(group: str, rows: list[tuple[str, str, str]]) -> None:
    print(_style(group, "1;35"))
    method_width = max(len(method) for method, _path, _note in rows)
    path_width = max(len(path) for _method, path, _note in rows)
    for method, path, note in rows:
        print(
            "  "
            f"{_style(method.ljust(method_width), _endpoint_method_color(method))}  "
            f"{_style(path.ljust(path_width), '36')}  "
            f"{_style(note, '2')}"
        )
    print()


def _endpoint_method_color(method: str) -> str:
    if "DELETE" in method:
        return "31"
    if "POST" in method:
        return "32"
    if "HEAD" in method:
        return "33"
    return "34"


def _ensure_setup(config_path: Path) -> None:
    first_run = not config_path.exists()
    result = _run_setup(config_path, install_system=first_run, quiet=True)
    if result.missing_python:
        missing = ", ".join(result.missing_python)
        raise SystemExit(f"Missing required Python packages: {missing}. Run `llama setup`.")
    _sync_config_clone_from_root(config_path)
    if not config_path.exists():
        example_path = _example_config_path(config_path)
        raise SystemExit(
            f"Missing {config_path.name}. Edit {example_path.name}, add your API keys/models, "
            f"then rename it to {config_path.name}."
        )


def _run_setup(
    config_path: Path,
    install_system: bool = False,
    quiet: bool = False,
) -> SetupResult:
    config_exists = config_path.exists()
    example_path = _example_config_path(config_path)
    if config_exists:
        result_config_path = config_path
        notes = ["env.yml already exists"]
    else:
        example_created = not example_path.exists()
        result_config_path = write_default_config(example_path)
        notes = [
            "created config.example.yml" if example_created else "config.example.yml already exists",
            "edit API keys/models, then rename config.example.yml to env.yml",
        ]

    api_settings_path = config_path.parent / DEFAULT_API_SETTINGS_PATH.name
    api_created = not api_settings_path.exists()
    notes.append("created Api.json" if api_created else "Api.json already exists")
    notes.extend(_ensure_launcher_environment(config_path.parent))

    missing_specs = _missing_python_specs()
    installed_python: list[str] = []
    if missing_specs and not getattr(sys, "frozen", False):
        installed_python = _install_python_specs(missing_specs, quiet=quiet)

    missing_python = _missing_python_specs()
    if getattr(sys, "frozen", False) and missing_python:
        notes.append("running as packaged exe, Python packages must be bundled into the exe")

    if not missing_python and config_exists:
        notes.extend(_check_configured_services(config_path, install_system=install_system))
    elif api_created:
        write_claude_api_settings(api_settings_path)

    return SetupResult(
        config_path=result_config_path,
        api_settings_path=api_settings_path,
        installed_python=installed_python,
        missing_python=missing_python,
        notes=notes,
    )


def _example_config_path(config_path: Path) -> Path:
    return config_path.parent / DEFAULT_EXAMPLE_CONFIG_PATH.name


def _missing_python_specs() -> list[str]:
    missing = []
    for module_name, requirement in PYTHON_REQUIREMENTS.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(requirement)
    return missing


def _install_python_specs(specs: list[str], quiet: bool = False) -> list[str]:
    command = [sys.executable, "-m", "pip", "install", *specs]
    process = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    if process.returncode != 0:
        return []
    return specs


def _check_configured_services(config_path: Path, install_system: bool = False) -> list[str]:
    notes: list[str] = []
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - setup should explain config problems.
        return [f"config check failed: {exc}"]

    active_provider_names = {alias.provider for alias in config.anthropic_models.values()}
    active_providers = [
        provider
        for provider_name, provider in config.providers.items()
        if provider_name in active_provider_names
    ]

    needs_ollama = any(provider.type in {"ollama", "ollama_local"} for provider in active_providers)
    if needs_ollama:
        if shutil.which("ollama"):
            notes.append("ollama command: ok")
        elif install_system and os.name == "nt" and shutil.which("winget"):
            _install_ollama_with_winget()
            notes.append("ollama install requested through winget")
        else:
            notes.append("ollama command: missing, install Ollama or run `llama setup`")

        ollama_health = _http_status("http://127.0.0.1:11434", path="/")
        notes.append(f"ollama service: {ollama_health}")

    needs_lm_studio = any(provider.type == "lm_studio" for provider in active_providers)
    if needs_lm_studio:
        lm_studio_health = _http_status("http://127.0.0.1:1234", path="/")
        notes.append(f"lm studio service: {lm_studio_health}")

    cloud_providers = [
        provider.name
        for provider in active_providers
        if provider.type not in {"ollama", "ollama_local", "lm_studio"}
        and (not provider.api_key or provider.api_key.startswith("${"))
    ]
    if cloud_providers:
        notes.append(f"api keys needed: {', '.join(cloud_providers)}")

    return notes


def _install_ollama_with_winget() -> None:
    subprocess.run(
        [
            "winget",
            "install",
            "--id",
            "Ollama.Ollama",
            "-e",
            "--accept-source-agreements",
            "--accept-package-agreements",
        ],
        check=False,
    )


def _ensure_launcher_environment(app_dir: Path) -> list[str]:
    resolved_app_dir = str(app_dir.resolve())
    os.environ["LLAMA_HOME"] = resolved_app_dir
    os.environ["PATH"] = _prepend_path(os.environ.get("PATH", ""), resolved_app_dir)

    if os.name != "nt":
        return [f"LLAMA_HOME set for this process: {resolved_app_dir}"]

    try:
        changed = _set_windows_user_environment(resolved_app_dir)
    except OSError as exc:
        return [f"could not update Windows user environment: {exc}"]

    if changed:
        _broadcast_windows_environment_change()
        return [
            f"LLAMA_HOME saved: {resolved_app_dir}",
            "app directory added to user PATH; open a new terminal to use `llama` everywhere",
        ]
    return [f"LLAMA_HOME already saved: {resolved_app_dir}", "app directory already in user PATH"]


def _set_windows_user_environment(app_dir: str) -> bool:
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        "Environment",
        0,
        winreg.KEY_READ | winreg.KEY_WRITE,
    ) as key:
        changed = False
        current_home = _read_registry_value(key, "LLAMA_HOME")
        if current_home != app_dir:
            winreg.SetValueEx(key, "LLAMA_HOME", 0, winreg.REG_EXPAND_SZ, app_dir)
            changed = True

        current_path = _read_registry_value(key, "Path") or ""
        new_path = _prepend_path(current_path, app_dir)
        if new_path != current_path:
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
            changed = True
        return changed


def _read_registry_value(key, name: str) -> str | None:
    import winreg

    try:
        value, _value_type = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    return str(value)


def _prepend_path(path_value: str, new_entry: str) -> str:
    parts = [part for part in path_value.split(os.pathsep) if part]
    normalized_new = os.path.normcase(os.path.normpath(new_entry))
    if any(os.path.normcase(os.path.normpath(part)) == normalized_new for part in parts):
        return path_value
    return os.pathsep.join([new_entry, *parts])


def _broadcast_windows_environment_change() -> None:
    import ctypes

    hwnd_broadcast = 0xFFFF
    wm_settingchange = 0x001A
    smto_abortifhung = 0x0002
    ctypes.windll.user32.SendMessageTimeoutW(
        hwnd_broadcast,
        wm_settingchange,
        0,
        "Environment",
        smto_abortifhung,
        5000,
        None,
    )


def _cmd_serve_foreground(
    config_path: Path,
    log_file: str | None = None,
    idle_timeout_seconds: int = 0,
    idle_after_file: Path | None = None,
) -> None:
    os.environ.setdefault("LLAMA_DEV_LOG", "1")
    _sync_config_clone_from_root(config_path)
    _configure_server_event_loop()
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = stream
        sys.stderr = stream

    import uvicorn
    from .server import create_app

    config = load_config(config_path)
    app = create_app(
        config_path,
        idle_timeout_seconds=idle_timeout_seconds,
        idle_after_file=idle_after_file,
        include_tools=True,
    )

    openwebui_port = config.server.openwebui_port
    if openwebui_port is not None:
        app_no_tools = create_app(
            config_path,
            idle_timeout_seconds=0,
            idle_after_file=None,
            include_tools=False,
        )
        _print_state("dual", f"tools on {config.server.host}:{config.server.port}, llm-only on {config.server.host}:{openwebui_port}", "36")

        async def _run_dual() -> None:
            server_main = uvicorn.Server(
                uvicorn.Config(app, host=config.server.host, port=config.server.port, log_level="info")
            )
            server_openwebui = uvicorn.Server(
                uvicorn.Config(app_no_tools, host=config.server.host, port=openwebui_port, log_level="info")
            )
            async with asyncio.TaskGroup() as tg:
                tg.create_task(server_main.serve())
                tg.create_task(server_openwebui.serve())

        asyncio.run(_run_dual())
    else:
        uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")


def _configure_server_event_loop() -> None:
    if os.name != "nt":
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if selector_policy is None:
            return
        asyncio.set_event_loop_policy(selector_policy())


def _cmd_start(
    config_path: Path,
    pid_path: Path,
    log_path: Path,
    idle_timeout_seconds: int = 0,
    idle_after_file: Path | None = None,
    verbose: bool = True,
) -> None:
    ensure_default_dirs(pid_path.parent)
    ensure_default_dirs(log_path.parent)
    _sync_config_clone_from_root(config_path)
    cfg = load_config(config_path)
    already_running, running_url = _server_is_running(config_path, pid_path)
    if _is_running(pid_path):
        if verbose:
            _print_state("run", f"llama server is already running with pid {pid_path.read_text().strip()}", "32")
            _print_note(f"MCP server URL: {_server_url(cfg.server.host, cfg.server.port).rstrip('/')}/mcp")
        _write_active_server_state(config_path, pid_path, log_path)
        return
    if already_running and running_url is not None:
        if verbose:
            _print_state("run", f"llama server is already running at {running_url}", "32")
            _print_note(f"MCP server URL: {running_url.rstrip('/')}/mcp")
        _write_active_server_state(config_path, pid_path, log_path)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    (config_path.parent / "llama.dev.log").write_text("", encoding="utf-8")
    if verbose:
        _title("llama start")
    if os.name == "nt":
        process_id = _start_windows_background(
            config_path,
            log_path,
            idle_timeout_seconds=idle_timeout_seconds,
            idle_after_file=idle_after_file,
        )
        if verbose:
            _pause_with_spinner("starting background server", 1)
        else:
            time.sleep(1)
        if not _pid_alive(process_id):
            _print_state("fail", f"llama failed to start, see log: {log_path}", "31")
            return
        pid_path.write_text(str(process_id), encoding="utf-8")
        _write_active_server_state(config_path, pid_path, log_path)
        if verbose:
            _print_state("ok", f"llama started in background on pid {process_id}", "32")
            _kv_rows([("log", str(log_path)), ("logs", "llama logs")])
            _print_note(f"MCP server URL: {_server_url(cfg.server.host, cfg.server.port).rstrip('/')}/mcp")
            if idle_timeout_seconds == 0:
                _print_note("Server will stay up until you run `llama stop`.")
            else:
                _print_note(f"Server will stop after {_format_idle_duration(idle_timeout_seconds)} of inactivity.")
            openwebui_port = _try_openwebui_port(config_path)
            if openwebui_port is not None:
                _print_note(f"LLM-only (no tools) server on {_server_url(cfg.server.host, openwebui_port)} for Open Web UI")
        return

    with log_path.open("a", encoding="utf-8") as handle:
        env = {**os.environ, "LLAMA_DEV_LOG": "1"}
        process = subprocess.Popen(
            _serve_command(config_path, log_path, idle_timeout_seconds, idle_after_file),
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            cwd=str(config_path.parent),
            start_new_session=True,
            env=env,
        )
    if verbose:
        _pause_with_spinner("starting background server", 1)
    else:
        time.sleep(1)
    if process.poll() is not None:
        _print_state("fail", f"llama failed to start, see log: {log_path}", "31")
        return
    pid_path.write_text(str(process.pid), encoding="utf-8")
    _write_active_server_state(config_path, pid_path, log_path)
    if verbose:
        _print_state("ok", f"llama started in background on pid {process.pid}", "32")
        _kv_rows([("log", str(log_path)), ("logs", "llama logs")])
        _print_note(f"MCP server URL: {_server_url(cfg.server.host, cfg.server.port).rstrip('/')}/mcp")
        if idle_timeout_seconds == 0:
            _print_note("Server will stay up until you run `llama stop`.")
        else:
            _print_note(f"Server will stop after {_format_idle_duration(idle_timeout_seconds)} of inactivity.")
        openwebui_port = _try_openwebui_port(config_path)
        if openwebui_port is not None:
            _print_note(f"LLM-only (no tools) server on {_server_url(cfg.server.host, openwebui_port)} for Open Web UI")


def _cmd_ngrok_start(config_path: Path) -> None:
    config = load_config(config_path)
    pid_path = config_path.parent / DEFAULT_NGROK_PID_PATH.name
    log_path = config_path.parent / DEFAULT_NGROK_LOG_PATH.name
    existing_pid = _read_pid(pid_path)
    if existing_pid is not None and _pid_alive(existing_pid):
        public_url = _wait_for_ngrok_url(timeout_seconds=2.0)
        if public_url:
            _print_state("online", f"public URL: {public_url}", "32")
            _print_note(f"MCP server URL: {public_url.rstrip('/')}/mcp")
        else:
            _print_state("online", f"ngrok is already running with pid {existing_pid}", "32")
        return
    if existing_pid is not None:
        pid_path.unlink(missing_ok=True)

    running, _running_url = _server_is_running(config_path, config_path.parent / DEFAULT_PID_PATH.name)
    if not running:
        _print_state("fail", "llama server is not running, so ngrok was not started", "31")
        return

    ngrok_exe = shutil.which("ngrok")
    if not ngrok_exe:
        _print_state("fail", "ngrok command was not found on PATH", "31")
        _print_note("Install ngrok, then set ngrok.auth_token in env.yml or NGROK_AUTHTOKEN.")
        return

    log_path.write_text("", encoding="utf-8")
    target_url = _ngrok_target_url(config.server.host, config.server.port)
    command = [ngrok_exe, "http", target_url]
    if config.ngrok.region:
        command.extend(["--region", str(config.ngrok.region)])

    env = dict(os.environ)
    auth_token = str(config.ngrok.auth_token or "").strip()
    if auth_token and not auth_token.startswith("${"):
        env["NGROK_AUTHTOKEN"] = auth_token
        command.extend(["--authtoken", auth_token])
    if config.server.auth_token == "change-me":
        _print_state("warn", "server.auth_token is still 'change-me'; change it before sharing the public URL", "33")

    popen_kwargs: dict[str, Any] = {
        "cwd": str(config_path.parent),
        "stdin": subprocess.DEVNULL,
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True

    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=handle,
            **popen_kwargs,
        )
    pid_path.write_text(str(process.pid), encoding="utf-8")
    _print_state("online", f"ngrok started on pid {process.pid}", "32")
    public_url = _wait_for_ngrok_url()
    if public_url:
        _print_state("online", f"public URL: {public_url}", "32")
        _print_note(f"OpenAI-compatible base URL: {public_url.rstrip('/')}/v1")
        _print_note(f"MCP server URL: {public_url.rstrip('/')}/mcp")
        _print_note("Use server.auth_token as the API key.")
    elif process.poll() is not None:
        pid_path.unlink(missing_ok=True)
        _print_state("fail", f"ngrok exited early, see log: {log_path}", "31")
    else:
        _print_state("warn", f"ngrok started, but no public URL was reported yet; see log: {log_path}", "33")


def _cmd_ngrok_stop(pid_path: Path, *, verbose: bool = True) -> None:
    pid = _read_pid(pid_path)
    if pid is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    finally:
        pid_path.unlink(missing_ok=True)
    if verbose:
        _print_state("ok", f"ngrok stopped pid {pid}", "32")


def _ngrok_target_url(host: str, port: int) -> str:
    target_host = host
    if target_host in {"0.0.0.0", "::"}:
        target_host = "127.0.0.1"
    return _server_url(target_host, port)


def _wait_for_ngrok_url(timeout_seconds: float = 15.0) -> str | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        public_url = _ngrok_public_url()
        if public_url:
            return public_url
        time.sleep(0.5)
    return None


def _ngrok_public_url() -> str | None:
    try:
        request = Request("http://127.0.0.1:4040/api/tunnels", method="GET")
        with urlopen(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    tunnels = payload.get("tunnels") if isinstance(payload, dict) else None
    if not isinstance(tunnels, list):
        return None
    http_url = None
    for tunnel in tunnels:
        if not isinstance(tunnel, dict):
            continue
        public_url = str(tunnel.get("public_url") or "")
        if public_url.startswith("https://"):
            return public_url
        if public_url.startswith("http://") and http_url is None:
            http_url = public_url
    return http_url


def _try_openwebui_port(config_path: Path) -> int | None:
    try:
        config = load_config(config_path)
        return config.server.openwebui_port
    except Exception:
        return None


def _sync_config_clone_from_root(config_path: Path) -> bool:
    candidates = _llama_config_paths(config_path)
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return False

    source = max(existing, key=lambda path: path.stat().st_mtime_ns)
    source_content = source.read_bytes()
    changed: list[Path] = []

    for target in candidates:
        if target == source:
            continue
        if target.exists() and target.read_bytes() == source_content:
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_content)
        except OSError as exc:
            raise SystemExit(f"could not sync {target} from {source}: {exc}") from exc
        changed.append(target)

    if changed:
        _print_state("sync", f"using newest env.yml: {source}", "36")
        _kv_rows([("updated", str(path)) for path in changed])
    return bool(changed)


def _llama_config_paths(config_path: Path) -> list[Path]:
    candidates = [
        config_path,
        DEFAULT_CONFIG_PATH,
    ]
    candidates.extend(_llama_root_config_candidates())

    seen: set[Path] = set()
    paths: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)
    return paths


def _llama_root_config_candidates() -> list[Path]:
    candidates: list[Path] = []
    llama_home = os.environ.get("LLAMA_HOME")
    if llama_home:
        candidates.append(Path(llama_home) / DEFAULT_CONFIG_PATH.name)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / DEFAULT_CONFIG_PATH.name)
    candidates.append(Path(__file__).resolve().parent.parent / DEFAULT_CONFIG_PATH.name)
    return candidates


def _cmd_stop(pid_path: Path) -> None:
    _cmd_ngrok_stop(pid_path.parent / DEFAULT_NGROK_PID_PATH.name, verbose=False)
    pid = _read_pid(pid_path)
    if pid is None:
        config_path = pid_path.parent / DEFAULT_CONFIG_PATH.name
        pid = _listening_pid_for_config(config_path)
    if pid is None:
        _print_state("stop", "llama server is not running", "33")
        return
    try:
        if os.name == "nt":
            _run_with_spinner(
                f"stopping server pid {pid}",
                lambda: subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _print_state("stop", "llama process was not found", "33")
    finally:
        pid_path.unlink(missing_ok=True)
        _clear_active_server_state(pid_path)
    _print_state("ok", f"llama stopped pid {pid}", "32")


def _cmd_restart(
    config_path: Path,
    pid_path: Path,
    log_path: Path,
    idle_timeout_seconds: int = 0,
) -> None:
    _title("llama restart")
    _cmd_stop(pid_path)
    _cmd_start(config_path, pid_path, log_path, idle_timeout_seconds=idle_timeout_seconds, verbose=False)


def _cmd_agent(args: argparse.Namespace) -> None:
    """Start, stop, or inspect the installed llama_agent."""
    action = "logs" if args.logs else "status" if args.status else "stop" if args.stop else args.agent_action
    home = _agent_home(Path(args.home) if args.home else None)
    pid_path = Path(args.pid_file) if args.pid_file else home / ".llama-agent.pid"
    log_path = Path(args.log_file) if args.log_file else home / ".llama-agent.log"
    if action == "logs":
        _cmd_agent_logs(log_path, pid_path, follow=args.follow, tail=args.tail)
        return
    if action == "status":
        _cmd_agent_status(home, pid_path, log_path)
        return
    if action == "stop":
        _cmd_agent_stop(pid_path)
        return
    _cmd_agent_start(home, pid_path, log_path, foreground=args.foreground, start_bridge=not args.no_start_bridge)


def _agent_home(override: Path | None = None) -> Path:
    """Resolve the installed llama_agent directory."""
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override)
    if os.environ.get("LLAMA_AGENT_HOME"):
        candidates.append(Path(os.environ["LLAMA_AGENT_HOME"]))
    if os.environ.get("LLAMA_HOME"):
        candidates.append(Path(os.environ["LLAMA_HOME"]) / "agent")
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "agent")
    candidates.append(Path(__file__).resolve().parent.parent / "agent")
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if (path / "package.json").exists():
            return path
    return candidates[0].expanduser().resolve() if candidates else Path.cwd()


def _cmd_agent_start(home: Path, pid_path: Path, log_path: Path, *, foreground: bool, start_bridge: bool) -> None:
    """Start llama_agent in the background or foreground."""
    _title("llama agent")
    _ensure_agent_home(home)
    if _is_running(pid_path):
        _print_state("run", f"llama_agent is already running with pid {pid_path.read_text().strip()}", "32")
        _kv_rows([("home", str(home)), ("log", str(log_path))])
        return
    if start_bridge:
        config_path = _default_config_path()
        _ensure_setup(config_path)
        config = load_config(config_path)
        _cmd_start(
            config_path,
            _arg_path(None, DEFAULT_PID_PATH, config_path),
            _arg_path(None, DEFAULT_LOG_PATH, config_path),
            _configured_idle_timeout_seconds(config),
            verbose=False,
        )
    npm = _find_npm()
    command = _npm_agent_command(npm, home)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if foreground:
        _print_state("run", "starting llama_agent in the current terminal", "36")
        raise SystemExit(subprocess.run(command, cwd=home, check=False).returncode)
    log_path.write_text("", encoding="utf-8")
    popen_kwargs: dict[str, Any] = {
        "cwd": str(home),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "env": {**os.environ, "LLAMA_AGENT_BACKGROUND": "1"},
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = _background_creationflags()
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        popen_kwargs["startupinfo"] = startupinfo
    else:
        popen_kwargs["start_new_session"] = True
    with log_path.open("a", encoding="utf-8") as handle:
        popen_kwargs["stdout"] = handle
        popen_kwargs["stderr"] = handle
        process = subprocess.Popen(
            command,
            **popen_kwargs,
        )
    env = _read_agent_env(home)
    port = env.get("PORT", "3456")
    if not _wait_for_agent_start(process, port, timeout_seconds=10):
        _print_state("fail", f"llama_agent failed to start, see log: {log_path}", "31")
        return
    pid_path.write_text(str(process.pid), encoding="utf-8")
    _print_state("ok", f"llama_agent started in background on pid {process.pid}", "32")
    _kv_rows([
        ("home", str(home)),
        ("log", str(log_path)),
        ("logs", "llama agent --logs"),
        ("status", "llama agent --status"),
        ("stop", "llama agent --stop"),
    ])


def _cmd_agent_status(home: Path, pid_path: Path, log_path: Path) -> None:
    """Show llama_agent status."""
    _title("llama agent status")
    pid = _read_pid(pid_path)
    running = pid is not None and _pid_alive(pid)
    if pid is not None and not running:
        pid_path.unlink(missing_ok=True)
        pid = None
    env = _read_agent_env(home)
    port = env.get("PORT", "3456")
    bridge_url = env.get("LLAMA_BRIDGE_URL", "http://127.0.0.1:8089")
    _kv_rows(
        [
            ("Agent", _status_label(running)),
            ("Agent PID", pid or "-"),
            ("Agent health", _http_status(f"http://127.0.0.1:{port}")),
            ("Bridge", _http_status(bridge_url)),
            ("Home", str(home)),
            ("Log", str(log_path)),
        ]
    )


def _cmd_agent_logs(log_path: Path, pid_path: Path, *, follow: bool = True, tail: int = 200) -> None:
    """Show or follow llama_agent logs."""
    running = _is_running(pid_path)
    if not running and follow:
        _print_state("stop", "llama_agent is not running; showing saved log and exiting", "33")
        follow = False
    log_path = _resolve_agent_log_path(log_path)
    if not log_path.exists():
        _print_state("warn", f"no llama_agent log found at {log_path}", "33")
        return
    _title("llama agent logs")
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        if tail > 0:
            lines = handle.readlines()
            for line in lines[-tail:]:
                print(_format_log_line(line), end="")
        elif not follow:
            pass
        try:
            while True:
                line = handle.readline()
                if line:
                    print(_format_log_line(line), end="")
                    continue
                if not follow:
                    return
                time.sleep(1)
        except KeyboardInterrupt:
            return


def _wait_for_agent_start(process: subprocess.Popen[Any], port: str, timeout_seconds: float) -> bool:
    """Wait for the agent health endpoint or process failure."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        if _http_status(f"http://127.0.0.1:{port}").startswith("ok"):
            return True
        time.sleep(0.5)
    return process.poll() is None


def _cmd_agent_stop(pid_path: Path) -> None:
    """Stop the background llama_agent process."""
    pid = _read_pid(pid_path)
    if pid is None:
        _print_state("stop", "llama_agent is not running", "33")
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _print_state("stop", "llama_agent process was not found", "33")
    finally:
        pid_path.unlink(missing_ok=True)
    _print_state("ok", f"llama_agent stopped pid {pid}", "32")


def _ensure_agent_home(home: Path) -> None:
    """Validate that llama_agent is installed."""
    if not (home / "package.json").exists():
        raise SystemExit(
            f"llama_agent is not installed at {home}.\n"
            "Run the Windows setup again and choose the llama_agent option, or pass --home."
        )


def _find_npm() -> str:
    """Return the npm executable path."""
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm and os.name == "nt":
        npm_path = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "npm.cmd"
        if npm_path.exists():
            npm = str(npm_path)
    if not npm:
        raise SystemExit("npm was not found. Install Node.js LTS and try again.")
    return npm


def _npm_agent_command(npm: str, home: Path) -> list[str]:
    """Return a Windows-safe command for the full llama_agent stack."""
    script = "start"
    script_command = ""
    package_path = home / "package.json"
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
        scripts = package.get("scripts") if isinstance(package, dict) else {}
        if isinstance(scripts, dict) and "dev" in scripts:
            script = "dev"
        if isinstance(scripts, dict):
            script_command = str(scripts.get(script) or "")
    except Exception:
        script = "start"
        script_command = ""

    direct_command = _direct_agent_node_command(home, script_command)
    if direct_command is not None:
        return direct_command

    if os.name == "nt" and Path(npm).suffix.lower() in {".cmd", ".bat"}:
        node = _find_node()
        npm_cli = _npm_cli_js(Path(npm))
        if node and npm_cli is not None:
            return [node, str(npm_cli), "run", script]
        comspec = os.environ.get("COMSPEC") or "cmd.exe"
        return [comspec, "/d", "/s", "/c", subprocess.list2cmdline([npm, "run", script])]
    return [npm, "run", script]


def _direct_agent_node_command(home: Path, script_command: str) -> list[str] | None:
    """Bypass npm for simple Node scripts so Windows does not leave a shell window open."""
    node = _find_node()
    if not node:
        return None
    match = re.fullmatch(r"\s*node\s+([^\s]+\.mjs)\s*", script_command)
    if not match:
        return None
    script_path = (home / match.group(1)).resolve()
    try:
        script_path.relative_to(home.resolve())
    except ValueError:
        return None
    if not script_path.exists():
        return None
    return [node, str(script_path)]


def _find_node() -> str | None:
    """Return the Node executable path when available."""
    node = shutil.which("node") or shutil.which("node.exe")
    if node:
        return node
    if os.name == "nt":
        node_path = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "node.exe"
        if node_path.exists():
            return str(node_path)
    return None


def _npm_cli_js(npm_cmd: Path) -> Path | None:
    """Resolve npm.cmd to npm's JS entrypoint so no command shell window is needed."""
    candidates = [
        npm_cmd.parent / "node_modules" / "npm" / "bin" / "npm-cli.js",
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "npm" / "node_modules" / "npm" / "bin" / "npm-cli.js")
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(Path(program_files) / "nodejs" / "node_modules" / "npm" / "bin" / "npm-cli.js")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    try:
        text = npm_cmd.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"(?i)([^\"\r\n]*node_modules[\\/]+npm[\\/]+bin[\\/]+npm-cli\.js)", text)
    if match:
        candidate = (npm_cmd.parent / match.group(1)).resolve()
        if candidate.exists():
            return candidate
    return None


def _resolve_agent_log_path(log_path: Path) -> Path:
    """Prefer the configured capture log, then common app-created logs."""
    if log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    home = log_path.parent
    for candidate in (
        log_path,
        home / "llama-agent.log",
        home / "llama_agent.log",
        home / "agent.log",
        home / "logs" / "agent.log",
        home / "logs" / "dev.log",
    ):
        if candidate.exists() and candidate.stat().st_size > 0:
            if candidate != log_path:
                _print_state("info", f"showing agent log at {candidate}", "36")
            return candidate
    return log_path


def _background_creationflags() -> int:
    """Return Windows background process flags when available."""
    if os.name != "nt":
        return 0
    return (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
        | subprocess.DETACHED_PROCESS
        | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    )


def _read_agent_env(home: Path) -> dict[str, str]:
    """Read simple KEY=value entries from llama_agent env files."""
    values: dict[str, str] = {}
    for name in (".env", ".env.local"):
        path = home / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _cmd_status(config_path: Path, pid_path: Path, log_path: Path) -> None:
    api_settings_path = config_path.parent / DEFAULT_API_SETTINGS_PATH.name
    config = None
    config_error = None
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - status should explain config problems.
        config_error = exc

    pid = _read_pid(pid_path)
    process_running = pid is not None and _pid_alive(pid)
    if pid is not None and not process_running:
        pid_path.unlink(missing_ok=True)
        pid = None

    if config is not None:
        main_url = _server_url(config.server.host, config.server.port)
        main_http = _http_status(main_url)
        if not process_running and main_http.startswith("ok"):
            process_running = True
        ngrok_pid = _read_pid(config_path.parent / DEFAULT_NGROK_PID_PATH.name)
        ngrok_running = ngrok_pid is not None and _pid_alive(ngrok_pid)
        ngrok_url = _ngrok_public_url() if ngrok_running else None

    _title("llama status")
    rows: list[tuple[str, str | int]] = [("process", _status_label(process_running))]
    if pid is not None:
        rows.append(("pid", pid))
    elif process_running:
        rows.append(("pid", "unknown"))

    if config is not None:
        rows.append(("url (tools)", f"{main_url} ({main_http})" if main_http else main_url))
        rows.append(("mcp", f"{main_url.rstrip('/')}/mcp"))
        if ngrok_running:
            rows.append(("url (online)", ngrok_url or "ngrok running, URL not available yet"))
            if ngrok_url:
                rows.append(("mcp (online)", f"{ngrok_url.rstrip('/')}/mcp"))
            rows.append(("ngrok pid", ngrok_pid or "unknown"))
        if config.server.openwebui_port is not None:
            owui_url = _server_url(config.server.host, config.server.openwebui_port)
            owui_http = _http_status(owui_url)
            rows.append(("url (llm-only)", f"{owui_url} ({owui_http})" if owui_http else owui_url))
        rows.extend(
            [
                ("config", str(config.source_path)),
                ("claude settings", str(api_settings_path)),
                ("connect", f"claude --settings {api_settings_path}"),
                ("providers", len(config.providers)),
                ("models", ", ".join(sorted(config.anthropic_models))),
            ]
        )
    else:
        rows.extend(
            [
                ("config", str(config_path)),
                ("claude settings", str(api_settings_path)),
                ("config error", str(config_error)),
            ]
        )

    rows.extend([("pid file", str(pid_path)), ("log file", str(log_path))])
    if log_path.exists():
        rows.append(("log size", f"{log_path.stat().st_size} bytes"))
    _kv_rows(rows)


def _cmd_usage(config_path: Path | None, watch: bool, json_output: bool) -> None:
    if config_path is None:
        config_path = _default_config_path()
    usage_file = config_path.parent / "llama.usage.json"

    if json_output:
        if not usage_file.exists():
            print("{}")
            return
        try:
            print(usage_file.read_text(encoding="utf-8"))
        except Exception:
            print("{}")
        return

    if not usage_file.exists() or usage_file.stat().st_size == 0:
        print("No usage data yet. Run some requests first.")
        return

    try:
        data = json.loads(usage_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print("No usage data yet. Run some requests first.")
        return

    models = data.get("models", {})
    if not models:
        print("No usage data yet. Run some requests first.")
        return

    if watch:
        _run_usage_watch(usage_file)
    else:
        _print_usage_table(data)


def _run_usage_watch(usage_file: Path) -> None:
    import time

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print("llama usage  (refreshing every 2s — press Ctrl+C to stop)\n")
            if usage_file.exists():
                try:
                    data = json.loads(usage_file.read_text(encoding="utf-8"))
                    _print_usage_table(data)
                except (json.JSONDecodeError, OSError):
                    print("No usage data yet. Run some requests first.")
            else:
                print("No usage data yet. Run some requests first.")
            time.sleep(2)
    except KeyboardInterrupt:
        return


def _print_usage_table(data: dict[str, Any]) -> None:
    models = data.get("models", {})
    if not models:
        print("No usage data yet. Run some requests first.")
        return

    def fmt_num(n: int) -> str:
        return f"{n:,}"

    col_widths = {
        "model": 30,
        "input": 14,
        "output": 14,
        "total": 14,
        "requests": 12,
    }

    rows = []
    totals = {"input": 0, "output": 0, "total": 0, "requests": 0}
    for name, info in sorted(models.items()):
        input_t = info.get("input_tokens", 0)
        output_t = info.get("output_tokens", 0)
        total_t = info.get("total_tokens", 0)
        reqs = info.get("request_count", 0)
        rows.append((name, input_t, output_t, total_t, reqs))
        totals["input"] += input_t
        totals["output"] += output_t
        totals["total"] += total_t
        totals["requests"] += reqs

    name_width = col_widths["model"]
    for name, _, _, _, _ in rows:
        name_width = max(name_width, len(name) + 2)
    col_widths["model"] = name_width

    bar = "═" * (name_width + 1 + col_widths["input"] + 1 + col_widths["output"] + 1 + col_widths["total"] + 1 + col_widths["requests"] + 1)

    print(f"╔{bar}╗")
    header = f"║ {'Model':<{name_width}} ║ {'Input Tokens':^{col_widths['input']}} ║ {'Output Tokens':^{col_widths['output']}} ║ {'Total Tokens':^{col_widths['total']}} ║ {'Requests':^{col_widths['requests']}} ║"
    print(header)
    print(f"╠{bar}╣")

    for name, input_t, output_t, total_t, reqs in rows:
        row = f"║ {name:<{name_width}} ║ {fmt_num(input_t):>{col_widths['input']}} ║ {fmt_num(output_t):>{col_widths['output']}} ║ {fmt_num(total_t):>{col_widths['total']}} ║ {fmt_num(reqs):>{col_widths['requests']}} ║"
        print(row)

    print(f"╠{bar}╣")
    total_row = f"║ {'TOTAL':<{name_width}} ║ {fmt_num(totals['input']):>{col_widths['input']}} ║ {fmt_num(totals['output']):>{col_widths['output']}} ║ {fmt_num(totals['total']):>{col_widths['total']}} ║ {fmt_num(totals['requests']):>{col_widths['requests']}} ║"
    print(total_row)
    print(f"╚{bar}╝")

    updated_at = data.get("updated_at", "")
    if updated_at:
        print(f"Updated: {updated_at}")


def _cmd_info(config_path: Path) -> None:
    print_llamafetch(config_path)


def _cmd_api_status(config_path: Path, timeout: float = 90.0) -> None:
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - CLI should show config errors plainly.
        _print_state("fail", f"could not load config: {exc}", "31")
        return

    _title("llama api status")
    _print_note(f"checking saved models from {config.source_path}")

    results = asyncio.run(_check_saved_model_apis(config, timeout, progress=True))
    _print_api_status_table(results)


def _cmd_api_limits(config_path: Path) -> None:
    while True:
        try:
            config = load_config(config_path)
        except Exception as exc:  # noqa: BLE001 - CLI should show config errors plainly.
            _print_state("fail", f"could not load config: {exc}", "31")
            return

        _render_api_limits_screen(config)
        if not sys.stdin.isatty():
            return
        try:
            choice = input("\n[r] refresh  [q] quit > ").strip().lower()
        except EOFError:
            return
        if choice in {"", "r", "refresh"}:
            continue
        if choice in {"q", "quit", "exit"}:
            return


def _cmd_openapi_export(config_path: Path, output_path: str | None) -> None:
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - CLI should show config errors plainly.
        _print_state("fail", f"could not load config: {exc}", "31")
        return

    spec = _generate_openapi_spec(config)

    if output_path:
        output_file = Path(output_path)
        output_file.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        _title("llama openapi export")
        _print_state("ok", f"OpenAPI spec written to {output_file}", "32")
    else:
        _title("llama openapi export")
        print(json.dumps(spec, indent=2, ensure_ascii=False))


def _cmd_openapi_validate(config_path: Path) -> None:
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - CLI should show config errors plainly.
        _print_state("fail", f"could not load config: {exc}", "31")
        return

    spec = _generate_openapi_spec(config)
    errors: list[str] = []

    if "openapi" not in spec:
        errors.append("Missing 'openapi' version field")
    elif not str(spec["openapi"]).startswith("3."):
        errors.append(f"Invalid OpenAPI version: {spec['openapi']}")

    if "info" not in spec:
        errors.append("Missing 'info' object")
    else:
        info = spec["info"]
        if "title" not in info:
            errors.append("Missing info.title")
        if "version" not in info:
            errors.append("Missing info.version")

    if "paths" not in spec:
        errors.append("Missing 'paths' object")
    elif not isinstance(spec["paths"], dict):
        errors.append("'paths' must be an object")
    elif not spec["paths"]:
        errors.append("No paths defined in 'paths'")

    _title("llama openapi validate")
    if errors:
        _print_state("fail", "Validation failed:", "31")
        for err in errors:
            print(f"  {_style('*', '31')} {err}")
    else:
        path_count = len(spec["paths"])
        _print_state("ok", "OpenAPI spec is valid", "32")
        _kv_rows([
            ("version", spec.get("openapi", "unknown")),
            ("title", spec.get("info", {}).get("title", "unknown")),
            ("version", spec.get("info", {}).get("version", "unknown")),
            ("paths", path_count),
        ])


def _generate_openapi_spec(config) -> dict[str, Any]:
    server = config.server
    base_url = f"http://{server.host}:{server.port}"

    spec: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {
            "title": "llama bridge",
            "description": "AI model bridge with multi-provider support",
            "version": "0.1.0",
        },
        "servers": [{"url": base_url}],
        "paths": {},
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "token",
                }
            }
        },
        "security": [{"BearerAuth": []}],
    }

    paths = spec["paths"]

    paths["/"] = {
        "get": {
            "summary": "Service probe",
            "responses": {"200": {"description": "Service is healthy"}},
        }
    }

    paths["/health"] = {
        "get": {
            "summary": "Bridge health status",
            "responses": {"200": {"description": "Bridge is healthy"}},
        }
    }

    paths["/v1/models"] = {
        "get": {
            "summary": "List configured models",
            "tags": ["OpenAI Compatible"],
            "responses": {
                "200": {
                    "description": "List of available models",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "data": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Model"},
                                    }
                                },
                            }
                        }
                    },
                }
            },
        }
    }

    paths["/v1/chat/completions"] = {
        "post": {
            "summary": "Chat completions",
            "tags": ["OpenAI Compatible"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ChatCompletionRequest"}
                    }
                },
            },
            "responses": {
                "200": {"description": "Chat completion response"}
            },
        }
    }

    paths["/v1/completions"] = {
        "post": {
            "summary": "Text completions (legacy)",
            "tags": ["OpenAI Compatible"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CompletionRequest"}
                    }
                },
            },
            "responses": {
                "200": {"description": "Completion response"}
            },
        }
    }

    paths["/v1/responses"] = {
        "post": {
            "summary": "OpenAI Responses API",
            "tags": ["OpenAI Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": {"description": "Response from model"}
            },
        }
    }

    paths["/v1/messages"] = {
        "post": {
            "summary": "Anthropic Messages API",
            "tags": ["Anthropic Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": {"description": "Message response"}
            },
        }
    }

    paths["/v1/messages/batches"] = {
        "get": {
            "summary": "List message batches",
            "tags": ["Anthropic Compatible"],
            "responses": {"200": {"description": "List of batches"}},
        },
        "post": {
            "summary": "Create message batch",
            "tags": ["Anthropic Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {"200": {"description": "Batch created"}},
        },
    }

    paths["/v1/tools"] = {
        "get": {
            "summary": "List bridge tools",
            "tags": ["Bridge Tools"],
            "responses": {
                "200": {
                    "description": "List of available tools",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "tools": {"type": "array"},
                                },
                            }
                        }
                    },
                }
            },
        }
    }

    paths["/v1/tools/call"] = {
        "post": {
            "summary": "Call a bridge tool",
            "tags": ["Bridge Tools"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {"type": "object"},
                            },
                            "required": ["name"],
                        }
                    }
                },
            },
            "responses": {"200": {"description": "Tool call result"}},
        }
    }

    paths["/api/tags"] = {
        "get": {
            "summary": "List available models (Ollama compatible)",
            "tags": ["Ollama Compatible"],
            "responses": {"200": {"description": "List of models"}},
        }
    }

    paths["/api/version"] = {
        "get": {
            "summary": "Bridge version",
            "tags": ["Ollama Compatible"],
            "responses": {"200": {"description": "Version info"}},
        }
    }

    paths["/api/chat"] = {
        "post": {
            "summary": "Chat messages (Ollama compatible)",
            "tags": ["Ollama Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {"200": {"description": "Chat response"}},
        }
    }

    paths["/api/generate"] = {
        "post": {
            "summary": "Prompt completion (Ollama compatible)",
            "tags": ["Ollama Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {"200": {"description": "Completion response"}},
        }
    }

    paths["/api/embed"] = {
        "post": {
            "summary": "Embedding (Ollama compatible)",
            "tags": ["Ollama Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {"200": {"description": "Embedding response"}},
        }
    }

    for alias_name in config.anthropic_models:
        alias = config.anthropic_models[alias_name]
        model = alias.model or config.providers[alias.provider].default_model or alias_name

        paths[f"/v1/models/{model}"] = {
            "get": {
                "summary": f"Get model: {model}",
                "tags": ["Models"],
                "responses": {
                    "200": {"description": f"Model {model}"}
                },
            }
        }

    paths["/v1/assistants"] = {
        "get": {
            "summary": "List assistants",
            "tags": ["OpenAI Compatible"],
            "responses": {"200": {"description": "List of assistants"}},
        },
        "post": {
            "summary": "Create assistant",
            "tags": ["OpenAI Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {"200": {"description": "Assistant created"}},
        },
    }

    paths["/v1/threads"] = {
        "get": {
            "summary": "List threads",
            "tags": ["OpenAI Compatible"],
            "responses": {"200": {"description": "List of threads"}},
        },
        "post": {
            "summary": "Create thread",
            "tags": ["OpenAI Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {"200": {"description": "Thread created"}},
        },
    }

    paths["/v1/files"] = {
        "get": {
            "summary": "List files",
            "tags": ["Anthropic Compatible"],
            "responses": {"200": {"description": "List of files"}},
        },
        "post": {
            "summary": "Upload file",
            "tags": ["Anthropic Compatible"],
            "requestBody": {
                "required": True,
                "content": {"multipart/form-data": {"schema": {"type": "object"}}},
            },
            "responses": {"200": {"description": "File uploaded"}},
        },
    }

    paths["/v1/skills"] = {
        "get": {
            "summary": "List skills",
            "tags": ["Anthropic Compatible"],
            "responses": {"200": {"description": "List of skills"}},
        },
        "post": {
            "summary": "Create skill",
            "tags": ["Anthropic Compatible"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {"200": {"description": "Skill created"}},
        },
    }

    spec["components"]["schemas"] = {
        "Model": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "display_name": {"type": "string"},
            },
        },
        "ChatCompletionRequest": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
                "max_tokens": {"type": "integer"},
                "temperature": {"type": "number"},
                "stream": {"type": "boolean"},
                "tools": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["model", "messages"],
        },
        "CompletionRequest": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "prompt": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                "max_tokens": {"type": "integer"},
                "temperature": {"type": "number"},
                "stream": {"type": "boolean"},
            },
            "required": ["model", "prompt"],
        },
    }

    return spec


def _cmd_telegram_status(config_path: Path) -> None:
    config = load_config(config_path)
    telegram = config.telegram
    _title("llama telegram")
    _kv_rows(
        [
            ("enabled", str(telegram.enabled)),
            ("token", _configured_label(telegram.bot_token)),
            ("provider", telegram.provider),
            ("model", telegram.model or config.providers[telegram.provider].default_model or "-"),
            ("allowed chats", ", ".join(telegram.allowed_chat_ids) or "all"),
        ]
    )


def _telegram_state_path(config_path: Path) -> Path:
    return config_path.parent / "llama.telegram.json"


def _read_last_telegram_chat_id(config_path: Path) -> str | None:
    state_path = _telegram_state_path(config_path)
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    chat_id = str(payload.get("chat_id") or "").strip()
    if chat_id:
        return chat_id
    chats = payload.get("chats") or {}
    if not isinstance(chats, dict) or not chats:
        return None
    latest_entry = max(
        (entry for entry in chats.values() if isinstance(entry, dict)),
        key=lambda entry: str(entry.get("updated_at") or ""),
        default=None,
    )
    if latest_entry is None:
        return None
    chat_id = str(latest_entry.get("chat_id") or "").strip()
    return chat_id or None


def _default_telegram_chat_id(config, config_path: Path) -> str | None:
    last_chat_id = _read_last_telegram_chat_id(config_path)
    if last_chat_id:
        return last_chat_id
    numeric_allowed = [
        str(item).strip()
        for item in (config.telegram.allowed_chat_ids or [])
        if re.fullmatch(r"-?\d+", str(item).strip())
    ]
    if len(numeric_allowed) == 1:
        return numeric_allowed[0]
    return None


def _cmd_bot_send(config_path: Path, message: str, chat_id: str | None = None) -> None:
    config = load_config(config_path)
    telegram = config.telegram
    if not telegram.enabled:
        raise SystemExit("Telegram bot is disabled in env.yml.")
    if not telegram.bot_token or telegram.bot_token.startswith("${"):
        raise SystemExit("Telegram bot token is not configured.")
    if not message.strip():
        raise SystemExit("Message text is required.")

    target_chat_id = str(chat_id or _default_telegram_chat_id(config, config_path) or "").strip()
    if not target_chat_id:
        raise SystemExit(
            "No Telegram chat target found. Use `llama bot send --chat-id <id> <message>` "
            "or send the bot a message first so it can remember the last chat."
        )

    response = httpx.post(
        f"https://api.telegram.org/bot{telegram.bot_token}/sendMessage",
        json={"chat_id": target_chat_id, "text": message},
        timeout=httpx.Timeout(30.0, connect=10.0),
    )
    response.raise_for_status()

    _title("llama bot send")
    _print_state("ok", "Telegram message sent", "32")
    _kv_rows(
        [
            ("chat", target_chat_id),
            ("message", message[:80] + ("..." if len(message) > 80 else "")),
        ]
    )


def _cmd_bot_send_ai(config_path: Path, prompt: str, chat_id: str | None = None) -> None:
    config = load_config(config_path)
    telegram = config.telegram
    if not telegram.enabled or not telegram.bot_token or telegram.bot_token.startswith("${"):
        raise SystemExit("Telegram bot is not configured.")
    if not prompt.strip():
        raise SystemExit("Prompt text is required.")

    target_chat_id = str(chat_id or _default_telegram_chat_id(config, config_path) or "").strip()
    if not target_chat_id:
        raise SystemExit("No Telegram chat target found.")

    from .providers import build_provider
    provider = build_provider(config.providers[telegram.provider])
    messages = [
        {"role": "system", "content": telegram.system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = provider.chat_completion(
        messages=messages,
        model=telegram.model or config.providers[telegram.provider].default_model,
        max_output_tokens=telegram.max_output_tokens,
    )
    content = response.choices[0].message.content if hasattr(response, "choices") else str(response)

    resp = httpx.post(
        f"https://api.telegram.org/bot{telegram.bot_token}/sendMessage",
        json={"chat_id": target_chat_id, "text": content},
        timeout=httpx.Timeout(30.0, connect=10.0),
    )
    resp.raise_for_status()
    _title("llama bot send-ai")
    _print_state("ok", "AI message sent", "32")
    _kv_rows([("chat", target_chat_id), ("prompt", prompt[:60] + ("..." if len(prompt) > 60 else ""))])


def _cmd_bot_test_token(config_path: Path) -> None:
    config = load_config(config_path)
    token = config.telegram.bot_token
    if not token or token.startswith("${"):
        raise SystemExit("Bot token not configured.")
    from .telegram_launcher import test_telegram_token
    result = test_telegram_token(token)
    if result.get("ok"):
        info = result["result"]
        _title("llama bot test-token")
        _print_state("ok", "Token is valid", "32")
        _kv_rows([
            ("username", f"@{info.get('username', '?')}"),
            ("id", info.get("id", "?")),
            ("first_name", info.get("first_name", "?")),
        ])
    else:
        raise SystemExit(f"Token test failed: {result.get('error', 'unknown')}")


def _cmd_bot_logs(config_path: Path) -> None:
    from .telegram_launcher import follow_telegram_log
    lines = follow_telegram_log(config_path, n_lines=50)
    _title("llama bot logs")
    for line in lines:
        print(line)


def _render_api_limits_screen(config) -> None:
    _clear_screen()
    _title("llama api limits")
    providers = list(config.providers.values())
    configured = sum(1 for provider in providers if provider.base_url)
    with_limits = sum(1 for provider in providers if provider.usage_limits)
    _kv_rows(
        [
            ("providers", len(providers)),
            ("configured", configured),
            ("with limits", with_limits),
        ]
    )

    sections: dict[str, list[tuple[Any, str | None, str | None, dict[str, Any] | None]]] = {}
    for provider in providers:
        limits = provider.usage_limits or {}
        model_limits = provider.model_limits or {}
        if not limits and not model_limits:
            sections.setdefault("untracked", []).append((provider, None, None, None))
            continue
        for period, entry in limits.items():
            sections.setdefault(_limit_period_label(period), []).append((provider, None, period, entry))
        for model_name, per_model_limits in model_limits.items():
            for period, entry in (per_model_limits or {}).items():
                sections.setdefault(_limit_period_label(period), []).append((provider, model_name, period, entry))

    for section_name in ["hourly", "daily", "weekly", "monthly", "yearly", "other", "untracked"]:
        entries = sections.get(section_name, [])
        if not entries:
            continue
        print()
        print(_style(section_name.upper(), "1;36"))
        _print_limit_table(entries)

    print()
    _print_note("Edit `providers.<name>.usage_limits` in env.yml to track quota, usage, and remaining budget.")


def _print_limit_table(entries: list[tuple[Any, str | None, str | None, dict[str, Any] | None]]) -> None:
    headers = ["provider", "type", "unit", "limit", "used", "left", "key", "model"]
    rows: list[list[str]] = []
    for provider, model_name, period, entry in entries:
        if entry is None:
            rows.append(
                [
                    provider.name,
                    provider.type,
                    "-",
                    "-",
                    "-",
                    "-",
                    "set" if provider.api_key else "missing",
                    model_name or provider.default_model or "-",
                ]
            )
            continue

        unit = str(entry.get("unit", "requests"))
        limit = _to_number(entry.get("limit"))
        used = _to_number(entry.get("used"))
        left = None if limit is None else max(limit - (used or 0), 0)
        rows.append(
            [
                provider.name,
                f"{provider.type}/{period}",
                unit,
                _format_limit_number(limit),
                _format_limit_number(used),
                _format_limit_number(left),
                "set" if provider.api_key else "missing",
                model_name or provider.default_model or "-",
            ]
        )

    widths = [
        max(len(headers[index]), max((len(row[index]) for row in rows), default=0))
        for index in range(len(headers))
    ]
    print("  " + "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  " + "  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def _limit_period_label(period: str) -> str:
    normalized = (period or "").strip().lower()
    if normalized in {"hourly", "hour"}:
        return "hourly"
    if normalized in {"daily", "day"}:
        return "daily"
    if normalized in {"weekly", "week"}:
        return "weekly"
    if normalized in {"monthly", "month"}:
        return "monthly"
    if normalized in {"yearly", "year", "annual"}:
        return "yearly"
    return "other"


def _to_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_limit_number(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isclose(value, round(value)):
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def _clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def _prompt_choice(title: str, options: list[str], default: str) -> str:
    print()
    print(_style(title, "1;35"))
    for index, option in enumerate(options, start=1):
        marker = "*" if option == default else " "
        print(f"  {marker} {index}. {option}")
    try:
        entered = input(f"Choose [default: {default}]: ").strip()
    except KeyboardInterrupt as exc:
        print()
        raise SetupCanceled() from exc
    if not entered:
        return default
    if entered.isdigit():
        selected_index = int(entered) - 1
        if 0 <= selected_index < len(options):
            return options[selected_index]
    if entered in options:
        return entered
    _print_note(f"Unknown option `{entered}`, keeping {default}.")
    return default


def _prompt_text(
    title: str,
    default: str = "",
    *,
    secret: bool = False,
    allow_blank: bool = True,
) -> str:
    prompt = f"{title}"
    if default:
        preview = "<hidden>" if secret else default
        prompt += f" [default: {preview}]"
    prompt += ": "
    try:
        entered = input(prompt)
    except KeyboardInterrupt as exc:
        print()
        raise SetupCanceled() from exc
    if not entered.strip():
        if default:
            return default
        if allow_blank:
            return ""
        return _prompt_text(title, default, secret=secret, allow_blank=allow_blank)
    return entered.strip()


def _prompt_yes_no(title: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    try:
        entered = input(f"{title} [{suffix}]: ").strip().lower()
    except KeyboardInterrupt as exc:
        print()
        raise SetupCanceled() from exc
    if not entered:
        return default
    return entered in {"y", "yes"}


def _cmd_tools_list(config_path: Path) -> None:
    config = load_config(config_path)
    registry = ToolRegistry(config)
    try:
        tools = registry.openai_tools()
        _title("llama tools")
        _kv_rows(
            [
                ("enabled", str(config.tools.enabled)),
                ("relevance filter", str(config.tools.relevance_filter)),
                ("keyword forcing", str(config.tools.force_for_keywords)),
                ("max exposed", config.tools.max_exposed),
                ("default search", config.tools.default_search_provider),
                ("tavily key", _configured_label(config.tools.tavily.api_key)),
                ("serpapi key", _configured_label(config.tools.serpapi.api_key)),
                ("weather", str(config.tools.weather.enabled)),
                ("wikipedia", str(config.tools.wikipedia.enabled)),
            ]
        )
        print()
        if tools:
            print("available:")
        for tool in tools:
            function = tool["function"]
            print(f"- {function['name']}")
            print(f"  {function.get('description', '').splitlines()[0]}")
        unavailable = registry.unavailable_tools()
        if unavailable:
            print()
            print("unavailable:")
            for name, reason in sorted(unavailable.items()):
                print(f"- {name}")
                print(f"  {reason}")
    finally:
        asyncio.run(registry.aclose())


def _cmd_tools_score(config_path: Path, query: str) -> None:
    config = load_config(config_path)
    registry = ToolRegistry(config)
    try:
        tools = registry.openai_tools()
        selected, scores = select_relevant_tools(
            tools,
            query,
            max_tools=config.tools.max_exposed,
            min_score=config.tools.confidence_threshold,
            force_for_keywords=config.tools.force_for_keywords,
            default_search_provider=config.tools.default_search_provider,
        )
        selected_names = {(tool.get("function") or {}).get("name") for tool in selected}
        _title("llama tools score")
        _print_note(query)
        for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            marker = "*" if name in selected_names else " "
            print(f"{marker} {name.ljust(18)} {score:.2f}")
    finally:
        asyncio.run(registry.aclose())


def _cmd_tools_test(config_path: Path, name: str, arguments: str) -> None:
    config = load_config(config_path)
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"arguments must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("arguments must be a JSON object")

    registry = ToolRegistry(config)
    try:
        result = asyncio.run(registry.call_structured(name, parsed))
    finally:
        asyncio.run(registry.aclose())
    print(json.dumps(result, indent=2, ensure_ascii=True))


def _cmd_tools_diagnose(config_path: Path, query: str) -> None:
    config = load_config(config_path)
    registry = ToolRegistry(config)
    try:
        tools = registry.openai_tools()
        selected, scores = select_relevant_tools(
            tools,
            query,
            max_tools=config.tools.max_exposed,
            min_score=config.tools.confidence_threshold,
            force_for_keywords=config.tools.force_for_keywords,
            default_search_provider=config.tools.default_search_provider,
        )
        selected_names = {(tool.get("function") or {}).get("name") for tool in selected}
        available_names = [(tool.get("function") or {}).get("name") for tool in tools]
        report = {
            "query": query,
            "intent": classify_query_intent(query),
            "config": {
                "relevance_filter": config.tools.relevance_filter,
                "max_exposed": config.tools.max_exposed,
                "confidence_threshold": config.tools.confidence_threshold,
                "force_for_keywords": config.tools.force_for_keywords,
                "default_search_provider": config.tools.default_search_provider,
            },
            "provider_availability": {
                "weather": config.tools.weather.enabled,
                "wikipedia": config.tools.wikipedia.enabled,
                "tavily": config.tools.tavily.enabled and _configured_label(config.tools.tavily.api_key) == "configured",
                "serpapi": config.tools.serpapi.enabled and _configured_label(config.tools.serpapi.api_key) == "configured",
            },
            "selected_tools": list(selected_names),
            "rejected_tools": [name for name in available_names if name not in selected_names],
            "scores": dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
        }
        print(json.dumps(report, indent=2, ensure_ascii=True))
    finally:
        asyncio.run(registry.aclose())




def _configured_label(value: str | None) -> str:
    if value and not value.startswith("${"):
        return "configured"
    return "missing"


async def _check_saved_model_apis(
    config,
    timeout: float,
    progress: bool = False,
) -> list[ApiStatusResult]:
    cache: dict[tuple[str, str], tuple[bool, str, str]] = {}
    results: list[ApiStatusResult] = []

    for alias_name in sorted(config.anthropic_models):
        alias = config.anthropic_models[alias_name]
        provider_config = config.providers[alias.provider]
        model = alias.model or provider_config.default_model or ""
        cache_key = (provider_config.name, model)

        if cache_key not in cache:
            label = f"{alias.alias} -> {provider_config.name}/{model}"
            cache[cache_key] = await _run_api_check_with_progress(
                label,
                _check_model_api(provider_config, model, timeout),
                progress,
            )
        elif progress:
            ok, status, _detail = cache[cache_key]
            state = "ok" if ok else status
            _print_state("reuse", f"{alias.alias} -> {provider_config.name}/{model}: {state}", "2")

        ok, status, detail = cache[cache_key]
        results.append(
            ApiStatusResult(
                alias=alias.alias,
                provider=provider_config.name,
                model=model,
                ok=ok,
                status=status,
                detail=detail,
            )
        )

    return results


async def _run_api_check_with_progress(
    label: str,
    check,
    progress: bool,
) -> tuple[bool, str, str]:
    if not progress:
        return await check

    if not sys.stdout.isatty():
        _print_state("check", label, "36")
        result = await check
        _print_api_check_result(label, result)
        return result

    task = asyncio.create_task(check)
    frames = "-\\|/"
    started = time.monotonic()
    index = 0
    while not task.done():
        elapsed = time.monotonic() - started
        sys.stdout.write(
            f"\r{_style(frames[index % len(frames)], '36')} checking {label} ({elapsed:.1f}s)"
        )
        sys.stdout.flush()
        index += 1
        await asyncio.sleep(0.1)

    result = await task
    sys.stdout.write("\r" + " " * (len(label) + 32) + "\r")
    sys.stdout.flush()
    _print_api_check_result(label, result)
    return result


def _print_api_check_result(label: str, result: tuple[bool, str, str]) -> None:
    ok, status, detail = result
    if ok:
        _print_state("ok", label, "32")
        return
    color = "33" if status in {"missing", "timeout"} else "31"
    _print_state(status, f"{label}: {detail}", color)


async def _check_model_api(provider_config, model: str, timeout: float) -> tuple[bool, str, str]:
    import httpx

    if _api_key_missing(provider_config):
        return False, "missing", "api key is not configured"

    from .providers import build_provider

    provider = build_provider(_api_probe_provider_config(provider_config, timeout))
    url = provider._chat_completions_url()
    try:
        async with provider._client.stream(
            "POST",
            url,
            headers=provider._headers(),
            json=provider._payload(
                {
                    "model": model,
                    "max_tokens": 2,
                    "messages": [{"role": "user", "content": "Reply ok."}],
                },
                True,
            ),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    return True, str(response.status_code), "stream started"
        return True, str(response.status_code), "ok"
    except httpx.HTTPStatusError as exc:
        return False, str(exc.response.status_code), _response_summary(exc.response)
    except httpx.TimeoutException:
        detail = f"generation did not start within {timeout:g}s from {url}"
        if provider_config.type == "nvidia_nim":
            detail += "; NVIDIA model catalog may still be valid while generation is queued or slow"
        return False, "timeout", detail
    except httpx.RequestError as exc:
        return False, "offline", str(exc)
    except Exception as exc:  # noqa: BLE001 - status command should not crash.
        return False, "error", str(exc)
    finally:
        await provider.aclose()

def _api_probe_provider_config(provider_config, timeout: float):
    probe_extra_body = {
        key: value
        for key, value in provider_config.extra_body.items()
        if key not in {"max_completion_tokens", "max_new_tokens", "max_tokens", "stream"}
    }
    return replace(provider_config, timeout=timeout, extra_body=probe_extra_body)


def _api_key_missing(provider_config) -> bool:
    if provider_config.type in {"ollama", "ollama_local", "lm_studio"}:
        return False
    api_key = provider_config.api_key or ""
    return not api_key or api_key.startswith("${")


def _response_summary(response) -> str:
    try:
        payload = response.json()
    except Exception:
        text = response.text.strip()
        return text[:140] if text else response.reason_phrase

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or str(error)
            return str(message)[:140]
        if isinstance(error, str):
            return error[:140]
        message = payload.get("message") or payload.get("detail")
        if message:
            return str(message)[:140]
    return str(payload)[:140]


def _print_api_status_table(results: list[ApiStatusResult]) -> None:
    if not results:
        _print_state("warn", "no saved model aliases found", "33")
        return

    headers = ["alias", "provider", "model", "status", "detail"]
    rows = [
        [
            result.alias,
            result.provider,
            result.model,
            _style("ok", "32") if result.ok else _style(result.status, "31" if result.status not in {"missing", "timeout"} else "33"),
            result.detail,
        ]
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(_strip_ansi(row[index])) for row in rows))
        for index in range(len(headers))
    ]

    header = "  " + "  ".join(
        _style(headers[index].ljust(widths[index]), "2")
        for index in range(len(headers))
    )
    print(header)
    for row in rows:
        print(
            "  "
            + "  ".join(
                row[index].ljust(widths[index] + len(row[index]) - len(_strip_ansi(row[index])))
                for index in range(len(row))
            )
        )


def _strip_ansi(value: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", value)


def _cmd_logs(
    log_path: Path,
    pid_path: Path,
    follow: bool = False,
    clear: bool = False,
    dev: bool = False,
    config_path: Path | None = None,
    tail: int = 200,
    use_active: bool = False,
) -> None:
    config_path = config_path or pid_path.parent / DEFAULT_CONFIG_PATH.name
    if use_active:
        active_paths = _active_server_paths()
        if active_paths is not None:
            active_config, active_pid, active_log = active_paths
            active_running, _active_url = _server_is_running(active_config, active_pid)
            if active_running:
                config_path = active_config
                pid_path = active_pid
                log_path = active_log

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        if follow:
            _print_state("stop", "llama server is not running; showing saved log and exiting", "33")
            follow = False

    main_log_path = log_path
    if dev:
        log_path = config_path.parent / "llama.dev.log"

    if clear:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_path.write_text("", encoding="utf-8")
        except PermissionError as exc:
            raise SystemExit(
                f"could not clear log while it is in use: {log_path}. "
                "Run `llama stop` first, then `llama logs --clear`."
            ) from exc

    requested_dev_log = dev
    if dev and (not log_path.exists() or log_path.stat().st_size == 0):
        if main_log_path.exists() and main_log_path.stat().st_size > 0:
            _print_state("info", f"no dev log entries yet at {log_path}; showing normal log at {main_log_path}", "36")
            log_path = main_log_path
            dev = False
        else:
            _print_state("info", f"no dev log entries yet at {log_path}", "36")
            return

    if not log_path.exists() or (not dev and log_path.stat().st_size == 0):
        fallback_log_path = config_path.parent / "llama.dev.log"
        if not dev and fallback_log_path.exists() and fallback_log_path.stat().st_size > 0:
            log_path = fallback_log_path
            dev = True
            _print_state("info", f"llama.log is empty; showing dev log at {log_path}", "36")
        elif not log_path.exists():
            _print_state("warn", f"no llama log found at {log_path}", "33")
            return
        else:
            _print_state(
                "info",
                f"{log_path} is empty; new server output will appear here",
                "36",
            )

    if follow:
        _title("llama dev logs" if dev and requested_dev_log else "llama logs")
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        if tail > 0:
            lines = handle.readlines()
            for line in lines[-tail:]:
                if dev:
                    print(_format_dev_log_line(line), end="")
                else:
                    print(_format_log_line(line), end="")
        elif not follow:
            pass
        try:
            while True:
                line = handle.readline()
                if line:
                    if dev:
                        print(_format_dev_log_line(line), end="")
                    else:
                        print(_format_log_line(line), end="")
                    continue
                if not follow:
                    return
                time.sleep(1)
        except KeyboardInterrupt:
            return


def _active_server_state_path() -> Path:
    return _launcher_state_dir() / "llama.active.json"


def _launcher_state_dir() -> Path:
    return Path.home() / ".llama_bridge"


def _write_active_server_state(config_path: Path, pid_path: Path, log_path: Path) -> None:
    state_path = _active_server_state_path()
    payload = {
        "config_path": str(config_path.resolve()),
        "pid_path": str(pid_path.resolve()),
        "log_path": str(log_path.resolve()),
    }
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def _clear_active_server_state(pid_path: Path) -> None:
    state_path = _active_server_state_path()
    active_paths = _active_server_paths()
    if active_paths is None:
        return
    _active_config, active_pid, _active_log = active_paths
    if active_pid.resolve() != pid_path.resolve():
        return
    try:
        state_path.unlink(missing_ok=True)
    except OSError:
        return


def _active_server_paths() -> tuple[Path, Path, Path] | None:
    state_path = _active_server_state_path()
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        config_path = Path(data["config_path"])
        pid_path = Path(data["pid_path"])
        log_path = Path(data["log_path"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not config_path.exists():
        return None
    return config_path, pid_path, log_path


def _cmd_pi(
    config_path: Path,
    pid_path: Path,
    log_path: Path,
    provider_override: str | None,
    model_override: str | None,
    pi_args: list[str],
    install_pi: bool = True,
    dev: bool = False,
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.pi.closed"
        idle_after_file.unlink(missing_ok=True)
        if dev:
            _print_state("start", "llama server is not running, starting it for Pi", "36")
        idle_timeout_seconds = _configured_idle_timeout_seconds(config)
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=idle_timeout_seconds,
            idle_after_file=idle_after_file,
            verbose=dev,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        if dev:
            _print_note(_idle_timeout_note("Pi", idle_timeout_seconds))
    else:
        idle_after_file = None
        if dev:
            _print_state("run", "using existing llama server", "32")

    provider_name = provider_override or config.pi.provider
    if provider_name not in config.providers:
        available = ", ".join(sorted(config.providers))
        raise SystemExit(f"Unknown Pi provider '{provider_name}'. Available providers: {available}")

    model = resolve_pi_model(config, provider_name=provider_name, model_override=model_override)
    if not model:
        raise SystemExit(pi_model_error(config, provider_name))

    pi_executable = _ensure_pi(install=install_pi, package=config.pi.install_package)
    shell_path = _ensure_pi_shell(install=install_pi)
    models_path, settings_path = _write_pi_settings(config, provider_name, model, shell_path)
    _ensure_pi_extensions(config, verbose=dev)

    if dev:
        _title("llama pi")
        _print_state("ok", "Pi configuration is ready", "32")
        _kv_rows(
            [
                ("provider", provider_name),
                ("model", model),
                ("models", str(models_path)),
                ("settings", str(settings_path)),
                *([("shell", shell_path)] if shell_path else []),
            ]
        )

    passthrough_args = pi_args
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    return_code = subprocess.run([pi_executable, *passthrough_args], check=False).returncode
    if idle_after_file is not None:
        idle_after_file.write_text("closed\n", encoding="utf-8")
    raise SystemExit(return_code)


def _write_pi_settings(
    config,
    provider_name: str,
    model: str,
    shell_path: str | None = None,
) -> tuple[Path, Path]:
    config_dir = Path(os.path.expanduser(config.pi.config_dir))
    config_dir.mkdir(parents=True, exist_ok=True)
    models_path = config_dir / "models.json"
    settings_path = config_dir / "settings.json"

    bridge_provider_name = "llama_bridge"
    provider_entry = {
        "baseUrl": f"{_server_url(config.server.host, config.server.port)}/v1",
        "api": config.pi.api,
        "models": [{"id": model}],
        "apiKey": config.server.auth_token,
    }
    models_data = {"providers": {bridge_provider_name: provider_entry}}
    _write_json(models_path, models_data)

    settings_data = _pi_settings_data(
        _read_json_object(settings_path),
        bridge_provider_name,
        model,
        shell_path,
    )
    _write_json(settings_path, settings_data)
    return models_path, settings_path


def _pi_settings_data(
    settings_data: dict,
    bridge_provider_name: str,
    model: str,
    shell_path: str | None = None,
) -> dict:
    updated = dict(settings_data)
    updated["defaultProvider"] = bridge_provider_name
    updated["defaultModel"] = model
    if shell_path:
        updated["shellPath"] = shell_path
    return updated


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _write_json(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except PermissionError as exc:
        raise SystemExit(
            f"could not update {path}. Close any running Pi session and try again."
        ) from exc


def _ensure_claude_tool_plugin(config, config_path: Path, verbose: bool = True) -> Path:
    plugin_dir = config_path.parent / "plugins" / "llama_bridge_tools_claude"
    _write_claude_tool_plugin(plugin_dir, config)
    short_commands_dir = _write_claude_short_commands()
    if verbose:
        _print_state("ok", f"Claude Code llama bridge tools plugin: {plugin_dir}", "32")
        _print_state("ok", f"Claude Code short slash commands: {short_commands_dir}", "32")
    return plugin_dir


def _ensure_codex_tool_extension(config, verbose: bool = True) -> tuple[Path, Path]:
    codex_config_path = Path(os.path.expanduser(config.codex.config_path))
    codex_config_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_dir = codex_config_path.parent / "plugins" / "llama_bridge_tools"
    _write_codex_tool_plugin(plugin_dir, config)
    _write_codex_mcp_config(codex_config_path, config)
    if verbose:
        _print_state("ok", f"Codex llama bridge tools plugin: {plugin_dir}", "32")
        _print_state("ok", f"Codex MCP tools config: {codex_config_path}", "32")
    return plugin_dir, codex_config_path


def _ensure_copilot_tool_extension(config, verbose: bool = True) -> Path:
    config_path = Path(os.path.expanduser("~/.copilot/mcp-config.json"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_json_object(config_path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    command, args = _mcp_server_command()
    servers["llama_bridge_tools"] = {
        "type": "local",
        "command": command,
        "args": args,
        "env": _mcp_server_env(config),
        "tools": ["*"],
        "timeout": 300000,
    }
    data["mcpServers"] = servers
    _write_json(config_path, data)
    if verbose:
        _print_state("ok", f"Copilot CLI MCP tools config: {config_path}", "32")
        _verify_mcp_server_tools(config)
    return config_path


def _write_claude_tool_plugin(plugin_dir: Path, config) -> None:
    (plugin_dir / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "skills" / "llama-bridge-tools").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "commands").mkdir(parents=True, exist_ok=True)
    _write_json(
        plugin_dir / ".claude-plugin" / "plugin.json",
        {
            "name": "llama-bridge-tools",
            "version": "0.1.0",
            "description": "Expose local llama bridge HTTP tools to Claude Code through MCP.",
            "author": {"name": "llama bridge"},
            "license": "Apache-2.0",
            "keywords": ["llama", "bridge", "mcp", "tools"],
            "skills": "./skills/",
            "mcpServers": "./.mcp.json",
        },
    )
    _write_json(plugin_dir / ".mcp.json", _mcp_json_config(config))
    (plugin_dir / "skills" / "llama-bridge-tools" / "SKILL.md").write_text(
        _bridge_tools_skill("Claude Code"),
        encoding="utf-8",
    )
    _write_claude_bridge_commands(plugin_dir / "commands")


def _write_codex_tool_plugin(plugin_dir: Path, config) -> None:
    (plugin_dir / ".codex-plugin").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "skills" / "llama-bridge-tools").mkdir(parents=True, exist_ok=True)
    _write_json(
        plugin_dir / ".codex-plugin" / "plugin.json",
        {
            "name": "llama-bridge-tools",
            "version": "0.1.0",
            "description": "Expose local llama bridge HTTP tools to Codex through MCP.",
            "author": {"name": "llama bridge"},
            "license": "Apache-2.0",
            "keywords": ["llama", "bridge", "mcp", "tools"],
            "skills": "./skills/",
            "mcpServers": "./.mcp.json",
            "interface": {
                "displayName": "Llama Bridge Tools",
                "shortDescription": "Local web, research, image, weather, and citation tools",
                "longDescription": "Connects Codex to the local llama bridge MCP adapter so enabled bridge tools can be called directly.",
                "developerName": "llama bridge",
                "category": "Productivity",
                "capabilities": ["Read", "Interactive"],
                "defaultPrompt": "Use llama bridge tools for current search, source research, images, weather, Wikipedia, or time lookups.",
                "brandColor": "#111111",
                "screenshots": [],
            },
        },
    )
    _write_json(plugin_dir / ".mcp.json", _mcp_json_config(config))
    (plugin_dir / "skills" / "llama-bridge-tools" / "SKILL.md").write_text(
        _bridge_tools_skill("Codex"),
        encoding="utf-8",
    )


def _write_codex_mcp_config(codex_config_path: Path, config) -> None:
    existing = codex_config_path.read_text(encoding="utf-8") if codex_config_path.exists() else ""
    command, args = _mcp_server_command()
    section = "\n".join(
        [
            f'command = "{_toml_escape(command)}"',
            f"args = {_toml_string_array(args)}",
            f"env = {_toml_inline_table(_mcp_server_env(config))}",
            "startup_timeout_sec = 30",
            "tool_timeout_sec = 300",
            "enabled = true",
        ]
    )
    updated = _replace_toml_section(existing, "mcp_servers.llama_bridge_tools", section)
    if updated and not updated.endswith("\n"):
        updated += "\n"
    codex_config_path.write_text(updated, encoding="utf-8")


def _mcp_json_config(config) -> dict:
    command, args = _mcp_server_command()
    return {
        "mcpServers": {
            "llama_bridge_tools": {
                "type": "stdio",
                "command": command,
                "args": args,
                "env": _mcp_server_env(config),
            }
        }
    }


def _mcp_server_env(config) -> dict[str, str]:
    server_url = _server_url(config.server.host, config.server.port)
    return {
        "LLAMA_BRIDGE_BASE_URL": server_url,
        "LLAMA_BRIDGE_API_KEY": config.server.auth_token,
    }


def _verify_mcp_server_tools(config) -> None:
    """Verify MCP server is reachable and return discovered tool names."""
    import json
    import subprocess

    command, args = _mcp_server_command()
    proc = None
    try:
        proc = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        init_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "copilot-cli-verify", "version": "1.0"},
            },
        }) + "\n"
        proc.stdin.write(init_msg)
        proc.stdin.flush()

        proc.stdout.readline()
        import select
        try:
            if select.select([proc.stdout], [], [], 1.0)[0]:
                proc.stdout.readline()
        except Exception:
            pass

        list_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        }) + "\n"
        proc.stdin.write(list_msg)
        proc.stdin.flush()

        response_line = proc.stdout.readline()
        if response_line:
            data = json.loads(response_line)
            tools = (data.get("result") or {}).get("tools") or []
            tool_names = [t.get("name") for t in tools if t.get("name")]
            if tool_names:
                _print_state("ok", f"MCP tools discovered: {', '.join(tool_names)}", "32")
            else:
                _print_state("warn", "MCP server returned no tools", "33")
        else:
            _print_state("warn", "MCP server did not respond to tools/list", "33")

    except Exception as exc:
        _print_state("warn", f"Could not verify MCP server: {exc}", "33")
    finally:
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                pass


def _bridge_tools_skill(host_name: str) -> str:
    return f"""---
name: llama-bridge-tools
description: Use when {host_name} needs llama bridge MCP tools for current web search, source verification, image research, weather, Wikipedia, or date/time lookups.
---

# Llama Bridge Tools

Use the `llama_bridge_tools` MCP server when a task needs current information,
source verification, image candidates, weather, Wikipedia, or local date/time
lookups.

Prefer the highest-level bridge tool that fits the task:

- `source_research` for cited factual research and evidence gathering.
- `image_research` for compact sourced image candidates.
- `tavily_search` or `serpapi_search` for current web results.
- `wikipedia_search` and `wikipedia_page` for encyclopedia context.
- `weather_current` for live weather.
- `datetime_now` for current time or timezone questions.

The MCP server calls the local llama bridge HTTP tool endpoints, so the llama
server must be running for tool calls to succeed.
"""


def _mcp_server_command() -> tuple[str, list[str]]:
    command, args = _llama_self_command()
    if getattr(sys, "frozen", False):
        return command, ["mcp-tools"]
    return command, [*args[:-1], "llama_bridge.mcp_tools"]


def _llama_self_command() -> tuple[str, list[str]]:
    if getattr(sys, "frozen", False):
        return sys.executable, []
    return sys.executable, ["-m", "llama_bridge.cli"]


def _write_claude_bridge_commands(commands_dir: Path) -> None:
    for filename, (description, hint, body) in _claude_bridge_commands().items():
        _write_claude_command_file(commands_dir / filename, description, hint, body, force=True)


def _write_claude_short_commands() -> Path:
    commands_dir = Path(os.path.expanduser("~/.claude/commands"))
    commands_dir.mkdir(parents=True, exist_ok=True)
    for filename, (description, hint, body) in _claude_bridge_commands().items():
        _write_claude_command_file(commands_dir / filename, description, hint, body, force=False)
    return commands_dir


def _claude_bridge_commands() -> dict[str, tuple[str, str, str]]:
    serp = (
        "Run a SerpAPI web search through the llama bridge.",
        "[query]",
        _bridge_command_body(
            "search query",
            "Use `mcp__llama_bridge_tools__serpapi_search` for the user input and summarize the best results with URLs.",
        ),
    )
    web = (
        "Run a web search through the llama bridge.",
        "[query]",
        _bridge_command_body(
            "web search query",
            "Use the best available llama bridge web search MCP tool for the user input and include URLs.",
        ),
    )
    fetch = (
        "Fetch a URL through the llama bridge.",
        "[url]",
        _bridge_command_body(
            "URL",
            "Use the llama bridge web fetch or source research tools to fetch and summarize the user input.",
        ),
    )
    image = (
        "Find sourced image candidates through the llama bridge.",
        "[query]",
        _bridge_command_body(
            "image search topic",
            "Use `mcp__llama_bridge_tools__image_research` to find 2-3 compact sourced image candidates for the user input.",
        ),
    )
    manim = (
        "Generate a short Manim animation video through the llama bridge.",
        "[animation prompt]",
        _bridge_command_body(
            "animation prompt",
            "Use `mcp__llama_bridge_tools__manim_render` to create a short Python Manim animation video and return the scene_path and video_path.",
        ),
    )
    return {
        "serp.md": serp,
        "web.md": web,
        "fetch.md": fetch,
        "image.md": image,
        "manim.md": manim,
    }


def _write_claude_command_file(
    path: Path,
    description: str,
    hint: str,
    body: str,
    *,
    force: bool,
) -> None:
    content = "\n".join(
        [
            "<!-- Generated by llama bridge. Safe to replace. -->",
            "---",
            f"description: {description}",
            f"argument-hint: {hint}",
            "allowed-tools: mcp__llama_bridge_tools",
            "---",
            "",
            body,
            "",
        ]
    )
    if not force and path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
        if "Generated by llama bridge" not in existing:
            return
    path.write_text(content, encoding="utf-8")


def _bridge_command_body(input_label: str, action: str) -> str:
    return "\n".join(
        [
            "User input: $ARGUMENTS",
            "",
            f"If the user input is empty, ask the user for the {input_label} in one short question and wait for their answer. Do not call tools yet.",
            "",
            f"If the user input is not empty, {action}",
        ]
    )


def _toml_inline_table(values: dict[str, str]) -> str:
    parts = [f'{key} = "{_toml_escape(value)}"' for key, value in values.items()]
    return "{ " + ", ".join(parts) + " }"


def _toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(f'"{_toml_escape(value)}"' for value in values) + "]"


def _cmd_claude(
    config_path: Path,
    pid_path: Path,
    log_path: Path,
    claude_args: list[str],
    install_claude: bool = True,
    dev: bool = False,
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)
    api_settings_path = config_path.parent / DEFAULT_API_SETTINGS_PATH.name
    claude_executable = _ensure_claude_code(install=install_claude)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.claude.closed"
        idle_after_file.unlink(missing_ok=True)
        if dev:
            _print_state("start", "llama server is not running, starting it for Claude Code", "36")
        idle_timeout_seconds = _configured_idle_timeout_seconds(config)
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=idle_timeout_seconds,
            idle_after_file=idle_after_file,
            verbose=dev,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        if dev:
            _print_note(_idle_timeout_note("Claude Code", idle_timeout_seconds))
    else:
        idle_after_file = None
        if dev:
            _print_state("run", "using existing llama server", "32")

    passthrough_args = claude_args
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    plugin_dir = _ensure_claude_tool_plugin(config, config_path, verbose=dev)

    command = [
        claude_executable,
        "--settings",
        str(api_settings_path),
        "--plugin-dir",
        str(plugin_dir),
        *passthrough_args,
    ]
    return_code = subprocess.run(command, check=False).returncode
    if idle_after_file is not None:
        idle_after_file.write_text("closed\n", encoding="utf-8")
    raise SystemExit(return_code)


def _cmd_codex(
    config_path: Path,
    pid_path: Path,
    log_path: Path,
    provider_override: str | None,
    model_override: str | None,
    codex_args: list[str],
    install_codex: bool = True,
    dev: bool = False,
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.codex.closed"
        idle_after_file.unlink(missing_ok=True)
        if dev:
            _print_state("start", "llama server is not running, starting it for Codex", "36")
        idle_timeout_seconds = _configured_idle_timeout_seconds(config)
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=idle_timeout_seconds,
            idle_after_file=idle_after_file,
            verbose=dev,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        if dev:
            _print_note(_idle_timeout_note("Codex", idle_timeout_seconds))
    else:
        idle_after_file = None
        if dev:
            _print_state("run", "using existing llama server", "32")

    provider_name = provider_override or config.codex.provider
    if provider_name not in config.providers:
        available = ", ".join(sorted(config.providers))
        raise SystemExit(f"Unknown Codex provider '{provider_name}'. Available providers: {available}")

    model = resolve_codex_model(
        config,
        provider_name=provider_name,
        model_override=model_override,
    )
    if not model:
        raise SystemExit(codex_model_error(config, provider_name))

    codex_executable = _ensure_codex(install=install_codex, package=config.codex.install_package)
    codex_config_path, model_catalog_path = _write_codex_config(config, provider_name, model)
    codex_plugin_dir, codex_mcp_config_path = _ensure_codex_tool_extension(config, verbose=dev)

    if dev:
        _title("llama codex")
        _print_state("ok", "Codex configuration is ready", "32")
        _kv_rows(
            [
                ("provider", provider_name),
                ("model", model),
                ("profile", config.codex.profile),
                ("config", str(codex_config_path)),
                ("models", str(model_catalog_path)),
                ("tools plugin", str(codex_plugin_dir)),
                ("mcp tools", str(codex_mcp_config_path)),
            ]
        )

    passthrough_args = codex_args
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    env = os.environ.copy()
    env["LLAMA_BRIDGE_API_KEY"] = config.server.auth_token
    command = [codex_executable, "--profile", config.codex.profile, *passthrough_args]
    return_code = subprocess.run(command, check=False, env=env).returncode
    if idle_after_file is not None:
        idle_after_file.write_text("closed\n", encoding="utf-8")
    raise SystemExit(return_code)


def _cmd_copilot(
    config_path: Path,
    pid_path: Path,
    log_path: Path,
    provider_override: str | None,
    model_override: str | None,
    copilot_args: list[str],
    install_copilot: bool = True,
    dev: bool = False,
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.copilot.closed"
        idle_after_file.unlink(missing_ok=True)
        if dev:
            _print_state("start", "llama server is not running, starting it for Copilot CLI", "36")
        idle_timeout_seconds = _configured_idle_timeout_seconds(config)
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=idle_timeout_seconds,
            idle_after_file=idle_after_file,
            verbose=dev,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        if dev:
            _print_note(_idle_timeout_note("Copilot CLI", idle_timeout_seconds))
    else:
        idle_after_file = None
        if dev:
            _print_state("run", "using existing llama server", "32")

    provider_name = provider_override or config.copilot_cli.provider
    if provider_name not in config.providers:
        available = ", ".join(sorted(config.providers))
        raise SystemExit(
            f"Unknown Copilot CLI provider '{provider_name}'. Available providers: {available}"
        )

    model = resolve_copilot_cli_model(
        config,
        provider_name=provider_name,
        model_override=model_override,
    )
    if not model:
        raise SystemExit(copilot_cli_model_error(config, provider_name))

    copilot_executable = _ensure_copilot_cli(
        install=install_copilot,
        package=config.copilot_cli.install_package,
    )
    copilot_mcp_config_path = _ensure_copilot_tool_extension(config, verbose=dev)

    if dev:
        _title("llama copilot")
        _print_state("ok", "Copilot CLI environment is ready", "32")
        _kv_rows(
            [
                ("provider", provider_name),
                ("model", model),
                ("base url", f"{_server_url(config.server.host, config.server.port)}/v1"),
                ("wire api", config.copilot_cli.wire_api),
                ("prompt tokens", config.copilot_cli.max_prompt_tokens),
                ("output tokens", config.copilot_cli.max_output_tokens),
                ("mcp tools", str(copilot_mcp_config_path)),
            ]
        )

    passthrough_args = copilot_args
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    env = os.environ.copy()
    env["COPILOT_PROVIDER_BASE_URL"] = f"{_server_url(config.server.host, config.server.port)}/v1"
    env["COPILOT_PROVIDER_API_KEY"] = config.server.auth_token
    env["COPILOT_PROVIDER_WIRE_API"] = config.copilot_cli.wire_api
    env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] = str(config.copilot_cli.max_prompt_tokens)
    env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] = str(config.copilot_cli.max_output_tokens)
    env["COPILOT_MODEL"] = model
    return_code = subprocess.run(
        [copilot_executable, *passthrough_args],
        check=False,
        env=env,
    ).returncode
    if idle_after_file is not None:
        idle_after_file.write_text("closed\n", encoding="utf-8")
    raise SystemExit(return_code)


def _cmd_opencode(
    config_path: Path,
    pid_path: Path,
    log_path: Path,
    provider_override: str | None,
    model_override: str | None,
    opencode_args: list[str],
    install_opencode: bool = True,
    project_config: bool = False,
    dev: bool = False,
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.opencode.closed"
        idle_after_file.unlink(missing_ok=True)
        if dev:
            _print_state("start", "llama server is not running, starting it for OpenCode", "36")
        idle_timeout_seconds = _configured_idle_timeout_seconds(config)
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=idle_timeout_seconds,
            idle_after_file=idle_after_file,
            verbose=dev,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        if dev:
            _print_note(_idle_timeout_note("OpenCode", idle_timeout_seconds))
    else:
        idle_after_file = None
        if dev:
            _print_state("run", "using existing llama server", "32")

    provider_name = provider_override or config.opencode.provider
    if provider_name not in config.providers:
        available = ", ".join(sorted(config.providers))
        raise SystemExit(
            f"Unknown OpenCode provider '{provider_name}'. Available providers: {available}"
        )

    model = resolve_opencode_model(
        config,
        provider_name=provider_name,
        model_override=model_override,
    )
    if not model:
        raise SystemExit(opencode_model_error(config, provider_name))

    opencode_executable = _ensure_opencode(
        install=install_opencode,
        package=config.opencode.install_package,
    )

    if dev:
        _title("llama opencode")
        _print_state("ok", "OpenCode environment is ready", "32")
        _kv_rows(
            [
                ("provider", provider_name),
                ("model", model),
                ("base url", f"{_server_url(config.server.host, config.server.port)}/v1"),
            ]
        )

    passthrough_args = opencode_args
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = config.server.auth_token
    env["OPENAI_BASE_URL"] = f"{_server_url(config.server.host, config.server.port)}/v1"

    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"llama-bridge/{model}",
        "provider": {
            "llama-bridge": {
                "npm": "@ai-sdk/openai-compatible",
                "name": config.opencode.provider_name,
                "options": {
                    "apiKey": config.server.auth_token,
                    "baseURL": f"{_server_url(config.server.host, config.server.port)}/v1",
                },
                "models": {
                    model: {
                        "name": model,
                        "limit": {
                            "context": config.opencode.context_size,
                            "output": config.opencode.output_tokens,
                        },
                    }
                },
            }
        },
    }
    if config.opencode.small_model:
        opencode_config["small_model"] = f"llama-bridge/{config.opencode.small_model}"
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_config)

    return_code = subprocess.run(
        [opencode_executable] + passthrough_args,
        check=False,
        env=env,
    ).returncode
    if idle_after_file is not None:
        idle_after_file.write_text("closed\n", encoding="utf-8")
    raise SystemExit(return_code)


def _cmd_poolside(
    config_path: Path,
    pid_path: Path,
    log_path: Path,
    poolside_args: list[str],
    provider_override: str | None = None,
    model_override: str | None = None,
    install_poolside: bool = True,
    dev: bool = False,
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    provider_name = provider_override or config.poolside.provider
    if provider_name not in config.providers:
        available = ", ".join(sorted(config.providers))
        raise SystemExit(
            f"Unknown Poolside provider '{provider_name}'. Available providers: {available}"
        )

    provider = config.providers[provider_name]
    model = model_override or config.poolside.model or provider.default_model
    if not model:
        raise SystemExit(
            "Poolside model is not configured. Set poolside.model, set that provider's default_model, "
            "or pass `llama poolside --model ...`."
        )

    poolside_api_url = _poolside_api_url(config)
    uses_llama_bridge = bool(
        poolside_api_url and _is_llama_bridge_poolside_url(config, poolside_api_url)
    )
    standalone_base_url = _poolside_standalone_base_url(config, poolside_api_url)
    poolside_api_key = _poolside_auth_token(config, poolside_api_url)
    if uses_llama_bridge:
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            idle_after_file = pid_path.parent / "llama.poolside.closed"
            idle_after_file.unlink(missing_ok=True)
            if dev:
                _print_state("start", "llama server is not running, starting it for Poolside", "36")
            idle_timeout_seconds = _configured_idle_timeout_seconds(config)
            _cmd_start(
                config_path,
                pid_path,
                log_path,
                idle_timeout_seconds=idle_timeout_seconds,
                idle_after_file=idle_after_file,
                verbose=dev,
            )
            server_running, _running_url = _server_is_running(config_path, pid_path)
            if not server_running:
                raise SystemExit(f"llama server failed to start, see log: {log_path}")
            if dev:
                _print_note(_idle_timeout_note("Poolside", idle_timeout_seconds))
        else:
            idle_after_file = None
            if dev:
                _print_state("run", "using existing llama server", "32")
    else:
        idle_after_file = None

    poolside_executable = _ensure_poolside(
        install=install_poolside,
        install_command=config.poolside.install_command,
        windows_install_command=config.poolside.windows_install_command,
    )
    poolside_agent_config_path = _write_poolside_agent_config()
    poolside_settings_path = _write_poolside_config(config)
    poolside_skill_path = _write_poolside_bridge_skill(config)

    if dev:
        _title("llama poolside")
        _print_state("ok", "Poolside environment is ready", "32")
        _kv_rows(
            [
                ("provider", provider_name),
                ("model", model),
                ("api url", poolside_api_url or standalone_base_url or "poolside default"),
                ("auth", "configured" if poolside_api_key else "stored login or interactive setup"),
                ("agent config", str(poolside_agent_config_path)),
                ("config", str(poolside_settings_path)),
                ("skill", str(poolside_skill_path)),
                ("command", poolside_executable),
            ]
        )

    passthrough_args = poolside_args
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    command = [poolside_executable]
    if "--model" not in passthrough_args and "-m" not in passthrough_args:
        command.extend(["--model", model])
    command.extend(passthrough_args)

    env = os.environ.copy()
    if poolside_api_url:
        env["POOLSIDE_API_URL"] = poolside_api_url
    if uses_llama_bridge and standalone_base_url:
        env["POOLSIDE_STANDALONE_BASE_URL"] = standalone_base_url
    if poolside_api_key:
        env["POOLSIDE_API_KEY"] = poolside_api_key
    if config.poolside.token and not str(config.poolside.token).startswith("${"):
        env["POOLSIDE_TOKEN"] = str(config.poolside.token)
    env["LLAMA_BRIDGE_API_KEY"] = config.server.auth_token
    if uses_llama_bridge and standalone_base_url:
        env["OPENAI_API_KEY"] = "ollama"
        env["OPENAI_BASE_URL"] = standalone_base_url

    return_code = subprocess.run(
        command,
        check=False,
        env=env,
    ).returncode
    if idle_after_file is not None:
        idle_after_file.write_text("closed\n", encoding="utf-8")
    raise SystemExit(return_code)


def _cmd_poolside_acp_proxy(poolside_acp_args: list[str]) -> None:
    _configure_proxy_stdio()
    poolside_executable = _find_poolside_executable()
    if not poolside_executable:
        raise SystemExit("Poolside CLI was not found. Install Poolside and try again.")
    passthrough_args = poolside_acp_args
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]
    command = [poolside_executable, "acp", *passthrough_args]
    env = _poolside_acp_proxy_env()
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if proc.stdin is None or proc.stdout is None or proc.stderr is None:
        raise SystemExit("Could not start Poolside ACP proxy.")

    pending_session_requests: set[Any] = set()
    pending_lock = threading.Lock()

    def forward_stderr() -> None:
        for line in proc.stderr or []:
            sys.stderr.write(line)
            sys.stderr.flush()

    def forward_client_to_agent() -> None:
        try:
            for line in sys.stdin:
                message = _json_line(line)
                if isinstance(message, dict) and message.get("method") in {"session/new", "session/load"}:
                    message_id = message.get("id")
                    if message_id is not None:
                        with pending_lock:
                            pending_session_requests.add(message_id)
                if isinstance(message, dict):
                    line = json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"
                proc.stdin.write(line)
                proc.stdin.flush()
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    def forward_agent_to_client() -> None:
        for line in proc.stdout or []:
            message = _json_line(line)
            if isinstance(message, dict):
                message = _with_poolside_bridge_commands(message)
                line = json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"
            sys.stdout.write(line)
            sys.stdout.flush()
            if not isinstance(message, dict):
                continue
            message_id = message.get("id")
            with pending_lock:
                is_session_response = message_id in pending_session_requests
                if is_session_response:
                    pending_session_requests.discard(message_id)
            if not is_session_response:
                continue
            result = message.get("result")
            session_id = result.get("sessionId") if isinstance(result, dict) else None
            if isinstance(session_id, str) and session_id:
                update = _poolside_available_commands_update(session_id)
                sys.stdout.write(json.dumps(update, separators=(",", ":")) + "\n")
                sys.stdout.flush()

    stderr_thread = threading.Thread(target=forward_stderr, daemon=True)
    stdin_thread = threading.Thread(target=forward_client_to_agent, daemon=True)
    stderr_thread.start()
    stdin_thread.start()
    try:
        forward_agent_to_client()
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
    raise SystemExit(proc.wait())


def _configure_proxy_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _poolside_acp_proxy_env() -> dict[str, str]:
    env = os.environ.copy()
    try:
        config_path = DEFAULT_CONFIG_PATH
        if config_path.exists():
            config = load_config(config_path)
            poolside_api_url = _poolside_api_url(config)
            standalone_base_url = _poolside_standalone_base_url(config, poolside_api_url)
            poolside_api_key = _poolside_auth_token(config, poolside_api_url)
            if poolside_api_url and "POOLSIDE_API_URL" not in env:
                env["POOLSIDE_API_URL"] = poolside_api_url
            if poolside_api_key and "POOLSIDE_API_KEY" not in env:
                env["POOLSIDE_API_KEY"] = poolside_api_key
            if config.poolside.token and not str(config.poolside.token).startswith("${") and "POOLSIDE_TOKEN" not in env:
                env["POOLSIDE_TOKEN"] = str(config.poolside.token)
            if "LLAMA_BRIDGE_API_KEY" not in env:
                env["LLAMA_BRIDGE_API_KEY"] = config.server.auth_token
            if poolside_api_url and _is_llama_bridge_poolside_url(config, poolside_api_url) and standalone_base_url:
                env.setdefault("POOLSIDE_STANDALONE_BASE_URL", standalone_base_url)
                env.setdefault("OPENAI_API_KEY", "ollama")
                env.setdefault("OPENAI_BASE_URL", standalone_base_url)
    except Exception:
        pass
    return env


def _json_line(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _poolside_available_commands_update(session_id: str) -> dict[str, Any]:
    available_commands = _poolside_bridge_available_commands()
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "available_commands_update",
                "availableCommands": available_commands,
            },
        },
    }


def _poolside_bridge_available_commands() -> list[dict[str, Any]]:
    commands = [
        ("model", "Open the agent/model selector.", "model name"),
        ("mode", "List or switch session mode.", "mode name"),
        ("plan", "Switch to plan mode.", None),
        ("clear", "Clear conversation history and free context.", None),
        ("rewind", "Roll back to a previous turn.", None),
        ("share", "Get a trajectory sharing link.", None),
        ("skills", "Refresh and list available skills.", None),
        ("usage", "Show token usage for the current session.", None),
        ("serp", "Search the web with SerpAPI through llama bridge.", "search query"),
        ("tavily", "Search the web with Tavily through llama bridge.", "search query"),
        ("web", "Search the web through llama bridge.", "search query"),
        ("image", "Find sourced image candidates through llama bridge.", "image query"),
        ("wiki", "Search Wikipedia through llama bridge.", "Wikipedia query"),
        ("manim", "Generate a short Manim animation video from text.", "animation prompt"),
    ]
    available_commands = []
    for name, description, hint in commands:
        command: dict[str, Any] = {"name": name, "description": description}
        if hint:
            command["input"] = {"hint": hint}
        available_commands.append(command)
    return available_commands


def _with_poolside_bridge_commands(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("method") != "session/update":
        return message
    params = message.get("params")
    if not isinstance(params, dict):
        return message
    update = params.get("update")
    if not isinstance(update, dict) or update.get("sessionUpdate") != "available_commands_update":
        return message
    commands = update.get("availableCommands")
    if not isinstance(commands, list):
        return message

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for command in [*commands, *_poolside_bridge_available_commands()]:
        if not isinstance(command, dict):
            continue
        name = command.get("name")
        if not isinstance(name, str) or not name or name in seen:
            continue
        seen.add(name)
        merged.append(command)

    return {
        **message,
        "params": {
            **params,
            "update": {
                **update,
                "availableCommands": merged,
            },
        },
    }




def _write_poolside_agent_config() -> Path:
    config_path = Path(os.path.expanduser("~/.config/poolside/pool.json"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    except json.JSONDecodeError:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    servers_value = existing.get("agent_servers")
    servers = dict(servers_value) if isinstance(servers_value, dict) else {}
    direct = servers.get("llama_bridge_poolside_direct")
    if not isinstance(direct, dict):
        direct = {"command": "{{SELF}}", "args": ["acp"]}
    servers["llama_bridge_poolside_direct"] = direct
    command, args = _llama_self_command()
    servers["default"] = {
        "command": command,
        "args": [*args, "poolside-acp-proxy"],
    }
    existing["agent_servers"] = servers
    _write_json(config_path, existing)
    return config_path


def _write_poolside_config(config) -> Path:
    import yaml

    settings_path = _resolved_poolside_settings_path(config)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    existing = yaml.safe_load(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    if not isinstance(existing, dict):
        existing = {}

    poolside_api_url = _poolside_api_url(config)
    pool_section_value = existing.get("pool")
    pool_section = dict(pool_section_value) if isinstance(pool_section_value, dict) else {}
    pool_section.pop("api_key", None)
    pool_section.pop("token", None)
    if poolside_api_url and not _is_llama_bridge_poolside_url(config, poolside_api_url):
        pool_section["api_url"] = poolside_api_url
        existing["pool"] = pool_section
    else:
        if _is_llama_bridge_poolside_url(config, pool_section.get("api_url")):
            pool_section.pop("api_url", None)
            if pool_section:
                existing["pool"] = pool_section
            else:
                existing.pop("pool", None)
        if _is_llama_bridge_poolside_url(config, existing.get("api_url")):
            existing.pop("api_url", None)

    mcp_servers = dict(existing.get("mcp_servers") or {})
    command, args = _mcp_server_command()
    mcp_servers["llama_bridge_tools"] = {
        "command": command,
        "args": args,
        "env": _mcp_server_env(config),
    }
    existing["mcp_servers"] = mcp_servers

    tools_value = existing.get("tools")
    tools = dict(tools_value) if isinstance(tools_value, dict) else {}
    shell_value = tools.get("shell")
    shell_tool = dict(shell_value) if isinstance(shell_value, dict) else {}
    shell_tool["disabled"] = False
    tools["shell"] = shell_tool
    for legacy_key in ("enabled", "allow_shell", "allow_bash"):
        tools.pop(legacy_key, None)
    existing["tools"] = tools

    try:
        settings_path.write_text(
            yaml.safe_dump(existing, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    except PermissionError:
        _print_state(
            "warn",
            f"Could not update Poolside settings at {settings_path}; continuing with environment auth",
            "33",
        )
    return settings_path


def _write_poolside_bridge_skill(config) -> Path:
    skills_dir = _resolved_poolside_settings_path(config).parent / "skills" / "llama-bridge-tools"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skills_dir / "SKILL.md"
    try:
        skill_path.write_text(_poolside_bridge_tools_skill(), encoding="utf-8")
    except PermissionError:
        _print_state(
            "warn",
            f"Could not update Poolside skill at {skill_path}; MCP tools remain configured",
            "33",
        )
    return skill_path


def _poolside_bridge_tools_skill() -> str:
    return """---
name: llama-bridge-tools
description: Use when Poolside needs llama bridge MCP tools for current web search, SerpAPI, Tavily, source verification, image search, Wikipedia, Manim animation videos, weather, or date/time lookups. Use when the user types shortcut-style prompts such as /serp, /tavily, /web, /image, /wiki, or /manim.
---

# Llama Bridge Tools

Use the `llama_bridge_tools` MCP server when a task needs current information,
source verification, image candidates, weather, Wikipedia, or local date/time
lookups.

Poolside exposes custom workflows through `/skills`. If the user types a
shortcut-like prompt directly, treat it as an instruction to use the matching
MCP tool:

- `/serp`: use `serpapi_search`.
- `/tavily`: use `tavily_search`.
- `/web`: use the best available bridge web search tool.
- `/image`: use `image_research`.
- `/wiki`: use `wikipedia_search`; follow with `wikipedia_page` when a
  specific page is needed.
- `/manim`: use `manim_render` to create a short Manim Community animation
  video from the user's text. Return the generated scene path and video path.
  If Manim is missing, show the install guidance returned by the tool.
Prefer the highest-level bridge tool that fits the task:

- `source_research` for cited factual research and evidence gathering.
- `image_research` for compact sourced image candidates.
- `tavily_search` or `serpapi_search` for current web results.
- `wikipedia_search` and `wikipedia_page` for encyclopedia context.
- `manim_render` for short Python Manim animation videos.
- `weather_current` for live weather.
- `datetime_now` for current time or timezone questions.

The MCP server calls the local llama bridge HTTP tool endpoints, so the llama
server must be running for tool calls to succeed.
"""


def _poolside_api_url(config) -> str | None:
    api_url = str(config.poolside.api_url or "").strip()
    if not api_url or api_url.startswith("${"):
        return None
    api_url = api_url.rstrip("/")
    bridge_v1_urls = {
        f"{_server_url(config.server.host, config.server.port).rstrip('/')}/v1",
        f"http://127.0.0.1:{config.server.port}/v1",
        f"http://localhost:{config.server.port}/v1",
    }
    if api_url in bridge_v1_urls:
        return api_url.removesuffix("/v1")
    return api_url


def _configured_poolside_api_key(config) -> str | None:
    for value in (config.poolside.api_key, config.poolside.token):
        key = str(value or "").strip()
        if key and not key.startswith("${"):
            return key
    return None


def _poolside_auth_token(config, api_url: str | None) -> str | None:
    if api_url and _is_llama_bridge_poolside_url(config, api_url):
        return "ollama"
    return _configured_poolside_api_key(config)


def _poolside_standalone_base_url(config, api_url: str | None) -> str | None:
    if not api_url:
        return None
    return api_url.rstrip("/")


def _is_llama_bridge_poolside_url(config, value: Any) -> bool:
    api_url = str(value or "").strip().rstrip("/")
    if not api_url:
        return False
    bridge_base = f"{_server_url(config.server.host, config.server.port)}/v1".rstrip("/")
    bridge_root = _server_url(config.server.host, config.server.port).rstrip("/")
    localhost_base = f"http://127.0.0.1:{config.server.port}/v1"
    localhost_root = f"http://127.0.0.1:{config.server.port}"
    return api_url in {
        bridge_base,
        bridge_root,
        localhost_base,
        localhost_root,
        f"http://localhost:{config.server.port}/v1",
        f"http://localhost:{config.server.port}",
    }


def _resolved_poolside_settings_path(config) -> Path:
    raw_path = str(config.poolside.config_path or "~/.config/poolside/settings.yaml")
    return Path(os.path.expanduser(raw_path))


def _cmd_cli(
    config_path: Path,
    list_only: bool = False,
    support_only: bool = False,
    remove_target: str | None = None,
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)
    targets = _cli_targets(config)

    if support_only:
        _print_supported_cli_targets(targets)
        return

    if remove_target is not None:
        if remove_target == "__prompt__":
            remove_target = _prompt_cli_target_name(targets)
        if not remove_target:
            raise SystemExit(1)
        _remove_cli_target(targets, remove_target)
        return

    _print_cli_targets(targets, heading="llama cli")
    if not list_only:
        _print_note("Use `llama cli --rm` to choose one to remove.")


def _write_codex_config(config, provider_name: str, model: str) -> tuple[Path, Path]:
    codex_config_path = Path(os.path.expanduser(config.codex.config_path))
    codex_config_path.parent.mkdir(parents=True, exist_ok=True)
    existing = codex_config_path.read_text(encoding="utf-8") if codex_config_path.exists() else ""
    model_catalog_path = codex_config_path.parent / "llama_bridge_models.json"
    _write_json(model_catalog_path, _codex_model_catalog(model))

    provider_section = "\n".join(
        [
            'name = "llama bridge"',
            f'base_url = "{_server_url(config.server.host, config.server.port)}/v1"',
            'env_key = "LLAMA_BRIDGE_API_KEY"',
            'wire_api = "responses"',
        ]
    )
    profile_section = "\n".join(
        [
            f'model = "{_toml_escape(model)}"',
            'model_provider = "llama_bridge"',
            "model_context_window = 65536",
            f'model_catalog_json = "{_toml_escape(str(model_catalog_path))}"',
        ]
    )
    updated = _replace_toml_section(
        existing,
        "model_providers.llama_bridge",
        provider_section,
    )
    updated = _replace_toml_section(
        updated,
        f"profiles.{config.codex.profile}",
        profile_section,
    )
    if updated and not updated.endswith("\n"):
        updated += "\n"
    codex_config_path.write_text(updated, encoding="utf-8")
    return codex_config_path, model_catalog_path


def _codex_model_catalog(model: str) -> dict:
    return {
        "models": [
            {
                "slug": model,
                "display_name": model,
                "description": "Llama bridge model",
                "context_window": 65536,
                "max_context_window": 65536,
                "auto_compact_token_limit": None,
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {
                        "effort": "low",
                        "description": "Fast responses with lighter reasoning",
                    },
                    {
                        "effort": "medium",
                        "description": "Balanced reasoning",
                    },
                    {
                        "effort": "high",
                        "description": "Greater reasoning depth",
                    },
                    {
                        "effort": "xhigh",
                        "description": "Extra reasoning depth",
                    },
                ],
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 100,
                "input_modalities": ["text"],
                "supports_parallel_tool_calls": True,
                "supports_reasoning_summaries": False,
                "default_reasoning_summary": "none",
                "support_verbosity": False,
                "default_verbosity": "medium",
                "apply_patch_tool_type": "freeform",
                "web_search_tool_type": "text_and_image",
                "supports_image_detail_original": False,
                "truncation_policy": {"mode": "tokens", "limit": 10000},
                "experimental_supported_tools": [],
                "supports_search_tool": True,
                "additional_speed_tiers": [],
                "base_instructions": "",
            }
        ]
    }


def _replace_toml_section(content: str, section: str, body: str) -> str:
    replacement = f"[{section}]\n{body}\n"
    pattern = re.compile(
        rf"(?ms)^\[{re.escape(section)}\]\r?\n.*?(?=^\[[^\]]+\]\s*$|\Z)"
    )
    if pattern.search(content):
        return pattern.sub(lambda _match: replacement, content).strip() + "\n"
    separator = "\n\n" if content.strip() else ""
    return content.rstrip() + separator + replacement


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _ensure_claude_code(install: bool = True) -> str:
    claude_executable = _find_claude_executable()
    if claude_executable:
        return claude_executable
    if not install:
        raise SystemExit("Claude Code CLI was not found. Install Claude Code and try again.")

    print("Claude Code CLI was not found, installing it now...")
    _ensure_node_and_npm()
    npm_executable = _find_npm_executable()
    if not npm_executable:
        raise SystemExit("npm was not found after setup. Install Node.js 18+ and try again.")

    process = subprocess.run(
        [npm_executable, "install", "-g", "@anthropic-ai/claude-code"],
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit("Claude Code install failed. Try `npm install -g @anthropic-ai/claude-code`.")

    claude_executable = _find_claude_executable()
    if not claude_executable:
        raise SystemExit("Claude Code installed, but `claude` is not on PATH. Restart your terminal and try again.")
    return claude_executable


def _ensure_pi(install: bool = True, package: str = "@mariozechner/pi-coding-agent") -> str:
    pi_executable = _find_pi_executable()
    if pi_executable:
        return pi_executable
    if not install:
        raise SystemExit("Pi CLI was not found. Install Pi and try again.")

    _print_state("install", "Pi CLI was not found, installing it now", "36")
    _ensure_node_and_npm()
    npm_executable = _find_npm_executable()
    if not npm_executable:
        raise SystemExit("npm was not found after setup. Install Node.js 18+ and try again.")

    process = subprocess.run(
        [npm_executable, "install", "-g", package],
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit(f"Pi install failed. Try `npm install -g {package}`.")

    pi_executable = _find_pi_executable()
    if not pi_executable:
        raise SystemExit("Pi installed, but `pi` is not on PATH. Restart your terminal and try again.")
    return pi_executable


def _ensure_pi_shell(install: bool = True) -> str | None:
    if os.name != "nt":
        return None

    bash_path = _find_git_bash()
    if bash_path:
        return bash_path
    if not install:
        raise SystemExit(
            "Git Bash was not found. Install Git for Windows or set shellPath in Pi settings.json."
        )
    if not shutil.which("winget"):
        raise SystemExit(
            "Git Bash was not found and winget is not available. Install Git for Windows "
            "or set shellPath in Pi settings.json."
        )

    _print_state("install", "Git Bash was not found, installing Git for Windows with winget", "36")
    process = subprocess.run(
        [
            "winget",
            "install",
            "--id",
            "Git.Git",
            "-e",
            "--accept-source-agreements",
            "--accept-package-agreements",
        ],
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit("Git for Windows install failed. Try `winget install --id Git.Git -e`.")

    bash_path = _find_git_bash()
    if not bash_path:
        raise SystemExit(
            "Git for Windows was installed, but Git Bash was not found. Restart your terminal "
            "or set shellPath in Pi settings.json."
        )
    return bash_path


def _ensure_codex(install: bool = True, package: str = "@openai/codex") -> str:
    codex_executable = _find_codex_executable()
    if codex_executable:
        return codex_executable
    if not install:
        raise SystemExit("Codex CLI was not found. Install Codex and try again.")

    _print_state("install", "Codex CLI was not found, installing it now", "36")
    _ensure_node_and_npm()
    npm_executable = _find_npm_executable()
    if not npm_executable:
        raise SystemExit("npm was not found after setup. Install Node.js 18+ and try again.")

    process = subprocess.run(
        [npm_executable, "install", "-g", package],
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit(f"Codex install failed. Try `npm install -g {package}`.")

    codex_executable = _find_codex_executable()
    if not codex_executable:
        raise SystemExit("Codex installed, but `codex` is not on PATH. Restart your terminal and try again.")
    return codex_executable


def _ensure_copilot_cli(install: bool = True, package: str = "@github/copilot") -> str:
    copilot_executable = _find_copilot_executable()
    if copilot_executable:
        return copilot_executable
    if not install:
        raise SystemExit("GitHub Copilot CLI was not found. Install Copilot CLI and try again.")

    _print_state("install", "GitHub Copilot CLI was not found, installing it now", "36")
    _ensure_node_and_npm()
    npm_executable = _find_npm_executable()
    if not npm_executable:
        raise SystemExit("npm was not found after setup. Install Node.js 18+ and try again.")

    process = subprocess.run(
        [npm_executable, "install", "-g", package],
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit(f"Copilot CLI install failed. Try `npm install -g {package}`.")

    copilot_executable = _find_copilot_executable()
    if not copilot_executable:
        raise SystemExit(
            "Copilot CLI installed, but `copilot` is not on PATH. Restart your terminal and try again."
        )
    return copilot_executable


def _ensure_pi_extensions(config, verbose: bool = True) -> list[Path]:
    paths: list[Path] = []
    web_tools = _ensure_pi_web_tools(config, verbose=verbose)
    if web_tools is not None:
        paths.append(web_tools)
    return paths


def _ensure_pi_web_tools(config, verbose: bool = True) -> Path | None:
    if not config.pi.web_search:
        return
    config_dir = Path(os.path.expanduser(config.pi.config_dir))
    extension_dir = config_dir / "extensions" / "llama_bridge_web_tools"
    extension_path = extension_dir / "index.ts"
    extension_dir.mkdir(parents=True, exist_ok=True)
    extension_path.write_text(_pi_web_tools_extension(config), encoding="utf-8")

    settings_path = config_dir / "settings.json"
    settings_data = _read_json_object(settings_path)
    packages = [
        package
        for package in settings_data.get("packages", [])
        if package != "npm:@ollama/pi-web-search"
    ]
    if packages:
        settings_data["packages"] = packages
    else:
        settings_data.pop("packages", None)
    _write_json(settings_path, settings_data)
    if verbose:
        _print_state("ok", f"Pi web search tools: {extension_path}", "32")
    return extension_path


def _pi_web_tools_extension(config) -> str:
    bridge_url = _server_url(config.server.host, config.server.port)
    api_key = config.server.auth_token
    timeout_seconds = 10.0
    timeout_ms = int(timeout_seconds * 1000)
    return f"""import type {{ ExtensionAPI }} from "@mariozechner/pi-coding-agent";
import {{ Type }} from "typebox";

const BRIDGE_URL = {json.dumps(bridge_url)};
const API_KEY = {json.dumps(api_key)};
const BRIDGE_TOOL_TIMEOUT_MS = {timeout_ms};

function bridgeTimeoutSignal(parentSignal: AbortSignal) {{
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(new Error("llama bridge tool timed out after {timeout_seconds:g} seconds")), BRIDGE_TOOL_TIMEOUT_MS);
  const abort = () => controller.abort(parentSignal.reason);
  if (parentSignal.aborted) abort();
  else parentSignal.addEventListener("abort", abort, {{ once: true }});
  return {{ signal: controller.signal, cleanup: () => {{
    clearTimeout(timeout);
    parentSignal.removeEventListener("abort", abort);
  }} }};
}}

async function postJson(path: string, body: unknown, signal: AbortSignal) {{
  const timed = bridgeTimeoutSignal(signal);
  let response: Response;
  try {{
    response = await fetch(`${{BRIDGE_URL}}${{path}}`, {{
      method: "POST",
      headers: {{
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
      }},
      body: JSON.stringify(body),
      signal: timed.signal,
    }});
  }} catch (error) {{
    return {{ ok: false, error: String(error), timeout_budget_seconds: {timeout_seconds:g}, path }};
  }} finally {{
    timed.cleanup();
  }}
  const text = await response.text().catch(() => "");
  let data: any = null;
  try {{
    data = text ? JSON.parse(text) : null;
  }} catch (error) {{
    data = null;
  }}
  if (!response.ok) {{
    if (data && typeof data === "object") return data;
    throw new Error(`llama bridge ${{path}} failed (${{response.status}}): ${{text || response.statusText}}`);
  }}
  return data;
}}

async function callBridgeTool(name: string, args: unknown, signal: AbortSignal) {{
  const data = await postJson(`/api/tools/${{name}}`, args, signal);
  return data.data ?? data.result?.data ?? data.result ?? data;
}}

function pretty(value: unknown) {{
  return JSON.stringify(value, null, 2);
}}

function toolErrorText(toolName: string, result: any) {{
  if (!result || result.ok !== false) return null;
  const error = result.error?.message ?? result.error ?? "unknown error";
  return `${{toolName}} failed: ${{String(error)}}\\n\\n${{pretty(result)}}`;
}}

function sendAutoFollowUp(piApi: any, text: string) {{
  try {{
    if (typeof piApi?.sendUserMessage !== "function") return false;
    setTimeout(() => {{
      try {{
        piApi.sendUserMessage(text, {{ deliverAs: "followUp" }});
      }} catch (error) {{
        try {{ piApi.sendUserMessage(text); }} catch (_ignored) {{}}
      }}
    }}, 250);
    return true;
  }} catch (error) {{
    return false;
  }}
}}

function formatSearchResults(results: any[]) {{
  return results.map((item: any, index: number) => {{
    const title = item.title ?? item.name ?? item.heading ?? "Untitled";
    const url = item.url ?? item.link ?? item.source_url ?? "";
    const snippet = item.snippet ?? item.content ?? item.summary ?? item.extract ?? "";
    return `${{index + 1}}. ${{title}}\\n   URL: ${{url}}\\n   ${{snippet}}`;
  }}).join("\\n\\n");
}}

function formatImageResults(result: any) {{
  const images = Array.isArray(result.images) ? result.images : [];
  const examples = Array.isArray(result.markdown_examples) ? result.markdown_examples : [];
  const lines = images.map((image: any, index: number) => [
    `${{index + 1}}. ${{image.title ?? "Untitled image"}}`,
    `   Image: ${{image.image_url ?? ""}}`,
    image.thumbnail ? `   Thumbnail: ${{image.thumbnail}}` : "",
    image.source_url ? `   Source: ${{image.source_url}}` : "",
    image.provider ? `   Provider: ${{image.provider}}` : "",
  ].filter(Boolean).join("\\n"));
  if (examples.length) {{
    if (result.markdown_css) {{
      lines.push("", "Compact image CSS:", result.markdown_css);
    }}
    lines.push("", "Markdown examples:");
    lines.push('<div class="image-grid">');
    for (const item of examples.slice(0, 3)) {{
      lines.push(item.markdown ?? pretty(item));
      if (item.source_note) lines.push(item.source_note);
    }}
    lines.push("</div>");
  }}
  if (Array.isArray(result.guardrails) && result.guardrails.length) {{
    lines.push("", "Guardrails:", ...result.guardrails.map((item: string) => `- ${{item}}`));
  }}
  if (Array.isArray(result.errors) && result.errors.length) {{
    lines.push("", "Provider errors:", pretty(result.errors));
  }}
  return lines.join("\\n");
}}

export default function (pi: ExtensionAPI) {{
  pi.registerTool({{
    name: "web_search",
    label: "Web Search",
    description: "Search the web with Ollama through the local llama bridge.",
    parameters: Type.Object({{
      query: Type.String({{ description: "The search query to execute" }}),
      max_results: Type.Optional(Type.Number({{ description: "Maximum number of search results to return", default: 5 }})),
    }}),
    async execute(_toolCallId, params, signal) {{
      const data = await postJson("/api/web_search", {{
        query: params.query,
        max_results: params.max_results ?? 5,
      }}, signal);
      const results = Array.isArray(data.results) ? data.results : [];
      const text = results.map((r: any, i: number) =>
        `${{i + 1}}. ${{r.title ?? "Untitled"}}\\n   URL: ${{r.url ?? ""}}\\n   ${{r.content ?? ""}}`
      ).join("\\n\\n");
      return {{
        content: [{{ type: "text", text: text || "No results found." }}],
        details: data,
      }};
    }},
  }});

  pi.registerTool({{
    name: "web_fetch",
    label: "Web Fetch",
    description: "Fetch a web page with Ollama through the local llama bridge.",
    parameters: Type.Object({{
      url: Type.String({{ description: "URL to fetch and extract content from" }}),
    }}),
    async execute(_toolCallId, params, signal) {{
      const data = await postJson("/api/web_fetch", {{ url: params.url }}, signal);
      if (data?.error || data?.ok === false) {{
        const error = data?.error?.message ?? data?.error?.error ?? data?.error ?? data?.result?.error ?? "Fetch failed";
        return {{
          content: [{{ type: "text", text: `Fetch failed for ${{params.url}}: ${{String(error)}}` }}],
          details: data,
        }};
      }}
      const links = Array.isArray(data.links) ? data.links : [];
      const text = [
        `Title: ${{data.title ?? params.url}}`,
        "",
        "Content:",
        data.content ?? "",
        "",
        `Links found: ${{links.length}}`,
        ...links.slice(0, 10).map((link: string) => `  - ${{link}}`),
      ].join("\\n");
      return {{
        content: [{{ type: "text", text }}],
        details: data,
      }};
    }},
  }});

  pi.registerTool({{
    name: "datetime_now",
    label: "Current Time",
    description: "Get the current date and time for a timezone through the local llama bridge.",
    parameters: Type.Object({{
      timezone: Type.Optional(Type.String({{ description: "IANA timezone such as UTC, Asia/Calcutta, or America/New_York." }})),
      country: Type.Optional(Type.String({{ description: "Country name to choose the local timezone." }})),
    }}),
    async execute(_toolCallId, params, signal) {{
      const result = await callBridgeTool("datetime_now", {{
        timezone: params.timezone,
        country: params.country,
      }}, signal);
      return {{
        content: [{{ type: "text", text: pretty(result) }}],
        details: result,
      }};
    }},
  }});

  pi.registerTool({{
    name: "wikipedia_search",
    label: "Wikipedia Search",
    description: "Search Wikipedia pages through the local llama bridge.",
    parameters: Type.Object({{
      query: Type.String({{ description: "Search query." }}),
      limit: Type.Optional(Type.Number({{ description: "Maximum results to return.", default: 5 }})),
      language: Type.Optional(Type.String({{ description: "Wikipedia language code.", default: "en" }})),
    }}),
    async execute(_toolCallId, params, signal) {{
      const result = await callBridgeTool("wikipedia_search", {{
        query: params.query,
        limit: params.limit ?? 5,
        language: params.language ?? "en",
      }}, signal);
      const results = Array.isArray(result.results) ? result.results : [];
      const text = results.map((item: any, index: number) =>
        `${{index + 1}}. ${{item.title ?? "Untitled"}}\\n   URL: ${{item.url ?? ""}}\\n   ${{item.snippet ?? ""}}`
      ).join("\\n\\n");
      return {{
        content: [{{ type: "text", text: text || pretty(result) }}],
        details: result,
      }};
    }},
  }});

  pi.registerTool({{
    name: "wikipedia_page",
    label: "Wikipedia Page",
    description: "Fetch a Wikipedia page summary through the local llama bridge.",
    parameters: Type.Object({{
      title: Type.String({{ description: "Wikipedia page title." }}),
      language: Type.Optional(Type.String({{ description: "Wikipedia language code.", default: "en" }})),
    }}),
    async execute(_toolCallId, params, signal) {{
      const result = await callBridgeTool("wikipedia_page", {{
        title: params.title,
        language: params.language ?? "en",
      }}, signal);
      const text = [
        `Title: ${{result.title ?? params.title}}`,
        result.url ? `URL: ${{result.url}}` : "",
        "",
        result.extract ?? pretty(result),
      ].filter(Boolean).join("\\n");
      return {{
        content: [{{ type: "text", text }}],
        details: result,
      }};
    }},
  }});

  pi.registerTool({{
    name: "weather_current",
    label: "Current Weather",
    description: "Get current weather by location or coordinates through the local llama bridge.",
    parameters: Type.Object({{
      location: Type.Optional(Type.String({{ description: "Place name to geocode." }})),
      latitude: Type.Optional(Type.Number({{ description: "WGS84 latitude." }})),
      longitude: Type.Optional(Type.Number({{ description: "WGS84 longitude." }})),
      temperature_unit: Type.Optional(Type.String({{ description: "celsius or fahrenheit.", default: "celsius" }})),
      wind_speed_unit: Type.Optional(Type.String({{ description: "kmh, ms, mph, or kn.", default: "kmh" }})),
    }}),
    async execute(_toolCallId, params, signal) {{
      const result = await callBridgeTool("weather_current", {{
        location: params.location,
        latitude: params.latitude,
        longitude: params.longitude,
        temperature_unit: params.temperature_unit ?? "celsius",
        wind_speed_unit: params.wind_speed_unit ?? "kmh",
      }}, signal);
      return {{
        content: [{{ type: "text", text: pretty(result) }}],
        details: result,
      }};
    }},
  }});

  pi.registerTool({{
    name: "serpapi_search",
    label: "SerpAPI Search",
    description: "Search current web results through SerpAPI via the local llama bridge.",
    parameters: Type.Object({{
      query: Type.String({{ description: "Search query." }}),
      engine: Type.Optional(Type.String({{ description: "Search engine: google, bing, or baidu.", default: "google" }})),
      location: Type.Optional(Type.String({{ description: "Optional location for localized results." }})),
      hl: Type.Optional(Type.String({{ description: "Language code for results." }})),
      gl: Type.Optional(Type.String({{ description: "Country code for results." }})),
      num: Type.Optional(Type.Number({{ description: "Number of results to return.", default: 5 }})),
    }}),
    async execute(_toolCallId, params, signal) {{
      const result = await callBridgeTool("serpapi_search", {{
        query: params.query,
        engine: params.engine ?? "google",
        location: params.location,
        hl: params.hl,
        gl: params.gl,
        num: params.num ?? 5,
      }}, signal);
      const errorText = toolErrorText("serpapi_search", result);
      if (errorText) {{
        return {{
          content: [{{ type: "text", text: errorText }}],
          details: result,
        }};
      }}
      const results = Array.isArray(result.organic_results) ? result.organic_results : [];
      const answerBox = result.answer_box ? `Answer box:\\n${{pretty(result.answer_box)}}\\n\\n` : "";
      return {{
        content: [{{ type: "text", text: answerBox + (formatSearchResults(results) || pretty(result)) }}],
        details: result,
      }};
    }},
  }});

  pi.registerTool({{
    name: "tavily_search",
    label: "Tavily Search",
    description: "Search current factual web results through Tavily via the local llama bridge.",
    parameters: Type.Object({{
      query: Type.String({{ description: "Search query." }}),
      search_depth: Type.Optional(Type.String({{ description: "basic or advanced.", default: "basic" }})),
      topic: Type.Optional(Type.String({{ description: "general, news, or finance.", default: "general" }})),
      max_results: Type.Optional(Type.Number({{ description: "Maximum results to return.", default: 5 }})),
      include_answer: Type.Optional(Type.Boolean({{ description: "Include Tavily's synthesized answer.", default: true }})),
      include_raw_content: Type.Optional(Type.Boolean({{ description: "Include raw page content.", default: false }})),
    }}),
    async execute(_toolCallId, params, signal) {{
      const result = await callBridgeTool("tavily_search", {{
        query: params.query,
        search_depth: params.search_depth ?? "basic",
        topic: params.topic ?? "general",
        max_results: params.max_results ?? 5,
        include_answer: params.include_answer ?? true,
        include_raw_content: params.include_raw_content ?? false,
      }}, signal);
      const errorText = toolErrorText("tavily_search", result);
      if (errorText) {{
        return {{
          content: [{{ type: "text", text: errorText }}],
          details: result,
        }};
      }}
      const results = Array.isArray(result.results) ? result.results : [];
      const answer = result.answer ? `Answer:\\n${{result.answer}}\\n\\n` : "";
      return {{
        content: [{{ type: "text", text: answer + (formatSearchResults(results) || pretty(result)) }}],
        details: result,
      }};
    }},
  }});

  pi.registerTool({{
    name: "image_research",
    label: "Image Research",
    description: "Find 2-3 compact image candidates with source/provenance metadata. SerpAPI image search is used only as fallback when the preferred image path fails or returns no images.",
    parameters: Type.Object({{
      query: Type.String({{ description: "Image search query, including subject and visual context." }}),
      max_results: Type.Optional(Type.Number({{ description: "Maximum image candidates to return.", default: 3 }})),
      include_domains: Type.Optional(Type.Array(Type.String({{ description: "Optional domains to include or prefer." }}))),
      exclude_domains: Type.Optional(Type.Array(Type.String({{ description: "Optional domains to exclude." }}))),
    }}),
    async execute(_toolCallId, params, signal) {{
      const result = await callBridgeTool("image_research", {{
        query: params.query,
        max_results: params.max_results ?? 3,
        include_domains: params.include_domains,
        exclude_domains: params.exclude_domains,
      }}, signal);
      const errorText = toolErrorText("image_research", result);
      if (errorText) {{
        return {{
          content: [{{ type: "text", text: errorText }}],
          details: result,
        }};
      }}
      sendAutoFollowUp(
        pi as any,
        [
          "Image research is complete. Use the images to help answer the user's query."
        ].join("\\n")
      );
      return {{
        content: [{{ type: "text", text: formatImageResults(result) || pretty(result) }}],
        details: result,
      }};
    }},
  }});

  function commandSignal(ctx: any): AbortSignal {{
    return ctx?.signal ?? new AbortController().signal;
  }}

  function sendCommandResult(title: string, text: string, details: unknown) {{
    pi.sendMessage({{
      customType: "llama_bridge_tool_result",
      content: `## ${{title}}\\n\\n${{text}}`,
      display: true,
      details,
    }}, {{ triggerTurn: false }});
  }}

  async function sendToolCommand(args: string, ctx: any, options: {{
    title: string;
    placeholder: string;
    emptyMessage: string;
    execute: (input: string, signal: AbortSignal) => Promise<{{ text: string; details: unknown }}>;
  }}) {{
    let input = args.trim();
    if (!input) {{
      const entered = await ctx.ui.input(options.title, options.placeholder);
      input = String(entered ?? "").trim();
      if (!input) {{
        ctx.ui.notify(options.emptyMessage, "warning");
        return;
      }}
    }}
    try {{
      ctx.ui.notify(`${{options.title}} running...`, "info");
      const result = await options.execute(input, commandSignal(ctx));
      sendCommandResult(options.title, result.text, result.details);
    }} catch (error) {{
      ctx.ui.notify(`${{options.title}} failed: ${{String(error)}}`, "error");
    }}
  }}

  function registerToolCommand(name: string, description: string, options: {{
    title: string;
    placeholder: string;
    emptyMessage: string;
    execute: (input: string, signal: AbortSignal) => Promise<{{ text: string; details: unknown }}>;
  }}) {{
    pi.registerCommand(name, {{
      description,
      handler: async (args, ctx) => sendToolCommand(args, ctx, options),
    }});
  }}

  registerToolCommand("web", "Run llama bridge web_search", {{
    title: "Web search query",
    placeholder: "Search query",
    emptyMessage: "Web search cancelled: no query entered.",
    execute: async (input, signal) => {{
      const details = await postJson("/api/web_search", {{ query: input, max_results: 5 }}, signal);
      return {{ text: formatSearchResults(details.results || []) || pretty(details), details }};
    }},
  }});
  registerToolCommand("fetch", "Fetch a URL through llama bridge web_fetch", {{
    title: "URL to fetch",
    placeholder: "https://example.com/page",
    emptyMessage: "Web fetch cancelled: no URL entered.",
    execute: async (input, signal) => {{
      const details = await postJson("/api/web_fetch", {{ url: input }}, signal);
      const text = details.text ? String(details.text).slice(0, 12000) : pretty(details);
      return {{ text, details }};
    }},
  }});
  registerToolCommand("serp", "Run SerpAPI search through llama bridge", {{
    title: "SerpAPI search query",
    placeholder: "Search query",
    emptyMessage: "SerpAPI search cancelled: no query entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("serpapi_search", {{ query: input, num: 5 }}, signal);
      const errorText = toolErrorText("serpapi_search", details);
      return {{ text: errorText || formatSearchResults(details.organic_results || details.results || []) || pretty(details), details }};
    }},
  }});
  registerToolCommand("tavily", "Run Tavily search through llama bridge", {{
    title: "Tavily search query",
    placeholder: "Search query",
    emptyMessage: "Tavily search cancelled: no query entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("tavily_search", {{ query: input, max_results: 5, include_answer: true }}, signal);
      const errorText = toolErrorText("tavily_search", details);
      const answer = details.answer ? `Answer:\\n${{details.answer}}\\n\\n` : "";
      return {{ text: errorText || answer + (formatSearchResults(details.results || []) || pretty(details)), details }};
    }},
  }});
  registerToolCommand("image", "Run image_research through llama bridge", {{
    title: "Image research query",
    placeholder: "Image search query",
    emptyMessage: "Image research cancelled: no query entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("image_research", {{ query: input, max_results: 3 }}, signal);
      const errorText = toolErrorText("image_research", details);
      return {{ text: errorText || formatImageResults(details) || pretty(details), details }};
    }},
  }});
  registerToolCommand("wiki", "Search Wikipedia through llama bridge", {{
    title: "Wikipedia search query",
    placeholder: "Topic or title",
    emptyMessage: "Wikipedia search cancelled: no query entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("wikipedia_search", {{ query: input, limit: 5, language: "en" }}, signal);
      const errorText = toolErrorText("wikipedia_search", details);
      return {{ text: errorText || formatSearchResults(details.results || []) || pretty(details), details }};
    }},
  }});
  registerToolCommand("manim", "Generate a Manim animation video through llama bridge", {{
    title: "Manim animation prompt",
    placeholder: "Explain a concept with a short animation",
    emptyMessage: "Manim render cancelled: no animation prompt entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("manim_render", {{ prompt: input, quality: "low", render: true }}, signal);
      const errorText = toolErrorText("manim_render", details);
      const text = errorText || `Scene: ${{details.scene_path || "(missing)"}}\\nVideo: ${{details.video_path || "(not rendered)"}}\\n\\n${{pretty(details)}}`;
      return {{ text, details }};
    }},
  }});
  registerToolCommand("weather", "Get current weather through llama bridge", {{
    title: "Weather location",
    placeholder: "City or place",
    emptyMessage: "Weather lookup cancelled: no location entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("weather_current", {{ location: input }}, signal);
      const errorText = toolErrorText("weather_current", details);
      return {{ text: errorText || pretty(details), details }};
    }},
  }});
  registerToolCommand("time", "Get current time through llama bridge", {{
    title: "Timezone or country",
    placeholder: "Asia/Calcutta, UTC, or a country",
    emptyMessage: "Time lookup cancelled: no timezone or country entered.",
    execute: async (input, signal) => {{
      const looksLikeTimezone = input.includes("/") || input.toUpperCase() === "UTC" || /^[+-]\\d{{2}}:?\\d{{2}}$/.test(input);
      const details = await callBridgeTool("datetime_now", looksLikeTimezone ? {{ timezone: input }} : {{ country: input }}, signal);
      const errorText = toolErrorText("datetime_now", details);
      return {{ text: errorText || pretty(details), details }};
    }},
  }});

}}
"""




def _ensure_node_and_npm() -> None:
    if _find_npm_executable():
        return

    manager_command = _node_install_command()
    if manager_command is None:
        raise SystemExit("npm is missing and no supported package manager was found.")

    print(f"npm was not found, installing Node.js with {_package_manager_name(manager_command)}...")
    process = subprocess.run(manager_command, check=False)
    if process.returncode != 0:
        raise SystemExit("Node.js install failed. Install Node.js and run the command again.")


def _find_npm_executable() -> str | None:
    npm_executable = shutil.which("npm")
    if npm_executable:
        return npm_executable
    candidates = []
    app_data = os.environ.get("APPDATA")
    program_files = os.environ.get("ProgramFiles")
    if app_data:
        candidates.append(Path(app_data) / "npm" / "npm.cmd")
    if program_files:
        candidates.append(Path(program_files) / "nodejs" / "npm.cmd")
    return _first_existing(candidates)


def _find_git_bash() -> str | None:
    if os.name != "nt":
        return None
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Git" / "bin" / "bash.exe",
    ]
    bash_executable = shutil.which("bash.exe") or shutil.which("bash")
    if bash_executable and _is_usable_pi_bash(Path(bash_executable)):
        candidates.insert(0, Path(bash_executable))
    return _first_existing(candidates)


def _is_usable_pi_bash(path: Path) -> bool:
    normalized = str(path).lower()
    if "\\windowsapps\\bash.exe" in normalized:
        return False
    return path.exists()


def _find_claude_executable() -> str | None:
    claude_executable = shutil.which("claude")
    if claude_executable:
        return claude_executable
    candidates = []
    app_data = os.environ.get("APPDATA")
    if app_data:
        candidates.extend(
            [
                Path(app_data) / "npm" / "claude.cmd",
                Path(app_data) / "npm" / "claude.exe",
            ]
        )
    return _first_existing(candidates)


def _find_pi_executable() -> str | None:
    pi_executable = shutil.which("pi")
    if pi_executable:
        return pi_executable
    candidates = []
    app_data = os.environ.get("APPDATA")
    if app_data:
        candidates.extend(
            [
                Path(app_data) / "npm" / "pi.cmd",
                Path(app_data) / "npm" / "pi.exe",
            ]
        )
    return _first_existing(candidates)


def _find_codex_executable() -> str | None:
    codex_executable = shutil.which("codex")
    if codex_executable:
        return codex_executable
    candidates = []
    app_data = os.environ.get("APPDATA")
    if app_data:
        candidates.extend(
            [
                Path(app_data) / "npm" / "codex.cmd",
                Path(app_data) / "npm" / "codex.exe",
            ]
        )
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidates.extend(Path(user_profile).glob(".vscode/extensions/openai.chatgpt-*/bin/windows-*/codex.exe"))
    return _first_existing(candidates)


def _find_copilot_executable() -> str | None:
    copilot_executable = shutil.which("copilot")
    if copilot_executable:
        return copilot_executable
    candidates = []
    app_data = os.environ.get("APPDATA")
    if app_data:
        candidates.extend(
            [
                Path(app_data) / "npm" / "copilot.cmd",
                Path(app_data) / "npm" / "copilot.exe",
            ]
        )
    return _first_existing(candidates)


def _find_opencode_executable() -> str | None:
    opencode_executable = shutil.which("opencode")
    if opencode_executable:
        return opencode_executable
    candidates = []
    app_data = os.environ.get("APPDATA")
    if app_data:
        candidates.extend(
            [
                Path(app_data) / "npm" / "opencode.cmd",
                Path(app_data) / "npm" / "opencode.exe",
            ]
        )
    return _first_existing(candidates)


def _find_poolside_executable() -> str | None:
    poolside_executable = shutil.which("pool")
    if poolside_executable:
        return poolside_executable
    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.extend(
            [
                Path(local_app_data) / "Programs" / "pool" / "bin" / "pool.exe",
                Path(local_app_data) / "Programs" / "pool" / "bin" / "pool.cmd",
            ]
        )
    home = Path.home()
    candidates.extend(
        [
            home / ".local" / "bin" / "pool",
            home / ".poolside" / "bin" / "pool",
        ]
    )
    return _first_existing(candidates)


def _ensure_opencode(install: bool = True, package: str | None = None) -> str:
    opencode_executable = _find_opencode_executable()
    if opencode_executable:
        return opencode_executable
    if not install:
        raise SystemExit("OpenCode was not found. Install OpenCode and try again.")

    _print_state("install", "OpenCode was not found, installing it now", "36")
    _ensure_node_and_npm()
    npm_executable = _find_npm_executable()
    if not npm_executable:
        raise SystemExit("npm was not found after setup. Install Node.js 18+ and try again.")

    if package is None:
        package = "opencode-ai"

    process = subprocess.run(
        [npm_executable, "install", "-g", package],
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit(f"OpenCode install failed. Try `npm install -g {package}`.")

    opencode_executable = _find_opencode_executable()
    if not opencode_executable:
        raise SystemExit(
            "OpenCode installed, but `opencode` is not on PATH. Restart your terminal and try again."
        )
    return opencode_executable


def _ensure_poolside(
    install: bool = True,
    install_command: str = "curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh",
    windows_install_command: str = "irm https://downloads.poolside.ai/pool/install.ps1 | iex",
) -> str:
    poolside_executable = _find_poolside_executable()
    if poolside_executable:
        return poolside_executable
    if not install:
        raise SystemExit("Poolside CLI was not found. Install Poolside and try again.")

    _print_state("install", "Poolside CLI was not found, installing it now", "36")
    if os.name == "nt":
        git_bash = _find_git_bash()
        if git_bash:
            _print_state("install", f"Git Bash found, running Poolside installer with {git_bash}", "36")
            process = subprocess.run(
                [git_bash, "-lc", install_command],
                check=False,
            )
            install_hint = install_command
            if process.returncode != 0:
                _print_state(
                    "warn",
                    "Poolside shell installer did not complete under Git Bash, falling back to PowerShell",
                    "33",
                )
                process = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        windows_install_command,
                    ],
                    check=False,
                )
                install_hint = windows_install_command
        else:
            process = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    windows_install_command,
                ],
                check=False,
            )
            install_hint = windows_install_command
    else:
        process = subprocess.run(
            ["sh", "-c", install_command],
            check=False,
        )
        install_hint = install_command
    if process.returncode != 0:
        raise SystemExit(f"Poolside install failed. Try `{install_hint}`.")

    poolside_executable = _find_poolside_executable()
    if not poolside_executable:
        raise SystemExit("Poolside installed, but `pool` is not on PATH. Restart your terminal and try again.")
    return poolside_executable


def _cli_targets(config) -> list[CliTarget]:
    return [
        CliTarget(
            name="claude",
            display_name="Claude Code",
            launcher_command="llama claude",
            finder=_find_claude_executable,
            install_method="npm",
            package="@anthropic-ai/claude-code",
            uninstall_hint="npm uninstall -g @anthropic-ai/claude-code",
        ),
        CliTarget(
            name="pi",
            display_name="Pi",
            launcher_command="llama pi",
            finder=_find_pi_executable,
            install_method="npm",
            package=config.pi.install_package,
            uninstall_hint=f"npm uninstall -g {config.pi.install_package}",
        ),
        CliTarget(
            name="codex",
            display_name="Codex",
            launcher_command="llama codex",
            finder=_find_codex_executable,
            install_method="npm",
            package=config.codex.install_package,
            uninstall_hint=f"npm uninstall -g {config.codex.install_package}",
        ),
        CliTarget(
            name="copilot",
            display_name="Copilot CLI",
            launcher_command="llama copilot",
            finder=_find_copilot_executable,
            install_method="npm",
            package=config.copilot_cli.install_package,
            uninstall_hint=f"npm uninstall -g {config.copilot_cli.install_package}",
        ),
        CliTarget(
            name="opencode",
            display_name="OpenCode",
            launcher_command="llama opencode",
            finder=_find_opencode_executable,
            install_method="npm",
            package=config.opencode.install_package,
            uninstall_hint=f"npm uninstall -g {config.opencode.install_package}",
        ),
        CliTarget(
            name="poolside",
            display_name="Poolside",
            launcher_command="llama poolside",
            finder=_find_poolside_executable,
            install_method="standalone",
            uninstall_hint="delete the Poolside installation directory",
        ),
    ]


def _print_cli_targets(targets: list[CliTarget], heading: str) -> None:
    _title(heading)
    headers = ["name", "status", "launcher", "install", "path"]
    rows: list[list[str]] = []
    for target in targets:
        path = target.finder()
        rows.append(
            [
                target.name,
                "usable" if path else "missing",
                target.launcher_command,
                target.install_method,
                path or "-",
            ]
        )

    widths = [
        max(len(headers[index]), max((len(row[index]) for row in rows), default=0))
        for index in range(len(headers))
    ]
    print("  " + "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  " + "  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def _print_supported_cli_targets(targets: list[CliTarget]) -> None:
    _title("llama cli --support")
    headers = ["name", "launcher", "install", "package"]
    rows = [
        [
            target.name,
            target.launcher_command,
            target.install_method,
            target.package or "-",
        ]
        for target in targets
    ]
    widths = [
        max(len(headers[index]), max((len(row[index]) for row in rows), default=0))
        for index in range(len(headers))
    ]
    print("  " + "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  " + "  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def _prompt_cli_target_name(targets: list[CliTarget]) -> str | None:
    installed = [target for target in targets if target.finder()]
    if not installed:
        _title("llama cli")
        _print_state("ok", "No managed CLI tools are currently installed", "32")
        return None

    _print_cli_targets(installed, heading="llama cli --rm")
    choice = input("\nChoose a CLI name to remove (or press Enter to cancel): ").strip().lower()
    if not choice:
        _print_note("Canceled.")
        return None
    return choice


def _remove_cli_target(targets: list[CliTarget], name: str) -> None:
    target = next((item for item in targets if item.name == name), None)
    if target is None:
        available = ", ".join(sorted(item.name for item in targets))
        raise SystemExit(f"Unknown CLI '{name}'. Available names: {available}")

    executable_path = target.finder()
    if not executable_path:
        _title("llama cli --rm")
        _print_state("ok", f"{target.display_name} is already not installed", "32")
        return

    _title("llama cli --rm")
    _print_state("run", f"Removing {target.display_name}", "36")
    _kv_rows(
        [
            ("name", target.name),
            ("path", executable_path),
        ]
    )

    if target.install_method == "npm":
        npm_executable = _find_npm_executable()
        if not npm_executable or not target.package:
            raise SystemExit(f"npm was not found. Try `{target.uninstall_hint}`.")
        process = subprocess.run(
            [npm_executable, "uninstall", "-g", target.package],
            check=False,
        )
        if process.returncode != 0:
            raise SystemExit(f"CLI remove failed. Try `{target.uninstall_hint}`.")
    elif target.name == "poolside":
        _remove_poolside_install(executable_path)
    else:
        raise SystemExit(f"Remove is not implemented for {target.name}.")

    _print_state("ok", f"Removed {target.display_name}", "32")


def _remove_poolside_install(executable_path: str) -> None:
    path = Path(executable_path)
    candidates = [path]
    if path.name.lower().startswith("pool") and path.parent.name.lower() == "bin":
        candidates.insert(0, path.parent.parent)

    for candidate in candidates:
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=False)
            return
        if candidate.is_file():
            candidate.unlink(missing_ok=True)
            return

    raise SystemExit("Could not determine which Poolside files to remove.")


def _first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path)
    return None


def _node_install_command() -> list[str] | None:
    if os.name == "nt" and shutil.which("winget"):
        return [
            "winget",
            "install",
            "--id",
            "OpenJS.NodeJS",
            "-e",
            "--accept-source-agreements",
            "--accept-package-agreements",
        ]
    if sys.platform == "darwin" and shutil.which("brew"):
        return ["brew", "install", "node"]
    if shutil.which("pacman"):
        return ["sudo", "pacman", "-S", "--needed", "nodejs", "npm"]
    return None


def _package_manager_name(command: list[str]) -> str:
    executable = Path(command[0]).name
    if executable == "sudo" and len(command) > 1:
        return Path(command[1]).name
    return executable


def _is_running(pid_path: Path) -> bool:
    pid = _read_pid(pid_path)
    if pid is None:
        return False
    try:
        return _pid_alive(pid)
    except OSError:
        pid_path.unlink(missing_ok=True)
        return False


def _server_is_running(config_path: Path, pid_path: Path) -> tuple[bool, str | None]:
    if _is_running(pid_path):
        return True, None
    try:
        config = load_config(config_path)
    except Exception:
        return False, None
    url = _server_url(config.server.host, config.server.port)
    if _http_status(url).startswith("ok"):
        return True, url
    return False, url


def _listening_pid_for_config(config_path: Path) -> int | None:
    try:
        config = load_config(config_path)
    except Exception:
        return None
    return _listening_pid(config.server.host, config.server.port)


def _listening_pid(host: str, port: int) -> int | None:
    if os.name != "nt":
        return None
    process = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return None
    suffix = f":{port}"
    candidates = {host, "0.0.0.0", "::", "[::]", "localhost"}
    for line in process.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_address, state, pid_text = parts[1], parts[3], parts[-1]
        if state.upper() != "LISTENING" or not local_address.endswith(suffix):
            continue
        address_host = local_address.rsplit(":", 1)[0].strip("[]")
        if address_host not in candidates:
            continue
        try:
            return int(pid_text)
        except ValueError:
            return None
    return None


def _read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return None


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _server_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


async def _http_status_async(url: str, path: str = "/health") -> str:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(f"{url.rstrip('/')}{path}")
            return f"ok ({response.status_code})"
    except httpx.TimeoutException:
        return "unreachable (timeout)"
    except httpx.RequestError as exc:
        return f"unreachable ({exc})"


def _http_status(url: str, path: str = "/health") -> str:
    return asyncio.run(_http_status_async(url, path))


def _serve_command(
    config_path: Path,
    log_path: Path,
    idle_timeout_seconds: int = 0,
    idle_after_file: Path | None = None,
) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [sys.executable, "serve"]
    else:
        command = [sys.executable, "-m", "llama_bridge", "serve"]
    command.extend(
        [
            "--config",
            str(config_path),
            "--log-file",
            str(log_path),
            "--foreground",
        ]
    )
    if idle_timeout_seconds > 0:
        command.extend(["--idle-timeout", str(idle_timeout_seconds)])
    if idle_after_file is not None:
        command.extend(["--idle-after-file", str(idle_after_file)])
    return command


def _start_windows_background(
    config_path: Path,
    log_path: Path,
    idle_timeout_seconds: int = 0,
    idle_after_file: Path | None = None,
) -> int:
    env = {**os.environ, "LLAMA_DEV_LOG": "1"}
    process = subprocess.Popen(
        _serve_command(config_path, log_path, idle_timeout_seconds, idle_after_file),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        cwd=str(config_path.parent),
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        ),
        env=env,
    )
    return process.pid


if __name__ == "__main__":
    main()

