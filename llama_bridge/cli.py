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
import time
import warnings
from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from pathlib import Path

import httpx

from .config import (
    DEFAULT_API_SETTINGS_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXAMPLE_CONFIG_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_PID_PATH,
    ensure_default_dirs,
    merge_missing_config_fields,
    load_config,
    codex_model_error,
    copilot_cli_model_error,
    openai_model_error,
    opencode_model_error,
    pi_model_error,
    resolve_codex_model,
    resolve_copilot_cli_model,
    resolve_openai_model,
    resolve_opencode_model,
    resolve_pi_model,
    write_claude_api_settings,
    write_default_config,
)
from .master import MasterReviewer
from .mcp_tools import main as mcp_tools_main
from .tools import ToolRegistry, classify_query_intent, select_relevant_tools


PYTHON_REQUIREMENTS = {
    "fastapi": "fastapi>=0.115.0",
    "httpx": "httpx>=0.27.0",
    "yaml": "pyyaml>=6.0.2",
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

        if args.command is None:
            _cmd_setup(_default_config_path(), install_system=True)
            _print_note("Run `llama serve` to start the server or `llama claude` to launch Claude Code.")
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
            _cmd_start(
                config_path,
                _arg_path(args.pid_file, DEFAULT_PID_PATH, config_path),
                _arg_path(args.log_file, DEFAULT_LOG_PATH, config_path),
                0 if args.forever else DEFAULT_START_IDLE_TIMEOUT_SECONDS,
            )
            return
        if args.command == "stop":
            config_path = _default_config_path()
            _cmd_stop(_arg_path(args.pid_file, DEFAULT_PID_PATH, config_path))
            return
        if args.command == "logs":
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
        if args.command == "master-review":
            _cmd_master_review(
                _arg_path(args.config),
                report_path=Path(args.report_json) if args.report_json else None,
                mode=args.mode,
                use_stdin=args.stdin,
                check_keys=args.check_keys,
                write_reviewed=Path(args.write_reviewed) if args.write_reviewed else None,
            )
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
            )
            return
        if args.command == "poolside":
            config_path = _arg_path(args.config)
            _cmd_poolside(
                config_path,
                args.poolside_args,
                provider_override=args.provider,
                model_override=args.model,
                install_poolside=not args.no_install_poolside,
            )
            return
        if args.command == "cli":
            config_path = _arg_path(args.config)
            _cmd_cli(
                config_path,
                list_only=args.list,
                remove_target=args.rm,
            )
            return
        if args.command == "telegram":
            config_path = _arg_path(args.config)
            if args.telegram_command == "status":
                _cmd_telegram_status(config_path)
                return
            parser.print_help()
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
        help="disable the default 3-minute idle auto-stop",
    )

    stop_cmd = subparsers.add_parser("stop", help="stop the background bridge")
    stop_cmd.add_argument("--pid-file")

    logs_cmd = subparsers.add_parser("logs", help="show bridge logs")
    logs_cmd.add_argument("--config")
    logs_cmd.add_argument("--log-file")
    logs_cmd.add_argument("--pid-file")
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

    subparsers.add_parser("mcp-tools", help="run the bridge tools MCP adapter")

    master_cmd = subparsers.add_parser(
        "master-review",
        help="review a deep/source research JSON report and print final improvement instructions",
    )
    master_cmd.add_argument("report_json", nargs="?", help="path to deep/source research JSON")
    master_cmd.add_argument("--config")
    master_cmd.add_argument(
        "--mode",
        choices=["fast", "balanced", "strict"],
        default=None,
        help="review depth override",
    )
    master_cmd.add_argument("--stdin", action="store_true", help="read research JSON from stdin")
    master_cmd.add_argument(
        "--check-keys",
        action="store_true",
        help="show configured Groq key slots without revealing key values",
    )
    master_cmd.add_argument(
        "--write-reviewed",
        help="write the revised draft to this path when the review produces one",
    )

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
        "--no-install-poolside",
        action="store_true",
        help="do not install Poolside automatically if it is missing",
    )
    poolside_cmd.add_argument(
        "poolside_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to pool",
    )

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
        "--rm",
        nargs="?",
        const="__prompt__",
        metavar="NAME",
        help="remove one installed CLI by name, or prompt to choose when no name is given",
    )

    telegram_cmd = subparsers.add_parser(
        "telegram",
        help="inspect the restricted Telegram bot configuration",
    )
    telegram_cmd.add_argument("--config")
    telegram_subparsers = telegram_cmd.add_subparsers(dest="telegram_command")
    telegram_subparsers.add_parser("status", help="show Telegram bot configuration status")

    bot_cmd = subparsers.add_parser(
        "bot",
        help="guided Telegram bot setup",
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

    return parser


def _cmd_init(config_path: Path, force: bool) -> None:
    if config_path.exists():
        created, changed = merge_missing_config_fields(config_path)
        _title("llama init")
        _print_state("ok", "existing config checked and merged forward", "32")
        _kv_rows(
            [
                ("config", str(created)),
                ("updated", "yes" if changed else "no"),
            ]
        )
        _print_note("Existing providers, API keys, and models were preserved.")
        return

    target_path = _example_config_path(config_path)
    created = write_default_config(target_path, force=force)
    _title("llama init")
    _print_state("ok", "example configuration is ready", "32")
    _kv_rows(
        [
            ("example", str(created)),
            ("next", f"edit API keys/models, then rename to {config_path.name}"),
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

    merge_missing_config_fields(config_path)
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
        for alias_name in ("haiku", "sonnet", "opus", "small_fast"):
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

    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

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

    merge_missing_config_fields(config_path)
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
        config_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
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

    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
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
    _print_note("Use http://127.0.0.1:8089 for Ollama-style clients.")
    _print_note("Use http://127.0.0.1:8089/v1 for OpenAI, LM Studio, Copilot, and Codex clients.")
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
    )
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
) -> None:
    ensure_default_dirs(pid_path.parent)
    ensure_default_dirs(log_path.parent)
    _sync_config_clone_from_root(config_path)
    load_config(config_path)
    already_running, running_url = _server_is_running(config_path, pid_path)
    if _is_running(pid_path):
        _print_state("run", f"llama server is already running with pid {pid_path.read_text().strip()}", "32")
        _write_active_server_state(config_path, pid_path, log_path)
        return
    if already_running and running_url is not None:
        _print_state("run", f"llama server is already running at {running_url}", "32")
        _write_active_server_state(config_path, pid_path, log_path)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    (config_path.parent / "llama.dev.log").write_text("", encoding="utf-8")
    _title("llama start")
    if os.name == "nt":
        process_id = _start_windows_background(
            config_path,
            log_path,
            idle_timeout_seconds=idle_timeout_seconds,
            idle_after_file=idle_after_file,
        )
        _pause_with_spinner("starting background server", 1)
        if not _pid_alive(process_id):
            _print_state("fail", f"llama failed to start, see log: {log_path}", "31")
            return
        pid_path.write_text(str(process_id), encoding="utf-8")
        _write_active_server_state(config_path, pid_path, log_path)
        _print_state("ok", f"llama started in background on pid {process_id}", "32")
        _kv_rows([("log", str(log_path)), ("logs", "llama logs")])
        if idle_timeout_seconds == 0:
            _print_note("Server will stay up until you run `llama stop`.")
        else:
            _print_note(f"Server will stop after {idle_timeout_seconds // 60} minutes of inactivity.")
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
    _pause_with_spinner("starting background server", 1)
    if process.poll() is not None:
        _print_state("fail", f"llama failed to start, see log: {log_path}", "31")
        return
    pid_path.write_text(str(process.pid), encoding="utf-8")
    _write_active_server_state(config_path, pid_path, log_path)
    _print_state("ok", f"llama started in background on pid {process.pid}", "32")
    _kv_rows([("log", str(log_path)), ("logs", "llama logs")])
    if idle_timeout_seconds == 0:
        _print_note("Server will stay up until you run `llama stop`.")
    else:
        _print_note(f"Server will stop after {idle_timeout_seconds // 60} minutes of inactivity.")


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


def _llama_root_config_path() -> Path | None:
    candidates = _llama_root_config_candidates()

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _cmd_stop(pid_path: Path) -> None:
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

    url = None
    http_status = None
    if config is not None:
        url = _server_url(config.server.host, config.server.port)
        http_status = _http_status(url)
        if not process_running and http_status.startswith("ok"):
            process_running = True

    _title("llama status")
    rows: list[tuple[str, str | int]] = [("process", _status_label(process_running))]
    if pid is not None:
        rows.append(("pid", pid))
    elif process_running:
        rows.append(("pid", "unknown"))

    if config is not None:
        rows.extend(
            [
                ("url", url or ""),
                ("http", http_status or ""),
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


def _cmd_master_review(
    config_path: Path,
    *,
    report_path: Path | None,
    mode: str | None,
    use_stdin: bool,
    check_keys: bool,
    write_reviewed: Path | None = None,
) -> None:
    config = load_config(config_path)
    if check_keys:
        _title("Master Review Keys")
        keys = config.master_review.groq.api_keys
        rows: list[tuple[str, str | int]] = [
            ("enabled", str(config.master_review.enabled)),
            ("groq enabled", str(config.master_review.groq.enabled)),
            ("model", config.master_review.groq.model),
            ("configured keys", len(keys)),
        ]
        _kv_rows(rows)
        for index, key in enumerate(keys, start=1):
            print(f"- groq_key_{index}: {_configured_label(key)}")
        if not keys:
            _print_state("warn", "no Groq keys configured; fallback review will be used", "33")
        return

    if use_stdin:
        raw_text = sys.stdin.read()
        output_dir = Path.cwd()
    elif report_path is not None:
        raw_text = report_path.read_text(encoding="utf-8")
        output_dir = report_path.resolve().parent
    else:
        raise SystemExit("Provide report_json, use --stdin, or pass --check-keys.")

    try:
        research_result = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"research JSON is invalid: {exc}") from exc
    if not isinstance(research_result, dict):
        raise SystemExit("research JSON must be an object")

    review = asyncio.run(_run_master_review_once(config.master_review, research_result, mode))

    data = review.get("data", {}) if isinstance(review.get("data"), dict) else {}
    final_review = data.get("final_review", {}) if isinstance(data.get("final_review"), dict) else {}
    metadata = review.get("metadata", {}) if isinstance(review.get("metadata"), dict) else {}
    instructions = str(data.get("final_llm_instructions") or "")
    instructions_path = output_dir / "master_review_instructions.txt"
    instructions_path.write_text(instructions + "\n", encoding="utf-8")
    if write_reviewed is not None and data.get("revised_draft"):
        write_reviewed.write_text(str(data.get("revised_draft")) + "\n", encoding="utf-8")

    _title("Master Review")
    _kv_rows(
        [
            ("Quality score", f"{data.get('quality_score', 'n/a')} / 10"),
            ("Risk level", str(data.get("risk_level", "unknown"))),
            ("Groq keys configured", len(config.master_review.groq.api_keys)),
            ("Groq keys used", ", ".join(metadata.get("groq_keys_used") or []) or "none"),
            ("Fallback used", str(metadata.get("fallback_used", False)).lower()),
        ]
    )
    _print_cli_list("Must fix", final_review.get("must_fix", []))
    _print_cli_list("Weak sources", final_review.get("weak_sources", []))
    _print_cli_list("Unsupported claims", final_review.get("unsupported_claims", []))
    print()
    print("Final LLM instructions:")
    print(instructions or "none")
    print()
    print(f"Final LLM instructions written to: {instructions_path}")
    if write_reviewed is not None and data.get("revised_draft"):
        print(f"Reviewed answer written to: {write_reviewed}")


def _print_cli_list(title: str, values: Any, *, limit: int = 8) -> None:
    items = values if isinstance(values, list) else []
    print()
    print(f"{title}:")
    if not items:
        print("- none")
        return
    for item in items[:limit]:
        print(f"- {str(item)[:500]}")


async def _run_master_review_once(master_config, research_result: dict[str, Any], mode: str | None) -> dict[str, Any]:
    reviewer = MasterReviewer(master_config)
    try:
        return await reviewer.review_deep_research(research_result, mode=mode)
    finally:
        await reviewer.aclose()


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
        _print_state("stop", "llama server is not running", "33")
        return

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

    if not log_path.exists():
        _print_state("warn", f"no llama log found at {log_path}", "33")
        return

    if follow:
        _title("llama dev logs" if dev else "llama logs")
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
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.pi.closed"
        idle_after_file.unlink(missing_ok=True)
        _print_state("start", "llama server is not running, starting it for Pi", "36")
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=180,
            idle_after_file=idle_after_file,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        _print_note("llama server will stop 3 minutes after Pi closes with no requests")
    else:
        idle_after_file = None
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
    _ensure_pi_extensions(config)

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


def _ensure_claude_tool_plugin(config, config_path: Path) -> Path:
    plugin_dir = config_path.parent / "plugins" / "llama_bridge_tools_claude"
    _write_claude_tool_plugin(plugin_dir, config)
    short_commands_dir = _write_claude_short_commands()
    _print_state("ok", f"Claude Code llama bridge tools plugin: {plugin_dir}", "32")
    _print_state("ok", f"Claude Code short slash commands: {short_commands_dir}", "32")
    return plugin_dir


def _ensure_codex_tool_extension(config) -> tuple[Path, Path]:
    codex_config_path = Path(os.path.expanduser(config.codex.config_path))
    codex_config_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_dir = codex_config_path.parent / "plugins" / "llama_bridge_tools"
    _write_codex_tool_plugin(plugin_dir, config)
    _write_codex_mcp_config(codex_config_path, config)
    _print_state("ok", f"Codex llama bridge tools plugin: {plugin_dir}", "32")
    _print_state("ok", f"Codex MCP tools config: {codex_config_path}", "32")
    return plugin_dir, codex_config_path


def _ensure_copilot_tool_extension(config) -> Path:
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
    _print_state("ok", f"Copilot CLI MCP tools config: {config_path}", "32")
    
    # Verify MCP server is reachable
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
    return {
        "LLAMA_BRIDGE_BASE_URL": _server_url(config.server.host, config.server.port),
        "LLAMA_BRIDGE_API_KEY": config.server.auth_token,
    }


def _verify_mcp_server_tools(config) -> None:
    """Verify MCP server is reachable and return discovered tool names."""
    import json
    import subprocess
    import sys
    from pathlib import Path

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

        # Send initialize request
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

        # Read initialize response (and possibly notification)
        init_response = proc.stdout.readline()
        # Try to read notification if present
        import select
        try:
            if select.select([proc.stdout], [], [], 1.0)[0]:
                notification = proc.stdout.readline()
        except Exception:
            pass

        # Send tools/list request
        list_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        }) + "\n"
        proc.stdin.write(list_msg)
        proc.stdin.flush()

        # Read tools/list response
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
description: Use when {host_name} needs current web search, source research, image research, weather, Wikipedia, or date/time lookups through the local llama bridge MCP tools.
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
    if getattr(sys, "frozen", False):
        return sys.executable, ["mcp-tools"]
    return sys.executable, ["-m", "llama_bridge.mcp_tools"]


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
    deep = (
        "Run deep sourced research through the llama bridge.",
        "[topic]",
        _bridge_command_body(
            "research topic",
            (
                "Use `mcp__llama_bridge_tools__source_research` first for the user input, then verify important claims "
                "with available llama bridge search tools, use `mcp__llama_bridge_tools__image_research` for 2-3 "
                "useful sourced images, and produce a concise cited research brief."
            ),
        ),
    )
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
    return {
        "deep.md": deep,
        "deep_research.md": deep,
        "deep-research.md": deep,
        "serp.md": serp,
        "serp_search.md": serp,
        "serp-sarch.md": serp,
        "web.md": web,
        "web_search.md": web,
        "fetch.md": fetch,
        "image.md": image,
        "image_search.md": image,
        "image-sarch.md": image,
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
            f"User input: $ARGUMENTS",
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
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)
    api_settings_path = config_path.parent / DEFAULT_API_SETTINGS_PATH.name
    claude_executable = _ensure_claude_code(install=install_claude)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.claude.closed"
        idle_after_file.unlink(missing_ok=True)
        _print_state("start", "llama server is not running, starting it for Claude Code", "36")
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=180,
            idle_after_file=idle_after_file,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        _print_note("llama server will stop 3 minutes after Claude Code closes with no requests")
    else:
        idle_after_file = None
        _print_state("run", "using existing llama server", "32")

    passthrough_args = claude_args
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    plugin_dir = _ensure_claude_tool_plugin(config, config_path)

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
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.codex.closed"
        idle_after_file.unlink(missing_ok=True)
        _print_state("start", "llama server is not running, starting it for Codex", "36")
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=180,
            idle_after_file=idle_after_file,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        _print_note("llama server will stop 3 minutes after Codex closes with no requests")
    else:
        idle_after_file = None
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
    codex_plugin_dir, codex_mcp_config_path = _ensure_codex_tool_extension(config)

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
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.copilot.closed"
        idle_after_file.unlink(missing_ok=True)
        _print_state("start", "llama server is not running, starting it for Copilot CLI", "36")
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=180,
            idle_after_file=idle_after_file,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        _print_note("llama server will stop 3 minutes after Copilot CLI closes with no requests")
    else:
        idle_after_file = None
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
    copilot_mcp_config_path = _ensure_copilot_tool_extension(config)

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
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)

    server_running, _running_url = _server_is_running(config_path, pid_path)
    if not server_running:
        idle_after_file = pid_path.parent / "llama.opencode.closed"
        idle_after_file.unlink(missing_ok=True)
        _print_state("start", "llama server is not running, starting it for OpenCode", "36")
        _cmd_start(
            config_path,
            pid_path,
            log_path,
            idle_timeout_seconds=180,
            idle_after_file=idle_after_file,
        )
        server_running, _running_url = _server_is_running(config_path, pid_path)
        if not server_running:
            raise SystemExit(f"llama server failed to start, see log: {log_path}")
        _print_note("llama server will stop 3 minutes after OpenCode closes with no requests")
    else:
        idle_after_file = None
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
    poolside_args: list[str],
    provider_override: str | None = None,
    model_override: str | None = None,
    install_poolside: bool = True,
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

    poolside_executable = _ensure_poolside(
        install=install_poolside,
        install_command=config.poolside.install_command,
        windows_install_command=config.poolside.windows_install_command,
    )
    poolside_settings_path = _write_poolside_config(config)

    _title("llama poolside")
    _print_state("ok", "Poolside environment is ready", "32")
    _kv_rows(
        [
            ("provider", provider_name),
            ("model", model),
            ("api url", f"{_server_url(config.server.host, config.server.port)}/v1"),
            ("config", str(poolside_settings_path)),
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
    env["POOLSIDE_API_URL"] = f"{_server_url(config.server.host, config.server.port)}/v1"
    env["POOLSIDE_API_KEY"] = config.server.auth_token
    env["LLAMA_BRIDGE_API_KEY"] = config.server.auth_token

    raise SystemExit(
        subprocess.run(
            command,
            check=False,
            env=env,
        ).returncode
    )


def _write_poolside_config(config) -> Path:
    import yaml

    settings_path = _resolved_poolside_settings_path(config)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    existing = yaml.safe_load(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    if not isinstance(existing, dict):
        existing = {}

    pool_section = dict(existing.get("pool") or {})
    pool_section["api_url"] = f"{_server_url(config.server.host, config.server.port)}/v1"
    existing["pool"] = pool_section

    mcp_servers = dict(existing.get("mcp_servers") or {})
    command, args = _mcp_server_command()
    mcp_servers["llama_bridge_tools"] = {
        "command": command,
        "args": args,
        "env": _mcp_server_env(config),
    }
    existing["mcp_servers"] = mcp_servers

    settings_path.write_text(
        yaml.safe_dump(existing, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return settings_path


def _resolved_poolside_settings_path(config) -> Path:
    raw_path = str(config.poolside.config_path or "~/.config/poolside/settings.yaml")
    if os.name == "nt" and raw_path == "~/.config/poolside/settings.yaml":
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "poolside" / "settings.yaml"
    return Path(os.path.expanduser(raw_path))


def _cmd_cli(
    config_path: Path,
    list_only: bool = False,
    remove_target: str | None = None,
) -> None:
    _ensure_setup(config_path)
    config = load_config(config_path)
    targets = _cli_targets(config)

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


def _ensure_pi_extensions(config) -> list[Path]:
    paths: list[Path] = []
    web_tools = _ensure_pi_web_tools(config)
    if web_tools is not None:
        paths.append(web_tools)
    deep_research = _ensure_pi_deep_research(config)
    if deep_research is not None:
        paths.append(deep_research)
    return paths


def _ensure_pi_web_tools(config) -> Path | None:
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
    _print_state("ok", f"Pi web search tools: {extension_path}", "32")
    return extension_path


def _ensure_pi_deep_research(config) -> Path | None:
    if not config.pi.web_search:
        return
    config_dir = Path(os.path.expanduser(config.pi.config_dir))
    extension_dir = config_dir / "extensions" / "llama_bridge_deep_research"
    extension_path = extension_dir / "index.ts"
    extension_dir.mkdir(parents=True, exist_ok=True)
    content = _pi_deep_research_extension(config)
    if not extension_path.exists() or extension_path.read_text(encoding="utf-8") != content:
        extension_path.write_text(content, encoding="utf-8")
    _print_state("ok", f"Pi deep research extension: {extension_path}", "32")
    return extension_path


def _pi_web_tools_extension(config) -> str:
    bridge_url = _server_url(config.server.host, config.server.port)
    api_key = config.server.auth_token
    return f"""import type {{ ExtensionAPI }} from "@mariozechner/pi-coding-agent";
import {{ Type }} from "typebox";

const BRIDGE_URL = {json.dumps(bridge_url)};
const API_KEY = {json.dumps(api_key)};

async function postJson(path: string, body: unknown, signal: AbortSignal) {{
  const response = await fetch(`${{BRIDGE_URL}}${{path}}`, {{
    method: "POST",
    headers: {{
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    }},
    body: JSON.stringify(body),
    signal,
  }});
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
          "Image research is complete. Continue the deep research workflow now: synthesize the report from the verified sources and selected images, then write the finished markdown file.",
          "Use the write tool exactly like this JSON shape: {{\\\"path\\\": \\\"report.md\\\", \\\"content\\\": \\\"<full markdown report>\\\"}}. The path field is required; do not call write with content only."
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
  registerToolCommand("web_search", "Run llama bridge web_search", {{
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
  registerToolCommand("serp_search", "Run SerpAPI search through llama bridge", {{
    title: "SerpAPI search query",
    placeholder: "Search query",
    emptyMessage: "SerpAPI search cancelled: no query entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("serpapi_search", {{ query: input, num: 5 }}, signal);
      const errorText = toolErrorText("serpapi_search", details);
      return {{ text: errorText || formatSearchResults(details.organic_results || details.results || []) || pretty(details), details }};
    }},
  }});
  registerToolCommand("serp-sarch", "Alias for /serp_search", {{
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
  registerToolCommand("image_search", "Run image_research through llama bridge", {{
    title: "Image research query",
    placeholder: "Image search query",
    emptyMessage: "Image research cancelled: no query entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("image_research", {{ query: input, max_results: 3 }}, signal);
      const errorText = toolErrorText("image_research", details);
      return {{ text: errorText || formatImageResults(details) || pretty(details), details }};
    }},
  }});
  registerToolCommand("image-sarch", "Alias for /image_search", {{
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
  registerToolCommand("wiki_page", "Fetch a Wikipedia page through llama bridge", {{
    title: "Wikipedia page title",
    placeholder: "Exact page title",
    emptyMessage: "Wikipedia page cancelled: no title entered.",
    execute: async (input, signal) => {{
      const details = await callBridgeTool("wikipedia_page", {{ title: input, language: "en" }}, signal);
      const errorText = toolErrorText("wikipedia_page", details);
      const text = details.summary ? `${{details.title || input}}\\n${{details.url || ""}}\\n\\n${{details.summary}}` : pretty(details);
      return {{ text: errorText || text, details }};
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


def _pi_deep_research_extension(config) -> str:
    bridge_url = _server_url(config.server.host, config.server.port)
    api_key = config.server.auth_token
    return f"""import type {{ ExtensionAPI }} from "@mariozechner/pi-coding-agent";
import {{ Type }} from "typebox";

const BRIDGE_URL = {json.dumps(bridge_url)};
const API_KEY = {json.dumps(api_key)};
const SOLID_SOURCE_GUIDE = [
  "Prefer primary/official sources: government departments, election commissions, courts, regulators, police/health/education agencies, company filings, official datasets, parliamentary records, central-bank/statistics offices, and original reports.",
  "Prefer international institutions when relevant: UN agencies, World Bank, IMF, OECD, WHO, WTO, IEA, IPCC, ILO, UNESCO, and recognized regional bodies.",
  "Prefer academic and research sources when relevant: peer-reviewed papers, university pages, SSRN/arXiv with caution, think-tank reports with named authors and methodology, official survey datasets, and reputable fact-checkers.",
  "Prefer established news/wire sources for current events: Reuters, Associated Press, AFP, BBC, NPR, PBS, Financial Times, The Economist, The Guardian, New York Times, Washington Post, Wall Street Journal, Bloomberg, CNBC, Al Jazeera, DW, France24, Nikkei Asia, The Hindu, Indian Express, Hindustan Times, NDTV, India Today, Scroll, The Wire, Frontline, LiveMint, Business Standard, and Economic Times when relevant.",
  "Avoid using low-quality sources as main evidence: SEO pages, copied press releases, anonymous blogs, social-media posts, YouTube-only commentary, forums, content farms, AI-generated summaries, and pages without author/date/source transparency.",
].join("\\n");

const PREFERRED_SOURCE_DOMAINS = [
  "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org",
  "ft.com", "economist.com", "theguardian.com", "nytimes.com", "washingtonpost.com",
  "wsj.com", "bloomberg.com", "cnbc.com", "aljazeera.com", "dw.com", "france24.com",
  "nikkei.com", "thehindu.com", "indianexpress.com", "hindustantimes.com", "ndtv.com",
  "indiatoday.in", "scroll.in", "thewire.in", "frontline.thehindu.com", "livemint.com",
  "business-standard.com", "economictimes.indiatimes.com", "pib.gov.in", "eci.gov.in",
  "rbi.org.in", "mospi.gov.in", "data.gov.in", "prsindia.org", "adrindia.org",
  "worldbank.org", "imf.org", "oecd.org", "who.int", "un.org", "unesco.org", "ilo.org",
  "wto.org", "iea.org", "ipcc.ch", "nature.com", "science.org", "nejm.org", "thelancet.com",
  "jamanetwork.com", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "scholar.google.com",
];

const DEEP_RESEARCH_REPORT_INSTRUCTIONS = [
  "Deep research reporting workflow:",
  "- First run deep_research and use its fetched evidence as the base brief.",
  "- Deep research must use at least 10 SerpAPI search passes when serpapi_search is enabled.",
  "- Deep research must use at least 5 Tavily search passes when tavily_search is enabled.",
  "- After the primary deep-search passes finish, run at least 6 web_search verification/recheck passes against important claims, source titles, and conflicting points.",
  "- After deep_research completes, verify important claims and sources with separate web_search, serpapi_search, and tavily_search calls when those tools are available; compare providers instead of treating one provider as enough.",
  "- Source quality matters more than search rank. Prefer official/primary sources, international institutions, academic/research sources, reputable fact-checkers, and established news/wire outlets. Use weak sources only as leads, not as main citations.",
  SOLID_SOURCE_GUIDE,
  "- Use image_research after source verification to collect only 2-3 relevant image URLs with source/provenance metadata.",
  "- After image_research returns, continue immediately into synthesis and file writing; do not stop and wait for the user to type continue.",
  "- After all sources and images are collected, create report.md in the current working directory.",
  "- When using the write tool, call it with BOTH required fields: path and content. Correct shape: {{\\\"path\\\": \\\"report.md\\\", \\\"content\\\": \\\"<full markdown report>\\\"}}. Never call write with content only.",
  "- Use a relative path such as report.md unless the user explicitly asks for a different filename.",
  "- Write report.md as a detailed prepared research report, not a short answer or bullet-only summary.",
  "- Match this report structure: H1 title beginning with 'Research brief:' or 'Research report:', Executive summary, 5-8 numbered analytical sections, Evidence gaps/limitations/recommended sources, Conclusion (synthesis), and References.",
  "- Number analytical headings like '## 1. ...', '## 2. ...'; use paragraphs plus focused bullets inside sections when they improve readability.",
  "- Use inline numbered citations such as [1], [2], [3] throughout the report; every major factual claim, date, statistic, quote, or contested point needs a nearby citation.",
  "- End with a clean, well-structured References section. Put every source on its own line as a numbered Markdown list item: '1. [Source title](https://example.com/full-url) - publisher or site, date if known.' Never place multiple references on one paragraph line.",
  "- Keep citation numbers consistent: inline [1] must map to References item 1, inline [2] to item 2, and so on. Do not skip numbers unless the skipped source was removed everywhere.",
  "- Use descriptive link text in References, not raw bare URLs as the visible text. The URL must still be inside the Markdown link target.",
  "- Explicitly call out uncertainty, conflicting figures, missing official datasets, and evidence limitations instead of smoothing them over.",
  "- Write report.md like a normal sourced article: use the search/source data for article text, headings, citations, and claims; use images only as compact supporting visuals near the relevant section.",
  "- Include the compact image CSS returned by image_research once near the top of report.md, then wrap selected image figures in <div class=\\\"image-grid\\\">...</div>.",
  "- Attach images only if they are clean, clear, readable, well sourced, and help the report: maps, timelines, locations, people, products, charts, event photos, or visual comparisons. Skip blurry thumbnails, low-resolution previews, cropped charts/maps, decorative stock images, or weakly sourced images.",
  "- Prefer the full image_url from image_research over thumbnails. Do not use an image if the source page is missing unless the report explicitly warns that provenance is weak.",
  "- For each image, include a short caption and a source link/citation beside or inside the figure caption.",
  "- The report.md file must be detailed, include inline source URLs/citations, embed selected compact images with nearby source links, and clearly separate verified findings from uncertain or conflicting evidence.",
  "- End report.md with this exact warning: Warning: This report only uses available sources and may contain wrong or incomplete information. Do not blindly believe it; verify important claims independently.",
].join("\\n");

const DEEP_RESEARCH_REPORT_TEMPLATE = [
  "# Research brief: <specific topic>",
  "",
  "Executive summary",
  "<3-6 dense paragraphs summarizing the answer, the strongest evidence, major caveats, and what could not be verified. Use numbered citations like [1].>",
  "",
  "<!-- If images are useful, place the compact CSS from image_research here once. -->",
  "<!-- Then embed 2-3 clean, readable, sourced figures inside <div class=\\\"image-grid\\\">...</div> near the relevant section. Skip unclear or weakly sourced images. -->",
  "",
  "## 1. Background and context",
  "<Explain the baseline facts, timeline, actors, and why the topic matters.>",
  "",
  "## 2. Main findings",
  "<Develop the central findings with evidence, dates, figures, and citations.>",
  "",
  "## 3. Key actors, mechanisms, or causes",
  "<Explain stakeholders, causes, incentives, technical mechanisms, policy forces, or market dynamics as relevant.>",
  "",
  "## 4. Timeline, geography, data, or operational details",
  "<Use whichever dimensions fit the topic. Include compact tables only when they add clarity.>",
  "",
  "## 5. Results, implications, and consequences",
  "<Explain outcomes, short-term effects, long-term implications, and who is affected.>",
  "",
  "## Evidence gaps, limitations and recommended data sources",
  "<List missing primary data, conflicting reports, weak sources, and what should be checked next.>",
  "",
  "## Conclusion (synthesis)",
  "<Pull the evidence together without adding unsupported claims.>",
  "",
  "## References",
  "1. [Descriptive source title](https://example.com/source) - publisher/site, date if known.",
  "2. [Descriptive source title](https://example.com/source) - publisher/site, date if known.",
].join("\\n");

async function postJson(path: string, body: unknown, signal: AbortSignal) {{
  const response = await fetch(`${{BRIDGE_URL}}${{path}}`, {{
    method: "POST",
    headers: {{
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    }},
    body: JSON.stringify(body),
    signal,
  }});
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

async function getJson(path: string, signal: AbortSignal) {{
  const response = await fetch(`${{BRIDGE_URL}}${{path}}`, {{
    method: "GET",
    headers: {{
      "x-api-key": API_KEY,
    }},
    signal,
  }});
  if (!response.ok) {{
    const text = await response.text().catch(() => "");
    throw new Error(`llama bridge ${{path}} failed (${{response.status}}): ${{text || response.statusText}}`);
  }}
  return await response.json();
}}

async function callBridgeTool(name: string, args: unknown, signal: AbortSignal) {{
  const data = await postJson(`/api/tools/${{name}}`, args, signal);
  return data.data ?? data.result?.data ?? data.result ?? data;
}}

function asNumber(value: unknown, fallback: number, min: number, max: number) {{
  const number = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.max(min, Math.min(max, Math.floor(number)));
}}

function compactText(value: unknown, maxLength = 900) {{
  const text = String(value ?? "").replace(/\\s+/g, " ").trim();
  return text.length > maxLength ? `${{text.slice(0, maxLength)}}...` : text;
}}

function callMaybe(owner: any, method: unknown, args: unknown[]) {{
  if (typeof method !== "function") return false;
  try {{
    method.apply(owner, args);
    return true;
  }} catch (error) {{
    return false;
  }}
}}

function createResearchProgress(piApi: any, toolCallId: string) {{
  type StepState = "queued" | "running" | "done" | "error";
  type Step = {{ id: string; title: string; detail: string; state: StepState; startedAt: number; endedAt?: number }};
  const frames = ["|", "/", "-", "\\\\"];
  const steps: Step[] = [];
  let frame = 0;
  let lastText = "";
  let lastNotifyKey = "";

  function upsert(id: string, title: string, detail: string, state: StepState) {{
    let step = steps.find((item) => item.id === id);
    if (!step) {{
      step = {{ id, title, detail, state, startedAt: Date.now() }};
      steps.push(step);
    }}
    step.title = title;
    step.detail = detail;
    step.state = state;
    if (state === "done" || state === "error") step.endedAt = Date.now();
  }}

  function render() {{
    frame = (frame + 1) % frames.length;
    const lines = steps.slice(-8).map((step) => {{
      const marker = step.state === "running" ? frames[frame] : step.state === "done" ? "OK" : step.state === "error" ? "!!" : "..";
      const elapsed = Math.max(0, Math.round(((step.endedAt ?? Date.now()) - step.startedAt) / 1000));
      return `${{marker}} ${{step.title}}${{step.detail ? ` - ${{step.detail}}` : ""}} (${{elapsed}}s)`;
    }});
    return ["Deep research is working", ...lines].join("\\n");
  }}

  function emit() {{
    const text = render();
    if (text === lastText) return;
    lastText = text;
    const payload = {{
      status: "running",
      title: "Deep research",
      text,
      content: text,
      details: {{ steps: snapshot() }},
    }};
    const sent =
      callMaybe(piApi, piApi?.updateToolCall, [toolCallId, payload]) ||
      callMaybe(piApi?.ui, piApi?.ui?.updateToolCall, [toolCallId, payload]) ||
      callMaybe(piApi?.toolCalls, piApi?.toolCalls?.update, [toolCallId, payload]) ||
      callMaybe(piApi, piApi?.setToolStatus, [toolCallId, text]) ||
      callMaybe(piApi?.ui, piApi?.ui?.setStatus, [text]);
    if (!sent) {{
      const latest = steps[steps.length - 1];
      const notifyKey = latest ? `${{latest.id}}:${{latest.state}}:${{latest.detail}}` : text;
      if (notifyKey !== lastNotifyKey) {{
        lastNotifyKey = notifyKey;
        const level = latest?.state === "error" ? "error" : latest?.state === "done" ? "success" : "info";
        callMaybe(piApi?.ui, piApi?.ui?.notify, [latest ? `${{latest.title}}${{latest.detail ? ` - ${{latest.detail}}` : ""}}` : text, level]);
      }}
    }}
  }}

  function step(id: string, title: string, detail = "", state: StepState = "running") {{
    upsert(id, title, detail, state);
    emit();
  }}

  function pulse(id: string, detail?: string) {{
    const step = steps.find((item) => item.id === id);
    if (!step) return;
    if (detail !== undefined) step.detail = detail;
    emit();
  }}

  async function track<T>(id: string, title: string, detail: string, task: Promise<T>) {{
    step(id, title, detail, "running");
    const timer = setInterval(() => pulse(id), 700);
    try {{
      const value = await task;
      step(id, title, detail, "done");
      return value;
    }} catch (error) {{
      step(id, title, String(error), "error");
      throw error;
    }} finally {{
      clearInterval(timer);
    }}
  }}

  function snapshot() {{
    return steps.map((step) => ({{
      title: step.title,
      detail: step.detail,
      state: step.state,
      elapsed_seconds: Math.max(0, Math.round(((step.endedAt ?? Date.now()) - step.startedAt) / 1000)),
    }}));
  }}

  return {{ step, pulse, track, snapshot }};
}}

function pushUnique(values: string[], value: unknown) {{
  const text = String(value ?? "").trim();
  if (text && !values.some((existing) => existing.toLowerCase() === text.toLowerCase())) {{
    values.push(text);
  }}
}}

function hasProvider(providers: string[], provider: string) {{
  return providers.some((item) => item === provider);
}}

function expandQueries(baseQueries: string[], topic: string, minimum: number, variants: string[]) {{
  const expanded: string[] = [];
  for (const query of baseQueries) pushUnique(expanded, query);
  for (const variant of variants) pushUnique(expanded, variant.replaceAll("{{topic}}", topic));
  let index = 1;
  while (expanded.length < minimum) {{
    pushUnique(expanded, `${{topic}} verification angle ${{index}}`);
    index += 1;
  }}
  return expanded.slice(0, Math.max(minimum, baseQueries.length));
}}

function normalizeSearchResults(provider: string, query: string, data: any) {{
  const rawResults =
    Array.isArray(data?.results) ? data.results :
    Array.isArray(data?.organic_results) ? data.organic_results :
    Array.isArray(data?.search_results) ? data.search_results :
    [];
  return rawResults.map((item: any) => ({{
    provider,
    query,
    title: item.title ?? item.name ?? item.heading ?? "Untitled",
    url: item.url ?? item.link ?? item.source_url ?? "",
    snippet: compactText(item.snippet ?? item.content ?? item.summary ?? item.extract ?? ""),
  }})).filter((item: any) => item.title || item.url || item.snippet);
}}

function sourceDomain(url: string) {{
  try {{
    return new URL(url).hostname.replace(/^www\\./, "").toLowerCase();
  }} catch (error) {{
    return "";
  }}
}}

function sourceQualityScore(item: any) {{
  const domain = sourceDomain(String(item?.url || ""));
  const text = `${{item?.title || ""}} ${{item?.snippet || ""}} ${{item?.url || ""}}`.toLowerCase();
  let score = 0;
  if (domain) score += 1;
  if (PREFERRED_SOURCE_DOMAINS.some((preferred) => domain === preferred || domain.endsWith(`.${{preferred}}`))) score += 8;
  if (/\\.(gov|edu)(\\.|$)/i.test(domain) || domain.endsWith(".gov.in") || domain.endsWith(".ac.in")) score += 7;
  if (/(official|government|commission|ministry|department|regulator|court|parliament|dataset|statistics|filing|annual report)/i.test(text)) score += 4;
  if (/(reuters|associated press|\\bap news\\b|afp|bbc|bloomberg|financial times|the hindu|indian express|associated)/i.test(text)) score += 3;
  if (/(wikipedia|youtube|facebook|twitter|x\\.com|reddit|quora|medium\\.com|blogspot|wordpress|pinterest)/i.test(domain)) score -= 5;
  if (/(liveblog|live updates|opinion|editorial|youtube|watch\\?v=|viral|rumor|rumour)/i.test(text)) score -= 2;
  return score;
}}

function sortFindingsByQuality(findings: any[]) {{
  findings.sort((left, right) => {{
    const quality = sourceQualityScore(right) - sourceQualityScore(left);
    if (quality !== 0) return quality;
    return String(left.title || "").localeCompare(String(right.title || ""));
  }});
  return findings;
}}

async function searchProvider(provider: string, query: string, maxResults: number, signal: AbortSignal) {{
  if (provider === "web_search") {{
    const data = await postJson("/api/web_search", {{ query, max_results: maxResults }}, signal);
    if (data?.error || data?.ok === false) throw new Error(String(data?.error?.message ?? data?.error ?? "web_search failed"));
    return normalizeSearchResults(provider, query, data);
  }}
  if (provider === "serpapi_search") {{
    const data = await callBridgeTool("serpapi_search", {{ query, num: maxResults }}, signal);
    const errorText = toolErrorText("serpapi_search", data);
    if (errorText) throw new Error(errorText);
    return normalizeSearchResults(provider, query, data);
  }}
  if (provider === "tavily_search") {{
    const data = await callBridgeTool("tavily_search", {{
      query,
      max_results: maxResults,
      include_answer: true,
    }}, signal);
    const errorText = toolErrorText("tavily_search", data);
    if (errorText) throw new Error(errorText);
    const results = normalizeSearchResults(provider, query, data);
    if (data?.answer) {{
      results.unshift({{
        provider,
        query,
        title: "Tavily synthesized answer",
        url: "",
        snippet: compactText(data.answer, 1200),
      }});
    }}
    return results;
  }}
  if (provider === "wikipedia_search") {{
    const data = await callBridgeTool("wikipedia_search", {{
      query,
      limit: Math.min(maxResults, 10),
      language: "en",
    }}, signal);
    return normalizeSearchResults(provider, query, data);
  }}
  return [];
}}

async function defaultSearchProviders(signal: AbortSignal) {{
  const providers = ["web_search"];
  try {{
    const data = await getJson("/api/tools", signal);
    const enabledTools = new Set((Array.isArray(data?.tools) ? data.tools : [])
      .map((tool: any) => tool?.name)
      .filter((name: unknown): name is string => typeof name === "string"));
    for (const provider of ["serpapi_search", "tavily_search", "wikipedia_search"]) {{
      if (enabledTools.has(provider)) providers.push(provider);
    }}
  }} catch (error) {{
    providers.push("wikipedia_search");
  }}
  return providers;
}}

async function fetchPageEvidence(url: string, signal: AbortSignal) {{
  if (!url || !/^https?:\\/\\//i.test(url)) return null;
  try {{
    const data = await postJson("/api/web_fetch", {{ url }}, signal);
    if (data?.error || data?.ok === false) {{
      const error = data?.error?.message ?? data?.error?.error ?? data?.error ?? data?.result?.error ?? "Fetch failed";
      return {{ url, title: url, content: `Fetch failed: ${{String(error)}}` }};
    }}
    return {{
      url,
      title: data?.title ?? url,
      content: compactText(data?.content ?? data?.text ?? "", 2400),
    }};
  }} catch (error) {{
    return {{ url, title: url, content: `Fetch failed: ${{String(error)}}` }};
  }}
}}

export default function (pi: ExtensionAPI) {{
  pi.registerTool({{
    name: "deep_research",
    label: "Deep Research",
    description: "Pi-only research agent: run at least 10 SerpAPI searches, 5 Tavily searches when enabled, then 6 web_search verification passes, fetch top pages, and return a structured evidence brief for report.md.",
    parameters: Type.Object({{
      topic: Type.String({{ description: "Research topic, question, or claim to investigate." }}),
      questions: Type.Optional(Type.Array(Type.String({{ description: "Optional sub-questions to investigate in parallel." }}))),
      depth: Type.Optional(Type.Number({{ description: "1 quick, 2 balanced, 3 deep. Higher depth creates more query variants.", default: 2 }})),
      max_results_per_query: Type.Optional(Type.Number({{ description: "Results to request from each provider for each query.", default: 5 }})),
      fetch_top_pages: Type.Optional(Type.Number({{ description: "How many discovered URLs to fetch for deeper evidence.", default: 6 }})),
      providers: Type.Optional(Type.Array(Type.String({{ description: "Optional providers: web_search, serpapi_search, tavily_search, wikipedia_search." }}))),
    }}),
    async execute(_toolCallId, params, signal) {{
      const progress = createResearchProgress(pi as any, String(_toolCallId ?? "deep_research"));
      progress.step("plan", "Building research plan", String(params.topic ?? ""));
      const depth = asNumber(params.depth, 2, 1, 3);
      const maxResults = asNumber(params.max_results_per_query, 5, 1, 10);
      const fetchTopPages = asNumber(params.fetch_top_pages, 6, 0, 12);
      const providers = Array.isArray(params.providers) && params.providers.length
        ? params.providers
        : await progress.track("providers", "Checking available providers", "web, SerpAPI, Tavily, Wikipedia", defaultSearchProviders(signal));

      const queries: string[] = [];
      pushUnique(queries, params.topic);
      for (const question of Array.isArray(params.questions) ? params.questions : []) {{
        pushUnique(queries, question);
      }}
      if (depth >= 2) {{
        pushUnique(queries, `${{params.topic}} background evidence`);
        pushUnique(queries, `${{params.topic}} recent developments`);
        pushUnique(queries, `${{params.topic}} criticism limitations`);
      }}
      if (depth >= 3) {{
        pushUnique(queries, `${{params.topic}} primary sources data`);
        pushUnique(queries, `${{params.topic}} official report data`);
        pushUnique(queries, `${{params.topic}} Reuters OR AP OR BBC`);
        pushUnique(queries, `${{params.topic}} expert analysis`);
        pushUnique(queries, `${{params.topic}} timeline`);
        pushUnique(queries, `${{params.topic}} site:gov OR site:edu`);
      }}
      const baseQueries = queries.slice(0, 12);
      const serpapiQueries = hasProvider(providers, "serpapi_search")
        ? expandQueries(baseQueries, String(params.topic), 10, [
          "{{topic}} evidence",
          "{{topic}} sources",
          "{{topic}} facts",
          "{{topic}} latest",
          "{{topic}} expert analysis",
          "{{topic}} criticism",
          "{{topic}} timeline",
          "{{topic}} statistics",
          "{{topic}} official source",
          "{{topic}} official report data",
          "{{topic}} Reuters AP BBC",
          "{{topic}} site:gov OR site:edu",
          "{{topic}} independent source",
        ])
        : [];
      const tavilyQueries = hasProvider(providers, "tavily_search")
        ? expandQueries(baseQueries, String(params.topic), 5, [
          "{{topic}} research evidence",
          "{{topic}} recent developments",
          "{{topic}} primary sources",
          "{{topic}} official data reputable news",
          "{{topic}} analysis",
          "{{topic}} limitations controversy",
        ])
        : [];
      const wikipediaQueries = hasProvider(providers, "wikipedia_search") ? baseQueries.slice(0, 3) : [];
      progress.step(
        "plan",
        "Building research plan",
        `${{serpapiQueries.length}} SerpAPI, ${{tavilyQueries.length}} Tavily, ${{wikipediaQueries.length}} Wikipedia primary passes`,
        "done"
      );

      const searchJobs = [
        ...serpapiQueries.map((query) => progress.track(
          `search:serpapi_search:${{query}}`,
          "Searching serpapi_search",
          query,
          searchProvider("serpapi_search", query, maxResults, signal)
        )),
        ...tavilyQueries.map((query) => progress.track(
          `search:tavily_search:${{query}}`,
          "Searching tavily_search",
          query,
          searchProvider("tavily_search", query, maxResults, signal)
        )),
        ...wikipediaQueries.map((query) => progress.track(
          `search:wikipedia_search:${{query}}`,
          "Searching wikipedia_search",
          query,
          searchProvider("wikipedia_search", query, maxResults, signal)
        )),
      ];
      progress.step("parallel-search", "Running parallel searches", `${{searchJobs.length}} searches in flight`);
      const settled = await Promise.allSettled(searchJobs);
      progress.step("parallel-search", "Running parallel searches", `${{searchJobs.length}} searches complete`, "done");
      const searchErrors: string[] = [];
      const seenUrls = new Set<string>();
      const findings: any[] = [];
      progress.step("merge", "Merging and deduplicating sources", "Normalizing provider results");
      for (const result of settled) {{
        if (result.status === "rejected") {{
          searchErrors.push(String(result.reason));
          continue;
        }}
        for (const item of result.value) {{
          const key = String(item.url || `${{item.provider}}:${{item.title}}`).toLowerCase();
          if (seenUrls.has(key)) continue;
          seenUrls.add(key);
          findings.push(item);
        }}
      }}
      sortFindingsByQuality(findings);
      progress.step("merge", "Merging and deduplicating sources", `${{findings.length}} unique sources`, "done");

      const verificationQueries: string[] = [];
      pushUnique(verificationQueries, `${{params.topic}} verify key facts`);
      pushUnique(verificationQueries, `${{params.topic}} fact check`);
      pushUnique(verificationQueries, `${{params.topic}} source reliability`);
      pushUnique(verificationQueries, `${{params.topic}} conflicting evidence`);
      for (const item of findings.slice(0, 8)) {{
        pushUnique(verificationQueries, `${{item.title}} verification`);
      }}
      while (verificationQueries.length < 6) {{
        pushUnique(verificationQueries, `${{params.topic}} recheck source ${{verificationQueries.length + 1}}`);
      }}
      const verificationJobs = hasProvider(providers, "web_search")
        ? verificationQueries.slice(0, 6).map((query) => progress.track(
          `verify:web_search:${{query}}`,
          "Verifying with web_search",
          query,
          searchProvider("web_search", query, maxResults, signal)
        ))
        : [];
      progress.step("verify-search", "Running verification searches", `${{verificationJobs.length}} web_search rechecks`);
      const verificationSettled = await Promise.allSettled(verificationJobs);
      const verificationFindings: any[] = [];
      for (const result of verificationSettled) {{
        if (result.status === "rejected") {{
          searchErrors.push(String(result.reason));
          continue;
        }}
        for (const item of result.value) {{
          const key = String(item.url || `${{item.provider}}:${{item.title}}`).toLowerCase();
          if (seenUrls.has(key)) continue;
          seenUrls.add(key);
          verificationFindings.push(item);
          findings.push(item);
        }}
      }}
      sortFindingsByQuality(findings);
      progress.step("verify-search", "Running verification searches", `${{verificationFindings.length}} additional verification sources`, "done");

      const urlsToFetch = findings
        .map((item) => item.url)
        .filter((url) => typeof url === "string" && /^https?:\\/\\//i.test(url))
        .slice(0, fetchTopPages);
      progress.step("fetch", "Fetching source pages", `${{urlsToFetch.length}} pages selected`);
      const fetchedSettled = await Promise.allSettled(
        urlsToFetch.map((url) =>
          progress.track(
            `fetch:${{url}}`,
            "Fetching page evidence",
            compactText(url, 120),
            fetchPageEvidence(url, signal)
          )
        )
      );
      const fetchedPages = fetchedSettled
        .filter((result): result is PromiseFulfilledResult<any> => result.status === "fulfilled" && Boolean(result.value))
        .map((result) => result.value);
      progress.step("fetch", "Fetching source pages", `${{fetchedPages.length}} pages fetched`, "done");
      progress.step("brief", "Assembling evidence brief", "Formatting findings and source excerpts");

      const sourceLines = findings.slice(0, 24).map((item, index) =>
        `${{index + 1}}. [${{item.provider}}] ${{item.title}}\\n   URL: ${{item.url || "n/a"}}\\n   Query: ${{item.query}}\\n   Note: ${{item.snippet || "No snippet."}}`
      );
      const fetchedLines = fetchedPages.map((page, index) =>
        `Fetched source ${{index + 1}}: ${{page.title}}\\nURL: ${{page.url}}\\nExcerpt: ${{page.content || "No extractable text."}}`
      );
      progress.step("brief", "Assembling evidence brief", "Ready for synthesis", "done");
      const progressLines = progress.snapshot().map((step, index) =>
        `${{index + 1}}. [${{step.state}}] ${{step.title}}${{step.detail ? ` - ${{step.detail}}` : ""}} (${{step.elapsed_seconds}}s)`
      );
      const text = [
        `Deep research brief for: ${{params.topic}}`,
        "",
        "Work trace:",
        progressLines.join("\\n") || "No progress events recorded.",
        "",
        `Primary queries planned: ${{baseQueries.join(" | ")}}`,
        `Providers attempted: ${{providers.join(", ")}}`,
        `SerpAPI search passes: ${{serpapiQueries.length}}`,
        `Tavily search passes: ${{tavilyQueries.length}}`,
        `Web verification searches: ${{verificationJobs.length}}`,
        `Unique sources found: ${{findings.length}}`,
        `Additional verification sources: ${{verificationFindings.length}}`,
        `Fetched pages: ${{fetchedPages.length}}`,
        searchErrors.length ? `Provider errors: ${{searchErrors.slice(0, 5).join(" | ")}}` : "",
        "",
        "Source quality policy:",
        SOLID_SOURCE_GUIDE,
        "",
        `Preferred domains/examples: ${{PREFERRED_SOURCE_DOMAINS.slice(0, 36).join(", ")}}`,
        "",
        "Search findings:",
        sourceLines.join("\\n\\n") || "No search findings returned.",
        "",
        "Fetched evidence:",
        fetchedLines.join("\\n\\n") || "No pages fetched.",
        "",
        DEEP_RESEARCH_REPORT_INSTRUCTIONS,
        "",
        "Report template to follow:",
        DEEP_RESEARCH_REPORT_TEMPLATE,
        "",
        "Instruction for Pi: synthesize the answer from the evidence above, cite URLs inline, call this tool again with narrower questions if important gaps remain.",
      ].filter(Boolean).join("\\n");

      return {{
        content: [{{ type: "text", text }}],
        details: {{
          topic: params.topic,
          progress: progress.snapshot(),
          queries: baseQueries,
          serpapi_queries: serpapiQueries,
          tavily_queries: tavilyQueries,
          web_verification_queries: verificationQueries.slice(0, 6),
          providers,
          findings,
          verification_findings: verificationFindings,
          fetched_pages: fetchedPages,
          errors: searchErrors,
        }},
      }};
    }},
  }});

  async function startDeepResearch(args: string, ctx: any) {{
    let topic = args.trim();
    if (!topic) {{
      const entered = await ctx.ui.input(
        "Deep research topic",
        "Ask the question or topic for report.md"
      );
      topic = String(entered ?? "").trim();
      if (!topic) {{
        ctx.ui.notify("Deep research cancelled: no topic entered.", "warning");
        return;
      }}
    }}
    const prompt = [
      `Run deep research on: ${{topic}}`,
      "",
      "Call the deep_research tool first with depth 3 unless the topic is very small.",
      "After the tool returns, use the report policy and template included in that tool output.",
      "If images help, call image_research after source verification and include only compact image blocks.",
      "After image_research returns, continue automatically into synthesis and file writing; do not wait for the user to say continue.",
      "Create report.md only after the sources are verified.",
      "When writing the file, call the write tool with both required fields: {{\\\"path\\\": \\\"report.md\\\", \\\"content\\\": \\\"<full markdown report>\\\"}}. Never call write with content only.",
    ].join("\\n");
    if (ctx.isIdle()) {{
      pi.sendUserMessage(prompt);
    }} else {{
      pi.sendUserMessage(prompt, {{ deliverAs: "followUp" }});
      ctx.ui.notify("Deep research queued", "info");
    }}
  }}

  pi.registerCommand("deep_research", {{
    description: "Run Pi-only deep research with parallel web/search providers",
    handler: async (args, ctx) => startDeepResearch(args, ctx),
  }});

  pi.registerCommand("deep", {{
    description: "Open deep research topic prompt",
    handler: async (args, ctx) => startDeepResearch(args, ctx),
  }});

  pi.registerCommand("deep-research", {{
    description: "Alias for /deep_research",
    handler: async (args, ctx) => startDeepResearch(args, ctx),
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


def _http_status(url: str, path: str = "/health") -> str:
    try:
        request = Request(f"{url.rstrip('/')}{path}", method="GET")
        with urlopen(request, timeout=1) as response:
            return f"ok ({response.status})"
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        return f"unreachable ({reason})"
    except TimeoutError:
        return "unreachable (timeout)"
    except OSError as exc:
        return f"unreachable ({exc})"


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
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            _serve_command(config_path, log_path, idle_timeout_seconds, idle_after_file),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
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
