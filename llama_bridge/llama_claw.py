from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OPENCLAW_CONFIG_PATH = "~/.openclaw/llama-openclaw.json"
DEFAULT_OPENCLAW_WORKSPACE = "~/.openclaw/llama-workspace"
DEFAULT_OPENCLAW_MODEL = "qwen3.5:cloud"
DEFAULT_OPENCLAW_INSTALL_PACKAGE = "openclaw"


@dataclass(slots=True)
class OpenClawSafetyConfig:
    config_path: Path
    workspace: Path
    workspace_access: str
    sandbox_backend: str
    model: str


def find_ollama_executable() -> str | None:
    executable = shutil.which("ollama")
    if executable:
        return executable
    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    program_files = os.environ.get("ProgramFiles")
    if local_app_data:
        candidates.extend(
            [
                Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe",
                Path(local_app_data) / "Ollama" / "ollama.exe",
            ]
        )
    if program_files:
        candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
    return _first_existing(candidates)


def find_openclaw_executable() -> str | None:
    executable = shutil.which("openclaw")
    if executable:
        return executable
    candidates: list[Path] = []
    app_data = os.environ.get("APPDATA")
    if app_data:
        candidates.extend(
            [
                Path(app_data) / "npm" / "openclaw.cmd",
                Path(app_data) / "npm" / "openclaw.exe",
            ]
        )
    return _first_existing(candidates)


def ensure_openclaw_safety_config(
    *,
    config_path: str | Path | None = None,
    workspace: str | Path | None = None,
    workspace_access: str = "none",
    sandbox_backend: str = "docker",
    model: str = DEFAULT_OPENCLAW_MODEL,
) -> OpenClawSafetyConfig:
    resolved_config_path = _expand_path(config_path or DEFAULT_OPENCLAW_CONFIG_PATH)
    resolved_workspace = _expand_path(workspace or DEFAULT_OPENCLAW_WORKSPACE)
    workspace_access = _validate_choice(
        workspace_access,
        {"none", "ro", "rw"},
        "OpenClaw sandbox workspace access",
    )
    sandbox_backend = _validate_choice(
        sandbox_backend,
        {"docker", "ssh", "openshell"},
        "OpenClaw sandbox backend",
    )
    _validate_safe_workspace(resolved_workspace)

    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_workspace.mkdir(parents=True, exist_ok=True)
    config_data = _safe_openclaw_config(
        workspace=resolved_workspace,
        workspace_access=workspace_access,
        sandbox_backend=sandbox_backend,
        model=model,
    )
    resolved_config_path.write_text(
        json.dumps(config_data, indent=2) + "\n",
        encoding="utf-8",
    )
    return OpenClawSafetyConfig(
        config_path=resolved_config_path,
        workspace=resolved_workspace,
        workspace_access=workspace_access,
        sandbox_backend=sandbox_backend,
        model=model,
    )


def openclaw_environment(safety: OpenClawSafetyConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENCLAW_CONFIG_PATH"] = str(safety.config_path)
    env["OPENCLAW_STATE_DIR"] = str(safety.config_path.parent / "llama-state")
    env["OPENCLAW_SANDBOX"] = "1"
    return env


def run_ollama_openclaw(
    *,
    model: str,
    safety: OpenClawSafetyConfig,
    passthrough_args: list[str] | None = None,
    config_only: bool = False,
    yes: bool = True,
) -> int:
    ollama_executable = find_ollama_executable()
    if not ollama_executable:
        raise SystemExit("Ollama was not found. Install Ollama, then run `llama openclaw` again.")

    command = [ollama_executable, "launch", "openclaw"]
    if model:
        command.extend(["--model", model])
    if config_only:
        command.append("--config")
    if yes and model and not config_only:
        command.append("--yes")
    command.extend(_strip_passthrough_separator(passthrough_args or []))
    return subprocess.run(command, check=False, env=openclaw_environment(safety)).returncode


def run_openclaw_command(
    command_args: list[str],
    *,
    safety: OpenClawSafetyConfig,
) -> int:
    openclaw_executable = find_openclaw_executable()
    if not openclaw_executable:
        raise SystemExit(
            "OpenClaw was not found. Run `llama openclaw` once so Ollama can install it, "
            "or install it with `npm i -g openclaw`."
        )
    return subprocess.run(
        [openclaw_executable, *_strip_passthrough_separator(command_args)],
        check=False,
        env=openclaw_environment(safety),
    ).returncode


def _safe_openclaw_config(
    *,
    workspace: Path,
    workspace_access: str,
    sandbox_backend: str,
    model: str,
) -> dict[str, Any]:
    return {
        "agents": {
            "defaults": {
                "workspace": _as_posixish_path(workspace),
                "model": model,
                "sandbox": {
                    "mode": "all",
                    "backend": sandbox_backend,
                    "scope": "session",
                    "workspaceAccess": workspace_access,
                    "docker": {
                        "network": "none",
                        "binds": [],
                        "readOnlyRoot": True,
                        "dangerouslyAllowContainerNamespaceJoin": False,
                    },
                    "browser": {
                        "enabled": False,
                        "allowHostControl": False,
                    },
                },
            },
        },
        "tools": {
            "profile": "coding",
            "deny": [
                "gateway",
                "nodes",
                "browser",
                "canvas",
                "message",
                "group:messaging",
            ],
            "exec": {
                "host": "sandbox",
                "security": "allowlist",
                "ask": "on-miss",
                "applyPatch": {
                    "workspaceOnly": True,
                },
            },
        },
    }


def _validate_safe_workspace(path: Path) -> None:
    resolved = path.resolve()
    blocked = _protected_paths()
    for protected in blocked:
        try:
            if resolved == protected or protected in resolved.parents:
                raise SystemExit(
                    f"Refusing to use protected OpenClaw workspace: {resolved}. "
                    "Choose a dedicated project or sandbox directory."
                )
        except OSError:
            continue
    if _is_drive_or_filesystem_root(resolved):
        raise SystemExit(
            f"Refusing to use filesystem root as OpenClaw workspace: {resolved}. "
            "Choose a dedicated project or sandbox directory."
        )


def _protected_paths() -> list[Path]:
    raw_paths: list[str | None] = []
    if os.name == "nt":
        raw_paths.extend(
            [
                os.environ.get("SystemRoot"),
                os.environ.get("WINDIR"),
                os.environ.get("ProgramFiles"),
                os.environ.get("ProgramFiles(x86)"),
                os.environ.get("ProgramData"),
                os.environ.get("APPDATA"),
                os.environ.get("LOCALAPPDATA"),
            ]
        )
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            raw_paths.extend(
                [
                    str(Path(user_profile) / ".ssh"),
                    str(Path(user_profile) / ".aws"),
                    str(Path(user_profile) / ".docker"),
                    str(Path(user_profile) / ".gnupg"),
                    str(Path(user_profile) / ".config"),
                ]
            )
    else:
        raw_paths.extend(["/bin", "/boot", "/dev", "/etc", "/proc", "/root", "/sbin", "/sys", "/usr", "/var"])
        home = os.environ.get("HOME")
        if home:
            raw_paths.extend(
                [
                    str(Path(home) / ".ssh"),
                    str(Path(home) / ".aws"),
                    str(Path(home) / ".docker"),
                    str(Path(home) / ".gnupg"),
                    str(Path(home) / ".config"),
                ]
            )
    paths: list[Path] = []
    for raw_path in raw_paths:
        if not raw_path:
            continue
        try:
            candidate = Path(raw_path).expanduser().resolve()
        except OSError:
            continue
        paths.append(candidate)
    return paths


def _is_drive_or_filesystem_root(path: Path) -> bool:
    anchor = Path(path.anchor) if path.anchor else path
    try:
        return path.resolve() == anchor.resolve()
    except OSError:
        return path == anchor


def _validate_choice(value: str, allowed: set[str], label: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        raise SystemExit(f"{label} must be one of: {', '.join(sorted(allowed))}.")
    return normalized


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def _first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path)
    return None


def _strip_passthrough_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _as_posixish_path(path: Path) -> str:
    value = str(path)
    if sys.platform == "win32":
        return value.replace("\\", "/")
    return value
