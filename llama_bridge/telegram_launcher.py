from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from .config import DEFAULT_CONFIG_PATH, load_config

LOGGER = logging.getLogger("uvicorn.error.telegram_launcher")

TELEGRAM_STATE_PATH = Path("llama.telegram.active.json")
TELEGRAM_PID_PATH = Path("llama.telegram.pid")
TELEGRAM_LOG_PATH = Path("llama.telegram.log")


def _state_path(config_path: Path) -> Path:
    return config_path.parent / TELEGRAM_STATE_PATH


def _pid_path(config_path: Path) -> Path:
    return config_path.parent / TELEGRAM_PID_PATH


def _log_path(config_path: Path) -> Path:
    return config_path.parent / TELEGRAM_LOG_PATH


def _resolve_module_path() -> str:
    return "llama_bridge.teligram"


def start_telegram_bot(
    config_path: Path = DEFAULT_CONFIG_PATH,
    workspace: str | None = None,
) -> subprocess.Popen:
    config = load_config(config_path)
    if not config.telegram.enabled:
        raise RuntimeError("Telegram bot is disabled in config")
    if not config.telegram.bot_token:
        raise RuntimeError("Telegram bot token is not configured")

    if telegram_bot_status(config_path)["running"]:
        raise RuntimeError("Telegram bot is already running")

    st = _state_path(config_path)
    st.write_text(json.dumps({"started_at": time.time(), "config": str(config_path)}), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        _resolve_module_path(),
        "--config",
        str(config_path),
    ]
    if workspace:
        cmd.extend(["--workspace", workspace])

    log_path = _log_path(config_path)
    log_file = log_path.open("a", encoding="utf-8")

    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        startupinfo=startupinfo,
    )

    pid_path = _pid_path(config_path)
    pid_path.write_text(str(proc.pid), encoding="utf-8")

    LOGGER.info("Telegram bot started: pid=%s", proc.pid)
    return proc


def stop_telegram_bot(config_path: Path = DEFAULT_CONFIG_PATH) -> bool:
    pid_path = _pid_path(config_path)
    if not pid_path.exists():
        LOGGER.info("No telegram pid file found")
        _cleanup_state(config_path)
        return False

    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        _cleanup_state(config_path)
        return False

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
        else:
            os.kill(pid, signal.SIGTERM)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.warning("Could not stop telegram bot pid %s: %s", pid, exc)
        _cleanup_state(config_path)
        return False

    _cleanup_state(config_path)
    LOGGER.info("Telegram bot stopped: pid=%s", pid)
    return True


def _cleanup_state(config_path: Path) -> None:
    pid_path = _pid_path(config_path)
    if pid_path.exists():
        pid_path.unlink(missing_ok=True)
    st = _state_path(config_path)
    if st.exists():
        try:
            data = json.loads(st.read_text(encoding="utf-8"))
            data["stopped_at"] = time.time()
            data["running"] = False
            st.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            st.unlink(missing_ok=True)
    LOGGER.info("Telegram bot state cleaned up")


def restart_telegram_bot(config_path: Path = DEFAULT_CONFIG_PATH) -> subprocess.Popen | None:
    stop_telegram_bot(config_path)
    time.sleep(1)
    return start_telegram_bot(config_path)


def telegram_bot_status(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    pid_path = _pid_path(config_path)
    running = False
    pid = None

    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None

    if pid is not None:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            running = str(pid) in result.stdout
        else:
            try:
                os.kill(pid, 0)
                running = True
            except OSError:
                running = False

    st = _state_path(config_path)
    state = {}
    if st.exists():
        try:
            state = json.loads(st.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "running": running,
        "pid": pid,
        "state": state,
        "started_at": state.get("started_at"),
    }


def follow_telegram_log(config_path: Path = DEFAULT_CONFIG_PATH, n_lines: int = 50) -> list[str]:
    log_path = _log_path(config_path)
    if not log_path.exists():
        return ["[No log file found]"]
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-n_lines:]
    except OSError as exc:
        return [f"[Could not read log: {exc}]"]


def test_telegram_token(token: str, timeout: float = 10.0) -> dict[str, Any]:
    client = httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))
    try:
        response = client.get(f"https://api.telegram.org/bot{token}/getMe")
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            return {"ok": False, "error": data.get("description", "unknown error")}
        return {"ok": True, "result": data.get("result", {})}
    except httpx.HTTPStatusError as exc:
        return {"ok": False, "error": f"HTTP {exc.response.status_code}"}
    except httpx.RequestError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        client.close()


def send_forced_message(
    config_path: Path = DEFAULT_CONFIG_PATH,
    chat_id: str = "",
    text: str = "",
) -> dict[str, Any]:
    if not chat_id.strip():
        return {"ok": False, "error": "chat_id is required"}
    if not text.strip():
        return {"ok": False, "error": "message text is required"}

    config = load_config(config_path)
    token = config.telegram.bot_token
    if not token or token.startswith("${"):
        return {"ok": False, "error": "bot token is not configured"}

    allowed = set(config.telegram.allowed_chat_ids or [])
    owners = set(config.telegram.owner_chat_ids or [])
    admins = set(config.telegram.admin_chat_ids or [])
    allow_all = config.telegram.allow_all_chats

    if chat_id not in allowed and chat_id not in owners and chat_id not in admins and not allow_all:
        return {"ok": False, "error": f"chat_id {chat_id} is not allowed"}

    client = httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))
    try:
        response = client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            return {"ok": False, "error": data.get("description", "unknown error")}
        LOGGER.info("forced message sent: chat=%s chars=%s", chat_id, len(text))
        return {"ok": True, "result": data.get("result", {})}
    except httpx.HTTPStatusError as exc:
        error_detail = "unknown"
        try:
            error_detail = exc.response.json().get("description", str(exc))
        except Exception:
            error_detail = str(exc)
        return {"ok": False, "error": error_detail}
    except httpx.RequestError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        client.close()


def broadcast_message(config_path: Path = DEFAULT_CONFIG_PATH, text: str = "") -> dict[str, Any]:
    config = load_config(config_path)
    chat_ids = set(config.telegram.allowed_chat_ids or [])
    chat_ids.update(config.telegram.owner_chat_ids or [])
    chat_ids.update(config.telegram.admin_chat_ids or [])

    results: list[dict[str, Any]] = []
    errors = 0
    for cid in chat_ids:
        result = send_forced_message(config_path, str(cid).strip(), text)
        if result.get("ok"):
            results.append({"chat_id": cid, "ok": True})
        else:
            results.append({"chat_id": cid, "ok": False, "error": result.get("error")})
            errors += 1
        time.sleep(0.05)

    return {"ok": errors == 0, "total": len(chat_ids), "errors": errors, "results": results}
