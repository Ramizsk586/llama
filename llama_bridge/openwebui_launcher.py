from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH, load_config
from .openwebui_config import (
    OpenWebUIConfig,
    OpenWebUIDiscovery,
    generate_openwebui_env,
    read_pid,
    write_pid,
    clear_pid,
    pid_alive,
    discover_openwebui,
    get_effective_ports,
    get_conda_python_path,
)


LLAMA_PID_PATH = DEFAULT_CONFIG_DIR / "llama.pid"
OPENWEBUI_PID_PATH = DEFAULT_CONFIG_DIR / "openwebui.pid"
LLAMA_LOG_PATH = DEFAULT_CONFIG_DIR / "llama.log"
OPENWEBUI_LOG_PATH = DEFAULT_CONFIG_DIR / "openwebui.log"
OPENWEBUI_ACTIVE_STATE_PATH = DEFAULT_CONFIG_DIR / "openwebui.active.json"


def _compact_path_for_log(path: str, max_len: int = 72) -> str:
    """Shorten paths in log display only."""
    if not path or len(path) <= max_len:
        return path
    p = path.replace("/", "\\")
    if ":" in p:
        drive = p[:2]
        tail = p.split("\\")
        if len(tail) >= 4:
            return drive + "\\...\\" + "\\".join(tail[-2:])
    return "..." + p[-(max_len - 3):]


def _create_hidden_startupinfo() -> subprocess.STARTUPINFO:
    """Create STARTUPINFO that hides the console window on Windows."""
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return si


def _get_hidden_creationflags() -> int:
    """Get Windows process creation flags that hide the console."""
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    if hasattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB"):
        flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
    if hasattr(subprocess, "DETACHED_PROCESS"):
        flags |= subprocess.DETACHED_PROCESS
    return flags if os.name == "nt" else 0


def _read_process_output(
    proc: subprocess.Popen,
    log_path: Path,
    log_queue: queue.Queue[str] | None,
    prefix: str,
) -> None:
    """Read process stdout line by line, write to log file and queue."""
    try:
        with open(str(log_path), "a", encoding="utf-8") as lf:
            for raw_line in proc.stdout or []:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                lf.write(line + "\n")
                lf.flush()
                if log_queue is not None:
                    log_queue.put(f"[{prefix}] {line}")
    except Exception:
        pass


def hidden_popen(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path,
    log_queue: queue.Queue[str] | None = None,
    prefix: str,
) -> subprocess.Popen:
    """Start a hidden subprocess with output captured to log file and optional queue.

    No console/terminal window appears on Windows.
    Output is read by a background thread and written to log_path.
    If log_queue is provided, lines are also pushed as f"[{prefix}] {line}".
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    startupinfo = _create_hidden_startupinfo()
    creationflags = _get_hidden_creationflags()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        startupinfo=startupinfo,
        creationflags=creationflags,
        cwd=str(cwd) if cwd else None,
        env=env,
    )

    t = threading.Thread(
        target=_read_process_output,
        args=(proc, log_path, log_queue, prefix),
        daemon=True,
    )
    t.start()

    return proc


def _serve_command(config_path: Path, log_path: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "serve"]
    else:
        cmd = [sys.executable, "-m", "llama_bridge", "serve"]
    cmd.extend(["--config", str(config_path), "--log-file", str(log_path), "--foreground"])
    return cmd


def find_openwebui_command(python_exe: str | None = None) -> str | None:
    """Locate open-webui entry point using discovery."""
    disc = discover_openwebui()
    if disc.installed:
        if disc.python_exe and disc.package_path:
            main_entry = disc.package_path / "__main__.py"
            if main_entry.exists():
                return str(main_entry)
        if disc.command and disc.command.is_file():
            return str(disc.command)
    if python_exe:
        try:
            result = subprocess.run(
                [python_exe, "-c", "import open_webui; print(open_webui.__file__)"],
                capture_output=True, text=True, timeout=10,
                startupinfo=_create_hidden_startupinfo(),
                creationflags=_get_hidden_creationflags(),
            )
            if result.returncode == 0:
                pkg_path = result.stdout.strip()
                if pkg_path:
                    main_entry = Path(pkg_path).resolve().parent / "__main__.py"
                    if main_entry.exists():
                        return str(main_entry)
        except Exception:
            pass
    import shutil
    cmd = shutil.which("open-webui")
    if cmd:
        return cmd
    try:
        import open_webui
        pkg_dir = Path(open_webui.__file__).resolve().parent
        main_entry = pkg_dir / "__main__.py"
        if main_entry.exists():
            return str(main_entry)
    except ImportError:
        pass
    try:
        import openwebui
        pkg_dir = Path(openwebui.__file__).resolve().parent
        main_entry = pkg_dir / "__main__.py"
        if main_entry.exists():
            return str(main_entry)
    except ImportError:
        pass
    return None


def install_openwebui(
    quiet: bool = False,
    log_queue: queue.Queue[str] | None = None,
) -> bool:
    """Install Open WebUI using hidden subprocess with live log capture."""
    from .openwebui_config import discover_openwebui, clear_discovery_cache

    disc = discover_openwebui()
    if disc.installed:
        return True

    python = get_conda_python_path() or sys.executable

    cfg_path = DEFAULT_CONFIG_PATH
    try:
        cfg = load_config(cfg_path)
        preferred_env = cfg.openwebui.preferred_env_name
        from .openwebui_config import _get_conda_roots
        for root in _get_conda_roots():
            env_py = root / "envs" / preferred_env / "python.exe"
            if env_py.is_file():
                python = str(env_py)
                break
    except Exception:
        pass

    try:
        cmd = [python, "-m", "pip", "install", "open-webui"]
        log_path = DEFAULT_CONFIG_DIR / "openwebui_install.log"

        if quiet and log_queue is None:
            subprocess.run(
                cmd, check=False, capture_output=True, timeout=120,
                startupinfo=_create_hidden_startupinfo(),
                creationflags=_get_hidden_creationflags(),
            )
        else:
            proc = hidden_popen(
                cmd,
                log_path=log_path,
                log_queue=log_queue,
                prefix="pip",
            )
            proc.wait(timeout=120)

        clear_discovery_cache()
        disc = discover_openwebui()
        return disc.installed
    except Exception:
        return False


def _bridge_running(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[bool, int | None]:
    pid = read_pid(LLAMA_PID_PATH)
    if pid is not None and pid_alive(pid):
        return True, pid
    try:
        cfg = load_config(config_path)
        import httpx
        url = f"http://{cfg.server.host}:{cfg.server.port}/health"
        r = httpx.get(url, timeout=2.0)
        if r.status_code == 200:
            return True, None
    except Exception:
        pass
    return False, None


def _openwebui_running() -> tuple[bool, int | None]:
    pid = read_pid(OPENWEBUI_PID_PATH)
    if pid is not None and pid_alive(pid):
        return True, pid
    return False, None


def start_bridge(
    config_path: Path = DEFAULT_CONFIG_PATH,
    log_path: Path | None = None,
    log_queue: queue.Queue[str] | None = None,
) -> tuple[bool, str]:
    """Start bridge in hidden subprocess. No terminal window."""
    running, pid = _bridge_running(config_path)
    if running:
        return True, f"Bridge already running (pid {pid})" if pid else "Bridge already running"

    log_path = log_path or LLAMA_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "LLAMA_DEV_LOG": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = _serve_command(config_path, log_path)

    proc = hidden_popen(
        cmd,
        env=env,
        log_path=log_path,
        log_queue=log_queue,
        prefix="bridge",
        cwd=config_path.parent,
    )
    write_pid(LLAMA_PID_PATH, proc.pid)

    time.sleep(1.5)
    running, _pid = _bridge_running(config_path)
    if not running:
        return False, f"Bridge failed to start. Check log: {log_path}"
    return True, f"Bridge started (pid {proc.pid})"


def stop_bridge(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[bool, str]:
    pid = read_pid(LLAMA_PID_PATH)
    if pid is None:
        try:
            cfg = load_config(config_path)
            import httpx
            httpx.post(f"http://{cfg.server.host}:{cfg.server.port}/shutdown", timeout=3.0)
        except Exception:
            pass
        return True, "Bridge was not running (no pid file)"

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False, capture_output=True, timeout=5,
                startupinfo=_create_hidden_startupinfo(),
                creationflags=_get_hidden_creationflags(),
            )
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            if pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        clear_pid(LLAMA_PID_PATH)

    return True, f"Bridge stopped (pid {pid})"


def start_openwebui(
    config_path: Path = DEFAULT_CONFIG_PATH,
    log_path: Path | None = None,
    log_queue: queue.Queue[str] | None = None,
) -> tuple[bool, str]:
    """Start Open WebUI in hidden subprocess. No terminal window."""
    running, pid = _openwebui_running()
    if running:
        return True, f"Open WebUI already running (pid {pid})"

    config = load_config(config_path)
    owui = config.openwebui
    if not owui.enabled:
        return False, "Open WebUI is disabled in config"

    disc = discover_openwebui(
        preferred_env_name=owui.preferred_env_name,
        preferred_python=owui.preferred_python,
        preferred_command=owui.preferred_command,
    )
    if not disc.installed:
        return False, "Open WebUI is not installed. Run `llama openwebui configure` or `pip install open-webui`"

    env = generate_openwebui_env(owui, config)
    merged_env = {**os.environ}
    merged_env.update(env)

    log_path = log_path or OPENWEBUI_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_cmd = _resolve_openwebui_start_command_from_discovery(disc, owui)

    proc = hidden_popen(
        start_cmd,
        env=merged_env,
        log_path=log_path,
        log_queue=log_queue,
        prefix="openwebui",
        cwd=config_path.parent,
    )
    write_pid(OPENWEBUI_PID_PATH, proc.pid)
    _write_openwebui_active_state(config_path, proc.pid)

    time.sleep(2.5)
    running, _pid = _openwebui_running()
    if not running:
        return False, f"Open WebUI may not have started. Check log: {log_path}"
    return True, f"Open WebUI started (pid {proc.pid}) on http://{owui.host}:{owui.port}"


def _resolve_openwebui_start_command(
    cmd: str,
    owui: OpenWebUIConfig,
    python_exe: str | None = None,
) -> list[str]:
    if cmd.endswith("__main__.py"):
        python = python_exe or sys.executable
        return [python, cmd, "serve", "--host", owui.host, "--port", str(owui.port)]
    return [cmd, "serve", "--host", owui.host, "--port", str(owui.port)]


def _resolve_openwebui_start_command_from_discovery(
    disc: OpenWebUIDiscovery,
    owui: OpenWebUIConfig,
) -> list[str]:
    host = owui.host
    port = str(owui.port)

    if disc.python_exe and disc.package_path:
        main_entry = disc.package_path / "__main__.py"
        if main_entry.exists():
            return [str(disc.python_exe), str(main_entry), "serve", "--host", host, "--port", port]
        return [str(disc.python_exe), "-m", "open_webui", "serve", "--host", host, "--port", port]

    if disc.command:
        return [str(disc.command), "serve", "--host", host, "--port", port]

    if disc.python_exe:
        return [str(disc.python_exe), "-m", "open_webui", "serve", "--host", host, "--port", port]

    return [sys.executable, "-m", "open_webui", "serve", "--host", host, "--port", port]


def stop_openwebui() -> tuple[bool, str]:
    pid = read_pid(OPENWEBUI_PID_PATH)
    if pid is None:
        return True, "Open WebUI was not running (no pid file)"

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False, capture_output=True, timeout=5,
                startupinfo=_create_hidden_startupinfo(),
                creationflags=_get_hidden_creationflags(),
            )
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            if pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        clear_pid(OPENWEBUI_PID_PATH)
        _clear_openwebui_active_state()

    return True, f"Open WebUI stopped (pid {pid})"


def restart_openwebui(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[bool, str]:
    stop_openwebui()
    time.sleep(1)
    return start_openwebui(config_path)


def restart_bridge(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[bool, str]:
    stop_bridge(config_path)
    time.sleep(1)
    return start_bridge(config_path)


def status(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    bridge_running, bridge_pid = _bridge_running(config_path)
    ow_running, ow_pid = _openwebui_running()

    config = load_config(config_path)
    owui = config.openwebui
    ports = get_effective_ports(owui, config)

    disc = discover_openwebui(
        preferred_env_name=owui.preferred_env_name,
        preferred_python=owui.preferred_python,
        preferred_command=owui.preferred_command,
    )
    conda_python = get_conda_python_path()

    return {
        "bridge": {"running": bridge_running, "pid": bridge_pid},
        "openwebui": {
            "running": ow_running,
            "pid": ow_pid,
            "installed": disc.installed,
            "install_path": str(disc.package_path) if disc.package_path else None,
            "discovery": disc,
            "conda_python": conda_python,
        },
        "ports": ports,
        "config": {
            "auth_enabled": owui.auth_enabled,
            "web_search_enabled": owui.web_search_enabled,
            "web_search_provider": owui.web_search_provider,
            "host": owui.host,
            "port": owui.port,
        },
        "urls": {
            "openwebui": f"http://{owui.host}:{owui.port}",
            "bridge_tools": f"http://{config.server.host}:{ports['bridge_tools']}",
            "bridge_llm_only": f"http://{config.server.host}:{ports['bridge_llm_only']}",
        },
    }


def follow_log(log_path: Path, n_lines: int = 50) -> list[str]:
    if not log_path.exists():
        return ["(log file does not exist)"]
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n_lines:]
    except Exception as exc:
        return [f"(error reading log: {exc})"]


def _write_openwebui_active_state(config_path: Path, pid: int) -> None:
    state = {
        "config_path": str(config_path.resolve()),
        "pid": pid,
        "started_at": time.time(),
    }
    try:
        OPENWEBUI_ACTIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        OPENWEBUI_ACTIVE_STATE_PATH.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


def _clear_openwebui_active_state() -> None:
    try:
        OPENWEBUI_ACTIVE_STATE_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def read_openwebui_active_state() -> dict[str, Any] | None:
    if not OPENWEBUI_ACTIVE_STATE_PATH.exists():
        return None
    try:
        return json.loads(OPENWEBUI_ACTIVE_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def stop_all(config_path: Path = DEFAULT_CONFIG_PATH) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    ok, msg = stop_openwebui()
    results.append(("Open WebUI", msg))
    ok, msg = stop_bridge(config_path)
    results.append(("Bridge", msg))
    return results
