# Changelog:
# - Added ANSI structured logging, verbose/no-color modes, startup/failure/success boxes.
# - Replaced the dot spinner with step-aware progress bars and an in-place status panel.
# - Added disk, admin, install-path, clone-integrity, lockfile, registry-length, and exe checks.
# - Added dry-run support, rollback on failed installs, cleanup hardening, and a post-install smoke test.

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_URL = "https://github.com/Ramizsk586/llama.git"
AGENT_REPO_URL = "https://github.com/Ramizsk586/llama_agent.git"
APP_NAME = "llama"
MIN_PYTHON = (3, 11)
MIN_NODE_MAJOR = 20
MIN_TEMP_FREE = 2 * 1024**3
MIN_INSTALL_FREE = 500 * 1024**2
MAX_INSTALL_PATH = 200
MAX_USER_PATH = 2048
BUILD_MODULES = (
    "PyInstaller",
    "fastapi",
    "httpx",
    "pydantic",
    "uvicorn",
    "yaml",
)
STEP_LABELS = (
    "Checked prerequisites",
    "Cleaned stale temp dirs",
    "Cloned repository",
    "Created virtual environment",
    "Installed dependencies",
    "Built llama.exe",
    "Installed llama agent",
    "Copied runtime files",
    "Updated PATH & environment",
)
UNICODE_SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
ASCII_SPINNER = ("|", "/", "-", "\\")
SHELL_SPECIAL_CHARS = set("\0;&|<>`\"")
LOCK_PATH = Path(tempfile.gettempdir()) / "llama-setup.lock"
UNICODE_FALLBACKS = str.maketrans(
    {
        "✔": "OK",
        "✖": "X",
        "⚠": "!",
        "·": ".",
        "🦙": "LLAMA",
        "—": "-",
        "═": "=",
        "╔": "+",
        "╗": "+",
        "╚": "+",
        "╝": "+",
        "║": "|",
        "⠋": "|",
        "⠙": "/",
        "⠹": "-",
        "⠸": "\\",
        "⠼": "|",
        "⠴": "/",
        "⠦": "-",
        "⠧": "\\",
        "⠇": "|",
        "⠏": "/",
    }
)


class Log:
    """Print timestamped structured log lines with optional ANSI color."""

    color: bool = False
    verbose: bool = False
    _lock = threading.Lock()

    RESET = "\033[0m"
    DIM = "\033[90m"
    WHITE = "\033[37m"
    CYAN_BOLD = "\033[96;1m"
    GREEN = "\033[92;1m"
    YELLOW = "\033[93;1m"
    RED = "\033[91;1m"
    MAGENTA_BOLD = "\033[95;1m"

    @classmethod
    def configure(cls, no_color: bool, verbose: bool) -> None:
        """Configure output style and verbosity."""
        cls.verbose = verbose
        cls.color = (not no_color) and _terminal_supports_ansi()
        if cls.color and os.name == "nt":
            _enable_windows_ansi()

    @classmethod
    def step(cls, n: int, total: int, label: str) -> None:
        """Log the start of a major numbered step."""
        cls._emit(f"[{n}/{total}] {label}", cls.CYAN_BOLD)

    @classmethod
    def info(cls, msg: str) -> None:
        """Log neutral information."""
        cls._emit(msg, cls.WHITE)

    @classmethod
    def ok(cls, msg: str) -> None:
        """Log a successful action."""
        cls._emit(f"✔ {msg}", cls.GREEN)

    @classmethod
    def warn(cls, msg: str) -> None:
        """Log a non-fatal warning."""
        cls._emit(f"⚠ {msg}", cls.YELLOW)

    @classmethod
    def error(cls, msg: str) -> None:
        """Log a fatal error."""
        cls._emit(f"✖ {msg}", cls.RED)

    @classmethod
    def debug(cls, msg: str) -> None:
        """Log verbose diagnostic information."""
        if cls.verbose:
            cls._emit(msg, cls.DIM)

    @classmethod
    def section(cls, title: str) -> None:
        """Log a magenta section separator."""
        cls._emit(f"════════ {title} ════════", cls.MAGENTA_BOLD)

    @classmethod
    def box(cls, lines: list[str], color: str) -> None:
        """Log a styled box."""
        lines = [_safe_console_text(line) for line in lines]
        width = max(len(_strip_ansi(line)) for line in lines)
        top = "  ╔" + "═" * (width + 2) + "╗"
        bottom = "  ╚" + "═" * (width + 2) + "╝"
        cls._emit_raw(cls._style(top, color))
        for line in lines:
            cls._emit_raw(cls._style(f"  ║ {line.ljust(width)} ║", color))
        cls._emit_raw(cls._style(bottom, color))

    @classmethod
    def _emit(cls, msg: str, color: str) -> None:
        """Emit a single timestamped log line."""
        stamp = cls._style(datetime.now().strftime("%H:%M:%S"), cls.DIM)
        text = cls._style(msg, color)
        cls._emit_raw(f"{stamp} {text}")

    @classmethod
    def _emit_raw(cls, msg: str) -> None:
        """Emit a raw log line."""
        with cls._lock:
            print(_safe_console_text(msg), flush=True)

    @classmethod
    def _style(cls, msg: str, color: str) -> str:
        """Apply ANSI style when color is enabled."""
        if not cls.color:
            return _strip_ansi(msg)
        return f"{color}{msg}{cls.RESET}"


@dataclass
class StepEntry:
    """Track the current state of one installer step."""

    index: int
    label: str
    status: str = "pending"
    start_time: float | None = None
    elapsed: float = 0.0
    detail: str = ""


class StepCounter:
    """Advance and store installer step state."""

    def __init__(self, total: int, labels: tuple[str, ...]) -> None:
        self.total = total
        self.entries = [StepEntry(i + 1, labels[i] if i < len(labels) else f"Step {i + 1}") for i in range(total)]
        self._position = 0
        self._lock = threading.Lock()

    def next(self, label: str | None = None) -> StepEntry:
        """Advance to the next installer step."""
        with self._lock:
            if self._position >= self.total:
                entry = self.entries[-1]
            else:
                entry = self.entries[self._position]
                self._position += 1
            if label:
                entry.label = label
            entry.status = "running"
            entry.start_time = time.perf_counter()
            entry.elapsed = 0.0
            entry.detail = ""
            Log.step(entry.index, self.total, entry.label)
            return entry

    def complete(self, entry: StepEntry, detail: str = "") -> None:
        """Mark a step complete."""
        with self._lock:
            entry.status = "done"
            entry.elapsed = _elapsed(entry)
            entry.detail = detail

    def fail(self, entry: StepEntry, detail: str) -> None:
        """Mark a step failed."""
        with self._lock:
            entry.status = "failed"
            entry.elapsed = _elapsed(entry)
            entry.detail = detail

    @property
    def current_index(self) -> int:
        """Return the one-based current step index."""
        return min(self._position, self.total)


class StatusPanel:
    """Render the step summary panel in place."""

    def __init__(self, steps: StepCounter) -> None:
        self.steps = steps
        self.enabled = Log.color and sys.stdout.isatty()
        self._rendered_lines = 0
        self._lock = threading.Lock()
        self._frame_index = 0
        self._spinner = _spinner_frames()

    def render(self) -> None:
        """Redraw the panel without scrolling."""
        if not self.enabled:
            return
        with self._lock:
            lines = self._build_lines()
            if self._rendered_lines:
                print(f"\033[{self._rendered_lines}F", end="")
            for line in lines:
                print(_safe_console_text(f"\033[K{line}"))
            self._rendered_lines = len(lines)
            self._frame_index += 1

    def clear(self) -> None:
        """Move the cursor below the current panel."""
        if self.enabled and self._rendered_lines:
            print("", flush=True)

    def _build_lines(self) -> list[str]:
        """Build the rendered panel lines."""
        frame = self._spinner[self._frame_index % len(self._spinner)]
        width = 46
        lines = [
            " ═" * 23,
            f"  Llama Bridge Setup  —  Step {self.steps.current_index} of {self.steps.total}",
            " ═" * 23,
        ]
        for entry in self.steps.entries:
            icon = self._icon(entry, frame)
            elapsed = entry.elapsed if entry.status != "running" else _elapsed(entry)
            suffix = f" ({elapsed:.1f}s)" if entry.status in {"running", "done", "failed"} else ""
            label = entry.label + ("..." if entry.status == "running" else "")
            lines.append(f"  {icon}  [{entry.index}] {label:<30}{suffix}")
        lines.append(" ═" * 23)
        return [line[:width + 12] for line in lines]

    @staticmethod
    def _icon(entry: StepEntry, frame: str) -> str:
        """Return a styled status icon."""
        if entry.status == "done":
            return Log._style("✔", Log.GREEN)
        if entry.status == "failed":
            return Log._style("✖", Log.RED)
        if entry.status == "running":
            return Log._style(frame, Log.CYAN_BOLD)
        return Log._style("·", Log.DIM)


class _ProgressBar:
    """Animate one command or phase until its stop event is set."""

    def __init__(self, step: StepEntry, panel: StatusPanel, label: str) -> None:
        self.step = step
        self.panel = panel
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._index = 0
        self._last_width = 0
        self._spinner = _spinner_frames()

    def start(self) -> None:
        """Start the progress animation."""
        self._thread.start()

    def succeed(self) -> None:
        """Finish the progress line as successful."""
        self._finish(True, "done")

    def fail(self, detail: str) -> None:
        """Finish the progress line as failed."""
        self._finish(False, detail)

    def _finish(self, ok: bool, detail: str) -> None:
        """Stop animation and write a final progress state."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.7)
        elapsed = _elapsed(self.step)
        icon = Log._style("✔", Log.GREEN) if ok else Log._style("✖", Log.RED)
        suffix = f" {detail}" if detail and not ok else ""
        self._write_line(f"{icon} [{self.step.index}/{self.panel.steps.total}] {self.label}{suffix} ({elapsed:.1f}s)")
        if not self.panel.enabled:
            print()
        self.panel.render()

    def _animate(self) -> None:
        """Run the spinner loop."""
        while not self._stop.is_set():
            frame = self._spinner[self._index % len(self._spinner)]
            self._index += 1
            elapsed = _elapsed(self.step)
            self._write_line(f"{frame} [{self.step.index}/{self.panel.steps.total}] {self.label}  ({elapsed:.1f}s)")
            self.panel.render()
            time.sleep(0.25)

    def _write_line(self, line: str) -> None:
        """Overwrite the single progress line when the panel is disabled."""
        if self.panel.enabled:
            return
        line = "\r" + line
        padding = " " * max(0, self._last_width - len(_strip_ansi(line)))
        self._last_width = len(_strip_ansi(line))
        print(_safe_console_text(f"{line}{padding}"), end="", flush=True)


@dataclass
class InstallState:
    """Track install replacement state for rollback."""

    backup_dir: Path | None = None
    replace_started: bool = False
    install_succeeded: bool = False


def main() -> int:
    """Run the Llama Bridge installer."""
    parser = argparse.ArgumentParser(description="Install llama from source and keep only the packaged exe runtime.")
    parser.add_argument("--repo", default=REPO_URL, help="Git repository to clone.")
    parser.add_argument("--branch", help="Optional branch, tag, or commit to check out.")
    parser.add_argument("--install-dir", type=Path, default=_default_install_dir(), help="Where llama.exe will be installed.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary clone/build folder for debugging.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show debug logs and subprocess output.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate install actions without writing files or registry keys.")
    parser.add_argument("--install-agent", action="store_true", help="Install llama_agent without prompting.")
    parser.add_argument("--no-install-agent", action="store_true", help="Skip llama_agent without prompting.")
    args = parser.parse_args()

    Log.configure(no_color=args.no_color, verbose=args.verbose)
    started = time.perf_counter()
    steps = StepCounter(total=9, labels=STEP_LABELS)
    panel = StatusPanel(steps)
    temp_root: Path | None = None
    lock_created = False
    install_state = InstallState()
    install_dir = args.install_dir.expanduser().resolve()
    agent_installed = False

    _banner(install_dir)
    try:
        lock_created = _acquire_lock(args.dry_run)

        step = steps.next("Checked prerequisites")
        git = _ensure_git(args.dry_run)
        python = _ensure_python(args.dry_run)
        _refuse_admin()
        _validate_install_dir(install_dir)
        _check_existing_llama_install(install_dir, args, install_state)
        _check_disk_space(Path(tempfile.gettempdir()), install_dir)
        steps.complete(step)
        panel.render()

        step = steps.next("Cleaned stale temp dirs")
        _cleanup_stale_temp_dirs(skip=None, dry_run=args.dry_run)
        steps.complete(step)
        panel.render()

        if args.dry_run:
            _run_dry_install_steps(args, steps, panel, install_dir)
            _success_summary(install_dir, started, path_updated=True, dry_run=True, agent_installed=args.install_agent)
            return 0

        temp_root = Path(tempfile.mkdtemp(prefix="llama-setup-")).resolve()
        source_dir = temp_root / "source"
        venv_dir = temp_root / "venv"
        staging_dir = temp_root / "package"
        agent_source_dir = temp_root / "agent"

        step = steps.next("Cloned repository")
        clone_command = [git, "clone", "--depth", "1"]
        if args.branch:
            clone_command.extend(["--branch", args.branch])
        clone_command.extend([args.repo, str(source_dir)])
        try:
            _run(clone_command, "Cloning repository", panel, step)
        except RuntimeError:
            if not args.branch:
                raise
            Log.warn("Requested branch clone failed; retrying with a direct ref checkout.")
            step.status = "running"
            step.start_time = time.perf_counter()
            _run([git, "clone", "--depth", "1", args.repo, str(source_dir)], "Cloning repository", panel, step)
            _run([git, "-C", str(source_dir), "fetch", "--depth", "1", "origin", args.branch], "Fetching requested ref", panel, step, finish_step=False)
            _run([git, "-C", str(source_dir), "checkout", args.branch], "Checking out requested ref", panel, step)
        _verify_clone_integrity(source_dir)

        step = steps.next("Created virtual environment")
        _run([python, "-m", "venv", str(venv_dir)], "Creating build environment", panel, step)
        build_python = str(_venv_python(venv_dir))

        step = steps.next("Installed dependencies")
        _install_build_dependencies(build_python, source_dir, panel, step)

        step = steps.next("Built llama.exe")
        _build_llama_exe(build_python, source_dir, panel, step)
        built_dir = source_dir / "dist" / APP_NAME
        built_exe = built_dir / f"{APP_NAME}.exe"
        if not built_exe.exists():
            raise RuntimeError(f"Build did not create {built_exe}")
        _verify_exe(built_exe, source_dir)

        install_agent, agent_is_update = _choose_agent_install(args, install_dir)
        step = steps.next("Installed llama agent")
        if install_agent:
            _install_llama_agent(git, agent_source_dir, install_dir / "agent", args.dry_run, panel, step, agent_is_update)
            agent_installed = True
        else:
            Log.info("Skipping llama_agent install.")
            steps.complete(step, "skipped")
            panel.render()

        step = steps.next("Copied runtime files")
        _copy_tree(built_dir, staging_dir, agent_source_dir if install_agent else None, args.dry_run)
        _replace_install_dir(staging_dir, install_dir, install_state, args.dry_run)
        _smoke_test(install_dir, args.dry_run)
        steps.complete(step)
        panel.render()

        step = steps.next("Updated PATH & environment")
        path_updated = _add_to_user_path(install_dir, panel, step, args.dry_run)
        if install_agent:
            _set_user_env("LLAMA_AGENT_HOME", str(install_dir / "agent"), panel, step, args.dry_run, finish_step=False)
        _set_user_env("LLAMA_HOME", str(install_dir), panel, step, args.dry_run, finish_step=True)

        install_state.install_succeeded = True
        _success_summary(install_dir, started, path_updated, dry_run=False, agent_installed=agent_installed)
        return 0
    except Exception as exc:  # noqa: BLE001 - installer should report any failure plainly.
        Log.error(str(exc))
        _rollback_install_dir(install_dir, install_state, args.dry_run)
        _failure_summary(exc)
        return 1
    finally:
        if install_state.install_succeeded and install_state.backup_dir and install_state.backup_dir.exists():
            Log.warn(f"Removing old backup: {install_state.backup_dir}")
            shutil.rmtree(install_state.backup_dir, ignore_errors=True)
        if temp_root and temp_root.exists() and not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
        _cleanup_stale_temp_dirs(skip=temp_root if args.keep_temp else None, dry_run=args.dry_run)
        if lock_created:
            _release_lock(args.dry_run)
        panel.clear()
        if _launched_by_double_click():
            input("Press Enter to close...")


def _banner(install_dir: Path) -> None:
    """Print the startup banner and environment details."""
    Log.box(["      🦙  Llama Bridge Setup", "      Installing llama.exe runtime"], Log.CYAN_BOLD)
    Log.info(f"Python   : {platform.python_version()} ({sys.executable})")
    Log.info(f"OS       : {platform.platform()}")
    Log.info(f"Target   : {install_dir}")
    Log.info(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _success_summary(install_dir: Path, started: float, path_updated: bool, dry_run: bool, agent_installed: bool) -> None:
    """Print the final success summary."""
    elapsed = time.perf_counter() - started
    title = " ✔  Dry run complete!" if dry_run else " ✔  Installation complete!"
    lines = [
        title,
        "",
        f" Location : {install_dir}",
        " Command  : llama",
        f" Total time: {elapsed:.1f} seconds",
    ]
    if agent_installed:
        lines.extend(["", " Agent   : llama agent / --logs / --status / --stop"])
    Log.box(lines, Log.GREEN)
    if not path_updated:
        Log.warn(f"PATH was not changed. Run directly with: {install_dir / 'llama.exe'}")


def _failure_summary(exc: Exception) -> None:
    """Print the final failure summary."""
    Log.box([" ✖  Setup failed", "", f" Error: {exc}", " Re-run with --verbose for more details."], Log.RED)


def _default_install_dir() -> Path:
    """Return the default per-user install directory."""
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "Programs" / "llama"
    return Path.home() / "AppData" / "Local" / "Programs" / "llama"


def _ensure_git(dry_run: bool) -> str:
    """Find Git, installing it with winget when needed."""
    git = shutil.which("git")
    if git:
        Log.ok(f"Found Git: {git}")
        return git
    if dry_run:
        Log.warn("[DRY RUN] Would install Git with winget if missing.")
        return "git"
    _install_with_winget("Git.Git", "Git", dry_run)
    git = shutil.which("git") or _first_existing(
        [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "cmd" / "git.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Git" / "cmd" / "git.exe",
        ]
    )
    if not git:
        raise RuntimeError("Git was not found. Install Git for Windows, then run setup again.")
    return git


def _ensure_python(dry_run: bool) -> str:
    """Find a usable Python 3.11+ interpreter."""
    candidates = [
        [sys.executable] if not getattr(sys, "frozen", False) else [],
        ["py", "-3.12"],
        ["py", "-3.11"],
        ["python"],
        ["python3"],
    ]
    for command in candidates:
        if not command:
            continue
        executable = _usable_python(command)
        if executable:
            Log.ok(f"Found Python: {executable}")
            return executable
    if dry_run:
        Log.warn("[DRY RUN] Would install Python 3.12 with winget if missing.")
        return "python"
    _install_with_winget("Python.Python.3.12", "Python 3.12", dry_run)
    for command in (["py", "-3.12"], ["python"]):
        executable = _usable_python(command)
        if executable:
            return executable
    raise RuntimeError("Python 3.11+ was not found. Install Python 3.11 or newer, then run setup again.")


def _usable_python(command: list[str]) -> str | None:
    """Return the Python executable path if the command is Python 3.11+."""
    try:
        result = subprocess.run(
            [*command, "-c", "import sys; print(sys.executable); print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    major, minor = (int(part) for part in lines[-1].split(".", maxsplit=1))
    if (major, minor) < MIN_PYTHON:
        return None
    return lines[-2]


def _install_with_winget(package_id: str, display_name: str, dry_run: bool) -> None:
    """Install a missing prerequisite through winget."""
    winget = shutil.which("winget")
    if not winget:
        raise RuntimeError(f"{display_name} is missing and winget is not available to install it automatically.")
    if dry_run:
        Log.info(f"[DRY RUN] Would install {display_name} with winget.")
        return
    _run_without_panel(
        [
            winget,
            "install",
            "--id",
            package_id,
            "--exact",
            "--source",
            "winget",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ],
        f"Installing {display_name}",
    )


def _check_existing_llama_install(install_dir: Path, args: argparse.Namespace, install_state: InstallState) -> None:
    """Detect an existing llama.exe and ask whether it should be replaced."""
    llama_exe = install_dir / "llama.exe"
    if not llama_exe.exists():
        return

    installed_version = "unknown"
    try:
        result = subprocess.run(
            [str(llama_exe), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            installed_version = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass

    remote_version = "latest from GitHub"
    Log.info("Llama Bridge is already installed.")
    Log.info(f"  Installed version : {installed_version}")
    Log.info(f"  Available version : {remote_version}")

    if not sys.stdin.isatty():
        Log.warn("Existing install detected but input is not interactive; proceeding with update automatically.")
        _uninstall_existing_llama(install_dir, install_state, args.dry_run)
        return

    print()
    answer = input("Do you want to update to the latest version? [Y/n]: ").strip().lower()
    if answer in {"", "y", "yes"}:
        _uninstall_existing_llama(install_dir, install_state, args.dry_run)
        return

    Log.info("Keeping existing installation. Exiting.")
    raise SystemExit(0)


def _uninstall_existing_llama(install_dir: Path, state: InstallState, dry_run: bool) -> None:
    """Move the current install aside so a new build can replace it."""
    Log.warn(f"Removing existing Llama Bridge installation from {install_dir}...")
    if dry_run:
        Log.info(f"[DRY RUN] Would delete {install_dir}")
        return
    backup_dir = install_dir.with_name(f"{install_dir.name}.old")
    _set_install_backup_state(state, backup_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    install_dir.rename(backup_dir)
    Log.ok(f"Existing installation moved to backup: {backup_dir}")


def _choose_agent_install(args: argparse.Namespace, install_dir: Path) -> tuple[bool, bool]:
    """Ask whether llama_agent should be installed or updated."""
    if args.install_agent and args.no_install_agent:
        raise RuntimeError("Use only one of --install-agent or --no-install-agent.")
    agent_dir = install_dir / "agent"
    agent_installed = (agent_dir / "package.json").exists()

    if agent_installed:
        if args.install_agent:
            return True, True
        if args.no_install_agent:
            return False, False
        if not sys.stdin.isatty():
            Log.warn("Existing agent detected; skipping update in non-interactive mode.")
            return False, False
        print()
        Log.info(f"llama_agent is already installed at {agent_dir}.")
        answer = input("Do you want to update it to the latest version? [Y/n]: ").strip().lower()
        if answer in {"", "y", "yes"}:
            return True, True
        return False, False

    if args.install_agent:
        return True, False
    if args.no_install_agent:
        return False, False
    if not sys.stdin.isatty():
        Log.warn("Input is not interactive; skipping optional llama_agent install.")
        return False, False
    print()
    answer = input("Install llama_agent and enable `llama agent` commands? [Y/n]: ").strip().lower()
    return answer in {"", "y", "yes"}, False


def _install_llama_agent(
    git: str,
    agent_dir: Path,
    final_agent_dir: Path,
    dry_run: bool,
    panel: StatusPanel,
    step: StepEntry,
    is_update: bool,
) -> None:
    """Clone and configure llama_agent for use with Llama Bridge."""
    _ensure_node(dry_run)
    npm = _ensure_npm(dry_run)
    if is_update:
        Log.warn("Removing old agent install...")
        if dry_run:
            Log.info(f"[DRY RUN] Would delete existing agent at {final_agent_dir}")
        elif final_agent_dir.exists():
            shutil.rmtree(final_agent_dir)
    if dry_run:
        Log.info(f"[DRY RUN] Would clone {AGENT_REPO_URL} into {agent_dir}.")
        Log.info("[DRY RUN] Would run npm install, write .env.local, and launch npm run setup.")
        Log.info("[DRY RUN] Would check for npm dependency updates.")
        panel.steps.complete(step)
        panel.render()
        return
    _run([git, "clone", "--depth", "1", AGENT_REPO_URL, str(agent_dir)], "Cloning llama_agent", panel, step, finish_step=False)
    _run([npm, "install"], "Installing llama_agent dependencies", panel, step, cwd=agent_dir, finish_step=False)
    _check_npm_updates(npm, agent_dir, panel, step)
    _write_agent_env(agent_dir)
    if sys.stdin.isatty():
        _run_agent_setup_wizard(npm, agent_dir, final_agent_dir, panel, step)
    else:
        Log.warn("Input is not interactive; skipping llama_agent setup wizard.")
        Log.info(f"Finish agent setup later with: cd {final_agent_dir} && npm run setup")
        _try_agent_codegen(agent_dir, panel, step)
    panel.steps.complete(step)
    panel.render()
    Log.ok("llama_agent configured for Llama Bridge.")
    Log.info("Use `llama agent` to start it, `llama agent --logs` to watch it, `llama agent --status` to inspect it, and `llama agent --stop` to stop it.")


def _check_npm_updates(npm: str, agent_dir: Path, panel: StatusPanel, step: StepEntry) -> None:
    """Offer to update outdated llama_agent npm dependencies."""
    try:
        result = subprocess.run(
            [npm, "outdated", "--json"],
            cwd=agent_dir,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        Log.debug(f"Could not check npm package updates: {exc}")
        return

    output = (result.stdout or "").strip()
    if not output:
        Log.debug("No outdated npm packages found.")
        return
    try:
        outdated = json.loads(output)
    except json.JSONDecodeError:
        Log.debug("No outdated npm packages found.")
        return
    if not isinstance(outdated, dict) or not outdated:
        Log.debug("No outdated npm packages found.")
        return

    compatible_updates: dict[str, str] = {}
    major_updates: dict[str, str] = {}
    rows: list[tuple[str, str, str, str]] = []
    for name, details in sorted(outdated.items()):
        if not isinstance(details, dict):
            continue
        current = str(details.get("current") or "")
        wanted = str(details.get("wanted") or "")
        latest = str(details.get("latest") or "")
        if not current:
            continue
        if wanted and wanted != current:
            compatible_updates[name] = wanted
        if latest and latest != (wanted or current):
            major_updates[name] = latest
        if (wanted and wanted != current) or (latest and latest != current):
            rows.append((name, current, wanted or "-", latest or "-"))

    if not rows:
        Log.ok("All npm packages are up to date.")
        return

    Log.info("Outdated npm packages found:")
    Log.info("  Package                   Current   ->  Wanted    (Latest)")
    Log.info("  ----------------------------------------------------------")
    for name, current, wanted, latest in rows:
        Log.info(f"  {name:<25} {current:<9} ->  {wanted:<9} ({latest})")

    if not sys.stdin.isatty():
        Log.warn("Non-interactive mode; skipping npm dependency update.")
        return

    updated = False
    if compatible_updates:
        answer = input(f"Update {len(compatible_updates)} compatible package(s) to their wanted versions? [Y/n]: ").strip().lower()
        if answer in {"", "y", "yes"}:
            packages = [f"{name}@{version}" for name, version in compatible_updates.items()]
            _run([npm, "install", *packages, "--save"], "Updating compatible npm packages", panel, step, cwd=agent_dir, finish_step=False)
            updated = True

    if major_updates:
        answer = input(
            f"WARNING: {len(major_updates)} package(s) have MAJOR version updates available (may be breaking). Update those too? [y/N]: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            packages = [f"{name}@{version}" for name, version in major_updates.items()]
            _run([npm, "install", *packages, "--save"], "Updating major npm packages", panel, step, cwd=agent_dir, finish_step=False)
            updated = True

    if updated:
        Log.ok("npm packages updated.")


def _ensure_node(dry_run: bool) -> str:
    """Find Node.js 20+ or install it with winget."""
    node = shutil.which("node")
    if node and _node_major(node) >= MIN_NODE_MAJOR:
        Log.ok(f"Found Node.js: {node}")
        return node
    if dry_run:
        Log.warn("[DRY RUN] Would install Node.js LTS with winget if missing or too old.")
        return "node"
    _install_with_winget("OpenJS.NodeJS.LTS", "Node.js LTS", dry_run)
    node = shutil.which("node") or _first_existing(
        [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "node.exe"]
    )
    if not node or _node_major(node) < MIN_NODE_MAJOR:
        raise RuntimeError("Node.js 20+ was not found. Install Node.js LTS, then run setup again.")
    return node


def _node_major(node: str) -> int:
    """Return the detected Node.js major version."""
    try:
        result = subprocess.run([node, "--version"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return 0
    match = re.search(r"v?(\d+)", result.stdout.strip())
    return int(match.group(1)) if match else 0


def _ensure_npm(dry_run: bool) -> str:
    """Find npm for installing llama_agent dependencies."""
    npm = shutil.which("npm") or shutil.which("npm.cmd") or _first_existing(
        [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "npm.cmd"]
    )
    if npm:
        Log.ok(f"Found npm: {npm}")
        return npm
    if dry_run:
        Log.warn("[DRY RUN] Would use npm from Node.js LTS.")
        return "npm"
    raise RuntimeError("npm was not found after installing Node.js. Re-open your terminal and run setup again.")


def _write_agent_env(agent_dir: Path) -> None:
    """Write llama_agent .env.local with Llama Bridge defaults."""
    env_path = agent_dir / ".env.local"
    example_path = agent_dir / ".env.example"
    if not env_path.exists() and example_path.exists():
        shutil.copy2(example_path, env_path)
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")
    values = {
        "LLAMA_BRIDGE_URL": "http://127.0.0.1:8089",
        "LLAMA_BRIDGE_MODEL": "sonnet",
        "LLAMA_BRIDGE_API_KEY": "change-me",
        "PORT": "3456",
        "PUBLIC_URL": "http://localhost:3456",
    }
    text = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for key, value in values.items():
        text = _set_env_line(text, key, value)
    env_path.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")


def _run_agent_setup_wizard(npm: str, agent_dir: Path, final_agent_dir: Path, panel: StatusPanel, step: StepEntry) -> None:
    """Run llama_agent's interactive setup wizard with inherited stdio."""
    panel.clear()
    was_enabled = panel.enabled
    panel.enabled = False
    Log.section("llama_agent setup wizard")
    Log.info("Answer the prompts below to configure Telegram, Convex, Composio, memory search, and ngrok.")
    Log.info("When it finishes, `llama agent` will start the full Boop stack in the background.")
    try:
        command = [npm, "run", "setup"]
        Log.debug("Running: " + subprocess.list2cmdline(command))
        result = subprocess.run(command, cwd=agent_dir, check=False)
        if result.returncode != 0:
            Log.warn(f"llama_agent setup exited with code {result.returncode}.")
            Log.info(f"Finish or retry later with: cd {final_agent_dir} && npm run setup")
        else:
            Log.ok("llama_agent setup wizard completed.")
    finally:
        panel.enabled = was_enabled
        panel.render()


def _set_env_line(lines: list[str], key: str, value: str) -> list[str]:
    """Set or append one KEY=value line."""
    pattern = re.compile(rf"^\s*{re.escape(key)}=")
    replacement = f"{key}={value}"
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = replacement
            return lines
    return [*lines, replacement]


def _try_agent_codegen(agent_dir: Path, panel: StatusPanel, step: StepEntry) -> None:
    """Try to generate Convex client code without making agent install fatal."""
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        Log.warn("npx was not found; run `npm run setup` in the agent folder if startup asks for Convex codegen.")
        return
    Log.info("Generating llama_agent Convex client files when possible.")
    process = subprocess.run(
        [npx, "convex", "dev", "--once"],
        cwd=agent_dir,
        check=False,
        capture_output=not Log.verbose,
        text=True,
    )
    if process.returncode == 0:
        Log.ok("Convex client files generated.")
        return
    Log.warn("Convex setup needs attention; `llama agent` will show details if setup is incomplete.")
    Log.debug((process.stdout or "").rstrip())
    Log.debug((process.stderr or "").rstrip())


def _first_existing(paths: list[Path]) -> str | None:
    """Return the first existing path from a list."""
    for path in paths:
        if path.exists():
            return str(path)
    return None


def _venv_python(venv_dir: Path) -> Path:
    """Return the Python executable path inside a virtual environment."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _install_build_dependencies(build_python: str, source_dir: Path, panel: StatusPanel, step: StepEntry) -> None:
    """Install build tools and project dependencies into the build venv."""
    _run(
        [
            build_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "pyinstaller",
            "fastapi",
            "httpx",
            "pydantic",
            "uvicorn",
            "pyyaml",
        ],
        "Installing build tools",
        panel,
        step,
        finish_step=False,
    )
    _run([build_python, "-m", "pip", "install", "-e", str(source_dir)], "Downloading llama dependencies", panel, step, finish_step=False)
    missing = _missing_python_modules(build_python, BUILD_MODULES)
    if missing:
        package_map = {
            "PyInstaller": "pyinstaller",
            "fastapi": "fastapi",
            "httpx": "httpx",
            "pydantic": "pydantic",
            "uvicorn": "uvicorn",
            "yaml": "pyyaml",
        }
        packages = [package_map[name] for name in missing if name in package_map]
        if packages:
            _run([build_python, "-m", "pip", "install", *packages], f"Installing missing modules: {', '.join(missing)}", panel, step)
            return
    steps = panel.steps
    steps.complete(step)
    panel.render()


def _missing_python_modules(python: str, modules: tuple[str, ...]) -> list[str]:
    """Return missing importable modules from a Python interpreter."""
    script = (
        "import importlib.util, sys; "
        "missing=[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; "
        "print('\\n'.join(missing)); "
        "raise SystemExit(1 if missing else 0)"
    )
    try:
        result = subprocess.run(
            [python, "-c", script, *modules],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return list(modules)
    if result.returncode not in {0, 1}:
        return list(modules)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _build_llama_exe(build_python: str, source_dir: Path, panel: StatusPanel, step: StepEntry) -> None:
    """Build llama.exe with PyInstaller."""
    add_data_args: list[str] = []
    icon_args: list[str] = []
    icon_path = source_dir / "assets" / "llama_bridge.ico"
    if icon_path.exists():
        icon_args = ["--icon", str(icon_path)]
    for filename in (
        "IDENTITY.md",
        "SOUL.md",
        "USER.md",
        "AGENTS.md",
        "TOOLS.md",
        "MEMORY.md",
        "HEARTBEAT.md",
        "EVOLUTION.md",
    ):
        source_file = source_dir / "llama_bridge" / "bot_docs" / filename
        if source_file.exists():
            add_data_args.extend(["--add-data", f"{source_file}{os.pathsep}llama_bridge{os.sep}bot_docs"])

    command = [
        build_python,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        APP_NAME,
        *icon_args,
        "--collect-all",
        "fastapi",
        "--collect-all",
        "uvicorn",
        "--collect-all",
        "pydantic",
        "--hidden-import",
        "yaml",
        "--hidden-import",
        "llama_bridge.teligram",
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.protocols.websockets.auto",
        *add_data_args,
        str(source_dir / "llama_bridge" / "__main__.py"),
    ]
    _run(command, "Building llama.exe", panel, step, cwd=source_dir)


def _copy_tree(source: Path, target: Path, agent_source: Path | None, dry_run: bool) -> None:
    """Copy the built runtime tree into a staging directory."""
    if dry_run:
        Log.info(f"[DRY RUN] Would copy runtime files from {source} to {target}.")
        if agent_source:
            Log.info(f"[DRY RUN] Would include llama_agent from {agent_source}.")
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    if agent_source and agent_source.exists():
        shutil.copytree(agent_source, target / "agent")
    Log.ok("Runtime files prepared.")


def _replace_install_dir(staging_dir: Path, install_dir: Path, state: InstallState, dry_run: bool) -> None:
    """Atomically replace the install directory with the staged runtime."""
    backup_dir = state.backup_dir or install_dir.with_name(f"{install_dir.name}.old")
    if state.backup_dir is None:
        _set_install_backup_state(state, backup_dir)
    if dry_run:
        Log.info(f"[DRY RUN] Would replace {install_dir} with {staging_dir}.")
        return
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    if backup_dir.exists() and install_dir.exists():
        shutil.rmtree(backup_dir)
    if install_dir.exists():
        Log.warn(f"Moving existing install to backup: {backup_dir}")
        install_dir.rename(backup_dir)
    shutil.move(str(staging_dir), str(install_dir))
    Log.ok(f"Installed runtime files to {install_dir}")


def _set_install_backup_state(state: InstallState, backup_dir: Path) -> None:
    """Record the backup directory used for replacement and rollback."""
    state.replace_started = True
    state.backup_dir = backup_dir


def _rollback_install_dir(install_dir: Path, state: InstallState, dry_run: bool) -> None:
    """Restore the old install directory after a failed replacement."""
    if dry_run or not state.replace_started or state.install_succeeded:
        return
    backup_dir = state.backup_dir
    if not backup_dir or not backup_dir.exists():
        return
    Log.warn("Rolling back install directory after failure.")
    if install_dir.exists():
        Log.warn(f"Removing partial install: {install_dir}")
        shutil.rmtree(install_dir, ignore_errors=True)
    Log.warn(f"Restoring backup: {backup_dir}")
    backup_dir.rename(install_dir)


def _validate_install_dir(install_dir: Path) -> None:
    """Reject unsafe install targets."""
    raw = str(install_dir)
    if any(char in raw for char in SHELL_SPECIAL_CHARS):
        raise RuntimeError(f"Refusing install path with unsafe characters: {install_dir}")
    if len(raw) > MAX_INSTALL_PATH:
        raise RuntimeError(f"Install path is too long ({len(raw)} chars, max {MAX_INSTALL_PATH}): {install_dir}")
    if install_dir.anchor == raw:
        raise RuntimeError("Refusing to install into a drive root.")
    if install_dir.name.lower() in {"", "windows", "system32", "program files", "program files (x86)", "users"}:
        raise RuntimeError(f"Refusing unsafe install directory: {install_dir}")
    for prefix in {Path(sys.prefix).resolve(), Path(sys.exec_prefix).resolve()}:
        if _is_parent_or_same(install_dir, prefix):
            raise RuntimeError(f"Refusing to install into a parent of the active Python runtime: {install_dir}")
    system_path = _read_system_env("PATH")
    for part in [item for item in system_path.split(";") if item]:
        if _same_path(Path(os.path.expandvars(part)).expanduser(), install_dir):
            raise RuntimeError(f"Refusing install directory already present on the System PATH: {install_dir}")
    Log.ok("Install directory passed safety checks.")


def _is_parent_or_same(parent: Path, child: Path) -> bool:
    """Return whether parent is child or one of its parents."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _same_path(left: Path, right: Path) -> bool:
    """Compare two paths case-insensitively after resolving what exists."""
    try:
        left_resolved = left.resolve()
    except OSError:
        left_resolved = left.absolute()
    try:
        right_resolved = right.resolve()
    except OSError:
        right_resolved = right.absolute()
    return os.path.normcase(str(left_resolved)) == os.path.normcase(str(right_resolved))


def _add_to_user_path(path: Path, panel: StatusPanel, step: StepEntry, dry_run: bool) -> bool:
    """Add the install directory to the user PATH when safe."""
    if os.name != "nt":
        panel.steps.complete(step)
        panel.render()
        return False
    current = _read_user_env("PATH")
    parts = [part for part in current.split(";") if part]
    if any(_same_path(Path(os.path.expandvars(part)).expanduser(), path) for part in parts):
        Log.ok("Install directory is already on the user PATH.")
        return True
    new_value = ";".join([*parts, str(path)]) if parts else str(path)
    if len(new_value) > MAX_USER_PATH:
        Log.warn(f"Skipping PATH update because it would exceed {MAX_USER_PATH} characters.")
        Log.warn(f"Add this directory manually if needed: {path}")
        return False
    _set_user_env("PATH", new_value, panel, step, dry_run, finish_step=False)
    return True


def _read_user_env(name: str) -> str:
    """Read a user environment variable from the registry."""
    return _read_registry_env(r"HKCU\Environment", name)


def _read_system_env(name: str) -> str:
    """Read a system environment variable from the registry."""
    return _read_registry_env(r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment", name)


def _read_registry_env(root: str, name: str) -> str:
    """Read an environment variable from a registry key."""
    if os.name != "nt":
        return ""
    try:
        result = subprocess.run(
            ["reg", "query", root, "/v", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    for line in result.stdout.splitlines():
        match = re.match(rf"\s*{re.escape(name)}\s+REG_\w+\s+(.*)", line)
        if match:
            return match.group(1).strip()
    return ""


def _set_user_env(name: str, value: str, panel: StatusPanel, step: StepEntry, dry_run: bool, finish_step: bool) -> None:
    """Write a user environment variable."""
    if os.name != "nt":
        if finish_step:
            panel.steps.complete(step)
            panel.render()
        return
    if dry_run:
        Log.info(f"[DRY RUN] Would set user environment variable {name}.")
        if finish_step:
            panel.steps.complete(step)
            panel.render()
        return
    _run(
        ["reg", "add", r"HKCU\Environment", "/v", name, "/t", "REG_EXPAND_SZ", "/d", value, "/f"],
        f"Saving user {name}",
        panel,
        step,
        finish_step=finish_step,
    )


def _run(command: list[str], label: str, panel: StatusPanel, step: StepEntry, cwd: Path | None = None, finish_step: bool = True) -> None:
    """Run a subprocess with progress tracking."""
    Log.debug("Running: " + subprocess.list2cmdline(command))
    progress = _ProgressBar(step, panel, label)
    progress.start()
    try:
        stdout_target = None if Log.verbose else subprocess.PIPE
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout_target, stderr=subprocess.STDOUT, text=True)
        output = ""
        if process.stdout is not None:
            while True:
                line = process.stdout.readline()
                if line:
                    output += line
                    Log.debug(line.rstrip())
                    continue
                if process.poll() is not None:
                    rest = process.stdout.read()
                    if rest:
                        output += rest
                        Log.debug(rest.rstrip())
                    break
                time.sleep(0.05)
        exit_code = process.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command, output=output)
    except FileNotFoundError as exc:
        progress.fail("command not found")
        panel.steps.fail(step, "command not found")
        raise RuntimeError(f"Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        progress.fail(f"failed ({exc.returncode})")
        panel.steps.fail(step, f"exit code {exc.returncode}")
        raise RuntimeError(f"{label} failed with exit code {exc.returncode}") from exc
    else:
        if finish_step:
            panel.steps.complete(step)
        progress.succeed()


def _run_without_panel(command: list[str], label: str) -> None:
    """Run a subprocess before the status panel exists."""
    Log.info(label)
    Log.debug("Running: " + subprocess.list2cmdline(command))
    result = subprocess.run(command, check=False, text=True, capture_output=not Log.verbose)
    if result.stdout:
        Log.debug(result.stdout.rstrip())
    if result.stderr:
        Log.debug(result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    Log.ok(label)


def _cleanup_stale_temp_dirs(skip: Path | None, dry_run: bool) -> None:
    """Remove stale llama setup temporary directories."""
    temp_base = Path(tempfile.gettempdir())
    for path in temp_base.glob("llama-setup-*"):
        if skip and _same_path(path, skip):
            continue
        if not path.is_dir():
            continue
        if dry_run:
            Log.info(f"[DRY RUN] Would remove stale temp directory: {path}")
        else:
            shutil.rmtree(path, ignore_errors=True)
            Log.debug(f"Removed stale temp directory: {path}")


def _check_disk_space(temp_dir: Path, install_dir: Path) -> None:
    """Check free disk space for temp and install drives."""
    temp_free = shutil.disk_usage(temp_dir).free
    if temp_free < MIN_TEMP_FREE:
        raise RuntimeError(f"Not enough disk space in temp dir: {temp_free // 1024**2} MB available, 2048 MB required.")
    install_base = _existing_parent(install_dir)
    install_free = shutil.disk_usage(install_base).free
    if install_free < MIN_INSTALL_FREE:
        raise RuntimeError(f"Not enough disk space for install dir: {install_free // 1024**2} MB available, 500 MB required.")
    Log.ok("Disk space checks passed.")


def _existing_parent(path: Path) -> Path:
    """Return the nearest existing parent for disk usage checks."""
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _refuse_admin() -> None:
    """Refuse to run the user-space installer as Administrator."""
    if os.name != "nt":
        return
    try:
        is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001 - Windows API can fail in restricted shells.
        is_admin = False
    if is_admin:
        raise RuntimeError("This installer must NOT be run as Administrator. Re-run as a normal user.")
    Log.ok("Installer is running as a normal user.")


def _verify_clone_integrity(source_dir: Path) -> None:
    """Verify the cloned repository looks like a llama project."""
    has_project_file = any((source_dir / name).exists() for name in ("pyproject.toml", "setup.py", "setup.cfg"))
    has_entrypoint = (source_dir / "llama_bridge" / "__main__.py").exists()
    if not has_project_file or not has_entrypoint:
        raise RuntimeError("Cloned repository does not look like a valid llama project. Aborting.")
    Log.ok("Cloned repository passed integrity checks.")


def _verify_exe(path: Path, repo_root: Path) -> None:
    """Verify the built executable has expected basic integrity."""
    with path.open("rb") as file:
        magic = file.read(2)
    if magic != b"MZ":
        raise RuntimeError(f"Built file does not look like a Windows executable: {path}")
    size = path.stat().st_size
    if size < 5 * 1024 * 1024:
        raise RuntimeError(f"Built executable is suspiciously small ({size} bytes). Build may have failed silently.")
    _verify_optional_sha256(path, repo_root)
    Log.ok(f"Built executable verified ({size // 1024**2} MB).")


def _verify_optional_sha256(path: Path, repo_root: Path) -> None:
    """Verify SHA256SUMS if it contains an entry for the built executable."""
    sums = repo_root / "SHA256SUMS"
    if not sums.exists():
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest().lower()
    for line in sums.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        expected, filename = parts[0].lower(), parts[-1].lstrip("*")
        normalized = filename.replace("\\", "/")
        if Path(normalized).name == path.name:
            if digest != expected:
                raise RuntimeError(f"SHA256 mismatch for {path.name}.")
            Log.ok("SHA256SUMS verification passed.")
            return
    Log.debug("SHA256SUMS present but no llama.exe entry was found.")


def _smoke_test(install_dir: Path, dry_run: bool) -> None:
    """Run a quick post-install executable sanity check."""
    exe = install_dir / "llama.exe"
    if dry_run:
        Log.info(f"[DRY RUN] Would run smoke test: {exe} --version")
        return
    for args, timeout in ((["--version"], 10), (["--help"], 5)):
        try:
            result = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            Log.warn(f"Post-install smoke test failed to run: {exc}")
            return
        if result.returncode == 0:
            output = (result.stdout or result.stderr).strip().splitlines()
            suffix = f": {output[0]}" if output else ""
            Log.ok(f"Smoke test passed{suffix}")
            return
    Log.warn("Post-install smoke test failed — llama.exe ran but returned non-zero. Installation may be broken.")


def _acquire_lock(dry_run: bool) -> bool:
    """Create the setup lock file unless another live setup owns it."""
    if LOCK_PATH.exists():
        text = LOCK_PATH.read_text(encoding="utf-8", errors="ignore").strip()
        if text.isdigit() and _pid_is_running(int(text)):
            raise RuntimeError(f"Another llama setup process is already running (PID {text}). Please wait for it to finish.")
        Log.warn("Removing stale setup lock file.")
        if not dry_run:
            LOCK_PATH.unlink(missing_ok=True)
    if dry_run:
        Log.info(f"[DRY RUN] Would create lock file: {LOCK_PATH}")
        return False
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release_lock(dry_run: bool) -> None:
    """Delete the setup lock file."""
    if dry_run:
        Log.info(f"[DRY RUN] Would delete lock file: {LOCK_PATH}")
        return
    LOCK_PATH.unlink(missing_ok=True)


def _pid_is_running(pid: int) -> bool:
    """Return whether a PID appears to be alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _run_dry_install_steps(args: argparse.Namespace, steps: StepCounter, panel: StatusPanel, install_dir: Path) -> None:
    """Simulate install steps that would normally write to disk or registry."""
    agent_action = (
        "install llama_agent dependencies and launch its setup wizard"
        if args.install_agent
        else "skip llama_agent"
        if args.no_install_agent
        else "ask whether to install llama_agent and launch its setup wizard if selected"
    )
    dry_actions = [
        ("Cloned repository", f"clone {args.repo} into a temp directory"),
        ("Created virtual environment", "create a Python virtual environment"),
        ("Installed dependencies", "install build and llama dependencies"),
        ("Built llama.exe", "run PyInstaller to build llama.exe"),
        ("Installed llama agent", agent_action),
        ("Copied runtime files", f"copy runtime files into {install_dir}"),
        ("Updated PATH & environment", "update user PATH, LLAMA_HOME, and LLAMA_AGENT_HOME"),
    ]
    for label, description in dry_actions:
        step = steps.next(label)
        Log.info(f"[DRY RUN] Would: {description}.")
        time.sleep(0.05)
        steps.complete(step)
        panel.render()


def _terminal_supports_ansi() -> bool:
    """Return whether stdout likely supports ANSI escape sequences."""
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if term and term.lower() != "dumb":
        return True
    return bool(os.environ.get("WT_SESSION") or os.environ.get("ANSICON") or os.environ.get("ConEmuANSI"))


def _enable_windows_ansi() -> None:
    """Enable virtual-terminal processing in modern Windows consoles."""
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        Log.color = False


def _spinner_frames() -> tuple[str, ...]:
    """Return Unicode spinner frames when stdout can encode them."""
    encoding = sys.stdout.encoding or ""
    try:
        "⠋".encode(encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return ASCII_SPINNER
    return UNICODE_SPINNER


def _elapsed(entry: StepEntry) -> float:
    """Return elapsed seconds for a step."""
    if entry.start_time is None:
        return entry.elapsed
    return time.perf_counter() - entry.start_time


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _safe_console_text(text: str) -> str:
    """Return text encodable by the active stdout encoding."""
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (LookupError, UnicodeEncodeError):
        translated = text.translate(UNICODE_FALLBACKS)
        return translated.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _launched_by_double_click() -> bool:
    """Return whether the frozen installer was likely launched by double-click."""
    return bool(getattr(sys, "frozen", False)) and os.name == "nt"


if __name__ == "__main__":
    raise SystemExit(main())
