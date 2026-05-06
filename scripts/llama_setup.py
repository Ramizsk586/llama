from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


REPO_URL = "https://github.com/Ramizsk586/llama.git"
APP_NAME = "llama"
MIN_PYTHON = (3, 11)
BUILD_MODULES = (
    "PyInstaller",
    "fastapi",
    "httpx",
    "pydantic",
    "uvicorn",
    "yaml",
)
PROGRESS_FRAMES = ("   ", ".  ", ".. ", "...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install llama from source and keep only the packaged exe runtime.")
    parser.add_argument("--repo", default=REPO_URL, help="Git repository to clone.")
    parser.add_argument("--branch", help="Optional branch, tag, or commit to check out.")
    parser.add_argument("--install-dir", type=Path, default=_default_install_dir(), help="Where llama.exe will be installed.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary clone/build folder for debugging.")
    args = parser.parse_args()

    _banner()
    temp_root: Path | None = None
    try:
        _cleanup_stale_temp_dirs()
        git = _ensure_git()
        python = _ensure_python()
        install_dir = args.install_dir.expanduser().resolve()
        _validate_install_dir(install_dir)

        temp_root = Path(tempfile.mkdtemp(prefix="llama-setup-")).resolve()
        source_dir = temp_root / "source"
        venv_dir = temp_root / "venv"
        staging_dir = temp_root / "package"

        clone_command = [git, "clone", "--depth", "1"]
        if args.branch:
            clone_command.extend(["--branch", args.branch])
        clone_command.extend([args.repo, str(source_dir)])
        try:
            _run(clone_command, "Cloning llama")
        except RuntimeError:
            if not args.branch:
                raise
            _run([git, "clone", "--depth", "1", args.repo, str(source_dir)], "Cloning llama")
            _run([git, "-C", str(source_dir), "fetch", "--depth", "1", "origin", args.branch], "Fetching requested ref")
            _run([git, "-C", str(source_dir), "checkout", args.branch], "Checking out requested ref")

        build_python = _prepare_build_python(python, source_dir, venv_dir)
        _build_llama_exe(build_python, source_dir)

        built_dir = source_dir / "dist" / APP_NAME
        if not (built_dir / f"{APP_NAME}.exe").exists():
            raise RuntimeError(f"Build did not create {built_dir / f'{APP_NAME}.exe'}")

        _copy_tree(built_dir, staging_dir)
        _replace_install_dir(staging_dir, install_dir)
        _add_to_user_path(install_dir)
        _set_user_env("LLAMA_HOME", str(install_dir))

        print()
        print("llama is installed.")
        print(f"Installed files: {install_dir}")
        print(f"Run: {install_dir / 'llama.exe'}")
        print("New terminals can also run: llama")
        return 0
    except Exception as exc:  # noqa: BLE001 - installer should report any failure plainly.
        print()
        print(f"Setup failed: {exc}")
        return 1
    finally:
        if temp_root and temp_root.exists() and not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
        if _launched_by_double_click():
            print()
            input("Press Enter to close...")


def _banner() -> None:
    print()
    print("Llama Bridge Setup")
    print("------------------")
    print("Installing the packaged llama.exe runtime.")
    print()


def _default_install_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "Programs" / "llama"
    return Path.home() / "AppData" / "Local" / "Programs" / "llama"


def _ensure_git() -> str:
    git = shutil.which("git")
    if git:
        return git
    _install_with_winget("Git.Git", "Git")
    git = shutil.which("git") or _first_existing(
        [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "cmd" / "git.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Git" / "cmd" / "git.exe",
        ]
    )
    if not git:
        raise RuntimeError("Git was not found. Install Git for Windows, then run setup again.")
    return git


def _ensure_python() -> str:
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
            return executable

    _install_with_winget("Python.Python.3.12", "Python 3.12")
    for command in (["py", "-3.12"], ["python"]):
        executable = _usable_python(command)
        if executable:
            return executable
    raise RuntimeError("Python 3.11+ was not found. Install Python 3.11 or newer, then run setup again.")


def _usable_python(command: list[str]) -> str | None:
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


def _install_with_winget(package_id: str, display_name: str) -> None:
    winget = shutil.which("winget")
    if not winget:
        raise RuntimeError(f"{display_name} is missing and winget is not available to install it automatically.")
    _run(
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


def _first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path)
    return None


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _prepare_build_python(python: str, source_dir: Path, venv_dir: Path) -> str:
    _run([python, "-m", "venv", str(venv_dir)], "Creating build environment")
    venv_python = str(_venv_python(venv_dir))
    _run(
        [
            venv_python,
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
        "Installing build tools and dependencies",
    )
    _run([venv_python, "-m", "pip", "install", "-e", str(source_dir)], "Downloading llama dependencies")
    missing = _missing_python_modules(venv_python, BUILD_MODULES)
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
            _run([venv_python, "-m", "pip", "install", *packages], f"Installing missing modules: {', '.join(missing)}")
    return venv_python


def _missing_python_modules(python: str, modules: tuple[str, ...]) -> list[str]:
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


def _build_llama_exe(build_python: str, source_dir: Path) -> None:
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
    _run(command, "Building llama.exe", cwd=source_dir)


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    _status_line("COPY", f"Preparing runtime files: {target}")
    shutil.copytree(source, target)
    _status_line("OK", "Runtime files prepared")


def _replace_install_dir(staging_dir: Path, install_dir: Path) -> None:
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = install_dir.with_name(f"{install_dir.name}.old")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if install_dir.exists():
        install_dir.rename(backup_dir)
    try:
        shutil.move(str(staging_dir), str(install_dir))
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir, ignore_errors=True)
        if backup_dir.exists():
            backup_dir.rename(install_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


def _validate_install_dir(install_dir: Path) -> None:
    if install_dir.anchor == str(install_dir):
        raise RuntimeError("Refusing to install into a drive root.")
    if install_dir.name.lower() in {"", "windows", "system32", "program files", "program files (x86)", "users"}:
        raise RuntimeError(f"Refusing unsafe install directory: {install_dir}")


def _add_to_user_path(path: Path) -> None:
    if os.name != "nt":
        return
    current = _read_user_env("PATH")
    parts = [part for part in current.split(";") if part]
    if any(Path(part).expanduser() == path for part in parts):
        return
    new_value = ";".join([*parts, str(path)]) if parts else str(path)
    _set_user_env("PATH", new_value)


def _read_user_env(name: str) -> str:
    try:
        result = subprocess.run(
            ["reg", "query", r"HKCU\Environment", "/v", name],
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


def _set_user_env(name: str, value: str) -> None:
    if os.name != "nt":
        return
    _run(["reg", "add", r"HKCU\Environment", "/v", name, "/t", "REG_EXPAND_SZ", "/d", value, "/f"], f"Saving user {name}")


def _run(command: list[str], label: str, cwd: Path | None = None) -> None:
    progress = _ProgressLine(label)
    progress.start()
    try:
        process = subprocess.Popen(command, cwd=cwd)
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                if exit_code != 0:
                    raise subprocess.CalledProcessError(exit_code, command)
                break
            time.sleep(0.1)
    except FileNotFoundError as exc:
        progress.fail("command not found")
        raise RuntimeError(f"Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        progress.fail(f"failed ({exc.returncode})")
        raise RuntimeError(f"{label} failed with exit code {exc.returncode}") from exc
    else:
        progress.succeed()


def _cleanup_stale_temp_dirs() -> None:
    temp_base = Path(tempfile.gettempdir())
    for path in temp_base.glob("llama-setup-*"):
        if not path.is_dir():
            continue
        shutil.rmtree(path, ignore_errors=True)


def _status_line(status: str, label: str) -> None:
    print(f"{status:<6} {label}")


class _ProgressLine:
    def __init__(self, label: str) -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._index = 0
        self._last_width = 0

    def start(self) -> None:
        self._write("RUN   ", "")
        self._thread.start()

    def succeed(self) -> None:
        self._finish("OK    ")

    def fail(self, detail: str) -> None:
        self._finish("FAIL  ", detail)

    def _finish(self, status: str, detail: str = "") -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._write(status, detail)
        print()

    def _animate(self) -> None:
        while not self._stop.is_set():
            frame = PROGRESS_FRAMES[self._index % len(PROGRESS_FRAMES)]
            self._index += 1
            self._write("RUN   ", frame)
            time.sleep(0.35)

    def _write(self, status: str, detail: str) -> None:
        suffix = f" {detail}" if detail else ""
        line = f"\r{status} {self.label}{suffix}"
        padding = " " * max(0, self._last_width - len(line))
        self._last_width = len(line)
        print(f"{line}{padding}", end="", flush=True)


def _launched_by_double_click() -> bool:
    return bool(getattr(sys, "frozen", False)) and os.name == "nt"


if __name__ == "__main__":
    raise SystemExit(main())
