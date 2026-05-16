from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from . import __version__
from .config import load_config


RESET = "\033[0m"
KEY = "\033[1;36m"
GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"

DOT = "\u00b7"
BULLET = "\u2022"
RING = "\u25e6"
CIRCLE = "\u2218"
SUN = "\u2299"
PETAL = "\u274b"
SPARK = "\u2726"
STAR = "\u2727"
BAR = "\u2500"
CHECK = "\u2713"
CROSS = "\u2717"

GLYPH_COLOR = {
    DOT: "\033[92m",
    BULLET: "\033[32m",
    RING: "\033[93m",
    CIRCLE: "\033[33m",
    SUN: "\033[33m",
    PETAL: "\033[97m",
    SPARK: "\033[96m",
    STAR: "\033[1;97m",
}

CANOPY = [1, 3, 5, 7, 9, 11, 13, 15, 17, 15, 13, 11, 9, 7, 5, 3]
LOGO_WIDTH = 39
LOGO_COL_WIDTH = 44
LOGO_HEIGHT = len(CANOPY) + 2
LABEL_WIDTH = 9


def print_llamafetch(config_path: Path) -> None:
    """Print an animated llama bridge info screen."""
    try:
        terminal = shutil.get_terminal_size((100, 24))
        stacked = terminal.columns < 72
        info_width = terminal.columns if stacked else max(30, terminal.columns - LOGO_COL_WIDTH - 2)
        info_lines = _get_info_rows(config_path, info_width)

        if _can_animate():
            _animate(info_lines, stacked)
            return

        output = _compose(_build_logo(frame=0, progress=1.0), info_lines, stacked)
        if _plain_output():
            output = _strip_ansi(output)
        _write_output(output)
    except Exception:
        print("llama info unavailable")


def _animate(info_lines: list[str], stacked: bool) -> None:
    frames = 14
    previous_logo = _logo_canvas(_build_logo(frame=0, progress=1 / frames))
    initial_output = "\n".join(_compose_lines(previous_logo, info_lines, stacked))
    total_lines = initial_output.count("\n") + 1

    sys.stdout.write("\033[?25l")
    try:
        _write_output(initial_output)
        for frame in range(1, frames):
            progress = (frame + 1) / frames
            current_logo = _logo_canvas(_build_logo(frame=frame, progress=progress))
            _patch_logo(previous_logo, current_logo, stacked, total_lines)
            previous_logo = current_logo
            time.sleep(0.045)

        final_logo = _logo_canvas(_build_logo(frame=frames, progress=1.0))
        _patch_logo(previous_logo, final_logo, stacked, total_lines)
        sys.stdout.flush()
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def _build_logo(frame: int, progress: float) -> list[str]:
    visible_canopy = max(1, int(len(CANOPY) * progress + 0.999))
    lines = [_canopy_line(width, row, frame) for row, width in enumerate(CANOPY[:visible_canopy])]
    if progress > 0.86:
        lines.extend(_sprinkle_floor(frame)[: int(2 * ((progress - 0.86) / 0.14) + 0.999)])
    return lines


def _canopy_line(width: int, row: int, frame: int) -> str:
    pad = (LOGO_WIDTH - ((width * 2) - 1)) // 2
    cells = [" "] * LOGO_WIDTH
    center = width // 2
    for index in range(width):
        distance = abs(index - center)
        if distance == center:
            glyph = DOT
        elif distance == center - 1:
            glyph = BULLET
        elif distance == center - 2:
            glyph = RING
        elif distance == center - 3:
            glyph = CIRCLE
        elif distance == 0 and width >= 7:
            glyph = STAR if (row + frame) % 4 == 0 else SPARK
        elif distance <= 1 and width >= 9:
            glyph = SPARK if (index + frame) % 3 == 0 else PETAL
        else:
            glyph = SUN
        cells[pad + (index * 2)] = glyph
    _add_sprinkles(cells, row, frame)
    return _paint_line(cells).rstrip()


def _add_sprinkles(cells: list[str], row: int, frame: int) -> None:
    sprinkle_slots = [
        (2, DOT),
        (5, BULLET),
        (33, DOT),
        (36, BULLET),
    ]
    for col, glyph in sprinkle_slots:
        if _open_sprinkle_slot(cells, col) and (row + col + frame) % 9 == 0:
            cells[col] = glyph


def _open_sprinkle_slot(cells: list[str], col: int) -> bool:
    start = max(0, col - 1)
    end = min(len(cells), col + 2)
    return all(cell == " " for cell in cells[start:end])


def _sprinkle_floor(frame: int) -> list[str]:
    rows = []
    center = LOGO_WIDTH // 2
    for row, offset in enumerate((5, 10)):
        cells = [" "] * LOGO_WIDTH
        for i in range(3):
            col = center - offset + (i * offset)
            glyph = SPARK if (col + row + frame) % 3 == 0 else DOT
            cells[col] = glyph
        rows.append(_paint_line(cells).rstrip())
    return rows


def _paint_line(cells: list[str]) -> str:
    return "".join(_paint(char) if char != " " else char for char in cells)


def _paint(glyph: str) -> str:
    color = GLYPH_COLOR.get(glyph)
    return f"{color}{glyph}{RESET}" if color else glyph


def _compose(logo_lines: list[str], info_lines: list[str], stacked: bool) -> str:
    return "\n".join(_compose_lines(_logo_canvas(logo_lines), info_lines, stacked))


def _compose_lines(logo_lines: list[str], info_lines: list[str], stacked: bool) -> list[str]:
    if stacked:
        return [*logo_lines, "", *info_lines]
    return _render_side_by_side(logo_lines, info_lines).splitlines()


def _logo_canvas(logo_lines: list[str]) -> list[str]:
    return [*logo_lines[:LOGO_HEIGHT], *([""] * max(0, LOGO_HEIGHT - len(logo_lines)))]


def _patch_logo(previous: list[str], current: list[str], stacked: bool, total_lines: int) -> None:
    for row, (old_line, new_line) in enumerate(zip(previous, current)):
        if old_line != new_line:
            _rewrite_logo_row(row, new_line, stacked, total_lines)
    sys.stdout.flush()


def _rewrite_logo_row(row: int, logo_line: str, stacked: bool, total_lines: int) -> None:
    up = total_lines - row
    sys.stdout.write(f"\033[{up}A\r")
    if stacked:
        sys.stdout.write(f"{logo_line}\033[K")
    else:
        sys.stdout.write(_logo_cell(logo_line))
    sys.stdout.write(f"\033[{up}B\r")


def _logo_cell(logo_line: str) -> str:
    visible_len = len(_strip_ansi(logo_line))
    return logo_line + (" " * max(0, LOGO_COL_WIDTH - visible_len))


def _get_info_rows(config_path: Path, max_width: int) -> list[str]:
    cfg = None
    try:
        cfg = load_config(config_path)
    except Exception:
        pass

    user = _get_user()
    host = socket.gethostname()
    header = f"{user}@{host}"
    providers = sorted(getattr(cfg, "providers", {}) or {})
    models = getattr(cfg, "anthropic_models", {}) or {}
    server = getattr(cfg, "server", None)

    rows = [
        _header(header),
        _header(BAR * len(header)),
        *_row("Version", __version__, max_width),
        *_row("OS", f"{platform.system()} {platform.release()} {platform.machine()}".strip(), max_width),
        *_row("Host", platform.node() or host, max_width),
        *_row("Kernel", platform.uname().release, max_width),
        *_row("Uptime", _get_uptime(), max_width),
        *_row("Shell", os.environ.get("SHELL") or os.environ.get("COMSPEC", "unknown"), max_width),
        *_row("Python", sys.version.split()[0], max_width),
        *_row("Config", str(getattr(cfg, "source_path", config_path)) if cfg else "not loaded", max_width),
        *_row("Server", f"{server.host}:{server.port}" if server else "not loaded", max_width),
        *_row("Models", str(len(models)) if cfg else "not loaded", max_width),
        *_row("Providers", _summarize_names(providers) if providers else "not loaded", max_width),
        *_row("Ollama", _check_service("ollama"), max_width),
        *_row("CPU", _get_cpu(), max_width),
    ]
    memory = _get_memory()
    if memory:
        rows.extend(_row("Memory", memory, max_width))
    rows.extend(["", _swatch()])
    return rows


def _row(key: str, value: str, max_width: int) -> list[str]:
    prefix_plain = f"{key:<{LABEL_WIDTH}}:  "
    prefix = f"{KEY}{key:<{LABEL_WIDTH}}{RESET}:  "
    continuation = " " * len(prefix_plain)
    value_width = max(12, max_width - len(prefix_plain))
    plain_value = _strip_ansi(value)

    if len(prefix_plain) + len(plain_value) <= max_width:
        return [prefix + value]

    chunks = textwrap.wrap(
        plain_value,
        width=value_width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [plain_value]
    return [prefix + chunks[0], *[continuation + chunk for chunk in chunks[1:]]]


def _summarize_names(names: list[str], limit: int = 6) -> str:
    if len(names) <= limit:
        return ", ".join(names)
    visible = ", ".join(names[:limit])
    return f"{visible}, +{len(names) - limit} more"


def _header(value: str) -> str:
    return f"{KEY}{value}{RESET}"


def _get_user() -> str:
    try:
        return os.getlogin()
    except OSError:
        return os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"


def _get_uptime() -> str:
    try:
        if sys.platform.startswith("linux"):
            seconds = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
            return _format_duration(seconds)
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "kern.boottime"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
            match = re.search(r"sec = (\d+)", result.stdout)
            if match:
                return _format_duration(time.time() - int(match.group(1)))
        if os.name == "nt":
            return _format_duration(time.monotonic())
    except Exception:
        pass
    return "n/a"


def _format_duration(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    days, minutes = divmod(minutes, 1440)
    hours, minutes = divmod(minutes, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes or not parts:
        parts.append(f"{minutes} min{'s' if minutes != 1 else ''}")
    return ", ".join(parts[:2])


def _get_memory() -> str:
    try:
        import psutil

        mem = psutil.virtual_memory()
        used = int((mem.total - mem.available) / 1024 / 1024)
        total = int(mem.total / 1024 / 1024)
        return f"{used} / {total} MiB"
    except Exception:
        pass

    try:
        data = Path("/proc/meminfo").read_text(encoding="utf-8")
        values = {
            key: int(value)
            for key, value in re.findall(r"^(MemTotal|MemAvailable):\s+(\d+)", data, re.M)
        }
        total = values["MemTotal"] // 1024
        used = (values["MemTotal"] - values["MemAvailable"]) // 1024
        return f"{used} / {total} MiB"
    except Exception:
        return ""


def _get_cpu() -> str:
    cpu = platform.processor()
    if cpu:
        return cpu
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _check_service(name: str) -> str:
    if name == "ollama":
        if not shutil.which("ollama"):
            return f"{GRAY}not found{RESET}"
        if _http_ok("http://127.0.0.1:11434/api/version"):
            return f"{GREEN}running {CHECK}{RESET}"
        return f"{RED}stopped {CROSS}{RESET}"

    return "n/a"


def _http_ok(url: str) -> bool:
    try:
        request = Request(url, method="GET")
        with urlopen(request, timeout=0.5) as response:
            return 200 <= response.status < 500
    except (OSError, URLError, TimeoutError):
        return False


def _process_running(name: str) -> bool:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {name}.exe"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
            return f"{name}.exe" in result.stdout.lower()
        result = subprocess.run(
            ["pgrep", "-f", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
        return result.returncode == 0
    except Exception:
        return False


def _swatch() -> str:
    swatch_colors = [41, 42, 43, 44, 45, 46, 47, 100]
    return "  " + "".join(f"\033[{color}m   " for color in swatch_colors) + RESET


def _render_side_by_side(logo_lines: list[str], info_lines: list[str]) -> str:
    height = max(len(logo_lines), len(info_lines))
    logo = [*logo_lines, *([""] * (height - len(logo_lines)))]
    info = [*info_lines, *([""] * (height - len(info_lines)))]

    output = []
    for logo_line, info_line in zip(logo, info):
        visible_len = len(_strip_ansi(logo_line))
        padding = " " * max(0, LOGO_COL_WIDTH - visible_len)
        output.append(logo_line + padding + "  " + info_line)
    return "\n".join(output)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", s)


def _plain_output() -> bool:
    return os.environ.get("NO_COLOR") is not None or not sys.stdout.isatty()


def _can_animate() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _write_output(output: str) -> None:
    try:
        print(output)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(output.encode("utf-8", errors="replace") + b"\n")
