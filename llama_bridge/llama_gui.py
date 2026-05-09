from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, load_config, BridgeConfig

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False
    class FakeTK:
        class Tk:
            def __init__(self):
                raise RuntimeError("Tkinter not available")
    tk = FakeTK()


BG = "#050505"
BG_ALT = "#0B0B0B"
SURFACE = "#111111"
SURFACE2 = "#171717"
SURFACE3 = "#1D1D1D"
BORDER = "#2A2A2A"
TEXT = "#E0E0E0"
TEXT_BRIGHT = "#FAFAFA"
MUTED = "#9A9A9A"
GREEN = "#4FD1A1"
RED = "#FF6B6B"
YELLOW = "#F2C66D"
CARD_GREEN = "#102A28"
CARD_RED = "#34181B"
CARD_GREEN_BORDER = "#1E6054"
CARD_RED_BORDER = "#7B3840"
BUTTON_TOP = "#1A2840"
BUTTON_BOTTOM = "#152035"
BUTTON_BORDER = "#314766"

LAYOUT = {
    "HEADER_H": 90,
    "PAD": 24,
    "CARD_H": 64,
    "CARD_GAP": 10,
    "BTN_H": 38,
    "WIN_W": 620,
    "WIN_H": 640,
}
PAD = LAYOUT["PAD"]


class AppStatus(Enum):
    SETUP = auto()
    READY = auto()
    RUNNING = auto()
    ERROR = auto()


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._enter, add=True)
        widget.bind("<Leave>", self._leave, add=True)

    def _enter(self, _event: Any = None) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tip, text=self.text, bg=SURFACE3, fg=TEXT_BRIGHT,
                       font=("Segoe UI", 9), padx=8, pady=4, relief="solid", bd=1)
        lbl.pack()

    def _leave(self, _event: Any = None) -> None:
        if self.tip:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None


def _set_dark_titlebar(root: tk.Tk) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int),
        )
    except Exception:
        pass


def _center_window(root: tk.Tk, w: int, h: int) -> None:
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")


def _compact_path(value: str, max_chars: int = 72) -> str:
    if not value:
        return "none"
    text = str(value)
    if len(text) <= max_chars:
        return text
    normalized = text.replace("/", "\\")
    parts = normalized.split("\\")
    if len(parts) >= 4 and ":" in parts[0]:
        drive = parts[0]
        for keep in range(3, 0, -1):
            tail = parts[-keep:]
            candidate = drive + "\\...\\" + "\\".join(tail)
            if len(candidate) <= max_chars:
                return candidate
    keep_left = max(10, max_chars // 2 - 4)
    keep_right = max_chars - keep_left - 3
    return text[:keep_left] + "..." + text[-keep_right:]


def _read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class LlamaControlCenter:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self._stopped = threading.Event()

        W, H = LAYOUT["WIN_W"], LAYOUT["WIN_H"]
        self.root = tk.Tk()
        self.root.title("Llama Bridge - Control Center")
        _center_window(self.root, W, H)
        self.root.minsize(580, 560)
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        _set_dark_titlebar(self.root)

        self.status = AppStatus.READY
        self._config: BridgeConfig | None = None
        self._log_lines: list[str] = []
        self._server_pid: int | None = None

        self.header_canvas: tk.Canvas | None = None
        self.cards_canvas: tk.Canvas | None = None
        self.util_btns: dict[str, tk.Button] = {}

        self._card_data = [
            {"key": "server", "title": "Server", "ok": False, "subtitle": "Stopped", "status": "Stopped"},
            {"key": "providers", "title": "Providers", "ok": False, "subtitle": "None configured", "status": "None"},
            {"key": "cli_tools", "title": "CLI Tools", "ok": False, "subtitle": "0 configured", "status": "None"},
            {"key": "models", "title": "Anthropic Models", "ok": False, "subtitle": "0 aliases", "status": "None"},
        ]

        self._build_ui()
        self._load_config()
        self._refresh_all()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        self.header_canvas = tk.Canvas(main, bg=BG, highlightthickness=0)
        self.header_canvas.pack(fill="x", side="top")
        self.header_canvas.configure(height=LAYOUT["HEADER_H"])

        self.cards_canvas = tk.Canvas(main, bg=BG, highlightthickness=0)
        self.cards_canvas.pack(fill="both", expand=True, side="top")

        sep = tk.Frame(main, bg=BORDER, height=1)
        sep.pack(fill="x", side="bottom")

        self._build_footer(main)

        self.header_canvas.bind("<Configure>", lambda e: self._draw_header())
        self.cards_canvas.bind("<Configure>", lambda e: self._draw_cards())

    def _build_footer(self, parent: tk.Frame) -> None:
        footer = tk.Frame(parent, bg=BG)
        footer.pack(fill="x", side="bottom", pady=(6, 10))

        util = tk.Frame(footer, bg=BG)
        util.pack(fill="x", padx=PAD)

        util_items = [
            ("btn_server", "\u2699 Server Config", self._open_server_config),
            ("btn_providers", "\U0001f310 Providers", self._open_providers),
            ("btn_cli", "\U0001f528 CLI Tools", self._open_cli_tools),
            ("btn_models", "\U0001f916 Models", self._open_models),
            ("btn_logs", "\U0001f4cb Logs", self._open_logs),
            ("btn_details", "\u2139 Details", self._open_details),
        ]
        for i, (key, text, cmd) in enumerate(util_items):
            btn = tk.Button(
                util, text=text, command=cmd,
                bg=BG_ALT, fg=MUTED, font=("Segoe UI", 9),
                relief="flat", bd=0, padx=8, pady=2,
                activebackground=SURFACE3, activeforeground=TEXT_BRIGHT,
                cursor="hand2",
            )
            btn.pack(side="left" if i < 5 else "right", padx=(0, 12))
            self.util_btns[key] = btn
            tips = {
                "btn_server": "Configure server host, port, auth",
                "btn_providers": "View and manage API providers",
                "btn_cli": "Configure CLI tool settings",
                "btn_models": "Manage anthropic model aliases",
                "btn_logs": "View server logs",
                "btn_details": "Show full technical details",
            }
            ToolTip(btn, tips[key])

    def _draw_header(self) -> None:
        cv = self.header_canvas
        if not cv:
            return
        cv.delete("all")
        w = cv.winfo_width() or LAYOUT["WIN_W"]
        h = LAYOUT["HEADER_H"]
        p = PAD

        cv.create_text(p, 16, anchor="nw", text="Llama Bridge Control Center",
                       font=("Segoe UI", 16, "bold"), fill=TEXT_BRIGHT)

        subtitle = self._subtitle_text()
        cv.create_text(p, h // 2 + 10, anchor="nw", text=subtitle,
                       font=("Segoe UI", 10), fill=MUTED)

        badge_text, badge_color = self._get_badge_info()
        btxt = f"  {badge_text}  "
        font_sz = 9
        tw = len(btxt) * (font_sz + 1)
        bx = w - p - tw - 12
        by = 20
        bw = tw + 12
        bh = 26
        cv.create_rectangle(bx, by, bx + bw, by + bh, fill="", outline=badge_color, width=1)
        cv.create_text(bx + bw // 2, by + bh // 2, text=badge_text,
                       font=("Segoe UI", 9, "bold"), fill=badge_color)

        sep_color = self._get_separator_color()
        cv.create_rectangle(0, h - 3, w, h, fill=sep_color, outline="")

    def _draw_cards(self) -> None:
        cv = self.cards_canvas
        if not cv:
            return
        cv.delete("all")
        w = cv.winfo_width() or LAYOUT["WIN_W"]
        p = PAD
        card_w = w - 2 * p
        y = 12
        ch = LAYOUT["CARD_H"]

        for card in self._card_data:
            ok = card["ok"]
            fill = CARD_GREEN if ok else CARD_RED
            border = CARD_GREEN_BORDER if ok else CARD_RED_BORDER
            accent = GREEN if ok else RED

            cv.create_rectangle(p, y, p + card_w, y + ch, fill=fill, outline=border, width=1)
            cv.create_rectangle(p, y, p + 5, y + ch, fill=accent, outline=accent, width=0)

            cx, cy = p + 34, y + ch // 2
            cr = 14
            cv.create_oval(cx - cr, cy - cr, cx + cr, cy + cr, outline=accent, width=2)
            if ok:
                cv.create_line(cx - 7, cy, cx - 1, cy + 6, cx + 7, cy - 6,
                               fill=accent, width=2, capstyle="round", joinstyle="round")
            else:
                cv.create_line(cx - 5, cy - 5, cx + 5, cy + 5, fill=accent, width=2)
                cv.create_line(cx + 5, cy - 5, cx - 5, cy + 5, fill=accent, width=2)

            tx = p + 68
            cv.create_text(tx, y + 14, text=card["title"],
                           fill=TEXT_BRIGHT, font=("Segoe UI", 11, "bold"), anchor="w")

            sub = card.get("subtitle", "")
            safe_sub = self._ellipsize(cv, sub, ("Segoe UI", 9), card_w - 190)
            cv.create_text(tx, y + 37, text=safe_sub,
                           fill=MUTED, font=("Segoe UI", 9), anchor="w")

            st = card.get("status", "")
            cv.create_text(p + card_w - 14, y + ch // 2, text=st,
                           fill=accent, font=("Segoe UI", 9, "bold"), anchor="e")

            y += ch + LAYOUT["CARD_GAP"]

    @staticmethod
    def _ellipsize(cv: tk.Canvas, text: str, font: tuple[str, int], max_width: int) -> str:
        if not text or max_width < 20:
            return ""
        tid = cv.create_text(-9999, -9999, text=text, font=font, anchor="w")
        try:
            bbox = cv.bbox(tid)
            if bbox and (bbox[2] - bbox[0]) <= max_width:
                return text
        finally:
            cv.delete(tid)
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = text[:mid] + "..."
            tid = cv.create_text(-9999, -9999, text=candidate, font=font, anchor="w")
            bbox = cv.bbox(tid)
            cv.delete(tid)
            if bbox and (bbox[2] - bbox[0]) <= max_width:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "..." if lo > 0 else "..."

    def _subtitle_text(self) -> str:
        if self.status == AppStatus.READY:
            return "Server is ready."
        if self.status == AppStatus.RUNNING:
            return "Server is running."
        if self.status == AppStatus.ERROR:
            return f"Error \u2014 check Logs"
        return ""

    def _get_badge_info(self) -> tuple[str, str]:
        s = self.status
        if s == AppStatus.ERROR:
            return "ERROR", RED
        if s == AppStatus.READY:
            return "READY", GREEN
        if s == AppStatus.RUNNING:
            return "RUNNING", GREEN
        return "READY", GREEN

    def _get_separator_color(self) -> str:
        s = self.status
        if s == AppStatus.ERROR:
            return RED
        if s == AppStatus.RUNNING:
            return GREEN
        return GREEN

    def _update_card_data(self) -> None:
        cfg = self._config
        if not cfg:
            return

        # Server
        server = cfg.server
        running = self._server_pid is not None and _pid_alive(self._server_pid)
        self._card_data[0] = {
            "key": "server", "title": "Server",
            "ok": running,
            "subtitle": f"{server.host}:{server.port}" if not running else f"{server.host}:{server.port} (pid={self._server_pid})",
            "status": "Running" if running else "Stopped",
        }

        # Providers
        prov_count = len(cfg.providers)
        prov_configured = sum(1 for p in cfg.providers.values() if p.api_key and not p.api_key.startswith("${"))
        self._card_data[1] = {
            "key": "providers", "title": "Providers",
            "ok": prov_configured > 0,
            "subtitle": f"{prov_configured}/{prov_count} with keys" if prov_count > 0 else "None configured",
            "status": f"{prov_configured}/{prov_count}" if prov_count > 0 else "None",
        }

        # CLI Tools
        tools_configured = 0
        tool_names = []
        for name, tool in [("Pi", cfg.pi), ("Codex", cfg.codex), ("Copilot", cfg.copilot_cli),
                           ("OpenCode", cfg.opencode), ("OpenClaw", cfg.openclaw), ("Poolside", cfg.poolside)]:
            if tool.provider and tool.provider in cfg.providers:
                tools_configured += 1
                tool_names.append(name)
        self._card_data[2] = {
            "key": "cli_tools", "title": "CLI Tools",
            "ok": tools_configured > 0,
            "subtitle": f"{tools_configured} configured: {', '.join(tool_names[:3])}" + ("..." if len(tool_names) > 3 else "") if tools_configured > 0 else "0 configured",
            "status": str(tools_configured),
        }

        # Anthropic Models
        alias_count = len(cfg.anthropic_models)
        self._card_data[3] = {
            "key": "models", "title": "Anthropic Models",
            "ok": alias_count > 0,
            "subtitle": f"{alias_count} aliases" if alias_count > 0 else "0 aliases",
            "status": str(alias_count),
        }

    def _load_config(self) -> None:
        try:
            self._config = load_config(self.config_path)
        except Exception as exc:
            self._config = None
            self._log_lines.append(f"[Config load error: {exc}]")

    def _determine_status(self) -> None:
        if not self._config:
            self.status = AppStatus.READY
            return
        # Check if server is running by PID
        pid_path = self.config_path.parent / "llama.pid"
        pid = _read_pid(pid_path)
        self._server_pid = pid
        if pid is not None and _pid_alive(pid):
            self.status = AppStatus.RUNNING
        else:
            self.status = AppStatus.READY

    def _refresh_all(self) -> None:
        self._load_config()
        self._determine_status()
        self._update_card_data()
        self._draw_header()
        self._draw_cards()

    def _on_close(self) -> None:
        self._stopped.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()

    def _open_server_config(self) -> None:
        ServerConfigDialog(self.root, self.config_path, self._on_config_saved)

    def _open_providers(self) -> None:
        ProvidersDialog(self.root, self.config_path, self._on_config_saved)

    def _open_cli_tools(self) -> None:
        CliToolsDialog(self.root, self.config_path, self._on_config_saved)

    def _open_models(self) -> None:
        ModelsDialog(self.root, self.config_path, self._on_config_saved)

    def _on_config_saved(self) -> None:
        self._refresh_all()

    def _open_logs(self) -> None:
        LogsDialog(self.root, self.config_path)

    def _open_details(self) -> None:
        DetailsDialog(self.root, self.config_path)


class _StyledScrollbar(tk.Frame):
    """Custom dark-themed scrollbar matching GUI theme."""
    def __init__(self, master: tk.Widget, command=None, **kwargs):
        kwargs.setdefault("width", 12)
        super().__init__(master, bg=BG, **kwargs)
        self._cmd = command
        self._lo = 0.0
        self._hi = 1.0
        self.pack_propagate(False)
        self._c = tk.Canvas(self, bg=BG, highlightthickness=0, width=12)
        self._c.pack(fill="both", expand=True)
        self._c.bind("<Button-1>", self._on_scroll_click)
        self._c.bind("<B1-Motion>", self._on_scroll_drag)
        self._c.bind("<MouseWheel>", self._on_scroll_wheel)
        self._draw()

    def set(self, lo: float, hi: float) -> None:
        self._lo = float(lo)
        self._hi = float(hi)
        self._draw()

    def _draw(self) -> None:
        c = self._c
        c.delete("all")
        w = c.winfo_width() or 12
        h = c.winfo_height() or 200
        c.create_rectangle(0, 0, w, h, fill=BG, outline="")
        rh = max(20, int(h * (self._hi - self._lo)))
        ry = int(self._lo * (h - rh))
        c.create_rectangle(2, ry, w - 2, ry + rh, fill=SURFACE2, outline=BORDER, width=1)

    def _on_scroll_click(self, e: tk.Event) -> None:
        if self._cmd:
            h = self._c.winfo_height() or 200
            self._cmd("moveto", max(0.0, min(1.0, e.y / max(1, h))))

    def _on_scroll_drag(self, e: tk.Event) -> None:
        if self._cmd:
            h = self._c.winfo_height() or 200
            self._cmd("moveto", max(0.0, min(1.0, e.y / max(1, h))))

    def _on_scroll_wheel(self, e: tk.Event) -> None:
        if self._cmd:
            self._cmd("scroll", -1 if e.delta > 0 else 1, "units")


class ServerConfigDialog:
    def __init__(self, parent: tk.Widget, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Server Config")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("460x340")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        import yaml
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.server_raw = self.raw.setdefault("server", {})

    def _build(self) -> None:
        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        def add_label(row, text):
            tk.Label(main, text=text, bg=BG, fg=TEXT, font=("Segoe UI", 9), anchor="w").grid(
                row=row, column=0, sticky="w", pady=2, padx=(0, 8))

        row = 0
        add_label(row, "Host")
        self.host_var = tk.StringVar(value=str(self.server_raw.get("host", "127.0.0.1")))
        tk.Entry(main, textvariable=self.host_var, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=0, highlightbackground=BORDER, highlightthickness=1).grid(
            row=row, column=1, sticky="ew", pady=2)
        row += 1

        add_label(row, "Port")
        self.port_var = tk.StringVar(value=str(self.server_raw.get("port", 8089)))
        tk.Spinbox(main, from_=1024, to=65535, textvariable=self.port_var,
                   bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE2, relief="flat",
                   highlightbackground=BORDER, highlightthickness=1, width=10).grid(
            row=row, column=1, sticky="w", pady=2)
        row += 1

        add_label(row, "Auth Token")
        self.auth_var = tk.StringVar(value=str(self.server_raw.get("auth_token", "")))
        auth_frame = tk.Frame(main, bg=BG)
        auth_frame.grid(row=row, column=1, sticky="ew", pady=2)
        tk.Entry(auth_frame, textvariable=self.auth_var, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=0, highlightbackground=BORDER, highlightthickness=1,
                 show="*").pack(side="left", fill="x", expand=True)
        self.show_auth_var = tk.BooleanVar(value=False)
        tk.Checkbutton(auth_frame, text="Show", variable=self.show_auth_var,
                       bg=BG, fg=MUTED, selectcolor=SURFACE2, activebackground=BG,
                       command=lambda: self.auth_var).pack(side="left", padx=(4, 0))
        row += 1

        add_label(row, "Idle Timeout (s)")
        self.idle_var = tk.StringVar(value=str(self.server_raw.get("idle_timeout_seconds", 180)))
        tk.Spinbox(main, from_=0, to=3600, increment=30, textvariable=self.idle_var,
                   bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE2, relief="flat",
                   highlightbackground=BORDER, highlightthickness=1, width=10).grid(
            row=row, column=1, sticky="w", pady=2)
        row += 1

        add_label(row, "Open WebUI Port")
        ow_port = self.server_raw.get("openwebui_port")
        self.owui_port_var = tk.StringVar(value=str(ow_port) if ow_port else "")
        tk.Entry(main, textvariable=self.owui_port_var, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=0, highlightbackground=BORDER, highlightthickness=1).grid(
            row=row, column=1, sticky="ew", pady=2)
        row += 1

        btn_row = tk.Frame(main, bg=BG)
        btn_row.grid(row=row, column=0, columnspan=2, pady=(16, 0))
        tk.Button(btn_row, text="Save", command=self._save,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
                  relief="flat", bd=0, padx=16, pady=6,
                  activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="Cancel", command=self.dialog.destroy,
                  bg=BG_ALT, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", bd=0, padx=16, pady=6,
                  activebackground=SURFACE3, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="left")

        main.columnconfigure(1, weight=1)

    def _save(self) -> None:
        import yaml
        self.server_raw["host"] = self.host_var.get().strip()
        self.server_raw["port"] = int(self.port_var.get())
        token = self.auth_var.get().strip()
        if token:
            self.server_raw["auth_token"] = token
        self.server_raw["idle_timeout_seconds"] = int(self.idle_var.get())
        ow = self.owui_port_var.get().strip()
        if ow:
            self.server_raw["openwebui_port"] = int(ow)
        else:
            self.server_raw.pop("openwebui_port", None)
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class ProvidersDialog:
    def __init__(self, parent: tk.Widget, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Providers")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("760x520")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        import yaml
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.providers_raw = self.raw.setdefault("providers", {})

    def _build(self) -> None:
        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        scroll_outer = tk.Frame(main, bg=BG)
        scroll_outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_outer, bg=BG, highlightthickness=0)
        scrollbar = _StyledScrollbar(scroll_outer, command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        TYPE_OPTIONS = [
            "openai", "ollama_cloud", "ollama_local", "lm_studio",
            "groq", "gemini", "cohere", "mistral", "deepseek",
            "openrouter", "openai_compatible", "nvidia_nim",
        ]

        self._entries: dict[str, dict[str, Any]] = {}
        items = sorted(self.providers_raw.items())
        num_cols = 2
        for idx, (name, prov) in enumerate(items):
            row_idx = idx // num_cols
            col_idx = idx % num_cols

            card = tk.Frame(inner, bg=SURFACE, bd=1, relief="flat",
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=row_idx, column=col_idx, sticky="nsew", padx=4, pady=5)

            tk.Label(card, text=name, bg=SURFACE, fg=TEXT_BRIGHT,
                     font=("Segoe UI", 11, "bold"), anchor="w").pack(anchor="w", padx=12, pady=(8, 3))

            entries: dict[str, Any] = {}
            fields: list[tuple[str, Any, str]] = [
                ("type", TYPE_OPTIONS, "combobox"),
                ("base_url", None, "entry"),
                ("api_key", None, "key"),
                ("default_model", None, "entry"),
                ("supports_tools", None, "check"),
            ]

            for label, values, kind in fields:
                row = tk.Frame(card, bg=SURFACE)
                row.pack(fill="x", padx=12, pady=1)
                tk.Label(row, text=label.replace("_", " ").title(), bg=SURFACE, fg=TEXT,
                         font=("Segoe UI", 9), anchor="w", width=12).pack(side="left")
                if kind == "combobox":
                    var = tk.StringVar(value=str(prov.get(label, values[0])))
                    ttk.Combobox(row, textvariable=var, values=values,
                                 state="readonly").pack(side="left", fill="x", expand=True)
                elif kind == "key":
                    var = tk.StringVar(value=str(prov.get(label, "")))
                    ef = tk.Entry(row, textvariable=var, bg=SURFACE2, fg=TEXT,
                                  insertbackground=TEXT, relief="flat", bd=0,
                                  highlightbackground=BORDER, highlightthickness=1, show="*")
                    ef.pack(side="left", fill="x", expand=True)
                    sv = tk.BooleanVar(value=False)
                    def _toggle_key(entry=ef, show_var=sv):
                        entry.configure(show="" if show_var.get() else "*")
                    tk.Checkbutton(row, text="S", variable=sv, bg=SURFACE, fg=MUTED,
                                   selectcolor=SURFACE2, activebackground=SURFACE,
                                   command=_toggle_key).pack(side="left", padx=(2, 0))
                elif kind == "check":
                    var = tk.BooleanVar(value=bool(prov.get(label, True)))
                    tk.Checkbutton(row, variable=var, bg=SURFACE, fg=TEXT,
                                   selectcolor=SURFACE2, activebackground=SURFACE).pack(side="left")
                else:
                    var = tk.StringVar(value=str(prov.get(label, "")))
                    tk.Entry(row, textvariable=var, bg=SURFACE2, fg=TEXT,
                             insertbackground=TEXT, relief="flat", bd=0,
                             highlightbackground=BORDER, highlightthickness=1
                             ).pack(side="left", fill="x", expand=True)
                entries[label] = var
            self._entries[name] = entries
            tk.Frame(card, bg=SURFACE, height=4).pack(fill="x")

        for c in range(num_cols):
            inner.grid_columnconfigure(c, weight=1, uniform="col")
        inner.grid_rowconfigure(len(items) // num_cols + (1 if len(items) % num_cols else 0), weight=1)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_row = tk.Frame(main, bg=BG)
        btn_row.pack(fill="x", pady=(10, 0))
        tk.Button(btn_row, text="Save", command=self._save,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
                  relief="flat", bd=0, padx=20, pady=6,
                  activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="right", padx=(6, 0))
        tk.Button(btn_row, text="Cancel", command=self.dialog.destroy,
                  bg=BG_ALT, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", bd=0, padx=20, pady=6,
                  activebackground=SURFACE3, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="right")

    def _save(self) -> None:
        import yaml
        for name, entries in self._entries.items():
            prov = self.providers_raw.setdefault(name, {})
            for label, var in entries.items():
                if isinstance(var, tk.BooleanVar):
                    prov[label] = bool(var.get())
                else:
                    val = var.get().strip()
                    if val:
                        prov[label] = val
        self.config_path.write_text(
            yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class CliToolsDialog:
    def __init__(self, parent: tk.Widget, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("CLI Tools")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("760x520")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        import yaml
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        provider_names = list(self.raw.get("providers", {}).keys())
        self.provider_names = provider_names if provider_names else ["ollama_cloud"]

    def _build(self) -> None:
        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        scroll_outer = tk.Frame(main, bg=BG)
        scroll_outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_outer, bg=BG, highlightthickness=0)
        scrollbar = _StyledScrollbar(scroll_outer, command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self._entries = {}
        tool_keys = [
            ("Pi", "pi", ["provider", "model", "api", "config_dir"]),
            ("Codex", "codex", ["provider", "model", "config_path", "profile"]),
            ("Copilot CLI", "copilot_cli", ["provider", "model", "wire_api", "max_prompt_tokens", "max_output_tokens"]),
            ("OpenCode", "opencode", ["provider", "model", "config_path", "context_size", "output_tokens"]),
            ("OpenClaw", "openclaw", ["provider", "model", "config_path", "workspace", "sandbox_backend"]),
            ("Poolside", "poolside", ["provider", "model", "api_url", "config_path"]),
        ]

        num_cols = 2
        for idx, (tool_name, section_key, fields) in enumerate(tool_keys):
            row_idx = idx // num_cols
            col_idx = idx % num_cols
            section = self.raw.get(section_key, {}) or {}

            card = tk.Frame(inner, bg=SURFACE, bd=1, relief="flat",
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=row_idx, column=col_idx, sticky="nsew", padx=4, pady=5)

            tk.Label(card, text=tool_name, bg=SURFACE, fg=TEXT_BRIGHT,
                     font=("Segoe UI", 11, "bold"), anchor="w").pack(anchor="w", padx=12, pady=(8, 3))

            entries = {}
            for field in fields:
                row = tk.Frame(card, bg=SURFACE)
                row.pack(fill="x", padx=12, pady=1)
                tk.Label(row, text=field.replace("_", " ").title(), bg=SURFACE, fg=TEXT,
                         font=("Segoe UI", 9), anchor="w", width=16).pack(side="left")

                if field == "provider":
                    var = tk.StringVar(value=str(section.get(field, self.provider_names[0])))
                    ttk.Combobox(row, textvariable=var, values=self.provider_names,
                                 state="readonly").pack(side="left", fill="x", expand=True)
                elif field in ("max_prompt_tokens", "max_output_tokens", "context_size", "output_tokens"):
                    var = tk.StringVar(value=str(section.get(field, "")))
                    tk.Spinbox(row, from_=0, to=262144, textvariable=var,
                               bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE2, relief="flat",
                               highlightbackground=BORDER, highlightthickness=1, width=12).pack(side="left")
                else:
                    var = tk.StringVar(value=str(section.get(field, "")))
                    tk.Entry(row, textvariable=var, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                             relief="flat", bd=0, highlightbackground=BORDER,
                             highlightthickness=1).pack(side="left", fill="x", expand=True)
                entries[field] = var
            self._entries[section_key] = entries

        for c in range(num_cols):
            inner.grid_columnconfigure(c, weight=1, uniform="col")
        total_rows = (len(tool_keys) + num_cols - 1) // num_cols
        inner.grid_rowconfigure(total_rows, weight=1)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_row = tk.Frame(main, bg=BG)
        btn_row.pack(fill="x", pady=(10, 0))
        tk.Button(btn_row, text="Save", command=self._save,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
                  relief="flat", bd=0, padx=20, pady=6,
                  activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="right", padx=(6, 0))
        tk.Button(btn_row, text="Cancel", command=self.dialog.destroy,
                  bg=BG_ALT, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", bd=0, padx=20, pady=6,
                  activebackground=SURFACE3, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="right")

    def _save(self) -> None:
        import yaml
        for section_key, entries in self._entries.items():
            section = self.raw.setdefault(section_key, {})
            for field, var in entries.items():
                val = var.get().strip()
                if field in ("max_prompt_tokens", "max_output_tokens", "context_size", "output_tokens"):
                    section[field] = int(val) if val else 0
                elif field in ("provider", "api", "wire_api", "sandbox_backend"):
                    section[field] = val
                else:
                    if val and not val.startswith("${"):
                        section[field] = val
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class ModelsDialog:
    def __init__(self, parent: tk.Widget, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Anthropic Models")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("500x360")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        import yaml
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        provider_names = list(self.raw.get("providers", {}).keys())
        self.provider_names = provider_names if provider_names else ["ollama_cloud"]

    def _build(self) -> None:
        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        scroll_outer = tk.Frame(main, bg=BG)
        scroll_outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_outer, bg=BG, highlightthickness=0)
        scrollbar = _StyledScrollbar(scroll_outer, command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        aliases_raw = self.raw.get("anthropic_models", {}) or {}
        self._entries = {}
        for alias, value in sorted(aliases_raw.items()):
            if not isinstance(value, dict):
                value = {}
            card = tk.Frame(inner, bg=SURFACE, bd=1, relief="flat",
                            highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill="x", pady=3, padx=2)

            tk.Label(card, text=alias, bg=SURFACE, fg=TEXT_BRIGHT,
                     font=("Segoe UI", 10, "bold"), anchor="w", width=12).pack(side="left", padx=8, pady=4)

            prov_var = tk.StringVar(value=str(value.get("provider", self.provider_names[0])))
            ttk.Combobox(card, textvariable=prov_var, values=self.provider_names,
                         state="readonly", width=18).pack(side="left", padx=2, pady=4)

            model_var = tk.StringVar(value=str(value.get("model", "")))
            tk.Entry(card, textvariable=model_var, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                     relief="flat", bd=0, highlightbackground=BORDER,
                     highlightthickness=1).pack(side="left", fill="x", expand=True, padx=4, pady=4)

            self._entries[alias] = (prov_var, model_var)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_row = tk.Frame(main, bg=BG)
        btn_row.pack(fill="x", pady=(10, 0))
        tk.Button(btn_row, text="Save", command=self._save,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
                  relief="flat", bd=0, padx=20, pady=6,
                  activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="right", padx=(6, 0))
        tk.Button(btn_row, text="Cancel", command=self.dialog.destroy,
                  bg=BG_ALT, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", bd=0, padx=20, pady=6,
                  activebackground=SURFACE3, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="right")

    def _save(self) -> None:
        import yaml
        aliases = self.raw.setdefault("anthropic_models", {})
        for alias, (prov_var, model_var) in self._entries.items():
            entry = aliases.setdefault(alias, {})
            entry["provider"] = prov_var.get()
            model = model_var.get().strip()
            if model:
                entry["model"] = model
            else:
                entry.pop("model", None)
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class LogsDialog:
    def __init__(self, parent: tk.Widget, config_path: Path) -> None:
        self.config_path = config_path
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Server Logs")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("720x520")
        self.dialog.minsize(560, 360)
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self._poll_id = None
        self._build()

    def _build(self) -> None:
        text_frame = tk.Frame(self.dialog, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=12, pady=(12, 4))
        self.text = tk.Text(text_frame, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                             font=("Consolas", 9), relief="flat", bd=0,
                             highlightbackground=BORDER, highlightthickness=1)
        self.text.pack(fill="both", expand=True)
        self.text.config(state="disabled")
        self.text.bind("<MouseWheel>", lambda e: self.text.yview_scroll(-1 * (e.delta // 120), "units"))
        self.text.tag_configure("error", foreground=RED)
        self.text.tag_configure("warn", foreground=YELLOW)
        self.text.tag_configure("info", foreground=GREEN)
        self.text.tag_configure("muted", foreground=MUTED)

        btn_row = tk.Frame(self.dialog, bg=BG)
        btn_row.pack(fill="x", padx=12, pady=(4, 12))
        tk.Button(btn_row, text="Refresh", command=self._refresh,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10),
                  relief="flat", padx=14, pady=4, activebackground=BUTTON_BOTTOM,
                  cursor="hand2").pack(side="left", padx=4)
        tk.Button(btn_row, text="Clear", command=self._clear,
                  bg=SURFACE2, fg=TEXT, font=("Segoe UI", 10),
                  relief="flat", padx=14, pady=4, activebackground=SURFACE3,
                  cursor="hand2").pack(side="left", padx=4)
        tk.Button(btn_row, text="Close", command=self._close,
                  bg=SURFACE2, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", padx=14, pady=4, activebackground=SURFACE3,
                  cursor="hand2").pack(side="right", padx=4)

        self.dialog.bind("<Escape>", lambda e: self._close())
        self.dialog.protocol("WM_DELETE_WINDOW", self._close)
        self._load_log()
        self._start_polling()

    def _load_log(self) -> None:
        log_path = self.config_path.parent / "llama.log"
        lines = []
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()[-200:]
            except OSError:
                lines = ["[Could not read log]"]
        else:
            lines = ["[No log file found]"]
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        for line in lines:
            lower = line.lower()
            if "error" in lower or "exception" in lower or "traceback" in lower:
                tag = "error"
            elif "warn" in lower:
                tag = "warn"
            elif "info" in lower or "start" in lower or "running" in lower or "ready" in lower:
                tag = "info"
            else:
                tag = "muted"
            self.text.insert("end", line + "\n", tag)
        self.text.config(state="disabled")
        self.text.see("end")

    def _start_polling(self) -> None:
        self._poll()

    def _poll(self) -> None:
        self._reload_if_changed()
        self._poll_id = self.dialog.after(3000, self._poll)

    def _reload_if_changed(self) -> None:
        log_path = self.config_path.parent / "llama.log"
        try:
            if not log_path.exists():
                return
            current = self.text.get("1.0", "end-1c").splitlines()
            raw = log_path.read_text(encoding="utf-8", errors="replace")
            new_lines = raw.splitlines()
            if new_lines == current:
                return
            self.text.config(state="normal")
            self.text.delete("1.0", "end")
            for line in new_lines[-200:]:
                lower = line.lower()
                if "error" in lower or "exception" in lower or "traceback" in lower:
                    tag = "error"
                elif "warn" in lower:
                    tag = "warn"
                elif "info" in lower or "start" in lower or "running" in lower or "ready" in lower:
                    tag = "info"
                else:
                    tag = "muted"
                self.text.insert("end", line + "\n", tag)
            self.text.config(state="disabled")
            self.text.see("end")
        except Exception:
            pass

    def _refresh(self) -> None:
        self._load_log()

    def _clear(self) -> None:
        log_path = self.config_path.parent / "llama.log"
        try:
            log_path.write_text("", encoding="utf-8")
        except Exception:
            pass
        self._load_log()

    def _close(self) -> None:
        if self._poll_id:
            self.dialog.after_cancel(self._poll_id)
            self._poll_id = None
        self.dialog.destroy()


class _DetailItem:
    def __init__(self, label: str, display_value: str, full_value: str) -> None:
        self.label = label
        self.display_value = display_value
        self.full_value = full_value


class DetailsDialog:
    def __init__(self, parent: tk.Widget, config_path: Path) -> None:
        self.config_path = config_path
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Details")
        self.dialog.geometry("760x520")
        self.dialog.minsize(620, 420)
        self.dialog.configure(bg=BG)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)

        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        self.items = self._build_items()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=BG_ALT, foreground=TEXT,
                        fieldbackground=BG_ALT,
                        font=("Segoe UI", 9), rowheight=26, borderwidth=0)
        style.configure("Treeview.Heading",
                        background=SURFACE2, foreground=TEXT_BRIGHT,
                        font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Treeview",
                  background=[("selected", SURFACE3)],
                  foreground=[("selected", TEXT_BRIGHT)])
        style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

        self.tree = ttk.Treeview(main, columns=("field", "value"), show="headings",
                                  selectmode="browse")
        self.tree.heading("field", text="Field", anchor="w")
        self.tree.heading("value", text="Value", anchor="w")
        self.tree.column("field", width=180, minwidth=140, stretch=False)
        self.tree.column("value", width=500, minwidth=300, stretch=True)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<MouseWheel>", self._on_tree_mousewheel)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-3>", self._on_right_click)

        for idx, item in enumerate(self.items):
            self.tree.insert("", "end", iid=str(idx),
                             values=(item.label, item.display_value))
        self._selected_idx: str | None = None

        btn_row = tk.Frame(self.dialog, bg=BG)
        btn_row.pack(fill="x", padx=16, pady=(8, 10))
        tk.Button(btn_row, text="Copy Selected Value", command=self._copy_selected,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=12, pady=4, activebackground=BUTTON_BOTTOM,
                  cursor="hand2").pack(side="left", padx=(0, 6))
        tk.Button(btn_row, text="Copy All Details", command=self._copy_all,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=12, pady=4, activebackground=BUTTON_BOTTOM,
                  cursor="hand2").pack(side="left", padx=(0, 6))
        tk.Button(btn_row, text="Close", command=self.dialog.destroy,
                  bg=SURFACE2, fg=MUTED, font=("Segoe UI", 9),
                  relief="flat", padx=16, pady=4, activebackground=SURFACE3,
                  cursor="hand2").pack(side="right", padx=4)
        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())

    def _build_items(self) -> list[_DetailItem]:
        import yaml
        items: list[_DetailItem] = []
        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}
        server_raw = raw.get("server", {}) or {}
        providers_raw = raw.get("providers", {}) or {}

        items.append(_DetailItem("Config path", _compact_path(str(self.config_path)), str(self.config_path)))
        items.append(_DetailItem("Host", str(server_raw.get("host", "127.0.0.1")), str(server_raw.get("host", "127.0.0.1"))))
        items.append(_DetailItem("Port", str(server_raw.get("port", 8089)), str(server_raw.get("port", 8089))))
        auth = server_raw.get("auth_token", "")
        items.append(_DetailItem("Auth token", "set" if (auth and auth != "change-me") else "change-me",
                                 "set" if (auth and auth != "change-me") else "change-me"))
        items.append(_DetailItem("Idle timeout", f"{server_raw.get('idle_timeout_seconds', 180)}s",
                                 str(server_raw.get('idle_timeout_seconds', 180))))
        items.append(_DetailItem("OpenWebUI port", str(server_raw.get("openwebui_port", "none")),
                                 str(server_raw.get("openwebui_port", "none"))))
        items.append(_DetailItem("Providers", str(len(providers_raw)), ", ".join(providers_raw.keys()) if providers_raw else "none"))
        for name, prov in sorted(providers_raw.items()):
            ptype = prov.get("type", "?")
            model = prov.get("default_model", "-") or "-"
            items.append(_DetailItem(f"  {name}", f"{ptype} / {model}", f"{ptype} / {model}"))

        tools_raw = raw.get("tools", {}) or {}
        items.append(_DetailItem("Tools enabled", str(tools_raw.get("enabled", True)), str(tools_raw.get("enabled", True))))
        items.append(_DetailItem("Tools include", ", ".join(tools_raw.get("include", []) or []) or "none",
                                 ", ".join(tools_raw.get("include", []) or []) or "none"))
        items.append(_DetailItem("Search", str(tools_raw.get("default_search_provider", "tavily")),
                                 str(tools_raw.get("default_search_provider", "tavily"))))
        return items

    def _on_tree_mousewheel(self, event: Any) -> None:
        self.tree.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_select(self, _event: Any = None) -> None:
        sel = self.tree.selection()
        self._selected_idx = sel[0] if sel else None

    def _on_right_click(self, event: Any) -> None:
        iid = self.tree.identify_row(event.y)
        if iid:
            self.tree.selection_set(iid)
            self._selected_idx = iid
            self._copy_selected()

    def _copy_selected(self) -> None:
        if self._selected_idx is None:
            sel = self.tree.selection()
            if not sel:
                return
            self._selected_idx = sel[0]
        try:
            idx = int(self._selected_idx)
            item = self.items[idx]
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(item.full_value)
        except (ValueError, IndexError):
            pass

    def _copy_all(self) -> None:
        lines = [f"{item.label}: {item.full_value}" for item in self.items]
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append("\n".join(lines))
