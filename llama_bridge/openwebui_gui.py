from __future__ import annotations

import json
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, load_config, BridgeConfig, OpenWebUIConfig
from .openwebui_config import (
    save_openwebui_config,
    generate_openwebui_env,
    check_openwebui_installed,
    discover_openwebui,
    OpenWebUIDiscovery,
    clear_discovery_cache,
    get_cached_discovery,
    test_search_provider,
    get_effective_ports,
    port_in_use,
    pid_alive,
    read_pid,
    VALID_SEARCH_PROVIDERS,
    get_conda_python_path,
)
from .openwebui_launcher import (
    start_bridge,
    start_openwebui,
    stop_openwebui,
    restart_openwebui,
    install_openwebui,
    follow_log,
    status as launcher_status,
    stop_all,
    LLAMA_PID_PATH,
    OPENWEBUI_PID_PATH,
    LLAMA_LOG_PATH,
    OPENWEBUI_LOG_PATH,
)

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


# ── Theme ──────────────────────────────────────────────────────────────────
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
    "WIN_W": 608,
    "WIN_H": 620,
}
PAD = LAYOUT["PAD"]


class Phase(Enum):
    INSTALL_PREREQS = auto()
    SETUP_ENV = auto()
    READY = auto()
    STARTING = auto()
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
        lbl = tk.Label(
            self.tip, text=self.text, bg=SURFACE3, fg=TEXT_BRIGHT,
            font=("Segoe UI", 9), padx=8, pady=4, relief="solid", bd=1,
        )
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


# ── Main UI Class ──────────────────────────────────────────────────────────
class OpenWebUISetupCenter:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self._stopped = threading.Event()

        W, H = LAYOUT["WIN_W"], LAYOUT["WIN_H"]
        self.root = tk.Tk()
        self.root.title("Llama Bridge - Open WebUI Setup")
        _center_window(self.root, W, H)
        self.root.minsize(560, 520)
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        _set_dark_titlebar(self.root)

        # State
        self.phase = Phase.READY
        self._config: BridgeConfig | None = None
        self._owui: OpenWebUIConfig | None = None
        self._last_msg: str = ""
        self._cmd_visible = False
        self._log_lines: list[str] = []
        self._detail_cards: list[dict[str, Any]] = []
        self._owui_health_ok = False
        self._discovery: OpenWebUIDiscovery = OpenWebUIDiscovery()

        # UI references
        self.header_canvas: tk.Canvas | None = None
        self.cards_canvas: tk.Canvas | None = None
        self.cmd_frame: tk.Frame | None = None
        self.cmd_text: tk.Text | None = None
        self.host_btn: tk.Button | None = None
        self.auth_btn: tk.Button | None = None
        self.primary_btn: tk.Button | None = None
        self.util_btns: dict[str, tk.Button] = {}
        self.status_label: tk.Label | None = None

        # Card data
        self._card_data: list[dict[str, Any]] = [
            {"key": "python", "title": "Python", "ok": True, "subtitle": "", "status": ""},
            {"key": "conda", "title": "Conda / Env", "ok": True, "subtitle": "", "status": ""},
            {"key": "ffmpeg", "title": "FFmpeg", "ok": True, "subtitle": "", "status": ""},
            {"key": "owui", "title": "Open WebUI", "ok": True, "subtitle": "", "status": ""},
        ]

        self._build_ui()
        self._load_config()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._poll_status)

    # ── UI Build ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # Header Canvas
        self.header_canvas = tk.Canvas(main, bg=BG, highlightthickness=0)
        self.header_canvas.pack(fill="x", side="top")
        self.header_canvas.configure(height=LAYOUT["HEADER_H"])

        # Cards Canvas
        self.cards_canvas = tk.Canvas(main, bg=BG, highlightthickness=0)
        self.cards_canvas.pack(fill="both", expand=True, side="top")

        # Command panel (hidden)
        self.cmd_frame = tk.Frame(main, bg=SURFACE, height=120)

        # Log panel (removed — logs go to Logs dialog only)

        # Status label (between cards and footer)
        self.status_label = tk.Label(
            main, text="", bg=BG, fg=MUTED, font=("Segoe UI", 9),
            anchor="w", padx=PAD,
        )
        self.status_label.pack(fill="x", side="top")

        # Separator
        sep = tk.Frame(main, bg=BORDER, height=1)
        sep.pack(fill="x", side="bottom")

        # Footer
        self._build_footer(main)

        # Bind resize
        self.header_canvas.bind("<Configure>", lambda e: self._draw_header())
        self.cards_canvas.bind("<Configure>", lambda e: self._draw_cards())

        self._bind_shortcuts()

    def _build_footer(self, parent: tk.Frame) -> None:
        footer = tk.Frame(parent, bg=BG)
        footer.pack(fill="x", side="bottom", pady=(6, 10))

        # Utility row
        util = tk.Frame(footer, bg=BG)
        util.pack(fill="x", padx=PAD)

        util_items = [
            ("btn_config", "\u2699 Config", self._open_config_dialog),
            ("btn_search", "\uD83D\uDD0D Web Search", self._open_websearch_dialog),
            ("btn_rescan", "\u21BB Rescan", self._action_rescan),
            ("btn_logs", "\uD83D\uDCCB Logs", self._open_logs_dialog),
            ("btn_details", "\u2139 Details", self._open_details_dialog),
            ("btn_cmd", "Preview Cmd", self._toggle_cmd_panel),
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
            ToolTip(btn, {"btn_config": "Open configuration dialog", "btn_search": "Web search settings", "btn_rescan": "Re-scan for Open WebUI installations", "btn_logs": "View logs", "btn_details": "Show full technical details", "btn_cmd": "Toggle command preview panel"}[key])

        # Action row
        action = tk.Frame(footer, bg=BG)
        action.pack(fill="x", padx=PAD, pady=(4, 0))

        self.host_btn = tk.Button(
            action, text="Host: local", command=self._toggle_host,
            bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=16, pady=8,
            activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
            cursor="hand2",
        )
        self.host_btn.pack(side="left", padx=(0, 8))
        ToolTip(self.host_btn, "Toggle between localhost (127.0.0.1) and LAN (0.0.0.0)")

        self.auth_btn = tk.Button(
            action, text="Auth: Off", command=self._toggle_auth,
            bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=16, pady=8,
            activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
            cursor="hand2",
        )
        self.auth_btn.pack(side="left")
        ToolTip(self.auth_btn, "Toggle authentication on/off")

        self.primary_btn = tk.Button(
            action, text="Start Server", command=self._primary_action,
            bg="#1A4030", fg=GREEN, font=("Segoe UI", 11, "bold"),
            relief="flat", bd=0, padx=24, pady=8,
            activebackground="#153828", activeforeground=GREEN,
            cursor="hand2",
        )
        self.primary_btn.pack(side="right")
        ToolTip(self.primary_btn, "Start or stop the Open WebUI server")

    # ── Canvas Drawing ─────────────────────────────────────────────────────

    def _draw_header(self) -> None:
        cv = self.header_canvas
        if not cv:
            return
        cv.delete("all")
        w = cv.winfo_width() or LAYOUT["WIN_W"]
        h = LAYOUT["HEADER_H"]
        p = PAD

        # Title
        cv.create_text(p, 16, anchor="nw", text="OpenWebUI Setup Center",
                       font=("Segoe UI", 16, "bold"), fill=TEXT_BRIGHT, tags="title")

        # Subtitle
        subtitle = self._get_subtitle()
        cv.create_text(p, h // 2 + 10, anchor="nw", text=subtitle,
                       font=("Segoe UI", 10), fill=MUTED, tags="subtitle")

        # Badge — simple clean rectangle
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
                       font=("Segoe UI", 9, "bold"), fill=badge_color, tags="badge")

        # Accent stripe at bottom of header
        sep_color = self._get_separator_color()
        cv.create_rectangle(0, h - 3, w, h, fill=sep_color, outline="", tags="accent")

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

        for i, card in enumerate(self._card_data):
            ok = card["ok"]
            fill = CARD_GREEN if ok else CARD_RED
            border = CARD_GREEN_BORDER if ok else CARD_RED_BORDER
            accent = GREEN if ok else RED

            # Main solid card rectangle
            cv.create_rectangle(p, y, p + card_w, y + ch, fill=fill, outline=border, width=1)

            # Left accent bar
            cv.create_rectangle(p, y, p + 5, y + ch, fill=accent, outline=accent, width=0)

            # Icon circle
            cx, cy = p + 34, y + ch // 2
            cr = 14
            cv.create_oval(cx - cr, cy - cr, cx + cr, cy + cr, outline=accent, width=2)

            # Check or X icon
            if ok:
                cv.create_line(cx - 7, cy, cx - 1, cy + 6, cx + 7, cy - 6,
                               fill=accent, width=2, capstyle="round", joinstyle="round")
            else:
                cv.create_line(cx - 5, cy - 5, cx + 5, cy + 5, fill=accent, width=2)
                cv.create_line(cx + 5, cy - 5, cx - 5, cy + 5, fill=accent, width=2)

            # Title
            tx = p + 68
            cv.create_text(tx, y + 14, text=card["title"],
                           fill=TEXT_BRIGHT, font=("Segoe UI", 11, "bold"), anchor="w")

            # Subtitle (ellipsized to prevent overflow)
            sub = card.get("subtitle", "")
            safe_sub = self._ellipsize(cv, sub, ("Segoe UI", 9), card_w - 190)
            cv.create_text(tx, y + 37, text=safe_sub,
                           fill=MUTED, font=("Segoe UI", 9), anchor="w")

            # Status right-aligned
            st = card.get("status", "")
            cv.create_text(p + card_w - 14, y + ch // 2, text=st,
                           fill=accent, font=("Segoe UI", 9, "bold"), anchor="e")

            y += ch + LAYOUT["CARD_GAP"]

    # ── State Helpers ──────────────────────────────────────────────────────

    def _get_subtitle(self) -> str:
        phase = self.phase
        if phase == Phase.INSTALL_PREREQS:
            return "Open WebUI package is missing \u2014 click Install."
        if phase == Phase.SETUP_ENV:
            return "Configure your environment, then click Next."
        if phase == Phase.READY:
            return "Environment is ready. Choose host/auth, then click Start Server."
        if phase == Phase.STARTING:
            return "Starting server\u2026"
        if phase == Phase.RUNNING:
            return "Server is running."
        if phase == Phase.ERROR:
            return f"Error: {self._last_msg}"
        return ""

    def _get_badge_info(self) -> tuple[str, str]:
        p = self.phase
        if p == Phase.ERROR:
            return "ERROR", RED
        if p in (Phase.INSTALL_PREREQS, Phase.SETUP_ENV):
            return "SETUP", YELLOW
        if p == Phase.READY:
            return "READY", GREEN
        if p == Phase.STARTING:
            return "STARTING", YELLOW
        if p == Phase.RUNNING:
            return "RUNNING", GREEN
        return "READY", GREEN

    def _get_separator_color(self) -> str:
        p = self.phase
        if p == Phase.ERROR:
            return RED
        if p == Phase.RUNNING:
            return GREEN
        if p in (Phase.READY,):
            return GREEN
        return BORDER

    # ── Config / Status ────────────────────────────────────────────────────

    def _load_config(self) -> None:
        try:
            self._config = load_config(self.config_path)
            self._owui = self._config.openwebui
            if self._owui:
                self.auth_mode = self._owui.auth_enabled
                self.host_mode = "lan" if self._owui.host in ("0.0.0.0", "") else "local"
        except Exception:
            self._config = None
            self._owui = None

    def _update_card_data(self) -> None:
        cfg, ow = self._config, self._owui
        if not cfg or not ow:
            return

        # 1. Python card
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        py_ok = sys.version_info >= (3, 11)
        self._card_data[0] = {"key": "python", "title": "Python", "ok": py_ok,
                              "subtitle": py_ver, "status": "OK" if py_ok else "Upgrade"}

        # 2. Conda / Env card
        conda_python = get_conda_python_path()
        venv = os.environ.get("CONDA_DEFAULT_ENV") or ""
        if conda_python:
            subtitle = "omx-open-webui"
            status = "CondA"
        elif venv:
            subtitle = venv
            status = "venv"
        else:
            subtitle = "system"
            status = "System"
        self._card_data[1] = {"key": "conda", "title": "Conda / Env", "ok": True,
                              "subtitle": subtitle, "status": status}

        # 3. FFmpeg card — NEVER show path
        import shutil
        ffmpeg_found = shutil.which("ffmpeg") is not None
        self._card_data[2] = {"key": "ffmpeg", "title": "FFmpeg", "ok": ffmpeg_found,
                              "subtitle": "Present" if ffmpeg_found else "Missing",
                              "status": "OK" if ffmpeg_found else "Missing"}

        # 4. Open WebUI card — use discovery
        disc = self._discovery
        if not disc.installed:
            # Run discovery
            disc = discover_openwebui(
                preferred_env_name=ow.preferred_env_name,
                preferred_python=ow.preferred_python,
                preferred_command=ow.preferred_command,
            )
            self._discovery = disc
        title, subtitle, status = disc.to_card_status()
        self._card_data[3] = {"key": "owui", "title": title,
                              "ok": disc.installed,
                              "subtitle": subtitle,
                              "status": status}

    def _determine_phase(self) -> None:
        if not self._config or not self._owui:
            self.phase = Phase.ERROR
            self._last_msg = "Config load failed"
            return

        # Don't override STARTING — background thread will resolve it
        if self.phase == Phase.STARTING:
            return

        disc = self._discovery
        if not disc.installed:
            disc = discover_openwebui(
                preferred_env_name=self._owui.preferred_env_name,
                preferred_python=self._owui.preferred_python,
                preferred_command=self._owui.preferred_command,
            )
            self._discovery = disc
        if not disc.installed:
            self.phase = Phase.INSTALL_PREREQS
            return

        op = read_pid(OPENWEBUI_PID_PATH)
        ow_alive = op is not None and pid_alive(op)

        if ow_alive:
            self.phase = Phase.RUNNING
        else:
            self.phase = Phase.READY

    def _update_footer_buttons(self) -> None:
        if not self.host_btn or not self.auth_btn or not self.primary_btn:
            return

        ow = self._owui
        host = ow.host if ow else "127.0.0.1"
        self.host_btn.configure(text=f"Host: {'LAN' if host in ('0.0.0.0',) else 'local'}")

        auth = ow.auth_enabled if ow else False
        self.auth_btn.configure(text=f"Auth: {'On' if auth else 'Off'}")

        if self.phase == Phase.INSTALL_PREREQS:
            self.primary_btn.configure(text="Install", state="normal", bg="#1A4030", fg=GREEN)
        elif self.phase == Phase.READY:
            self.primary_btn.configure(text="Start Server", state="normal", bg="#1A4030", fg=GREEN)
        elif self.phase == Phase.STARTING:
            self.primary_btn.configure(text="Starting\u2026", state="disabled", bg="#333", fg=MUTED)
        elif self.phase == Phase.RUNNING:
            self.primary_btn.configure(text="Stop Server", state="normal", bg="#402020", fg=RED)
        elif self.phase == Phase.ERROR:
            self.primary_btn.configure(text="Retry", state="normal", bg="#402020", fg=YELLOW)

        self.host_btn.configure(state="normal" if self.phase in (Phase.READY, Phase.INSTALL_PREREQS) else "disabled")
        self.auth_btn.configure(state="normal" if self.phase in (Phase.READY, Phase.INSTALL_PREREQS) else "disabled")

    def _update_status_label(self) -> None:
        if not self.status_label:
            return
        ow = self._owui
        if not ow:
            self.status_label.configure(text="")
            return

        ports = get_effective_ports(ow, self._config) if self._config else {}
        parts = []

        if self.phase == Phase.RUNNING:
            parts.append(f"http://{ow.host}:{ports.get('openwebui', 8080)}")
            parts.append(f"LLM port: {ports.get('bridge_llm_only', 11534)}")
        elif self.phase == Phase.INSTALL_PREREQS:
            parts.append("Open WebUI package is missing \u2014 click Install")
        elif self.phase == Phase.READY:
            parts.append("Ready to start")

        txt = "  |  ".join(parts) if parts else ""
        self.status_label.configure(text=txt, fg=GREEN if self.phase == Phase.RUNNING else MUTED)

    def _update_cmd_preview(self) -> None:
        if not self._owui:
            return
        env = generate_openwebui_env(self._owui, self._config)
        lines = []
        for k, v in sorted(env.items()):
            if any(secret in k.lower() for secret in ("key", "token", "secret", "auth")):
                v = "****" if v else ""
            lines.append(f"{k}={v}")
        cmd_txt = "\n".join(lines) if lines else "(no env vars)"
        if self.cmd_text:
            self.cmd_text.delete("1.0", "end")
            self.cmd_text.insert("1.0", cmd_txt)

    # ── Command / Log Panel ────────────────────────────────────────────────

    def _toggle_cmd_panel(self) -> None:
        if not self.cmd_frame:
            return
        self._cmd_visible = not self._cmd_visible
        if self._cmd_visible:
            self._update_cmd_preview()
            self._show_cmd_panel()
        else:
            self._hide_panels()
        self._update_util_btn()

    def _show_cmd_panel(self) -> None:
        if not self.cmd_frame:
            return
        self._hide_panels()
        self.cmd_frame.pack(fill="x", side="top", padx=PAD, pady=(0, 4), before=self.status_label)
        if not self.cmd_text:
            cmd_inner = tk.Frame(self.cmd_frame, bg=BG_ALT, bd=1, relief="flat",
                                 highlightbackground=BORDER, highlightthickness=1)
            cmd_inner.pack(fill="both", expand=True, padx=0, pady=4)
            self.cmd_text = tk.Text(
                cmd_inner, bg=BG_ALT, fg=TEXT, font=("Consolas", 9),
                relief="flat", bd=0, wrap="none", height=5, highlightthickness=0,
            )
            scroll_x = tk.Scrollbar(cmd_inner, orient="horizontal", command=self.cmd_text.xview)
            self.cmd_text.configure(xscrollcommand=scroll_x.set)
            scroll_x.pack(side="bottom", fill="x")
            self.cmd_text.pack(side="left", fill="both", expand=True)
        self.cmd_frame.configure(height=120)

    def _hide_panels(self) -> None:
        if self.cmd_frame:
            self.cmd_frame.pack_forget()

    def _update_util_btn(self) -> None:
        btn = self.util_btns.get("btn_cmd")
        if btn:
            if self._cmd_visible:
                btn.configure(text="Hide Cmd")
            else:
                btn.configure(text="Preview Cmd")

    def _append_log_line(self, line: str) -> None:
        pass  # Logs go to Logs dialog only

    # ── Toggles ────────────────────────────────────────────────────────────

    def _toggle_host(self) -> None:
        if not self._owui:
            return
        current = self._owui.host
        new_host = "0.0.0.0" if current in ("127.0.0.1",) else "127.0.0.1"
        self._owui = OpenWebUIConfig(
            enabled=self._owui.enabled, host=new_host, port=self._owui.port,
            bridge_tools_port=self._owui.bridge_tools_port,
            bridge_llm_only_port=self._owui.bridge_llm_only_port,
            auth_enabled=self._owui.auth_enabled, auto_login=self._owui.auto_login,
            web_search_enabled=self._owui.web_search_enabled,
            web_search_provider=self._owui.web_search_provider,
            web_search_providers=self._owui.web_search_providers,
            search_result_count=self._owui.search_result_count,
            concurrent_requests=self._owui.concurrent_requests,
            bypass_embedding_and_retrieval=self._owui.bypass_embedding_and_retrieval,
            bypass_web_loader=self._owui.bypass_web_loader,
            hf_token=self._owui.hf_token,
            openai_base_url_mode=self._owui.openai_base_url_mode,
            openwebui_data_dir=self._owui.openwebui_data_dir,
            extra_env=self._owui.extra_env,
            preferred_env_name=self._owui.preferred_env_name,
            preferred_python=self._owui.preferred_python,
            preferred_command=self._owui.preferred_command,
            auto_discover=self._owui.auto_discover,
        )
        if self._config:
            self._config.openwebui = self._owui
        self._save_config()
        self._draw_cards()
        self._update_footer_buttons()
        self._update_status_label()

    def _toggle_auth(self) -> None:
        if not self._owui:
            return
        new_auth = not self._owui.auth_enabled
        self._owui = OpenWebUIConfig(
            enabled=self._owui.enabled, host=self._owui.host, port=self._owui.port,
            bridge_tools_port=self._owui.bridge_tools_port,
            bridge_llm_only_port=self._owui.bridge_llm_only_port,
            auth_enabled=new_auth, auto_login=not new_auth,
            web_search_enabled=self._owui.web_search_enabled,
            web_search_provider=self._owui.web_search_provider,
            web_search_providers=self._owui.web_search_providers,
            search_result_count=self._owui.search_result_count,
            concurrent_requests=self._owui.concurrent_requests,
            bypass_embedding_and_retrieval=self._owui.bypass_embedding_and_retrieval,
            bypass_web_loader=self._owui.bypass_web_loader,
            hf_token=self._owui.hf_token,
            openai_base_url_mode=self._owui.openai_base_url_mode,
            openwebui_data_dir=self._owui.openwebui_data_dir,
            extra_env=self._owui.extra_env,
            preferred_env_name=self._owui.preferred_env_name,
            preferred_python=self._owui.preferred_python,
            preferred_command=self._owui.preferred_command,
            auto_discover=self._owui.auto_discover,
        )
        if self._config:
            self._config.openwebui = self._owui
        self._save_config()
        self._draw_cards()
        self._update_footer_buttons()
        self._update_status_label()

    def _primary_action(self) -> None:
        if self.phase == Phase.INSTALL_PREREQS:
            self._action_install()
        elif self.phase == Phase.READY:
            self._action_start()
        elif self.phase == Phase.RUNNING:
            self._action_stop()
        elif self.phase == Phase.ERROR:
            self.phase = Phase.READY
            self._refresh_all()

    # ── Actions ────────────────────────────────────────────────────────────

    def _action_install(self) -> None:
        def run():
            ok = install_openwebui()
            self.root.after(0, lambda: self._on_install_done(ok))

        self.primary_btn.configure(text="Installing\u2026", state="disabled")
        threading.Thread(target=run, daemon=True).start()

    def _on_install_done(self, ok: bool) -> None:
        if ok:
            self.phase = Phase.READY
            self._last_msg = ""
        else:
            self.phase = Phase.ERROR
            self._last_msg = "Installation failed. Check network / permissions."
        self._refresh_all()
        if not ok:
            messagebox.showerror("Install Error", self._last_msg)

    def _action_start(self) -> None:
        ow = self._owui
        if ow and not ow.auth_enabled and ow.host in ("0.0.0.0",):
            ok = messagebox.askyesno(
                "Security Warning",
                "Auth is OFF and host is LAN (0.0.0.0).\n"
                "Anyone on your network can access Open WebUI without login.\n\n"
                "Continue?",
                icon="warning",
            )
            if not ok:
                return

        self.phase = Phase.STARTING
        self._refresh_all()

        def run():
            try:
                b_ok, b_msg = start_bridge(self.config_path)
                if not b_ok:
                    self.root.after(0, lambda m=b_msg: self._on_start_failed(m))
                    return

                ow_ok, ow_msg = start_openwebui(self.config_path)
                if not ow_ok:
                    self.root.after(0, lambda m=ow_msg: self._on_start_failed(m))
                    return

                ports = get_effective_ports(ow, self._config) if self._config and ow else {}
                ow_port = ports.get("openwebui", 8080)
                ow_host = ow.host if ow else "127.0.0.1"
                for _ in range(20):
                    time.sleep(1)
                    used, _ = port_in_use(ow_port, ow_host)
                    if used:
                        url = f"http://{ow_host}:{ow_port}"
                        self.root.after(0, lambda u=url: self._on_start_ok(u))
                        return
                self.root.after(0, lambda: self._on_start_failed(f"Port {ow_port} not responding after 20s"))
            except Exception as e:
                self.root.after(0, lambda: self._on_start_failed(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _on_start_failed(self, msg: str) -> None:
        self.phase = Phase.ERROR
        self._last_msg = msg
        self._append_log_line(f"[error] {msg}")
        self._refresh_all(skip_phase=True)

    def _on_start_ok(self, url: str) -> None:
        self.phase = Phase.RUNNING
        self._append_log_line(f"[health] Server is healthy: {url}")
        self._refresh_all()
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def _action_stop(self) -> None:
        def run():
            self.root.after(0, lambda: self._append_log_line("[stop] Stopping Open WebUI\u2026"))
            stop_openwebui()
            self.root.after(0, lambda: self._on_stop_done())
        self.primary_btn.configure(text="Stopping\u2026", state="disabled")
        threading.Thread(target=run, daemon=True).start()

    def _on_stop_done(self) -> None:
        self.phase = Phase.READY
        self._append_log_line("[stop] Stopped")
        self._refresh_all()

    def _refresh_all(self, skip_phase: bool = False) -> None:
        self._load_config()
        self._update_card_data()
        if not skip_phase:
            self._determine_phase()
        self._draw_header()
        self._draw_cards()
        self._update_footer_buttons()
        self._update_status_label()

    def _save_config(self) -> None:
        if self._owui:
            save_openwebui_config(self._owui, self.config_path)

    # ── Polling ────────────────────────────────────────────────────────────

    def _poll_status(self) -> None:
        if self._stopped.is_set():
            return
        try:
            self._refresh_all()
        except Exception:
            pass
        self.root.after(3000, self._poll_status)

    # ── Dialogs ────────────────────────────────────────────────────────────

    def _open_config_dialog(self) -> None:
        ConfigDialog(self.root, self._owui, self._config, self.config_path, self._on_config_saved)

    def _open_websearch_dialog(self) -> None:
        WebSearchDialog(self.root, self._owui, self.config_path, self._on_config_saved)

    def _open_logs_dialog(self) -> None:
        LogsDialog(self.root)

    def _open_details_dialog(self) -> None:
        DetailsDialog(self.root, self._config, self._owui)

    def _on_config_saved(self) -> None:
        self._refresh_all()

    def _action_rescan(self) -> None:
        """Re-run discovery and refresh the UI."""
        clear_discovery_cache()
        self._discovery = OpenWebUIDiscovery()
        self._refresh_all()

    # ── Shortcuts ──────────────────────────────────────────────────────────

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-l>", lambda e: self._open_logs_dialog())
        self.root.bind("<Control-s>", lambda e: [self._save_config(), messagebox.showinfo("Saved", "Configuration saved.")])
        self.root.bind("<Control-r>", lambda e: self._action_stop() if self.phase == Phase.RUNNING else self._action_start())
        self.root.bind("<Escape>", lambda e: self._hide_panels())

    # ── Cleanup ────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stopped.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()


# ── Details Helpers ────────────────────────────────────────────────────────

def compact_path(value: str, max_chars: int = 72) -> str:
    """Shorten a filesystem path for display only. Never modifies the real value."""
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


def compact_value(value: str, max_chars: int = 72) -> str:
    """Shorten any value for display (path, URL, text)."""
    if not value:
        return "none"
    text = str(value)
    if len(text) <= max_chars:
        return text
    # If it looks like a path (has drive letter or / or \), use compact_path
    if ":" in text[:3] or "/" in text or "\\" in text:
        return compact_path(text, max_chars)
    # Otherwise truncate from end with ellipsis
    return text[:max_chars - 3] + "..."


# ── Details Dialog ─────────────────────────────────────────────────────────
class DetailsDialog:
    def __init__(self, parent: tk.Tk, config: BridgeConfig | None,
                 owui: OpenWebUIConfig | None) -> None:
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

        # Build data items
        self.items = self._build_items(config, owui)

        # Treeview
        columns = ("field", "value")
        self.tree = ttk.Treeview(
            main, columns=columns, show="headings",
            selectmode="browse",
        )
        self.tree.heading("field", text="Field", anchor="w")
        self.tree.heading("value", text="Value", anchor="w")
        self.tree.column("field", width=180, minwidth=140, stretch=False)
        self.tree.column("value", width=500, minwidth=300, stretch=True)

        # Style the treeview to dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=BG_ALT, foreground=TEXT,
                        fieldbackground=BG_ALT,
                        font=("Segoe UI", 9),
                        rowheight=26,
                        borderwidth=0)
        style.configure("Treeview.Heading",
                        background=SURFACE2, foreground=TEXT_BRIGHT,
                        font=("Segoe UI", 9, "bold"),
                        borderwidth=0)
        style.map("Treeview",
                  background=[("selected", SURFACE3)],
                  foreground=[("selected", TEXT_BRIGHT)])
        style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<MouseWheel>", self._on_tree_mousewheel)

        # Bind selection to show full value in status
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-3>", self._on_right_click)

        # Populate
        for idx, item in enumerate(self.items):
            self.tree.insert("", "end", iid=str(idx),
                             values=(item.label, item.display_value))

        # Selected item tracker
        self._selected_idx: str | None = None

        # Button row
        btn_row = tk.Frame(self.dialog, bg=BG)
        btn_row.pack(fill="x", padx=16, pady=(8, 10))

        tk.Button(btn_row, text="Copy Selected Value", command=self._copy_selected,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=12, pady=4, activebackground=BUTTON_BOTTOM,
                  ).pack(side="left", padx=(0, 6))
        tk.Button(btn_row, text="Copy All Details", command=self._copy_all,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=12, pady=4, activebackground=BUTTON_BOTTOM,
                  ).pack(side="left", padx=(0, 6))
        tk.Button(btn_row, text="Close", command=self.dialog.destroy,
                  bg=SURFACE2, fg=MUTED, font=("Segoe UI", 9),
                  relief="flat", padx=16, pady=4, activebackground=SURFACE3,
                  ).pack(side="right", padx=4)

        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())
        self.dialog.bind("<Control-c>", lambda e: self._copy_selected())

    def _build_items(self, config: BridgeConfig | None,
                     owui: OpenWebUIConfig | None) -> list[_DetailItem]:
        import shutil, sys
        items: list[_DetailItem] = []

        # Python
        items.append(_DetailItem("Python executable", compact_path(sys.executable), sys.executable))
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        items.append(_DetailItem("Python version", py_ver, py_ver))

        # Conda
        conda_py = get_conda_python_path()
        items.append(_DetailItem("Conda python",
                                 compact_path(conda_py) if conda_py else "not found",
                                 conda_py or "not found"))

        # FFmpeg
        ffmpeg_path = shutil.which("ffmpeg")
        items.append(_DetailItem("FFmpeg path",
                                 compact_path(ffmpeg_path) if ffmpeg_path else "not found",
                                 ffmpeg_path or "not found"))

        # Open WebUI installed check (current Python fallback)
        installed, _ = check_openwebui_installed()
        items.append(_DetailItem("Open WebUI (current)", "yes" if installed else "no",
                                 "yes" if installed else "no"))

        # Discovery info
        disc = get_cached_discovery()
        if disc:
            items.append(_DetailItem("Discovery source", disc.source or "none", disc.source or "none"))
            items.append(_DetailItem("Discovery env name", disc.env_name or "none", disc.env_name or "none"))
            items.append(_DetailItem("Discovery env path",
                                     compact_path(str(disc.env_path)) if disc.env_path else "none",
                                     str(disc.env_path) if disc.env_path else "none"))
            items.append(_DetailItem("Discovery python",
                                     compact_path(str(disc.python_exe)) if disc.python_exe else "none",
                                     str(disc.python_exe) if disc.python_exe else "none"))
            items.append(_DetailItem("Discovery command",
                                     compact_path(str(disc.command)) if disc.command else "none",
                                     str(disc.command) if disc.command else "none"))
            items.append(_DetailItem("Package path",
                                     compact_path(str(disc.package_path)) if disc.package_path else "none",
                                     str(disc.package_path) if disc.package_path else "none"))
            items.append(_DetailItem("Version", disc.version or "unknown", disc.version or "unknown"))
            # Discovery details as separate row with button to expand
            detail_text = "; ".join(disc.details[-3:]) if disc.details else "none"
            items.append(_DetailItem("Discovery log", compact_value(detail_text, 100), detail_text))
        else:
            items.append(_DetailItem("Discovery", "not yet scanned", "not yet scanned"))

        # Config / ports
        if owui:
            ports = get_effective_ports(owui, config) if config else {}
            items.append(_DetailItem("Open WebUI port", str(ports.get("openwebui", "?")),
                                     str(ports.get("openwebui", "?"))))
            llm_url = f"http://{owui.host}:{ports.get('bridge_llm_only', '?')}/v1"
            items.append(_DetailItem("LLM endpoint", llm_url, llm_url))
            tools_url = f"http://{config.server.host if config else '?'}:{ports.get('bridge_tools', '?')}/v1"
            items.append(_DetailItem("Tools endpoint", tools_url, tools_url))
            items.append(_DetailItem("Data dir",
                                     compact_path(owui.openwebui_data_dir) if owui.openwebui_data_dir else "default",
                                     owui.openwebui_data_dir or "default"))
            items.append(_DetailItem("Web search provider", owui.web_search_provider or "none",
                                     owui.web_search_provider or "none"))

            # Auth / secrets — NEVER reveal actual values
            hf = "set" if (owui.hf_token or os.environ.get("HF_TOKEN")) else "not set"
            items.append(_DetailItem("HF token", hf, hf))
            items.append(_DetailItem("Auth mode", "On" if owui.auth_enabled else "Off",
                                     "On" if owui.auth_enabled else "Off"))
            host_mode = "LAN (0.0.0.0)" if owui.host in ("0.0.0.0",) else "local"
            items.append(_DetailItem("Host mode", host_mode, host_mode))

            # Web search API keys — show set/missing only
            for prov in ("ollama", "tavily", "serpapi", "searchapi"):
                pcfg = owui.web_search_providers.get(prov)
                key_status = "set" if (pcfg and pcfg.api_key) else "missing"
                items.append(_DetailItem(f"  {prov} API key", key_status, key_status + f" ({prov})"))

        else:
            items.append(_DetailItem("Open WebUI config", "not loaded", "not loaded"))

        return items

    def _on_tree_mousewheel(self, event: Any) -> None:
        self.tree.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_select(self, _event: Any = None) -> None:
        sel = self.tree.selection()
        self._selected_idx = sel[0] if sel else None

    def _on_right_click(self, event: Any) -> None:
        item = self.tree.identify("item", event.x, event.y)
        if item:
            self.tree.selection_set(item)
            self._selected_idx = item
            self._copy_selected()

    def _get_full_value(self, idx: str) -> str:
        try:
            return self.items[int(idx)].full_value
        except (IndexError, ValueError):
            return ""

    def _copy_selected(self) -> None:
        idx = self._selected_idx
        if idx is None:
            sel = self.tree.selection()
            idx = sel[0] if sel else None
        if idx is None:
            messagebox.showinfo("Copy", "No item selected.", parent=self.dialog)
            return
        full = self._get_full_value(idx)
        if full:
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(full)
            label = self.items[int(idx)].label
            messagebox.showinfo("Copied", f'Copied "{label}" value.', parent=self.dialog)

    def _copy_all(self) -> None:
        parts: list[str] = []
        for item in self.items:
            parts.append(f"{item.label:<25} {item.full_value}")
        text = "\n".join(parts)
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(text)
        messagebox.showinfo("Copied", f"All {len(self.items)} detail items copied.", parent=self.dialog)


@dataclass
class _DetailItem:
    label: str
    display_value: str
    full_value: str


# ── Config Dialog ──────────────────────────────────────────────────────────
class ConfigDialog:
    def __init__(self, parent: tk.Tk, owui: OpenWebUIConfig | None,
                 config: BridgeConfig | None, config_path: Path,
                 on_save: Any) -> None:
        self.owui = owui or OpenWebUIConfig()
        self.config = config
        self.config_path = config_path
        self.on_save = on_save

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configuration")
        self.dialog.geometry("460x540")
        self.dialog.minsize(400, 400)
        self.dialog.configure(bg=BG)
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=16)

        row = 0

        def add_field(label: str, var_type: str = "entry", **kw: Any) -> Any:
            nonlocal row
            tk.Label(main, text=label, bg=BG, fg=TEXT_BRIGHT,
                     font=("Segoe UI", 10), anchor="w").grid(row=row, column=0, sticky="w", pady=3)
            if var_type == "entry":
                e = tk.Entry(main, bg=SURFACE2, fg=TEXT, font=("Segoe UI", 10),
                             relief="flat", bd=0, insertbackground=TEXT, **kw)
                e.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
                main.columnconfigure(1, weight=1)
                row += 1
                return e
            if var_type == "check":
                v = tk.BooleanVar(value=kw.get("value", False))
                cb = tk.Checkbutton(main, variable=v, bg=BG, fg=TEXT,
                                    selectcolor=SURFACE2, activebackground=BG,
                                    font=("Segoe UI", 10))
                cb.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
                row += 1
                return v
            return None

        self.e_host = add_field("Open WebUI Host", width=24)
        self.e_port = add_field("Open WebUI Port", width=10)
        self.e_tools_port = add_field("Bridge Tools Port", width=10)
        self.e_llm_port = add_field("Bridge LLM-only Port", width=10)
        self.v_auth = add_field("Auth Enabled", "check", value=self.owui.auth_enabled)
        self.v_auto = add_field("Auto Login", "check", value=self.owui.auto_login)
        self.e_data = add_field("Data Dir (empty=default)", width=40)
        self.e_hf = add_field("HF Token", width=40, show="*")
        self.e_url_mode = add_field("OpenAI URL Mode")
        row += 1

        # Populate
        self.e_host.insert(0, self.owui.host)
        self.e_port.insert(0, str(self.owui.port))
        if self.owui.bridge_tools_port:
            self.e_tools_port.insert(0, str(self.owui.bridge_tools_port))
        if self.owui.bridge_llm_only_port:
            self.e_llm_port.insert(0, str(self.owui.bridge_llm_only_port))
        if self.owui.openwebui_data_dir:
            self.e_data.insert(0, self.owui.openwebui_data_dir)
        if self.owui.hf_token:
            self.e_hf.insert(0, self.owui.hf_token)
        self.e_url_mode.insert(0, self.owui.openai_base_url_mode or "llm_only")

        # Buttons
        btn_row = tk.Frame(main, bg=BG)
        btn_row.grid(row=row, column=0, columnspan=2, pady=(12, 0))
        tk.Button(btn_row, text="Save", command=self._save,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=24, pady=6, activebackground=BUTTON_BOTTOM,
                  ).pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", command=self.dialog.destroy,
                  bg=SURFACE2, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", padx=16, pady=6, activebackground=SURFACE3,
                  ).pack(side="left", padx=4)

        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())

    def _save(self) -> None:
        try:
            from .config import ExternalToolProviderConfig
            providers = {}
            if self.owui:
                for k, v in self.owui.web_search_providers.items():
                    providers[k] = v
            new_owui = OpenWebUIConfig(
                enabled=self.owui.enabled if self.owui else True,
                host=self.e_host.get().strip() or "127.0.0.1",
                port=int(self.e_port.get().strip() or "8080"),
                bridge_tools_port=int(v) if (v := self.e_tools_port.get().strip()) else None,
                bridge_llm_only_port=int(v) if (v := self.e_llm_port.get().strip()) else None,
                auth_enabled=self.v_auth.get(),
                auto_login=self.v_auto.get(),
                web_search_enabled=self.owui.web_search_enabled if self.owui else False,
                web_search_provider=self.owui.web_search_provider if self.owui else "ollama",
                web_search_providers=providers,
                search_result_count=self.owui.search_result_count if self.owui else 3,
                concurrent_requests=max(1, self.owui.concurrent_requests if self.owui else 1),
                bypass_embedding_and_retrieval=self.owui.bypass_embedding_and_retrieval if self.owui else False,
                bypass_web_loader=self.owui.bypass_web_loader if self.owui else False,
                hf_token=self.e_hf.get().strip() or None,
                openai_base_url_mode=self.e_url_mode.get().strip() or "llm_only",
                openwebui_data_dir=self.e_data.get().strip() or None,
                extra_env=self.owui.extra_env if self.owui else {},
                preferred_env_name=self.owui.preferred_env_name if self.owui else "omx-open-webui",
                preferred_python=self.owui.preferred_python if self.owui else None,
                preferred_command=self.owui.preferred_command if self.owui else None,
                auto_discover=self.owui.auto_discover if self.owui else True,
            )
            save_openwebui_config(new_owui, self.config_path)
            self.dialog.destroy()
            messagebox.showinfo("Saved", "Configuration saved.")
            self.on_save()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save: {exc}")


# ── Web Search Dialog ──────────────────────────────────────────────────────
class WebSearchDialog:
    def __init__(self, parent: tk.Tk, owui: OpenWebUIConfig | None,
                 config_path: Path, on_save: Any) -> None:
        self.owui = owui or OpenWebUIConfig()
        self.config_path = config_path
        self.on_save = on_save

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Web Search Configuration")
        self.dialog.geometry("460x520")
        self.dialog.minsize(400, 380)
        self.dialog.configure(bg=BG)
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=16)

        row = 0

        def add_field(label: str, var_type: str = "entry", **kw: Any) -> Any:
            nonlocal row
            tk.Label(main, text=label, bg=BG, fg=TEXT_BRIGHT,
                     font=("Segoe UI", 10), anchor="w").grid(row=row, column=0, sticky="w", pady=3)
            if var_type == "entry":
                e = tk.Entry(main, bg=SURFACE2, fg=TEXT, font=("Segoe UI", 10),
                             relief="flat", bd=0, insertbackground=TEXT, **kw)
                e.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
                main.columnconfigure(1, weight=1)
                row += 1
                return e
            if var_type == "check":
                v = tk.BooleanVar(value=kw.get("value", False))
                cb = tk.Checkbutton(main, variable=v, bg=BG, fg=TEXT,
                                    selectcolor=SURFACE2, activebackground=BG,
                                    font=("Segoe UI", 10))
                cb.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
                row += 1
                return v
            if var_type == "combo":
                cb = ttk.Combobox(main, values=list(VALID_SEARCH_PROVIDERS),
                                  state="readonly", width=18, font=("Segoe UI", 10))
                cb.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
                row += 1
                return cb
            return None

        self.v_enabled = add_field("Enable Web Search", "check", value=self.owui.web_search_enabled)
        self.c_provider = add_field("Provider", "combo")
        self.c_provider.set(self.owui.web_search_provider)
        self.e_api_key = add_field("API Key", width=40, show="*")
        self.e_count = add_field("Search Result Count", width=10)
        self.e_concurrent = add_field("Concurrent Requests", width=10)
        self.v_bypass_embed = add_field("Bypass Embedding & Retrieval", "check",
                                        value=self.owui.bypass_embedding_and_retrieval)
        self.v_bypass_loader = add_field("Bypass Web Loader", "check",
                                         value=self.owui.bypass_web_loader)
        row += 1

        # Populate keys
        pcfg = self.owui.web_search_providers.get(self.owui.web_search_provider)
        if pcfg and pcfg.api_key:
            self.e_api_key.insert(0, pcfg.api_key)
        self.e_count.insert(0, str(self.owui.search_result_count))
        self.e_concurrent.insert(0, str(max(1, self.owui.concurrent_requests)))

        # Buttons
        btn_row = tk.Frame(main, bg=BG)
        btn_row.grid(row=row, column=0, columnspan=2, pady=(12, 0))
        tk.Button(btn_row, text="Test Search", command=self._test,
                  bg=SURFACE2, fg=TEXT, font=("Segoe UI", 10),
                  relief="flat", padx=14, pady=6, activebackground=SURFACE3,
                  ).pack(side="left", padx=4)
        tk.Button(btn_row, text="Save", command=self._save,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=20, pady=6, activebackground=BUTTON_BOTTOM,
                  ).pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", command=self.dialog.destroy,
                  bg=SURFACE2, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", padx=16, pady=6, activebackground=SURFACE3,
                  ).pack(side="left", padx=4)

        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())

    def _test(self) -> None:
        provider = self.c_provider.get()
        if provider == "disabled":
            messagebox.showinfo("Test Search", "Web search is disabled.")
            return
        api_key = self.e_api_key.get().strip() or os.environ.get(f"{provider.upper()}_API_KEY")
        if not api_key:
            messagebox.showwarning("Test Search", f"No API key for {provider}.")
            return

        def run():
            ok, msg = test_search_provider(provider, api_key, "Open WebUI test")
            self.dialog.after(0, lambda: messagebox.showinfo(
                "Test Search", f"Provider: {provider}\n\n{'OK' if ok else 'FAILED'}\n{msg}"
            ))

        threading.Thread(target=run, daemon=True).start()

    def _save(self) -> None:
        try:
            from .config import ExternalToolProviderConfig
            provider = self.c_provider.get()
            api_key = self.e_api_key.get().strip() or None
            providers = {}
            for prov in ("ollama", "tavily", "serpapi", "searchapi"):
                existing = self.owui.web_search_providers.get(prov)
                p_key = api_key if prov == provider else (existing.api_key if existing else None)
                providers[prov] = ExternalToolProviderConfig(
                    enabled=(existing.enabled if existing else True),
                    api_key=p_key,
                    base_url=(existing.base_url if existing else None),
                    defaults=(existing.defaults if existing else {}),
                )

            concurrent = 1
            try:
                concurrent = max(1, int(self.e_concurrent.get().strip() or "1"))
            except (ValueError, TypeError):
                concurrent = 1

            new_owui = OpenWebUIConfig(
                enabled=self.owui.enabled,
                host=self.owui.host, port=self.owui.port,
                bridge_tools_port=self.owui.bridge_tools_port,
                bridge_llm_only_port=self.owui.bridge_llm_only_port,
                auth_enabled=self.owui.auth_enabled,
                auto_login=self.owui.auto_login,
                web_search_enabled=self.v_enabled.get(),
                web_search_provider=provider,
                web_search_providers=providers,
                search_result_count=max(1, int(self.e_count.get().strip() or "3")),
                concurrent_requests=concurrent,
                bypass_embedding_and_retrieval=self.v_bypass_embed.get(),
                bypass_web_loader=self.v_bypass_loader.get(),
                hf_token=self.owui.hf_token,
                openai_base_url_mode=self.owui.openai_base_url_mode,
                openwebui_data_dir=self.owui.openwebui_data_dir,
                extra_env=self.owui.extra_env,
                preferred_env_name=self.owui.preferred_env_name,
                preferred_python=self.owui.preferred_python,
                preferred_command=self.owui.preferred_command,
                auto_discover=self.owui.auto_discover,
            )
            save_openwebui_config(new_owui, self.config_path)
            self.dialog.destroy()
            messagebox.showinfo("Saved", "Web search configuration saved.")
            self.on_save()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save: {exc}")


# ── Logs Dialog ────────────────────────────────────────────────────────────
class LogsDialog:
    def __init__(self, parent: tk.Tk) -> None:
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Logs")
        self.dialog.geometry("720x520")
        self.dialog.minsize(480, 320)
        self.dialog.configure(bg=BG)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)

        nb = ttk.Notebook(self.dialog)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        tab_owui = tk.Frame(nb, bg=BG)
        tab_bridge = tk.Frame(nb, bg=BG)
        nb.add(tab_owui, text=" Open WebUI Logs ")
        nb.add(tab_bridge, text=" Bridge Logs ")

        self._auto_refresh = True
        self._poll_id = None
        self._last_sizes: dict[str, int] = {"owui": 0, "bridge": 0}

        self._build_log_tab(tab_owui, OPENWEBUI_LOG_PATH, "owui")
        self._build_log_tab(tab_bridge, LLAMA_LOG_PATH, "bridge")

        btn_row = tk.Frame(self.dialog, bg=BG)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))
        self._auto_btn = tk.Button(btn_row, text="Auto: ON",
                                   command=self._toggle_auto,
                                   bg=BUTTON_TOP, fg=TEXT_BRIGHT,
                                   font=("Segoe UI", 10, "bold"),
                                   relief="flat", padx=14, pady=4,
                                   activebackground=BUTTON_BOTTOM)
        self._auto_btn.pack(side="left", padx=4)
        tk.Button(btn_row, text="Clear", command=self._clear_all,
                  bg=SURFACE2, fg=TEXT, font=("Segoe UI", 10),
                  relief="flat", padx=14, pady=4, activebackground=SURFACE3,
                  ).pack(side="left", padx=4)
        tk.Button(btn_row, text="Close", command=self._close,
                  bg=SURFACE2, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", padx=14, pady=4, activebackground=SURFACE3,
                  ).pack(side="right", padx=4)

        self.dialog.bind("<Escape>", lambda e: self._close())
        self.dialog.protocol("WM_DELETE_WINDOW", self._close)
        self._start_polling()

    def _build_log_tab(self, parent: tk.Frame, log_path: Path, key: str) -> None:
        text_w = tk.Text(parent, bg=BG_ALT, fg=TEXT, font=("Consolas", 9),
                         relief="flat", bd=0, wrap="none", highlightthickness=0)
        text_w.pack(fill="both", expand=True)
        text_w.bind("<MouseWheel>", lambda e: text_w.yview_scroll(-1 * (e.delta // 120), "units"))
        text_w.tag_configure("error", foreground=RED)
        text_w.tag_configure("warn", foreground=YELLOW)
        text_w.tag_configure("info", foreground=GREEN)
        text_w.tag_configure("muted", foreground=MUTED)
        setattr(self, f"_log_text_{key}", text_w)
        self._load_log(text_w, log_path)
        try:
            self._last_sizes[key] = log_path.stat().st_size if log_path.exists() else 0
        except OSError:
            self._last_sizes[key] = 0

    def _load_log(self, text_w: tk.Text, log_path: Path) -> None:
        try:
            lines = follow_log(log_path, 200)
            text_w.delete("1.0", "end")
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
                text_w.insert("end", line + "\n", tag)
            text_w.see("end")
        except Exception:
            text_w.delete("1.0", "end")
            text_w.insert("1.0", "(log unavailable)")

    def _start_polling(self) -> None:
        self._poll()

    def _poll(self) -> None:
        for key, path in (("bridge", LLAMA_LOG_PATH), ("owui", OPENWEBUI_LOG_PATH)):
            tw = getattr(self, f"_log_text_{key}", None)
            if tw:
                self._append_new_lines(tw, path, key)
        if self._auto_refresh:
            self._poll_id = self.dialog.after(1500, self._poll)

    def _append_new_lines(self, text_w: tk.Text, log_path: Path, key: str) -> None:
        try:
            if not log_path.exists():
                return
            current_size = log_path.stat().st_size
            if current_size < self._last_sizes[key]:
                self._last_sizes[key] = 0
                return
            if current_size == self._last_sizes[key]:
                return
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._last_sizes[key])
                new_lines = f.readlines()
                self._last_sizes[key] = f.tell()
            for line in new_lines:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                lower = line.lower()
                if "error" in lower or "exception" in lower or "traceback" in lower:
                    tag = "error"
                elif "warn" in lower:
                    tag = "warn"
                elif "info" in lower or "start" in lower or "running" in lower or "ready" in lower:
                    tag = "info"
                else:
                    tag = "muted"
                text_w.insert("end", line + "\n", tag)
            text_w.see("end")
        except Exception:
            pass

    def _toggle_auto(self) -> None:
        self._auto_refresh = not self._auto_refresh
        self._auto_btn.config(text=f"Auto: {'ON' if self._auto_refresh else 'OFF'}")
        if self._auto_refresh:
            self._start_polling()
        else:
            if self._poll_id:
                self.dialog.after_cancel(self._poll_id)
                self._poll_id = None

    def _clear_all(self) -> None:
        for path in (LLAMA_LOG_PATH, OPENWEBUI_LOG_PATH):
            try:
                path.write_text("", encoding="utf-8")
            except Exception:
                pass
        for key, path in (("bridge", LLAMA_LOG_PATH), ("owui", OPENWEBUI_LOG_PATH)):
            tw = getattr(self, f"_log_text_{key}", None)
            if tw:
                self._load_log(tw, path)
            try:
                self._last_sizes[key] = path.stat().st_size if path.exists() else 0
            except OSError:
                self._last_sizes[key] = 0

    def _close(self) -> None:
        if self._poll_id:
            self.dialog.after_cancel(self._poll_id)
            self._poll_id = None
        self.dialog.destroy()


# ── Entry Point ────────────────────────────────────────────────────────────
def launch_gui(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    if not HAS_TK:
        print("Tkinter is not available. Use `llama openwebui configure` for CLI setup.")
        sys.exit(1)
    app = OpenWebUISetupCenter(config_path)
    app.run()
