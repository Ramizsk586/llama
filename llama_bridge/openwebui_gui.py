from __future__ import annotations

import json
import os
import subprocess
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
    follow_log,
    status as launcher_status,
    stop_all,
    LLAMA_PID_PATH,
    OPENWEBUI_PID_PATH,
    LLAMA_LOG_PATH,
    OPENWEBUI_LOG_PATH,
)

try:
    import customtkinter as ctk
    HAS_TK = True
except ImportError:
    HAS_TK = False
    class _FakeCTk:
        class CTk:
            def __init__(self):
                raise RuntimeError("customtkinter not available")
    ctk = _FakeCTk()

import tkinter as tk
from tkinter import ttk, messagebox

GREEN = "#4FD1A1"
RED = "#FF6B6B"
YELLOW = "#F2C66D"

LAYOUT = {
    "PAD": 20,
    "CARD_H": 68,
    "CARD_GAP": 10,
    "WIN_W": 640,
    "WIN_H": 480,
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
    def __init__(self, widget: ctk.CTkBaseClass, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: ctk.CTkToplevel | None = None
        widget.bind("<Enter>", self._enter, add=True)
        widget.bind("<Leave>", self._leave, add=True)

    def _enter(self, _event: Any = None) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = ctk.CTkToplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = ctk.CTkLabel(self.tip, text=self.text, font=("Segoe UI", 9), padx=8, pady=4)
        lbl.pack()

    def _leave(self, _event: Any = None) -> None:
        if self.tip:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


def _center_window(root: ctk.CTk, w: int, h: int) -> None:
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")


def compact_path(value: str, max_chars: int = 72) -> str:
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
    if not value:
        return "none"
    text = str(value)
    if len(text) <= max_chars:
        return text
    if ":" in text[:3] or "/" in text or "\\" in text:
        return compact_path(text, max_chars)
    return text[:max_chars - 3] + "..."


class OpenWebUISetupCenter:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.config_path = config_path
        self._stopped = threading.Event()

        W, H = LAYOUT["WIN_W"], LAYOUT["WIN_H"]
        self.root = ctk.CTk()
        self.root.title("Llama Bridge - Open WebUI Setup")
        _center_window(self.root, W, H)
        self.root.minsize(520, 400)
        self.root.resizable(True, True)

        self.phase = Phase.READY
        self._config: BridgeConfig | None = None
        self._owui: OpenWebUIConfig | None = None
        self._last_msg: str = ""
        self._log_lines: list[str] = []
        self._detail_cards: list[dict[str, Any]] = []
        self._owui_health_ok = False
        self._discovery: OpenWebUIDiscovery = OpenWebUIDiscovery()

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self._card_data: list[dict[str, Any]] = [
            {"key": "python", "title": "Python", "ok": True, "subtitle": py_ver, "status": "OK"},
            {"key": "conda", "title": "Conda / Env", "ok": True, "subtitle": "", "status": ""},
            {"key": "ffmpeg", "title": "FFmpeg", "ok": True, "subtitle": "", "status": ""},
            {"key": "owui", "title": "Open WebUI", "ok": True, "subtitle": "", "status": ""},
        ]
        self._card_widgets: list[ctk.CTkFrame] = []

        self._build_ui()
        self._load_config()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._poll_status)

    def _build_ui(self) -> None:
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True)

        self._build_header(main)

        sep = ctk.CTkFrame(main, fg_color=GREEN, height=2, corner_radius=0)
        sep.pack(fill="x", padx=PAD, pady=(4, 0))

        self.status_label = ctk.CTkLabel(main, text="", font=("Segoe UI", 9),
                                          text_color=("#888888", "#888888"), anchor="w")
        self.status_label.pack(fill="x", padx=PAD, pady=(6, 4))

        self._build_footer(main)

        self.cards_area = ctk.CTkFrame(main, fg_color="transparent")
        self.cards_area.pack(fill="x", expand=False, padx=PAD, pady=(8, 0))

        self._bind_shortcuts()

    def _build_header(self, parent: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=PAD, pady=(PAD, 0))

        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(title_frame, text="OpenWebUI Setup Center",
                     font=("Segoe UI", 18, "bold"), anchor="w").pack(anchor="w")
        self.subtitle_label = ctk.CTkLabel(title_frame, text="",
                                           font=("Segoe UI", 10),
                                           text_color=("#888888", "#888888"), anchor="w")
        self.subtitle_label.pack(anchor="w", pady=(2, 0))

        self.badge = ctk.CTkLabel(header, text="  READY  ",
                                  font=("Segoe UI", 10, "bold"), corner_radius=6)
        self.badge.grid(row=0, column=1, sticky="ne", pady=(4, 0))

        self._update_header()

    def _update_header(self) -> None:
        self.subtitle_label.configure(text=self._get_subtitle())
        badge_text, badge_color = self._get_badge_info()
        badge_bg = {
            Phase.RUNNING: "#0a2a1a",
            Phase.READY: "#0a2a1a",
            Phase.INSTALL_PREREQS: "#2a2a0a",
            Phase.SETUP_ENV: "#2a2a0a",
            Phase.STARTING: "#2a2a0a",
            Phase.ERROR: "#2a0a0a",
        }.get(self.phase, "#0a2a1a")
        self.badge.configure(text=f"  {badge_text}  ",
                             fg_color=badge_bg, text_color=badge_color)

    def _build_footer(self, parent: ctk.CTkFrame) -> None:
        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.pack(fill="x", side="bottom", pady=(4, 12), padx=PAD)

        panels_frame = ctk.CTkFrame(footer, fg_color="transparent")
        panels_frame.pack(fill="x")
        for col in range(3):
            panels_frame.columnconfigure(col, weight=1, uniform="panel_col")

        sub_panels = [
            ("Config", [("\u2699 Config", self._open_config_dialog),
                        ("\U0001f9f0 Guide", self._open_guide_dialog)]),
            ("Search", [("\U0001f50d Web Search", self._open_websearch_dialog),
                        ("\u21bb Rescan", self._action_rescan)]),
            ("Info", [("\U0001f4cb Logs", self._open_logs_dialog),
                      ("\u2139 Details", self._open_details_dialog),
                      ("Preview Cmd", self._open_cmd_dialog)]),
        ]
        self.util_btns: dict[str, ctk.CTkButton] = {}
        tips = {
            "\u2699 Config": "Open configuration dialog",
            "\U0001f9f0 Guide": "Open setup guide with install commands",
            "\U0001f50d Web Search": "Web search settings",
            "\u21bb Rescan": "Re-scan for Open WebUI installations",
            "\U0001f4cb Logs": "View logs",
            "\u2139 Details": "Show full technical details",
            "Preview Cmd": "Open environment variable preview dialog",
        }
        for col, (panel_title, buttons) in enumerate(sub_panels):
            panel = ctk.CTkFrame(panels_frame, fg_color=("#e8e8e8", "#1e1e1e"), corner_radius=6)
            panel.grid(row=0, column=col, sticky="nsew", padx=4, pady=2)
            panel.columnconfigure(0, weight=1)
            panel.columnconfigure(1, weight=1)

            ctk.CTkLabel(panel, text=panel_title,
                         font=("Segoe UI", 8, "bold"),
                         text_color=("#666666", "#888888")).grid(
                row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 0))

            for i, (text, cmd) in enumerate(buttons):
                btn = ctk.CTkButton(
                    panel, text=text, command=cmd,
                    font=("Segoe UI", 9), height=26,
                    fg_color=("#e0e0e0", "#2a2a2a"),
                    text_color=("#333333", "#cccccc"),
                    hover_color=("#d0d0d0", "#3a3a3a"),
                    corner_radius=4,
                )
                r = 1 + i // 2
                c = i % 2
                span = 2 if (len(buttons) == 3 and i == 2) else 1
                btn.grid(row=r, column=c, columnspan=span, sticky="ew", padx=4, pady=(2, 4))
                self.util_btns[text] = btn
                ToolTip(btn, tips.get(text, ""))

        action = ctk.CTkFrame(footer, fg_color="transparent")
        action.pack(fill="x", pady=(8, 0))

        self.primary_btn = ctk.CTkButton(
            action, text="Start Server", command=self._primary_action,
            font=("Segoe UI", 11, "bold"), height=40,
            fg_color=("#1a4030", "#1a4030"), text_color=GREEN,
            hover_color=("#153828", "#153828"), corner_radius=6,
        )
        self.primary_btn.pack(side="right")
        ToolTip(self.primary_btn, "Start or stop the Open WebUI server")

    def _create_card(self, card_data: dict, parent: ctk.CTkFrame | None = None) -> ctk.CTkFrame:
        if parent is None:
            parent = self.cards_area
        ok = card_data["ok"]
        card_bg = "#0f2820" if ok else "#281414"
        accent_color = GREEN if ok else RED
        icon_text = "\u2713" if ok else "\u2717"

        card = ctk.CTkFrame(parent, fg_color=card_bg, corner_radius=10, height=LAYOUT["CARD_H"] + 4)
        card.grid_propagate(False)

        card.columnconfigure(0, weight=0)
        card.columnconfigure(1, weight=0)
        card.columnconfigure(2, weight=1)
        card.columnconfigure(3, weight=0)

        accent = ctk.CTkFrame(card, fg_color=accent_color, width=5, corner_radius=0)
        accent.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))

        ctk.CTkLabel(card, text=icon_text,
                     font=("Segoe UI", 14, "bold"),
                     text_color=accent_color).grid(row=0, column=1, rowspan=2, padx=(0, 6))

        ctk.CTkLabel(card, text=card_data["title"],
                     font=("Segoe UI", 11, "bold"), anchor="w").grid(
            row=0, column=2, sticky="w", pady=(7, 0))

        ctk.CTkLabel(card, text=card_data.get("subtitle", ""),
                     font=("Segoe UI", 10),
                     text_color=("#999999", "#999999"), anchor="w").grid(
            row=1, column=2, sticky="w", pady=(0, 7))

        ctk.CTkLabel(card, text=card_data.get("status", ""),
                     font=("Segoe UI", 10, "bold"),
                     text_color=accent_color).grid(
            row=0, column=3, rowspan=2, sticky="e", padx=(0, 18))

        return card

    def _rebuild_cards(self) -> None:
        for w in self._card_widgets:
            w.destroy()
        self._card_widgets.clear()

        self.cards_area.columnconfigure(0, weight=1, uniform="card_col")
        self.cards_area.columnconfigure(1, weight=1, uniform="card_col")

        GAP = LAYOUT["CARD_GAP"] // 2
        for idx, card_data in enumerate(self._card_data):
            r, c = divmod(idx, 2)
            px = (0, GAP) if c == 0 else (GAP, 0)
            card = self._create_card(card_data, parent=self.cards_area)
            card.grid(row=r, column=c, sticky="nsew", padx=px, pady=GAP)
            self._card_widgets.append(card)

        for r in range((len(self._card_data) + 1) // 2):
            self.cards_area.rowconfigure(r, weight=0)

    def _get_subtitle(self) -> str:
        phase = self.phase
        if phase == Phase.INSTALL_PREREQS:
            return "Open WebUI package is missing \u2014 configure in Config panel."
        if phase == Phase.SETUP_ENV:
            return "Configure your environment."
        if phase == Phase.READY:
            return "Ready to start."
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

    def _load_config(self) -> None:
        try:
            self._config = load_config(self.config_path)
            self._owui = self._config.openwebui
        except Exception:
            self._config = None
            self._owui = None

    def _update_card_data(self) -> None:
        cfg, ow = self._config, self._owui
        if not cfg or not ow:
            return

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        py_ok = sys.version_info >= (3, 11)
        self._card_data[0] = {"key": "python", "title": "Python", "ok": py_ok,
                              "subtitle": py_ver, "status": "OK" if py_ok else "Upgrade"}

        import shutil
        conda_exe = shutil.which("conda")
        conda_prefix = os.environ.get("CONDA_PREFIX") or ""
        venv = os.environ.get("CONDA_DEFAULT_ENV") or ""
        conda_python = get_conda_python_path()
        if conda_exe:
            try:
                r = subprocess.run([conda_exe, "--version"], capture_output=True, text=True, timeout=5)
                cv = r.stdout.strip()
            except Exception:
                cv = ""
            if conda_prefix:
                subtitle = f"{venv} {cv}" if cv else venv
                status = "Active"
            elif cv:
                subtitle = cv
                status = "Ready"
            else:
                subtitle = "available"
                status = "Ready"
        elif conda_python:
            subtitle = "env found (not on PATH)"
            status = "Inactive"
        elif venv:
            subtitle = venv
            status = "venv"
        else:
            subtitle = "system"
            status = "System"
        conda_ok = bool(conda_exe or conda_python)
        self._card_data[1] = {"key": "conda", "title": "Conda / Env", "ok": conda_ok,
                              "subtitle": subtitle, "status": status}

        import shutil
        ffmpeg_found = shutil.which("ffmpeg") is not None
        self._card_data[2] = {"key": "ffmpeg", "title": "FFmpeg", "ok": ffmpeg_found,
                              "subtitle": "Present" if ffmpeg_found else "Missing",
                              "status": "OK" if ffmpeg_found else "Missing"}

        disc = self._discovery
        if not disc.installed:
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
        if not self.primary_btn:
            return

        if self.phase == Phase.INSTALL_PREREQS:
            self.primary_btn.configure(text="Start Server", state="disabled",
                                       fg_color=("#444444", "#333333"),
                                       text_color=("#888888", "#888888"))
        elif self.phase == Phase.READY:
            self.primary_btn.configure(text="Start Server", state="normal", fg_color="#1a4030", text_color=GREEN)
        elif self.phase == Phase.STARTING:
            self.primary_btn.configure(text="Starting\u2026", state="disabled", fg_color=("#444444", "#333333"), text_color=("#888888", "#888888"))
        elif self.phase == Phase.RUNNING:
            self.primary_btn.configure(text="Stop Server", state="normal", fg_color="#281414", text_color=RED)
        elif self.phase == Phase.ERROR:
            self.primary_btn.configure(text="Retry", state="normal", fg_color="#281414", text_color=YELLOW)

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
            parts.append("Open WebUI package is missing")
        elif self.phase == Phase.READY:
            parts.append("Ready to start")

        txt = "  |  ".join(parts) if parts else ""
        self.status_label.configure(text=txt,
                                    text_color=GREEN if self.phase == Phase.RUNNING else ("#888888", "#888888"))

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
        if hasattr(self, 'cmd_text') and self.cmd_text:
            self.cmd_text.delete("0.0", "end")
            self.cmd_text.insert("0.0", cmd_txt)

    def _open_cmd_dialog(self) -> None:
        CmdPreviewDialog(self.root, self._owui, self._config)

    def _append_log_line(self, line: str) -> None:
        pass

    def _primary_action(self) -> None:
        if self.phase == Phase.READY:
            self._action_start()
        elif self.phase == Phase.RUNNING:
            self._action_stop()
        elif self.phase == Phase.ERROR:
            self.phase = Phase.READY
            self._refresh_all()

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
        self._update_header()
        self._rebuild_cards()
        self._update_footer_buttons()
        self._update_status_label()

    def _save_config(self) -> None:
        if self._owui:
            save_openwebui_config(self._owui, self.config_path)

    def _poll_status(self) -> None:
        if self._stopped.is_set():
            return
        try:
            self._refresh_all()
        except Exception:
            pass
        self.root.after(3000, self._poll_status)

    def _open_config_dialog(self) -> None:
        ConfigDialog(self.root, self._owui, self._config, self.config_path, self._on_config_saved)

    def _open_websearch_dialog(self) -> None:
        WebSearchDialog(self.root, self._owui, self.config_path, self._on_config_saved)

    def _open_logs_dialog(self) -> None:
        LogsDialog(self.root)

    def _open_details_dialog(self) -> None:
        DetailsDialog(self.root, self._config, self._owui)

    def _open_guide_dialog(self) -> None:
        SetupGuideDialog(self.root, self._owui)

    def _on_config_saved(self) -> None:
        self._refresh_all()

    def _action_rescan(self) -> None:
        clear_discovery_cache()
        self._discovery = OpenWebUIDiscovery()
        self._refresh_all()

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-l>", lambda e: self._open_logs_dialog())
        self.root.bind("<Control-s>", lambda e: [self._save_config(), messagebox.showinfo("Saved", "Configuration saved.")])
        self.root.bind("<Control-r>", lambda e: self._action_stop() if self.phase == Phase.RUNNING else self._action_start())
        self.root.bind("<Escape>", lambda e: None)

    def _on_close(self) -> None:
        self._stopped.set()
        try:
            if self.phase == Phase.RUNNING:
                self.root.withdraw()
            else:
                self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()


class DetailsDialog:
    def __init__(self, parent: ctk.CTk, config: BridgeConfig | None,
                 owui: OpenWebUIConfig | None) -> None:
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Details")
        self.dialog.geometry("780x540")
        self.dialog.minsize(640, 440)

        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)

        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        self.items = self._build_items(config, owui)

        style = ttk.Style()
        style.theme_use("clam")
        dark_bg = "#1a1a1a"
        dark_fg = "#e0e0e0"
        sel_bg = "#2a2a2a"
        heading_bg = "#222222"
        heading_fg = "#fafafa"
        style.configure("OWUI.Treeview",
                        background=dark_bg, foreground=dark_fg,
                        fieldbackground=dark_bg,
                        font=("Segoe UI", 9), rowheight=28, borderwidth=0)
        style.configure("OWUI.Treeview.Heading",
                        background=heading_bg, foreground=heading_fg,
                        font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("OWUI.Treeview",
                  background=[("selected", sel_bg)],
                  foreground=[("selected", dark_fg)])
        style.layout("OWUI.Treeview", [("OWUI.Treeview.treearea", {"sticky": "nswe"})])

        tree_frame = ctk.CTkFrame(main, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("field", "value"), show="headings",
                                 selectmode="browse", style="OWUI.Treeview")
        self.tree.heading("field", text="Field", anchor="w")
        self.tree.heading("value", text="Value", anchor="w")
        self.tree.column("field", width=200, minwidth=140, stretch=False)
        self.tree.column("value", width=500, minwidth=300, stretch=True)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<MouseWheel>", self._on_tree_mousewheel)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-3>", self._on_right_click)

        for idx, item in enumerate(self.items):
            self.tree.insert("", "end", iid=str(idx), values=(item.label, item.display_value))

        self._selected_idx: str | None = None

        btn_row = ctk.CTkFrame(self.dialog, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(btn_row, text="Copy Selected Value", command=self._copy_selected,
                      font=("Segoe UI", 9), height=30).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_row, text="Copy All Details", command=self._copy_all,
                      font=("Segoe UI", 9), height=30).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_row, text="Close", command=self.dialog.destroy,
                      font=("Segoe UI", 9), height=30,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right", padx=4)
        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())
        self.dialog.bind("<Control-c>", lambda e: self._copy_selected())

    def _build_items(self, config: BridgeConfig | None,
                     owui: OpenWebUIConfig | None) -> list[_DetailItem]:
        import shutil
        items: list[_DetailItem] = []

        items.append(_DetailItem("Python executable", compact_path(sys.executable), sys.executable))
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        items.append(_DetailItem("Python version", py_ver, py_ver))

        conda_py = get_conda_python_path()
        items.append(_DetailItem("Conda python",
                                 compact_path(conda_py) if conda_py else "not found",
                                 conda_py or "not found"))

        ffmpeg_path = shutil.which("ffmpeg")
        items.append(_DetailItem("FFmpeg path",
                                 compact_path(ffmpeg_path) if ffmpeg_path else "not found",
                                 ffmpeg_path or "not found"))

        installed, _ = check_openwebui_installed()
        items.append(_DetailItem("Open WebUI (current)", "yes" if installed else "no",
                                 "yes" if installed else "no"))

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
            detail_text = "; ".join(disc.details[-3:]) if disc.details else "none"
            items.append(_DetailItem("Discovery log", compact_value(detail_text, 100), detail_text))
        else:
            items.append(_DetailItem("Discovery", "not yet scanned", "not yet scanned"))

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

            hf = "set" if (owui.hf_token or os.environ.get("HF_TOKEN")) else "not set"
            items.append(_DetailItem("HF token", hf, hf))
            items.append(_DetailItem("Auth mode", "On" if owui.auth_enabled else "Off",
                                     "On" if owui.auth_enabled else "Off"))
            host_mode = "LAN (0.0.0.0)" if owui.host in ("0.0.0.0",) else "local"
            items.append(_DetailItem("Host mode", host_mode, host_mode))

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


class ConfigDialog:
    def __init__(self, parent: ctk.CTk, owui: OpenWebUIConfig | None,
                 config: BridgeConfig | None, config_path: Path,
                 on_save: Any) -> None:
        self.owui = owui or OpenWebUIConfig()
        self.config = config
        self.config_path = config_path
        self.on_save = on_save

        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Configuration")
        self.dialog.geometry("480x448")
        self.dialog.minsize(420, 336)
        self.dialog.resizable(True, True)

        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=16)

        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll._scrollbar.grid_remove()

        row = 0
        self._entries: list[ctk.CTkEntry] = []

        def add_field(label: str, var_type: str = "entry", **kw: Any) -> Any:
            nonlocal row
            ctk.CTkLabel(scroll, text=label, font=("Segoe UI", 10), anchor="w").grid(
                row=row, column=0, sticky="w", pady=3)
            if var_type == "entry":
                show = kw.get("show", "")
                w = kw.get("width", None)
                e = ctk.CTkEntry(scroll, font=("Segoe UI", 10), show=show, width=w)
                e.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
                self._entries.append(e)
                scroll.columnconfigure(1, weight=1)
                row += 1
                return e
            if var_type == "check":
                v = ctk.BooleanVar(value=kw.get("value", False))
                ctk.CTkCheckBox(scroll, variable=v, text="", font=("Segoe UI", 10)).grid(
                    row=row, column=1, sticky="w", padx=(8, 0), pady=3)
                row += 1
                return v
            return None

        self.c_host = ctk.CTkComboBox(scroll, values=["local (127.0.0.1)", "LAN (0.0.0.0)"],
                                       state="readonly", font=("Segoe UI", 10), width=160)
        ctk.CTkLabel(scroll, text="Server Type", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=3)
        self.c_host.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
        self.c_host.set("local (127.0.0.1)" if self.owui.host in ("127.0.0.1",) else "LAN (0.0.0.0)")
        row += 1
        self.e_port = add_field("Open WebUI Port", width=10)
        self.e_tools_port = add_field("Bridge Tools Port", width=10)
        self.e_llm_port = add_field("Bridge LLM-only Port", width=10)
        self.v_auth = add_field("Auth Enabled", "check", value=self.owui.auth_enabled)
        self.v_auto = add_field("Auto Login", "check", value=self.owui.auto_login)
        self.e_data = add_field("Data Dir (empty=default)", width=40)
        self.e_hf = add_field("HF Token", width=40, show="*")
        self.e_url_mode = add_field("OpenAI URL Mode", width=30)
        row += 1

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

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self._close,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

        self.dialog.protocol("WM_DELETE_WINDOW", self._close)
        self.dialog.bind("<Escape>", lambda e: self._close())

    def _close(self) -> None:
        for w in self._entries:
            try:
                w.destroy()
            except RuntimeError:
                pass
        self.dialog.destroy()

    def _save(self) -> None:
        try:
            from .config import ExternalToolProviderConfig
            providers = {}
            if self.owui:
                for k, v in self.owui.web_search_providers.items():
                    providers[k] = v
            new_owui = OpenWebUIConfig(
                enabled=self.owui.enabled if self.owui else True,
                host="0.0.0.0" if "LAN" in self.c_host.get() else "127.0.0.1",
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
            self._close()
            messagebox.showinfo("Saved", "Configuration saved.")
            self.on_save()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save: {exc}")


class WebSearchDialog:
    def __init__(self, parent: ctk.CTk, owui: OpenWebUIConfig | None,
                 config_path: Path, on_save: Any) -> None:
        self.owui = owui or OpenWebUIConfig()
        self.config_path = config_path
        self.on_save = on_save

        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Web Search Configuration")
        self.dialog.geometry("480x432")
        self.dialog.minsize(420, 320)
        self.dialog.resizable(True, True)

        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=16)

        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll._scrollbar.grid_remove()

        row = 0
        self._entries: list[ctk.CTkEntry] = []

        def add_field(label: str, var_type: str = "entry", **kw: Any) -> Any:
            nonlocal row
            ctk.CTkLabel(scroll, text=label, font=("Segoe UI", 10), anchor="w").grid(
                row=row, column=0, sticky="w", pady=3)
            if var_type == "entry":
                show = kw.get("show", "")
                w = kw.get("width", None)
                e = ctk.CTkEntry(scroll, font=("Segoe UI", 10), show=show, width=w)
                e.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
                self._entries.append(e)
                scroll.columnconfigure(1, weight=1)
                row += 1
                return e
            if var_type == "check":
                v = ctk.BooleanVar(value=kw.get("value", False))
                ctk.CTkCheckBox(scroll, variable=v, text="", font=("Segoe UI", 10)).grid(
                    row=row, column=1, sticky="w", padx=(8, 0), pady=3)
                row += 1
                return v
            if var_type == "combo":
                cb = ctk.CTkComboBox(scroll, values=list(VALID_SEARCH_PROVIDERS),
                                     state="readonly", font=("Segoe UI", 10))
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

        pcfg = self.owui.web_search_providers.get(self.owui.web_search_provider)
        if pcfg and pcfg.api_key:
            self.e_api_key.insert(0, pcfg.api_key)
        self.e_count.insert(0, str(self.owui.search_result_count))
        self.e_concurrent.insert(0, str(max(1, self.owui.concurrent_requests)))

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Test Search", command=self._test,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self._close,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

        self.dialog.protocol("WM_DELETE_WINDOW", self._close)
        self.dialog.bind("<Escape>", lambda e: self._close())

    def _close(self) -> None:
        for w in self._entries:
            try:
                w.destroy()
            except RuntimeError:
                pass
        self.dialog.destroy()

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
            self._close()
            messagebox.showinfo("Saved", "Web search configuration saved.")
            self.on_save()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save: {exc}")


class LogsDialog:
    def __init__(self, parent: ctk.CTk) -> None:
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Logs")
        self.dialog.geometry("740x540")
        self.dialog.minsize(500, 340)

        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)

        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        tabview = ctk.CTkTabview(main)
        tabview.pack(fill="both", expand=True)

        tab_owui = tabview.add(" Open WebUI Logs ")
        tab_bridge = tabview.add(" Bridge Logs ")

        self._auto_refresh = True
        self._poll_id = None
        self._last_sizes: dict[str, int] = {"owui": 0, "bridge": 0}

        self._build_log_tab(tab_owui, OPENWEBUI_LOG_PATH, "owui")
        self._build_log_tab(tab_bridge, LLAMA_LOG_PATH, "bridge")

        btn_row = ctk.CTkFrame(self.dialog, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        self._auto_btn = ctk.CTkButton(btn_row, text="Auto: ON",
                                       command=self._toggle_auto,
                                       font=("Segoe UI", 10, "bold"))
        self._auto_btn.pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Clear", command=self._clear_all,
                      font=("Segoe UI", 10),
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Close", command=self._close,
                      font=("Segoe UI", 10),
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right", padx=4)

        self.dialog.bind("<Escape>", lambda e: self._close())
        self.dialog.protocol("WM_DELETE_WINDOW", self._close)
        self._start_polling()

    def _build_log_tab(self, parent: ctk.CTkFrame, log_path: Path, key: str) -> None:
        text_w = ctk.CTkTextbox(parent, font=("Consolas", 10), wrap="none",
                                activate_scrollbars=True)
        text_w.pack(fill="both", expand=True)
        text_w.bind("<MouseWheel>", lambda e: text_w.yview_scroll(-1 * (e.delta // 120), "units"))
        setattr(self, f"_log_text_{key}", text_w)
        self._load_log(text_w, log_path)
        try:
            self._last_sizes[key] = log_path.stat().st_size if log_path.exists() else 0
        except OSError:
            self._last_sizes[key] = 0

    def _load_log(self, text_w: ctk.CTkTextbox, log_path: Path) -> None:
        try:
            lines = follow_log(log_path, 200)
            text_w.delete("0.0", "end")
            for line in lines:
                text_w.insert("end", line + "\n")
            text_w.see("end")
        except Exception:
            text_w.delete("0.0", "end")
            text_w.insert("0.0", "(log unavailable)")

    def _start_polling(self) -> None:
        self._poll()

    def _poll(self) -> None:
        for key, path in (("bridge", LLAMA_LOG_PATH), ("owui", OPENWEBUI_LOG_PATH)):
            tw = getattr(self, f"_log_text_{key}", None)
            if tw:
                self._append_new_lines(tw, path, key)
        if self._auto_refresh:
            self._poll_id = self.dialog.after(1500, self._poll)

    def _append_new_lines(self, text_w: ctk.CTkTextbox, log_path: Path, key: str) -> None:
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
                text_w.insert("end", line + "\n")
            text_w.see("end")
        except Exception:
            pass

    def _toggle_auto(self) -> None:
        self._auto_refresh = not self._auto_refresh
        self._auto_btn.configure(text=f"Auto: {'ON' if self._auto_refresh else 'OFF'}")
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


class CmdPreviewDialog:
    def __init__(self, parent: ctk.CTk, owui: OpenWebUIConfig | None,
                 config: BridgeConfig | None) -> None:
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Environment Preview")
        self.dialog.geometry("520x480")
        self.dialog.minsize(420, 320)
        self.dialog.resizable(True, True)

        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=16)

        ctk.CTkLabel(main, text="\U0001f4bb Environment Variables",
                     font=("Segoe UI", 14, "bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(main, text="These variables will be set when the server starts.",
                     font=("Segoe UI", 10), text_color=("#888888", "#888888"),
                     anchor="w").pack(anchor="w", pady=(2, 12))

        scroll = ctk.CTkScrollableFrame(main, corner_radius=6)
        scroll.pack(fill="both", expand=True)
        scroll._scrollbar.grid_remove()

        if owui and config:
            from .openwebui_config import generate_openwebui_env
            env = generate_openwebui_env(owui, config)
            text = ctk.CTkTextbox(scroll, font=("Consolas", 10), fg_color="transparent", wrap="none")
            text.pack(fill="both", expand=True, padx=6, pady=6)
            lines = []
            for k, v in sorted(env.items()):
                if any(secret in k.lower() for secret in ("key", "token", "secret", "auth")):
                    v = "****" if v else ""
                lines.append(f"{k}={v}")
            text.insert("0.0", "\n".join(lines) if lines else "(no env vars)")
            text.configure(state="disabled")
        else:
            ctk.CTkLabel(scroll, text="No configuration loaded.",
                         font=("Segoe UI", 10), anchor="w").pack(pady=20)

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Close", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")


class SetupGuideDialog:
    """Setup guide with requirements checklist and install commands."""

    REQS = [
        ("python", "Python 3.12+", "\U0001f40d"),
        ("conda", "Miniconda", "\U0001f300"),
        ("ffmpeg", "FFmpeg", "\U0001f3ac"),
        ("hf_token", "HF Token", "\U0001f510"),
    ]
    COMMANDS = [
        ("winget install -e --id Python.Python.3.12", "Python 3.12"),
        ("winget install -e --id Anaconda.Miniconda3", "Miniconda"),
        ("winget install -e --id Gyan.FFmpeg", "FFmpeg"),
    ]
    ENV_COMMANDS = [
        ('conda create -n omx-open-webui python=3.12 -y', "Create env"),
        ('conda activate omx-open-webui', "Activate env"),
        ('pip install open-webui', "Install Open WebUI"),
    ]

    def __init__(self, parent: ctk.CTk, owui: OpenWebUIConfig | None) -> None:
        self.owui = owui
        self.result: dict[str, bool] = {}

        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Setup Guide")
        self.dialog.geometry("560x580")
        self.dialog.minsize(480, 480)
        self.dialog.resizable(True, True)

        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=16)

        ctk.CTkLabel(main, text="\U0001f9f0 Setup Guide", font=("Segoe UI", 16, "bold"),
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(main, text="Complete all 4 requirements before starting the server.",
                     font=("Segoe UI", 10), text_color=("#888888", "#888888"),
                     anchor="w").pack(anchor="w", pady=(2, 12))

        # ── Requirements cards ──
        self.cards_frame = ctk.CTkFrame(main, fg_color="transparent")
        self.cards_frame.pack(fill="x")
        self.cards_frame.columnconfigure(0, weight=1)
        self.cards_frame.columnconfigure(1, weight=1)

        self.card_labels: dict[str, tuple[ctk.CTkLabel, ctk.CTkLabel, ctk.CTkLabel]] = {}
        for idx, (key, title, emoji) in enumerate(self.REQS):
            r, c = divmod(idx, 2)
            card = ctk.CTkFrame(self.cards_frame, fg_color=("#1a1a1a", "#1a1a1a"),
                                corner_radius=10, height=72)
            card.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            card.grid_propagate(False)

            card.columnconfigure(0, weight=0)
            card.columnconfigure(1, weight=1)
            card.columnconfigure(2, weight=0)

            icon_lbl = ctk.CTkLabel(card, text=emoji, font=("Segoe UI", 18), anchor="w")
            icon_lbl.grid(row=0, column=0, rowspan=2, padx=(10, 8), pady=6, sticky="ns")

            title_lbl = ctk.CTkLabel(card, text=title, font=("Segoe UI", 11, "bold"), anchor="w")
            title_lbl.grid(row=0, column=1, sticky="w", pady=(8, 0))

            status_lbl = ctk.CTkLabel(card, text="\u23f3 Checking\u2026",
                                      font=("Segoe UI", 10),
                                      text_color=("#888888", "#888888"), anchor="w")
            status_lbl.grid(row=1, column=1, sticky="w", pady=(0, 8))

            self.card_labels[key] = (icon_lbl, title_lbl, status_lbl)

        # ── Commands section ──
        ctk.CTkLabel(main, text="\U0001f4dd Commands", font=("Segoe UI", 12, "bold"),
                     anchor="w").pack(anchor="w", pady=(14, 2))
        self.cmd_label = ctk.CTkLabel(main, text="",
                                      font=("Segoe UI", 9),
                                      text_color=("#888888", "#888888"), anchor="w")
        self.cmd_label.pack(anchor="w")

        scroll = ctk.CTkScrollableFrame(main, fg_color=("#0d0d0d", "#0d0d0d"),
                                        corner_radius=8, height=160)
        scroll.pack(fill="x", pady=(4, 0))
        scroll._scrollbar.grid_remove()

        self.cmd_text = ctk.CTkTextbox(scroll, font=("Consolas", 11),
                                       fg_color="transparent", wrap="none")
        self.cmd_text.pack(fill="both", expand=True, padx=6, pady=6)

        # ── Footer buttons ──
        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(12, 0))

        ctk.CTkButton(btn_row, text="\U0001f4cb Copy", command=self._copy_commands,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_row, text="\u21bb Refresh", command=self._check_all,
                      font=("Segoe UI", 10), width=90).pack(side="left")
        ctk.CTkButton(btn_row, text="Close", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

        self.dialog.after(100, self._check_all)

    def _check_all(self) -> None:
        self.result = {}
        import shutil

        # 1. Python 3.12+
        py_ok = sys.version_info >= (3, 11)
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.result["python"] = py_ok
        self._update_card("python", py_ok, py_ver)

        # 2. Miniconda
        conda_on_path = shutil.which("conda") is not None
        conda_env = get_conda_python_path() is not None
        conda_active = os.environ.get("CONDA_PREFIX") or ""
        conda_ok = conda_on_path or conda_env
        if conda_on_path:
            try:
                r = subprocess.run(["conda", "--version"], capture_output=True, text=True, timeout=5)
                cv = r.stdout.strip()
            except Exception:
                cv = ""
            sub = f"{cv} (active)" if (cv and conda_active) else (cv or "on PATH")
        elif conda_env:
            sub = "env found, needs conda on PATH"
        else:
            sub = "not found"
        self.result["conda"] = conda_ok
        self._update_card("conda", conda_ok, sub)

        # 3. FFmpeg
        ffmpeg_ok = shutil.which("ffmpeg") is not None
        self.result["ffmpeg"] = ffmpeg_ok
        self._update_card("ffmpeg", ffmpeg_ok, "found" if ffmpeg_ok else "not found")

        # 4. HF Token
        hf = (self.owui and self.owui.hf_token) or os.environ.get("HF_TOKEN") or ""
        hf_ok = bool(hf.strip())
        self.result["hf_token"] = hf_ok
        self._update_card("hf_token", hf_ok, f"{hf[:20]}..." if hf_ok and len(hf) > 20 else ("set" if hf_ok else "missing \u2192 Config dialog"))

        self._update_commands()

    def _update_card(self, key: str, ok: bool, subtitle: str) -> None:
        icon_lbl, title_lbl, status_lbl = self.card_labels[key]
        icon = "\u2705" if ok else "\u274c"
        color = GREEN if ok else RED
        icon_lbl.configure(text=icon, text_color=color)
        status_lbl.configure(text=subtitle, text_color=color)

    def _update_commands(self) -> None:
        lines: list[str] = []
        missing = [k for k, v in self.result.items() if not v]

        if missing:
            self.cmd_label.configure(text="Run these commands in PowerShell, then click Refresh:")

            if "python" in missing:
                lines.append(f":: 1. {self.REQS[0][1]}")
                lines.append(self.COMMANDS[0][0])
                lines.append("")
            if "conda" in missing:
                lines.append(f":: 2. {self.REQS[1][1]}")
                lines.append(self.COMMANDS[1][0])
                lines.append("")
            if "ffmpeg" in missing:
                lines.append(f":: 3. {self.REQS[2][1]}")
                lines.append(self.COMMANDS[2][0])
                lines.append("")
            if "hf_token" in missing:
                lines.append(f":: 4. {self.REQS[3][1]}")
                lines.append("Open the Config dialog and set HF_TOKEN there.")
                lines.append("")
        else:
            self.cmd_label.configure(text="All requirements met! Next, set up the environment:")
            for i, (cmd, desc) in enumerate(self.ENV_COMMANDS, 1):
                lines.append(f":: {i}. {desc}")
                lines.append(cmd)
                lines.append("")

        self.cmd_text.delete("0.0", "end")
        self.cmd_text.insert("0.0", "\n".join(lines).strip())

    def _copy_commands(self) -> None:
        text = self.cmd_text.get("0.0", "end-1c").strip()
        if not text:
            return
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(text)
        self.dialog.after(100, lambda: None)


def launch_gui(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    if not HAS_TK:
        print("Tkinter is not available. Use `llama openwebui configure` for CLI setup.")
        sys.exit(1)
    app = OpenWebUISetupCenter(config_path)
    app.run()
