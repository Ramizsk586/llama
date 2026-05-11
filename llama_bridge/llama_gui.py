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
    import customtkinter as ctk
    HAS_TK = True
except ImportError:
    HAS_TK = False
    class _FakeCTk:
        class CTk:
            def __init__(self):
                raise RuntimeError("customtkinter not available")
    ctk = _FakeCTk()

GREEN = "#4FD1A1"
RED = "#FF6B6B"
YELLOW = "#F2C66D"

LAYOUT = {
    "PAD": 20,
    "HEADER_H": 80,
    "CARD_H": 68,
    "CARD_GAP": 8,
    "BTN_H": 36,
    "WIN_W": 640,
    "WIN_H": 420,
}
PAD = LAYOUT["PAD"]


class AppStatus(Enum):
    SETUP = auto()
    READY = auto()
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
        lbl = ctk.CTkLabel(self.tip, text=self.text,
                           font=("Segoe UI", 9), padx=8, pady=4)
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
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.config_path = config_path
        self._stopped = threading.Event()

        W, H = LAYOUT["WIN_W"], LAYOUT["WIN_H"]
        self.root = ctk.CTk()
        self.root.title("Llama Bridge - Control Center")
        _center_window(self.root, W, H)
        self.root.minsize(520, 380)
        self.root.resizable(True, True)

        self.status = AppStatus.READY
        self._config: BridgeConfig | None = None
        self._log_lines: list[str] = []
        self._server_pid: int | None = None

        self._card_data = [
            {"key": "server", "title": "Server", "ok": False, "subtitle": "Stopped", "status": "Stopped"},
            {"key": "providers", "title": "Providers", "ok": False, "subtitle": "None configured", "status": "None"},
            {"key": "cli_tools", "title": "CLI Tools", "ok": False, "subtitle": "0 configured", "status": "None"},
            {"key": "models", "title": "Anthropic Models", "ok": False, "subtitle": "0 aliases", "status": "None"},
        ]
        self._card_widgets: list[ctk.CTkFrame] = []

        self._build_ui()
        self._load_config()
        self._refresh_all()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True)

        self._build_header(main)

        sep = ctk.CTkFrame(main, fg_color=GREEN, height=2, corner_radius=0)
        sep.pack(fill="x", padx=PAD, pady=(4, 0))

        self._build_footer(main)

        self.cards_area = ctk.CTkFrame(main, fg_color="transparent")
        self.cards_area.pack(fill="x", expand=False, padx=PAD, pady=(PAD // 2, 0))

    def _build_header(self, parent: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=PAD, pady=(PAD, 0))

        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(title_frame, text="Llama Bridge Control Center",
                     font=("Segoe UI", 18, "bold"), anchor="w").pack(anchor="w")
        self.subtitle_label = ctk.CTkLabel(title_frame, text="",
                                           font=("Segoe UI", 10),
                                           text_color=("#888888", "#888888"), anchor="w")
        self.subtitle_label.pack(anchor="w", pady=(2, 0))

        self.badge = ctk.CTkLabel(header, text="  READY  ",
                                  font=("Segoe UI", 9, "bold"), corner_radius=4)
        self.badge.grid(row=0, column=1, sticky="ne", pady=(4, 0))

        self._update_header()

    def _update_header(self) -> None:
        self.subtitle_label.configure(text=self._subtitle_text())
        badge_text, badge_color = self._get_badge_info()
        badge_bg = {
            AppStatus.RUNNING: "#0a2a1a",
            AppStatus.READY: "#0a2a1a",
            AppStatus.ERROR: "#2a0a0a",
        }.get(self.status, "#0a2a1a")
        self.badge.configure(text=f"  {badge_text}  ",
                             fg_color=badge_bg,
                             text_color=badge_color)

    def _build_footer(self, parent: ctk.CTkFrame) -> None:
        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.pack(fill="x", side="bottom", pady=(6, 12), padx=PAD)

        for col in range(3):
            footer.columnconfigure(col, weight=1, uniform="ftr_col")

        sub_panels = [
            ("Config", [("\u2699 Server", self._open_server_config),
                        ("\U0001f310 Providers", self._open_providers)]),
            ("Tools", [("\U0001f528 CLI Tools", self._open_cli_tools),
                       ("\U0001f916 Models", self._open_models)]),
            ("Info", [("\U0001f4cb Logs", self._open_logs),
                      ("\u2139 Details", self._open_details)]),
        ]
        self.util_btns: dict[str, ctk.CTkButton] = {}
        for col, (panel_title, buttons) in enumerate(sub_panels):
            panel = ctk.CTkFrame(footer, fg_color=("#e8e8e8", "#1e1e1e"), corner_radius=6)
            panel.grid(row=0, column=col, sticky="nsew", padx=4, pady=2)
            panel.columnconfigure(0, weight=1)
            panel.columnconfigure(1, weight=1)

            ctk.CTkLabel(panel, text=panel_title,
                         font=("Segoe UI", 8, "bold"),
                         text_color=("#666666", "#888888")).grid(
                row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 0))

            tips = {
                "\u2699 Server": "Configure server host, port, auth",
                "\U0001f310 Providers": "View and manage API providers",
                "\U0001f528 CLI Tools": "Configure CLI tool settings",
                "\U0001f916 Models": "Manage anthropic model aliases",
                "\U0001f4cb Logs": "View server logs",
                "\u2139 Details": "Show full technical details",
            }
            for i, (text, cmd) in enumerate(buttons):
                btn = ctk.CTkButton(
                    panel, text=text, command=cmd,
                    font=("Segoe UI", 9), height=26,
                    fg_color=("#e0e0e0", "#2a2a2a"),
                    text_color=("#333333", "#cccccc"),
                    hover_color=("#d0d0d0", "#3a3a3a"),
                    corner_radius=4,
                )
                btn.grid(row=1, column=i, sticky="ew", padx=4, pady=(2, 4))
                self.util_btns[text] = btn
                ToolTip(btn, tips.get(text, ""))

    def _create_card(self, card_data: dict, parent: ctk.CTkFrame | None = None) -> ctk.CTkFrame:
        if parent is None:
            parent = self.cards_area
        ok = card_data["ok"]
        card_bg = "#0f2820" if ok else "#281414"
        accent_color = GREEN if ok else RED
        icon_text = "\u2713" if ok else "\u2717"

        card = ctk.CTkFrame(parent, fg_color=card_bg, corner_radius=8, height=LAYOUT["CARD_H"])
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
                     font=("Segoe UI", 10, "bold"), anchor="w").grid(
            row=0, column=2, sticky="w", pady=(6, 0))

        ctk.CTkLabel(card, text=card_data.get("subtitle", ""),
                     font=("Segoe UI", 9),
                     text_color=("#888888", "#888888"), anchor="w").grid(
            row=1, column=2, sticky="w", pady=(0, 6))

        ctk.CTkLabel(card, text=card_data.get("status", ""),
                     font=("Segoe UI", 9, "bold"),
                     text_color=accent_color).grid(
            row=0, column=3, rowspan=2, sticky="e", padx=(0, 16))

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

    def _subtitle_text(self) -> str:
        if self.status == AppStatus.READY:
            return "Server is ready."
        if self.status == AppStatus.RUNNING:
            return "Server is running."
        if self.status == AppStatus.ERROR:
            return "Error \u2014 check Logs"
        return ""

    def _get_badge_info(self) -> tuple[str, str]:
        s = self.status
        if s == AppStatus.ERROR:
            return "ERROR", RED
        if s in (AppStatus.READY, AppStatus.RUNNING):
            return "RUNNING" if s == AppStatus.RUNNING else "READY", GREEN
        return "READY", GREEN

    def _update_card_data(self) -> None:
        cfg = self._config
        if not cfg:
            return

        server = cfg.server
        running = self._server_pid is not None and _pid_alive(self._server_pid)
        self._card_data[0] = {
            "key": "server", "title": "Server",
            "ok": running,
            "subtitle": f"{server.host}:{server.port}" if not running else f"{server.host}:{server.port} (pid={self._server_pid})",
            "status": "Running" if running else "Stopped",
        }

        prov_count = len(cfg.providers)
        prov_configured = sum(1 for p in cfg.providers.values() if p.api_key and not p.api_key.startswith("${"))
        self._card_data[1] = {
            "key": "providers", "title": "Providers",
            "ok": prov_configured > 0,
            "subtitle": f"{prov_configured}/{prov_count} with keys" if prov_count > 0 else "None configured",
            "status": f"{prov_configured}/{prov_count}" if prov_count > 0 else "None",
        }

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
        self._update_header()
        self._rebuild_cards()

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


class ServerConfigDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Server Config")
        self.dialog.geometry("420x300")
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
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        fields: list[tuple[str, str, type]] = [
            ("host", "Host", str),
            ("port", "Port", int),
            ("auth_token", "Auth Token", str),
            ("idle_timeout_seconds", "Idle Timeout (s)", int),
            ("openwebui_port", "Open WebUI Port", str),
        ]
        self._vars: dict[str, ctk.StringVar] = {}
        row = 0
        for key, label, _ in fields:
            ctk.CTkLabel(main, text=label, font=("Segoe UI", 10), anchor="w").grid(
                row=row, column=0, sticky="w", pady=(0, 10), padx=(0, 8))

            if key == "idle_timeout_seconds":
                var = ctk.StringVar(value=str(self.server_raw.get(key, 180)))
                entry = ctk.CTkEntry(main, textvariable=var, width=100,
                                     font=("Segoe UI", 10))
                entry.grid(row=row, column=1, sticky="w", pady=(0, 10))
            elif key == "port":
                var = ctk.StringVar(value=str(self.server_raw.get(key, 8089)))
                entry = ctk.CTkEntry(main, textvariable=var, width=100,
                                     font=("Segoe UI", 10))
                entry.grid(row=row, column=1, sticky="w", pady=(0, 10))
            elif key == "auth_token":
                var = ctk.StringVar(value=str(self.server_raw.get(key, "")))
                auth_frame = ctk.CTkFrame(main, fg_color="transparent")
                auth_frame.grid(row=row, column=1, sticky="ew", pady=(0, 10))
                entry = ctk.CTkEntry(auth_frame, textvariable=var,
                                     font=("Segoe UI", 10), show="*")
                entry.pack(side="left", fill="x", expand=True)
                self.show_auth_var = ctk.BooleanVar(value=False)
                def _toggle_auth():
                    entry.configure(show="" if self.show_auth_var.get() else "*")
                ctk.CTkCheckBox(auth_frame, text="Show", variable=self.show_auth_var,
                                command=_toggle_auth, font=("Segoe UI", 10)).pack(side="left", padx=(6, 0))
            elif key == "openwebui_port":
                ow_port = self.server_raw.get(key)
                var = ctk.StringVar(value=str(ow_port) if ow_port else "")
                entry = ctk.CTkEntry(main, textvariable=var, width=120,
                                     font=("Segoe UI", 10))
                entry.grid(row=row, column=1, sticky="w", pady=(0, 10))
            else:
                var = ctk.StringVar(value=str(self.server_raw.get(key, "127.0.0.1")))
                entry = ctk.CTkEntry(main, textvariable=var, width=240,
                                     font=("Segoe UI", 10))
                entry.grid(row=row, column=1, sticky="ew", pady=(0, 10))

            self._vars[key] = var
            row += 1

        main.grid_columnconfigure(1, weight=1)

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.grid(row=row, column=0, columnspan=2, pady=(12, 0), sticky="e")
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self.dialog.destroy,
                      font=("Segoe UI", 10), fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555"), width=90).pack(side="right")

    def _save(self) -> None:
        import yaml
        self.server_raw["host"] = self._vars["host"].get().strip()
        self.server_raw["port"] = int(self._vars["port"].get())
        token = self._vars["auth_token"].get().strip()
        if token:
            self.server_raw["auth_token"] = token
        self.server_raw["idle_timeout_seconds"] = int(self._vars["idle_timeout_seconds"].get())
        ow = self._vars["openwebui_port"].get().strip()
        if ow:
            self.server_raw["openwebui_port"] = int(ow)
        else:
            self.server_raw.pop("openwebui_port", None)
        self.config_path.write_text(
            yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class ProvidersDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Providers")
        self.dialog.geometry("820x580")
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
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll._scrollbar.grid_remove()

        TYPE_OPTIONS = [
            "openai", "ollama_cloud", "ollama_local", "lm_studio",
            "groq", "gemini", "cohere", "mistral", "deepseek",
            "openrouter", "openai_compatible", "nvidia_nim",
        ]

        self._entries: dict[str, dict[str, Any]] = {}
        items = sorted(self.providers_raw.items())
        num_cols = 2

        for idx, (name, prov) in enumerate(items):
            card = ctk.CTkFrame(scroll, fg_color=("#1a1a1a", "#1a1a1a"), corner_radius=8)
            card.grid(row=idx // num_cols, column=idx % num_cols,
                      sticky="nsew", padx=4, pady=5)

            ctk.CTkLabel(card, text=name, font=("Segoe UI", 12, "bold"),
                         anchor="w").pack(anchor="w", padx=12, pady=(8, 4))
            ctk.CTkFrame(card, fg_color=("#333333", "#333333"), height=1,
                         corner_radius=0).pack(fill="x", padx=12, pady=(0, 6))

            entries: dict[str, Any] = {}
            fields: list[tuple[str, Any, str]] = [
                ("type", TYPE_OPTIONS, "combobox"),
                ("base_url", None, "entry"),
                ("api_key", None, "key"),
                ("default_model", None, "entry"),
                ("supports_tools", None, "check"),
            ]

            for label, values, kind in fields:
                row = ctk.CTkFrame(card, fg_color="transparent")
                row.pack(fill="x", padx=12, pady=2)
                ctk.CTkLabel(row, text=label.replace("_", " ").title(),
                             font=("Segoe UI", 9), anchor="w", width=100).pack(side="left")
                if kind == "combobox":
                    var = ctk.StringVar(value=str(prov.get(label, values[0])))
                    ctk.CTkComboBox(row, variable=var, values=values,
                                    state="readonly", font=("Segoe UI", 9)).pack(
                        side="left", fill="x", expand=True)
                elif kind == "key":
                    var = ctk.StringVar(value=str(prov.get(label, "")))
                    entry = ctk.CTkEntry(row, textvariable=var,
                                         font=("Segoe UI", 9), show="*")
                    entry.pack(side="left", fill="x", expand=True)
                    sv = ctk.BooleanVar(value=False)
                    def _toggle_key(e=entry, sv=sv):
                        e.configure(show="" if sv.get() else "*")
                    ctk.CTkCheckBox(row, text="Show", variable=sv,
                                    command=_toggle_key, width=52).pack(side="left", padx=(4, 0))
                elif kind == "check":
                    var = ctk.BooleanVar(value=bool(prov.get(label, True)))
                    ctk.CTkCheckBox(row, variable=var, text="", width=20).pack(side="left")
                else:
                    var = ctk.StringVar(value=str(prov.get(label, "")))
                    ctk.CTkEntry(row, textvariable=var,
                                 font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True)
                entries[label] = var
            self._entries[name] = entries

        for c in range(num_cols):
            scroll.grid_columnconfigure(c, weight=1, uniform="col")
        total_rows = (len(items) + num_cols - 1) // num_cols
        if total_rows > 0:
            scroll.grid_rowconfigure(total_rows, weight=1)

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

    def _save(self) -> None:
        import yaml
        for name, entries in self._entries.items():
            prov = self.providers_raw.setdefault(name, {})
            for label, var in entries.items():
                if isinstance(var, ctk.BooleanVar):
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
    def __init__(self, parent: ctk.CTk, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("CLI Tools")
        self.dialog.geometry("820x580")
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
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll._scrollbar.grid_remove()

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
            section = self.raw.get(section_key, {}) or {}

            card = ctk.CTkFrame(scroll, fg_color=("#1a1a1a", "#1a1a1a"), corner_radius=8)
            card.grid(row=idx // num_cols, column=idx % num_cols, sticky="nsew", padx=4, pady=5)

            ctk.CTkLabel(card, text=tool_name, font=("Segoe UI", 12, "bold"),
                         anchor="w").pack(anchor="w", padx=12, pady=(8, 4))
            ctk.CTkFrame(card, fg_color=("#333333", "#333333"), height=1,
                         corner_radius=0).pack(fill="x", padx=12, pady=(0, 6))

            entries = {}
            for field in fields:
                row = ctk.CTkFrame(card, fg_color="transparent")
                row.pack(fill="x", padx=12, pady=1)
                ctk.CTkLabel(row, text=field.replace("_", " ").title(),
                             font=("Segoe UI", 9), anchor="w", width=120).pack(side="left")

                if field == "provider":
                    var = ctk.StringVar(value=str(section.get(field, self.provider_names[0])))
                    ctk.CTkComboBox(row, variable=var, values=self.provider_names,
                                    state="readonly", font=("Segoe UI", 9)).pack(
                        side="left", fill="x", expand=True)
                elif field in ("max_prompt_tokens", "max_output_tokens", "context_size", "output_tokens"):
                    var = ctk.StringVar(value=str(section.get(field, "")))
                    ctk.CTkEntry(row, textvariable=var, width=100,
                                 font=("Segoe UI", 9)).pack(side="left")
                else:
                    var = ctk.StringVar(value=str(section.get(field, "")))
                    ctk.CTkEntry(row, textvariable=var, font=("Segoe UI", 9)).pack(
                        side="left", fill="x", expand=True)
                entries[field] = var
            self._entries[section_key] = entries

        for c in range(num_cols):
            scroll.grid_columnconfigure(c, weight=1, uniform="col")
        total_rows = (len(tool_keys) + num_cols - 1) // num_cols
        if total_rows > 0:
            scroll.grid_rowconfigure(total_rows, weight=1)

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

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
        self.config_path.write_text(
            yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class ModelsDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Anthropic Models")
        self.dialog.geometry("600x440")
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
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll._scrollbar.grid_remove()

        aliases_raw = self.raw.get("anthropic_models", {}) or {}
        self._entries: dict[str, tuple[ctk.StringVar, ctk.StringVar]] = {}
        for alias, value in sorted(aliases_raw.items()):
            if not isinstance(value, dict):
                value = {}
            card = ctk.CTkFrame(scroll, fg_color=("#1a1a1a", "#1a1a1a"), corner_radius=8)
            card.pack(fill="x", pady=4, padx=4)

            ctk.CTkLabel(card, text=alias, font=("Segoe UI", 10, "bold"),
                         anchor="w", width=90).pack(side="left", padx=(10, 4), pady=8)

            prov_var = ctk.StringVar(value=str(value.get("provider", self.provider_names[0])))
            ctk.CTkComboBox(card, variable=prov_var, values=self.provider_names,
                            state="readonly", font=("Segoe UI", 9), width=180).pack(
                side="left", padx=(4, 6), pady=8)

            model_var = ctk.StringVar(value=str(value.get("model", "")))
            ctk.CTkEntry(card, textvariable=model_var,
                         font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=4, pady=8)

            self._entries[alias] = (prov_var, model_var)

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

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
        self.config_path.write_text(
            yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class LogsDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path) -> None:
        self.config_path = config_path
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Server Logs")
        self.dialog.geometry("740x540")
        self.dialog.minsize(580, 380)
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self._poll_id = None
        self._build()

    def _build(self) -> None:
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        self.text = ctk.CTkTextbox(main, font=("Consolas", 10), wrap="none",
                                   activate_scrollbars=True)
        self.text.pack(fill="both", expand=True)
        self.text.bind("<MouseWheel>", lambda e: self.text.yview_scroll(-1 * (e.delta // 120), "units"))

        btn_row = ctk.CTkFrame(self.dialog, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(btn_row, text="Refresh", command=self._refresh,
                      font=("Segoe UI", 10), width=80).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Clear", command=self._clear,
                      font=("Segoe UI", 10), width=80,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Close", command=self._close,
                      font=("Segoe UI", 10), width=80,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right", padx=4)

        self.dialog.bind("<Escape>", lambda e: self._close())
        self.dialog.protocol("WM_DELETE_WINDOW", self._close)
        self._load_log()
        self._start_polling()

    def _load_log(self) -> None:
        log_path = self.config_path.parent / "llama.log"
        self.text.delete("0.0", "end")
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()[-200:]
                for line in lines:
                    self.text.insert("end", line + "\n")
                self.text.see("end")
            except OSError:
                self.text.insert("end", "[Could not read log]")
        else:
            self.text.insert("end", "[No log file found]")

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
            current = self.text.get("0.0", "end-1c").splitlines()
            raw = log_path.read_text(encoding="utf-8", errors="replace")
            new_lines = raw.splitlines()
            if new_lines == current:
                return
            self.text.delete("0.0", "end")
            for line in new_lines[-200:]:
                self.text.insert("end", line + "\n")
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


class DetailsDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path) -> None:
        self.config_path = config_path
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Details")
        self.dialog.geometry("780x540")
        self.dialog.minsize(640, 440)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)

        import tkinter as tk
        from tkinter import ttk

        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        items = self._build_items()

        style = ttk.Style()
        style.theme_use("clam")
        dark_bg = "#1a1a1a"
        dark_fg = "#e0e0e0"
        sel_bg = "#2a2a2a"
        heading_bg = "#222222"
        heading_fg = "#fafafa"
        style.configure("CTk.Treeview",
                        background=dark_bg, foreground=dark_fg,
                        fieldbackground=dark_bg,
                        font=("Segoe UI", 9), rowheight=28, borderwidth=0)
        style.configure("CTk.Treeview.Heading",
                        background=heading_bg, foreground=heading_fg,
                        font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("CTk.Treeview",
                  background=[("selected", sel_bg)],
                  foreground=[("selected", dark_fg)])
        style.layout("CTk.Treeview", [("CTk.Treeview.treearea", {"sticky": "nswe"})])

        tree_frame = ctk.CTkFrame(main, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("field", "value"), show="headings",
                                 selectmode="browse", style="CTk.Treeview")
        self.tree.heading("field", text="Field", anchor="w")
        self.tree.heading("value", text="Value", anchor="w")
        self.tree.column("field", width=200, minwidth=140, stretch=False)
        self.tree.column("value", width=500, minwidth=300, stretch=True)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<MouseWheel>", lambda e: self.tree.yview_scroll(-1 * (e.delta // 120), "units"))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-3>", self._on_right_click)

        for idx, item in enumerate(items):
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

    class _DetailItem:
        def __init__(self, label: str, display_value: str, full_value: str) -> None:
            self.label = label
            self.display_value = display_value
            self.full_value = full_value

    def _build_items(self) -> list[_DetailItem]:
        import yaml
        items: list[DetailsDialog._DetailItem] = []
        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}
        server_raw = raw.get("server", {}) or {}
        providers_raw = raw.get("providers", {}) or {}

        items.append(DetailsDialog._DetailItem("Config path", _compact_path(str(self.config_path)), str(self.config_path)))
        items.append(DetailsDialog._DetailItem("Host", str(server_raw.get("host", "127.0.0.1")), str(server_raw.get("host", "127.0.0.1"))))
        items.append(DetailsDialog._DetailItem("Port", str(server_raw.get("port", 8089)), str(server_raw.get("port", 8089))))
        auth = server_raw.get("auth_token", "")
        items.append(DetailsDialog._DetailItem("Auth token", "set" if (auth and auth != "change-me") else "change-me",
                                               "set" if (auth and auth != "change-me") else "change-me"))
        items.append(DetailsDialog._DetailItem("Idle timeout", f"{server_raw.get('idle_timeout_seconds', 180)}s",
                                               str(server_raw.get('idle_timeout_seconds', 180))))
        items.append(DetailsDialog._DetailItem("OpenWebUI port", str(server_raw.get("openwebui_port", "none")),
                                               str(server_raw.get("openwebui_port", "none"))))
        items.append(DetailsDialog._DetailItem("Providers", str(len(providers_raw)),
                                               ", ".join(providers_raw.keys()) if providers_raw else "none"))
        for name, prov in sorted(providers_raw.items()):
            ptype = prov.get("type", "?")
            model = prov.get("default_model", "-") or "-"
            items.append(DetailsDialog._DetailItem(f"  {name}", f"{ptype} / {model}", f"{ptype} / {model}"))

        tools_raw = raw.get("tools", {}) or {}
        items.append(DetailsDialog._DetailItem("Tools enabled", str(tools_raw.get("enabled", True)), str(tools_raw.get("enabled", True))))
        items.append(DetailsDialog._DetailItem("Tools include", ", ".join(tools_raw.get("include", []) or []) or "none",
                                               ", ".join(tools_raw.get("include", []) or []) or "none"))
        items.append(DetailsDialog._DetailItem("Search", str(tools_raw.get("default_search_provider", "tavily")),
                                               str(tools_raw.get("default_search_provider", "tavily"))))
        return items

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
            item = self._build_items()[idx]
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(item.full_value)
        except (ValueError, IndexError):
            pass

    def _copy_all(self) -> None:
        items = self._build_items()
        lines = [f"{item.label}: {item.full_value}" for item in items]
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append("\n".join(lines))
