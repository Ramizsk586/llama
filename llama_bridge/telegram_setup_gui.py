from __future__ import annotations

import os
import sys
import threading
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml

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

# Allow running as standalone script or as part of a package
if __package__ is None or __package__ == "":
    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    _this_pkg = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    _mod = import_module(f"{_this_pkg}.config")
    DEFAULT_CONFIG_PATH = _mod.DEFAULT_CONFIG_PATH
    load_config = _mod.load_config
    ToolPolicy = _mod.ToolPolicy
    _tl = import_module(f"{_this_pkg}.telegram_launcher")
    telegram_bot_status = _tl.telegram_bot_status
    follow_telegram_log = _tl.follow_telegram_log
    test_telegram_token = _tl.test_telegram_token
    send_forced_message = _tl.send_forced_message
else:
    from .config import DEFAULT_CONFIG_PATH, load_config, ToolPolicy
    from .telegram_launcher import (
        telegram_bot_status,
        follow_telegram_log,
        test_telegram_token,
        send_forced_message,
    )

GREEN = "#34D399"
RED = "#FB7185"
YELLOW = "#FBBF24"
ACCENT = "#60A5FA"
DARK_BG = "#0A0F16"
SIDEBAR_BG = "#101720"
PANEL_BG = "#121B26"
PANEL_BG_SOFT = "#162231"
BORDER = "#263241"
MUTED = "#8B949E"
TEXT = "#E6EDF3"
CARD_OK_BG = "#0E2B22"
CARD_ERR_BG = "#2A1519"

MAX_LOG_LINES = 500

LAYOUT = {
    "PAD": 24,
    "CARD_H": 96,
    "CARD_GAP": 14,
    "WIN_W": 880,
    "WIN_H": 560,
    "SIDEBAR_W": 232,
}
PAD = LAYOUT["PAD"]


class BotStatus(Enum):
    SETUP = auto()
    READY = auto()
    STARTING = auto()
    RUNNING = auto()
    ERROR = auto()


class ToolTip:
    def __init__(self, widget: ctk.CTkBaseClass, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: ctk.CTkToplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._enter, add=True)
        widget.bind("<Leave>", self._leave, add=True)

    def _enter(self, _event: Any = None) -> None:
        if self._after_id:
            return
        self._after_id = self.widget.after(400, self._show)

    def _show(self) -> None:
        self._after_id = None
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = ctk.CTkToplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        ctk.CTkLabel(self.tip, text=self.text, font=("Segoe UI", 9), padx=8, pady=4).pack()

    def _leave(self, _event: Any = None) -> None:
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
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


def _compact_path(path_str: str, max_len: int = 50) -> str:
    if not path_str:
        return "-"
    path = str(Path(path_str).resolve())
    if len(path) <= max_len:
        return path
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 4:
        drive = parts[0]
        tail = parts[-2:]
        return f"{drive}/.../{'/'.join(tail)}"
    return f"...{path[-max_len + 3:]}"


class TelegramSetupCenter:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.config_path = config_path
        self._stopped = threading.Event()

        W, H = LAYOUT["WIN_W"], LAYOUT["WIN_H"]
        self.root = ctk.CTk()
        self.root.title("Llama Bridge - Telegram Bot Setup")
        _center_window(self.root, W, H)
        self.root.minsize(760, 500)
        self.root.resizable(True, True)

        self.status = BotStatus.SETUP
        self._config = None
        self._last_msg = ""
        self._log_lines: list[str] = []
        self._bot_username: str | None = None
        self._bot_id: str | None = None
        self._test_ok = False

        self._card_data = [
            {"key": "token", "title": "Bot Token", "ok": False, "subtitle": "Missing", "status": "Missing"},
            {"key": "api", "title": "Telegram API", "ok": False, "subtitle": "Not tested", "status": "Test"},
            {"key": "model", "title": "AI Model", "ok": False, "subtitle": "Missing", "status": "Missing"},
            {"key": "access", "title": "Access & Tools", "ok": False, "subtitle": "Not configured", "status": "Warn"},
        ]
        self._card_widgets: list[ctk.CTkFrame] = []

        self._build_ui()
        self._load_config()
        self._refresh_all(reload_config=False)
        self.root.after(100, self._auto_test_token)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(2000, self._poll_status)

    def _build_ui(self) -> None:
        self._nav_stack: list[str] = []
        self._current_view: str = "home"

        self.root.configure(fg_color=DARK_BG)

        sidebar = ctk.CTkFrame(self.root, width=LAYOUT["SIDEBAR_W"], fg_color=SIDEBAR_BG, corner_radius=0)
        sidebar.pack(side="left", fill="y", padx=0, pady=0)
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="Telegram Bot",
                     font=("Segoe UI", 18, "bold"), anchor="w",
                     fg_color=SIDEBAR_BG, text_color=TEXT).pack(
            fill="x", padx=20, pady=(26, 0))
        ctk.CTkLabel(sidebar, text="Setup Center",
                     font=("Segoe UI", 10), anchor="w",
                     fg_color=SIDEBAR_BG, text_color=MUTED).pack(
            fill="x", padx=20, pady=(2, 16))

        sep = ctk.CTkFrame(sidebar, fg_color=BORDER, height=1)
        sep.pack(fill="x", padx=16, pady=(0, 14))

        self._sidebar_buttons: dict[str, ctk.CTkButton] = {}
        nav_items = [
            ("dashboard", "Dashboard", self._show_dashboard),
            ("config", "Config", self._show_config),
            ("access", "Access", self._show_access),
            ("tools", "Tools", self._show_tools),
            ("force_msg", "Force Msg", self._show_force_message),
            ("logs", "Logs", self._show_logs),
            ("details", "Details", self._show_details),
        ]
        for key, label, cmd in nav_items:
            btn = ctk.CTkButton(
                sidebar, text=label, command=cmd,
                font=("Segoe UI", 11), height=38,
                fg_color="transparent", text_color="#CBD5E1",
                hover_color="#1C2734", corner_radius=8,
                anchor="w",
            )
            btn.pack(fill="x", padx=12, pady=2)
            self._sidebar_buttons[key] = btn

        self._active_nav = "dashboard"

        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(side="left", fill="both", expand=True)

        self._build_header(main)

        self.status_label = ctk.CTkLabel(main, text="", font=("Segoe UI", 9),
                                          text_color=MUTED, anchor="w")
        self.status_label.pack(fill="x", padx=PAD, pady=(8, 0))

        self.cards_area = ctk.CTkFrame(main, fg_color="transparent")
        self.cards_area.pack(fill="x", expand=False, padx=PAD, pady=(18, 0))

        self.dashboard_body = ctk.CTkFrame(main, fg_color="transparent")
        self.dashboard_body.pack(fill="both", expand=True, padx=PAD, pady=(10, PAD))
        self._build_dashboard_body()

        self._content_area = ctk.CTkScrollableFrame(main, fg_color="transparent")
        self._content_area.pack(fill="both", expand=True, padx=PAD, pady=(8, 0))
        self._content_area._scrollbar.grid_remove()
        self._content_area.pack_forget()

        self._footer_frame = ctk.CTkFrame(main, fg_color="transparent")
        self._footer_frame.pack_forget()

        self._cards_area_visible = True
        self._show_dashboard()

    def _set_active_nav(self, key: str) -> None:
        for k, btn in self._sidebar_buttons.items():
            if k == key:
                btn.configure(fg_color="#238636", text_color="#ffffff", hover_color="#2EA043")
            else:
                btn.configure(fg_color="transparent", text_color="#CBD5E1", hover_color="#1C2734")

    def _show_dashboard(self) -> None:
        self._current_view = "home"
        self._nav_stack.clear()
        self._cards_area_visible = True
        self._header_row.pack_configure(padx=PAD, pady=(PAD, 0))
        self.cards_area.pack(fill="x", expand=False, padx=PAD, pady=(18, 0))
        self.dashboard_body.pack(fill="both", expand=True, padx=PAD, pady=(10, PAD))
        self._content_area.pack_forget()
        self._footer_frame.pack_forget()
        self._set_active_nav("dashboard")
        self.subtitle_label.configure(text=self._subtitle_text())
        self._update_dashboard_summary()

    def _show_config(self) -> None:
        self._nav_stack.append("config")
        self._set_active_nav("config")
        self._show_panel("config")

    def _show_access(self) -> None:
        self._nav_stack.append("access")
        self._set_active_nav("access")
        self._show_panel("access")

    def _show_tools(self) -> None:
        self._nav_stack.append("tools")
        self._set_active_nav("tools")
        self._show_panel("tools")

    def _show_force_message(self) -> None:
        self._nav_stack.append("force_msg")
        self._set_active_nav("force_msg")
        self._show_panel("force_msg")

    def _show_logs(self) -> None:
        self._nav_stack.append("logs")
        self._set_active_nav("logs")
        self._show_panel("logs")

    def _show_details(self) -> None:
        self._nav_stack.append("details")
        self._set_active_nav("details")
        self._show_panel("details")

    def _build_dashboard_body(self) -> None:
        self.dashboard_body.columnconfigure(0, weight=3, uniform="dash")
        self.dashboard_body.columnconfigure(1, weight=2, uniform="dash")

        actions = ctk.CTkFrame(
            self.dashboard_body,
            fg_color=PANEL_BG,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
        )
        actions.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=0)
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            actions,
            text="Quick Actions",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT,
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(14, 8))

        action_buttons = [
            ("Config", self._show_config),
            ("Access", self._show_access),
            ("Tools", self._show_tools),
            ("Force Msg", self._show_force_message),
            ("Logs", self._show_logs),
            ("Details", self._show_details),
        ]
        for idx, (label, command) in enumerate(action_buttons):
            btn = ctk.CTkButton(
                actions,
                text=label,
                command=command,
                height=34,
                font=("Segoe UI", 10, "bold"),
                fg_color=PANEL_BG_SOFT,
                text_color=TEXT,
                hover_color="#203046",
                corner_radius=7,
            )
            btn.grid(
                row=1 + idx // 2,
                column=idx % 2,
                sticky="ew",
                padx=(16 if idx % 2 == 0 else 6, 16 if idx % 2 == 1 else 6),
                pady=5,
            )

        snapshot = ctk.CTkFrame(
            self.dashboard_body,
            fg_color=PANEL_BG,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
        )
        snapshot.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=0)
        snapshot.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            snapshot,
            text="Bot Snapshot",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT,
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(14, 8))

        self.dashboard_summary_labels: dict[str, ctk.CTkLabel] = {}
        for row, (key, label) in enumerate([
            ("token", "Token"),
            ("model", "Model"),
            ("access", "Access"),
            ("tools", "Tools"),
        ], start=1):
            ctk.CTkLabel(
                snapshot,
                text=label,
                font=("Segoe UI", 10),
                text_color=MUTED,
                anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(16, 10), pady=4)

            value = ctk.CTkLabel(
                snapshot,
                text="-",
                font=("Segoe UI", 10, "bold"),
                text_color=TEXT,
                anchor="e",
            )
            value.grid(row=row, column=1, sticky="ew", padx=(0, 16), pady=4)
            self.dashboard_summary_labels[key] = value

    def _update_dashboard_summary(self) -> None:
        labels = getattr(self, "dashboard_summary_labels", None)
        if not labels:
            return

        telegram = self._config.telegram if self._config else None
        if not telegram:
            values = {
                "token": "missing",
                "model": "-",
                "access": "-",
                "tools": "-",
            }
        else:
            token_set = bool(telegram.bot_token and not telegram.bot_token.startswith("${"))
            tool_policy = telegram.tool_policy
            tool_count = len(tool_policy.ai_auto_tools) + len(tool_policy.command_tools)
            values = {
                "token": "set" if token_set else "missing",
                "model": f"{telegram.provider} / {telegram.model}" if telegram.provider and telegram.model else "missing",
                "access": "all chats" if telegram.allow_all_chats else f"{len(telegram.allowed_chat_ids)} allowed",
                "tools": f"{tool_count} enabled",
            }

        for key, value in values.items():
            labels[key].configure(text=value)

    def _build_header(self, parent: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=PAD, pady=(PAD, 0))

        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)
        header.columnconfigure(2, weight=0)

        self._header_row = header

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(title_frame, text="Telegram Bot Setup Center",
                     font=("Segoe UI", 22, "bold"), text_color=TEXT,
                     anchor="w").pack(anchor="w")
        self.subtitle_label = ctk.CTkLabel(title_frame, text="",
                                            font=("Segoe UI", 10),
                                            text_color=MUTED, anchor="w")
        self.subtitle_label.pack(anchor="w", pady=(2, 0))

        self.badge = ctk.CTkLabel(header, text="  SETUP  ",
                                  font=("Segoe UI", 9, "bold"), corner_radius=7,
                                  width=86, height=32)
        self.badge.grid(row=0, column=1, sticky="ne", pady=(4, 0))

        self._update_header()

    def _update_header(self) -> None:
        self.subtitle_label.configure(text=self._subtitle_text())
        badge_text, badge_color = self._get_badge_info()
        badge_bg = {
            BotStatus.RUNNING: "#123B2F",
            BotStatus.READY: "#123B2F",
            BotStatus.SETUP: "#3B3218",
            BotStatus.STARTING: "#3B3218",
            BotStatus.ERROR: "#3B1C22",
        }.get(self.status, "#123B2F")
        self.badge.configure(text=f"  {badge_text}  ",
                             fg_color=badge_bg, text_color=badge_color)

    def _show_panel(self, panel: str) -> None:
        self._current_view = panel
        self._cards_area_visible = False
        self._header_row.pack_configure(padx=PAD, pady=(18, 0))
        self.cards_area.pack_forget()
        self.dashboard_body.pack_forget()
        self._content_area.pack(fill="both", expand=True, padx=PAD, pady=(8, PAD))
        for w in self._content_area.winfo_children():
            w.destroy()
        self._footer_frame.pack_forget()

        if panel == "config":
            self.subtitle_label.configure(text="Telegram Config")
            self._build_config_form()
        elif panel == "access":
            self.subtitle_label.configure(text="Access Control")
            self._build_access_form()
        elif panel == "tools":
            self.subtitle_label.configure(text="Command & Tool Policy")
            self._build_tools_form()
        elif panel == "force_msg":
            self.subtitle_label.configure(text="Force Message")
            self._build_force_message_form()
        elif panel == "logs":
            self.subtitle_label.configure(text="Telegram Bot Logs")
            log_lines = follow_telegram_log(self.config_path)
            text = ctk.CTkTextbox(self._content_area, font=("Consolas", 10), wrap="none")
            text.pack(fill="both", expand=True)
            for line in log_lines[-200:]:
                text.insert("end", line + "\n")
            text.see("end")
        elif panel == "details":
            self.subtitle_label.configure(text="Telegram Bot Details")
            import yaml
            text = ctk.CTkTextbox(self._content_area, font=("Segoe UI", 10), wrap="word")
            text.pack(fill="both", expand=True)
            try:
                raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                raw = {}
            telegram_raw = raw.get("telegram", {}) or {}
            lines = [
                f"Bot Token: {'set' if telegram_raw.get('bot_token') else 'missing'}",
                f"Provider: {telegram_raw.get('provider', '-')}",
                f"Model: {telegram_raw.get('model', '-')}",
                f"Allow all chats: {telegram_raw.get('allow_all_chats', False)}",
                f"Allowed IDs: {len(telegram_raw.get('allowed_chat_ids', []))}",
                f"Owner IDs: {len(telegram_raw.get('owner_chat_ids', []))}",
                f"Admin IDs: {len(telegram_raw.get('admin_chat_ids', []))}",
            ]
            for line in lines:
                text.insert("end", line + "\n")

    def _build_config_form(self) -> None:
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self._cfg_raw = raw
        telegram_raw = raw.setdefault("telegram", {})
        self._telegram_raw = telegram_raw

        form = ctk.CTkFrame(self._content_area, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=0, pady=(0, 16))
        form.columnconfigure(0, weight=3, uniform="cfg")
        form.columnconfigure(1, weight=2, uniform="cfg")

        left = ctk.CTkFrame(form, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = ctk.CTkFrame(form, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        def section(parent: ctk.CTkFrame, title: str) -> ctk.CTkFrame:
            card = ctk.CTkFrame(
                parent,
                fg_color=PANEL_BG,
                corner_radius=8,
                border_width=1,
                border_color=BORDER,
            )
            card.pack(fill="x", pady=(0, 12))
            ctk.CTkLabel(
                card,
                text=title,
                font=("Segoe UI", 12, "bold"),
                text_color=TEXT,
                anchor="w",
            ).pack(fill="x", padx=16, pady=(14, 8))
            body = ctk.CTkFrame(card, fg_color="transparent")
            body.pack(fill="x", padx=16, pady=(0, 16))
            return body

        switches = section(left, "Bot Behavior")
        credentials = section(left, "Credentials & Model")
        limits = section(right, "Runtime Limits")
        prompt_section = section(right, "System Prompt")

        self._cfg_enabled_var = ctk.BooleanVar(value=bool(telegram_raw.get("enabled", False)))
        ctk.CTkCheckBox(switches, text="Enable Telegram bot", variable=self._cfg_enabled_var,
                        font=("Segoe UI", 10), height=30).pack(anchor="w", pady=3)

        self._cfg_allow_all_var = ctk.BooleanVar(value=bool(telegram_raw.get("allow_all_chats", False)))
        self._cfg_allow_all_cb = ctk.CTkCheckBox(
            switches, text="Allow all chats (open to everyone)",
            variable=self._cfg_allow_all_var,
            font=("Segoe UI", 10),
            text_color=RED if self._cfg_allow_all_var.get() else None,
            height=30,
        )
        self._cfg_allow_all_cb.pack(anchor="w", pady=3)

        def _on_allow_toggle():
            self._cfg_allow_all_cb.configure(text_color=RED if self._cfg_allow_all_var.get() else None)
        self._cfg_allow_all_var.trace_add("write", lambda *_: _on_allow_toggle())

        self._cfg_auto_var = ctk.BooleanVar(value=bool(telegram_raw.get("autonomous_enabled", False)))
        ctk.CTkCheckBox(switches, text="Autonomous mode",
                       variable=self._cfg_auto_var, font=("Segoe UI", 10), height=30).pack(anchor="w", pady=3)
        ctk.CTkLabel(
            switches,
            text="Lets the bot initiate conversations when explicitly enabled.",
            font=("Segoe UI", 9),
            text_color=MUTED,
            anchor="w",
        ).pack(fill="x", pady=(0, 2))

        provider_names = list(raw.get("providers", {}).keys())
        self._cfg_provider_var = ctk.StringVar(
            value=str(telegram_raw.get("provider", provider_names[0] if provider_names else "")))

        self._cfg_vars: dict[str, Any] = {}

        def field(parent: ctk.CTkFrame, label: str) -> ctk.CTkFrame:
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=label, font=("Segoe UI", 10),
                         text_color="#CBD5E1", width=112, anchor="w").pack(side="left")
            return row

        token_row = field(credentials, "Bot Token")
        token_entry = ctk.CTkEntry(token_row, font=("Segoe UI", 10), show="*")
        token_entry.pack(side="left", fill="x", expand=True)
        token_entry.insert(0, str(telegram_raw.get("bot_token", "")))
        token_show_var = ctk.BooleanVar(value=False)
        def _toggle_token() -> None:
            token_entry.configure(show="" if token_show_var.get() else "*")
        ctk.CTkCheckBox(token_row, text="Show", variable=token_show_var,
                        command=_toggle_token, font=("Segoe UI", 9), width=70).pack(side="left", padx=(8, 0))
        self._cfg_vars["token"] = token_entry

        provider_row = field(credentials, "Provider")
        provider_values = provider_names if provider_names else [self._cfg_provider_var.get()]
        ctk.CTkComboBox(provider_row, variable=self._cfg_provider_var, values=provider_values,
                        state="readonly", font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True)
        self._cfg_vars["provider"] = self._cfg_provider_var

        model_row = field(credentials, "Model")
        model_entry = ctk.CTkEntry(model_row, font=("Segoe UI", 10))
        model_entry.pack(side="left", fill="x", expand=True)
        model_entry.insert(0, str(telegram_raw.get("model", "")))
        self._cfg_vars["model"] = model_entry

        prompt_box = ctk.CTkTextbox(prompt_section, font=("Segoe UI", 10), height=154)
        prompt_box.pack(fill="x")
        prompt_box.insert("0.0", str(telegram_raw.get("system_prompt", "")))
        self._cfg_vars["prompt"] = prompt_box

        num_fields = [
            ("Max Input Chars", "max_input_chars", 4000),
            ("Max Output Tokens", "max_output_tokens", 512),
            ("Poll Interval (s)", "poll_interval_seconds", 2.0),
            ("Response Timeout (s)", "response_timeout_seconds", 180.0),
        ]
        for label, key, default_val in num_fields:
            row = field(limits, label)
            var = ctk.StringVar(value=str(telegram_raw.get(key, default_val)))
            ctk.CTkEntry(row, textvariable=var, width=120, font=("Segoe UI", 10)).pack(side="left")
            self._cfg_vars[key] = var

        self._cfg_test_result = ctk.StringVar(value="")

        btn_row = ctk.CTkFrame(form, fg_color="transparent")
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ctk.CTkButton(btn_row, text="Test Token", command=self._cfg_test_token,
                      font=("Segoe UI", 9, "bold"), width=112, height=36).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(btn_row, textvariable=self._cfg_test_result,
                     font=("Segoe UI", 9), text_color=MUTED).pack(side="left")
        ctk.CTkButton(btn_row, text="Save", command=self._cfg_save,
                      font=("Segoe UI", 10, "bold"), width=112, height=36).pack(side="right", padx=(8, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self._show_dashboard,
                      font=("Segoe UI", 10), width=112, height=36,
                      fg_color=("#555555", "#444444")).pack(side="right")

    def _cfg_test_token(self) -> None:
        token = self._cfg_vars.get("token")
        if token:
            token = token.get().strip()
        if not token or token.startswith("${"):
            self._cfg_test_result.set("Token not configured")
            return
        t = threading.Thread(target=self._cfg_test_token_thread, args=(token,), daemon=True)
        t.start()

    def _cfg_test_token_thread(self, token: str) -> None:
        result = test_telegram_token(token)
        if result.get("ok"):
            bot_info = result["result"]
            username = bot_info.get("username", "?")
            bot_id = bot_info.get("id", "?")
            self.root.after(0, lambda: self._cfg_test_result.set(f"OK @{username} (id={bot_id})"))
            self._bot_username = username
            self._bot_id = bot_id
            self._test_ok = True
        else:
            self.root.after(0, lambda: self._cfg_test_result.set(f"FAIL: {result.get('error', 'unknown')}"))

    def _cfg_save(self) -> None:
        self._telegram_raw["enabled"] = bool(self._cfg_enabled_var.get())
        self._telegram_raw["allow_all_chats"] = bool(self._cfg_allow_all_var.get())
        self._telegram_raw["autonomous_enabled"] = bool(self._cfg_auto_var.get())
        token = self._cfg_vars.get("token")
        if token:
            token = token.get().strip()
            if token:
                self._telegram_raw["bot_token"] = token
        self._telegram_raw["provider"] = self._cfg_provider_var.get()
        model = self._cfg_vars.get("model")
        if model:
            self._telegram_raw["model"] = model.get().strip()
        prompt = self._cfg_vars.get("prompt")
        if prompt:
            self._telegram_raw["system_prompt"] = prompt.get("0.0", "end-1c").strip()
        for key in ("max_input_chars", "max_output_tokens", "poll_interval_seconds", "response_timeout_seconds"):
            var = self._cfg_vars.get(key)
            if var:
                try:
                    val = var.get()
                    self._telegram_raw[key] = int(float(val)) if key in ("max_input_chars", "max_output_tokens") else float(val)
                except ValueError:
                    pass
        self.config_path.write_text(yaml.safe_dump(self._cfg_raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self._on_config_saved()

    def _build_access_form(self) -> None:
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self._acc_raw = raw
        telegram_raw = raw.setdefault("telegram", {})
        self._acc_telegram_raw = telegram_raw

        form = ctk.CTkFrame(self._content_area, fg_color="transparent")
        form.pack(fill="x", padx=0, pady=(0, 16))

        self._acc_allow_all_var = ctk.BooleanVar(value=bool(telegram_raw.get("allow_all_chats", False)))
        ctk.CTkCheckBox(form, text="Allow all chats (WARNING: open to everyone)",
                        variable=self._acc_allow_all_var, font=("Segoe UI", 10),
                        text_color=RED).pack(anchor="w", pady=4)

        ctk.CTkLabel(form, text="Allowed Chat IDs", font=("Segoe UI", 10, "bold"), anchor="w").pack(anchor="w", pady=(8, 0))
        self._acc_allowed = ctk.CTkTextbox(form, font=("Segoe UI", 10), height=60)
        self._acc_allowed.pack(fill="x", pady=2)
        self._acc_allowed.insert("0.0", "\n".join(str(x) for x in (telegram_raw.get("allowed_chat_ids") or [])))

        ctk.CTkLabel(form, text="Owner Chat IDs", font=("Segoe UI", 10, "bold"), anchor="w").pack(anchor="w", pady=(8, 0))
        self._acc_owner = ctk.CTkTextbox(form, font=("Segoe UI", 10), height=50)
        self._acc_owner.pack(fill="x", pady=2)
        self._acc_owner.insert("0.0", "\n".join(str(x) for x in (telegram_raw.get("owner_chat_ids") or [])))

        ctk.CTkLabel(form, text="Admin Chat IDs", font=("Segoe UI", 10, "bold"), anchor="w").pack(anchor="w", pady=(8, 0))
        self._acc_admin = ctk.CTkTextbox(form, font=("Segoe UI", 10), height=50)
        self._acc_admin.pack(fill="x", pady=2)
        self._acc_admin.insert("0.0", "\n".join(str(x) for x in (telegram_raw.get("admin_chat_ids") or [])))

        ctk.CTkLabel(form, text="To get your chat ID, send /myid to the bot in Telegram.",
                     font=("Segoe UI", 9), text_color=("#888888", "#888888")).pack(anchor="w", pady=4)

        self._acc_core_var = ctk.BooleanVar(value=bool(telegram_raw.get("core_editing_enabled", False)))
        ctk.CTkCheckBox(form, text="Core editing enabled", variable=self._acc_core_var,
                        font=("Segoe UI", 10)).pack(anchor="w", pady=4)

        btn_row = ctk.CTkFrame(form, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._acc_save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self._show_dashboard,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444")).pack(side="right")

    def _acc_save(self) -> None:
        self._acc_telegram_raw["allow_all_chats"] = bool(self._acc_allow_all_var.get())
        self._acc_telegram_raw["allowed_chat_ids"] = [x.strip() for x in self._acc_allowed.get("0.0", "end-1c").split("\n") if x.strip()]
        self._acc_telegram_raw["owner_chat_ids"] = [x.strip() for x in self._acc_owner.get("0.0", "end-1c").split("\n") if x.strip()]
        self._acc_telegram_raw["admin_chat_ids"] = [x.strip() for x in self._acc_admin.get("0.0", "end-1c").split("\n") if x.strip()]
        self._acc_telegram_raw["core_editing_enabled"] = bool(self._acc_core_var.get())
        self.config_path.write_text(yaml.safe_dump(self._acc_raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self._on_config_saved()

    def _build_tools_form(self) -> None:
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self._tools_raw = raw
        telegram_raw = raw.setdefault("telegram", {})
        self._tools_telegram_raw = telegram_raw
        self._tools_cmd_policy = telegram_raw.setdefault("command_policy", {})
        self._tools_tool_policy = telegram_raw.setdefault("tool_policy", {})

        form = ctk.CTkFrame(self._content_area, fg_color="transparent")
        form.pack(fill="both", expand=True)

        panel = ctk.CTkFrame(
            form,
            fg_color=PANEL_BG,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
        )
        panel.pack(fill="x", padx=4, pady=(0, 12))

        top_row = ctk.CTkFrame(panel, fg_color="transparent")
        top_row.pack(fill="x", padx=16, pady=(14, 8))
        ctk.CTkLabel(
            top_row,
            text="Command Policy",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT,
            anchor="w",
        ).pack(side="left")

        self._tools_segment = ctk.CTkSegmentedButton(
            top_row,
            values=["Commands", "AI Tools"],
            command=self._show_tools_tab,
            height=30,
            selected_color="#2563A5",
            selected_hover_color="#2B6DB3",
            unselected_color=PANEL_BG_SOFT,
            unselected_hover_color="#203046",
            text_color=TEXT,
        )
        self._tools_segment.pack(side="right")

        self._tools_content = ctk.CTkFrame(panel, fg_color="transparent")
        self._tools_content.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._tools_segment.set("Commands")
        self._show_tools_tab("Commands")

        btn_row = ctk.CTkFrame(form, fg_color="transparent")
        btn_row.pack(fill="x", padx=4, pady=(0, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._tools_save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self._show_dashboard,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444")).pack(side="right")

    def _show_tools_tab(self, value: str) -> None:
        for child in self._tools_content.winfo_children():
            child.destroy()
        if value == "AI Tools":
            self._build_tools_ai_tools(self._tools_content)
        else:
            self._build_tools_commands(self._tools_content)

    def _build_tools_commands(self, parent: ctk.CTkFrame) -> None:
        table = ctk.CTkFrame(parent, fg_color="transparent")
        table.pack(fill="x")
        table.columnconfigure(0, weight=1)

        header_row = ctk.CTkFrame(table, fg_color="transparent")
        header_row.pack(fill="x", pady=(2, 6))
        for i, h in enumerate(["Command", "Enabled", "Visible", "Permission"]):
            ctk.CTkLabel(header_row, text=h, font=("Segoe UI", 9, "bold"),
                         text_color=TEXT,
                         anchor="w", width=140 if i == 0 else 88).pack(side="left", padx=4)

        self._tools_cmd_widgets = {}
        for cmd in _COMMAND_LIST:
            policy = self._tools_cmd_policy.get(cmd, {})
            if not isinstance(policy, dict):
                policy = {}
            row = ctk.CTkFrame(table, fg_color=PANEL_BG_SOFT, corner_radius=7)
            row.pack(fill="x", pady=3)

            ctk.CTkLabel(row, text=f"/{cmd}", font=("Segoe UI", 10), text_color=TEXT,
                         anchor="w", width=140).pack(side="left", padx=(10, 4), pady=6)

            en_var = ctk.BooleanVar(value=bool(policy.get("enabled", True)))
            ctk.CTkCheckBox(row, variable=en_var, text="", width=42).pack(side="left", padx=(4, 46))

            vis_var = ctk.BooleanVar(value=bool(policy.get("visible", True)))
            ctk.CTkCheckBox(row, variable=vis_var, text="", width=42).pack(side="left", padx=(4, 46))

            perm_var = ctk.StringVar(value=str(policy.get("permission", "everyone")))
            ctk.CTkComboBox(row, variable=perm_var, values=_PERMISSION_LEVELS,
                            state="readonly", font=("Segoe UI", 9), width=150).pack(side="left", padx=4)

            self._tools_cmd_widgets[cmd] = (en_var, vis_var, perm_var)

    def _build_tools_ai_tools(self, parent: ctk.CTkFrame) -> None:
        tp_default = ToolPolicy()
        self._tools_ai_auto_var = ctk.StringVar(value=", ".join(self._tools_tool_policy.get("ai_auto_tools", tp_default.ai_auto_tools)))
        self._tools_cmd_tools_var = ctk.StringVar(value=", ".join(self._tools_tool_policy.get("command_tools", tp_default.command_tools)))
        self._tools_blocked_var = ctk.StringVar(value=", ".join(self._tools_tool_policy.get("blocked_tools", tp_default.blocked_tools)))
        self._tools_user_vis_var = ctk.StringVar(value=", ".join(self._tools_tool_policy.get("user_visible_tools", tp_default.user_visible_tools)))
        self._tools_req_admin_var = ctk.StringVar(value=", ".join(self._tools_tool_policy.get("require_admin_for", tp_default.require_admin_for)))
        self._tools_req_owner_var = ctk.StringVar(value=", ".join(self._tools_tool_policy.get("require_owner_for", tp_default.require_owner_for)))

        sections = [
            ("AI Auto Tools", "Tools the AI can call automatically", self._tools_ai_auto_var),
            ("Command-only Tools", "Tools callable only via explicit /tools", self._tools_cmd_tools_var),
            ("Blocked Tools (dangerous)", "Never allowed in Telegram", self._tools_blocked_var),
            ("User-visible Tools", "Shown in /tools list for non-admins", self._tools_user_vis_var),
            ("Require Admin", "These tools need admin role", self._tools_req_admin_var),
            ("Require Owner", "These tools need owner role", self._tools_req_owner_var),
        ]
        for title, sub, var in sections:
            section = ctk.CTkFrame(parent, fg_color=PANEL_BG_SOFT, corner_radius=7)
            section.pack(fill="x", pady=5)
            ctk.CTkLabel(section, text=title, font=("Segoe UI", 10, "bold"),
                         text_color=TEXT, anchor="w").pack(anchor="w", padx=10, pady=(8, 0))
            ctk.CTkLabel(section, text=sub, font=("Segoe UI", 8),
                         text_color=MUTED, anchor="w").pack(anchor="w", padx=10)
            ctk.CTkEntry(section, textvariable=var, font=("Segoe UI", 9)).pack(fill="x", padx=10, pady=(4, 10))

    def _tools_save(self) -> None:
        cmd_policy = {}
        for cmd, (en_var, vis_var, perm_var) in self._tools_cmd_widgets.items():
            cmd_policy[cmd] = {
                "enabled": bool(en_var.get()),
                "visible": bool(vis_var.get()),
                "permission": perm_var.get(),
            }
        self._tools_telegram_raw["command_policy"] = cmd_policy
        self._tools_tool_policy["ai_auto_tools"] = [x.strip() for x in self._tools_ai_auto_var.get().split(",") if x.strip()]
        self._tools_tool_policy["command_tools"] = [x.strip() for x in self._tools_cmd_tools_var.get().split(",") if x.strip()]
        self._tools_tool_policy["blocked_tools"] = [x.strip() for x in self._tools_blocked_var.get().split(",") if x.strip()]
        self._tools_tool_policy["user_visible_tools"] = [x.strip() for x in self._tools_user_vis_var.get().split(",") if x.strip()]
        self._tools_tool_policy["require_admin_for"] = [x.strip() for x in self._tools_req_admin_var.get().split(",") if x.strip()]
        self._tools_tool_policy["require_owner_for"] = [x.strip() for x in self._tools_req_owner_var.get().split(",") if x.strip()]
        self.config_path.write_text(yaml.safe_dump(self._tools_raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self._on_config_saved()

    def _build_force_message_form(self) -> None:
        import yaml
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        telegram_raw = raw.get("telegram", {}) or {}
        all_chats = list(set(
            str(x).strip() for x in (
                telegram_raw.get("allowed_chat_ids", []) +
                telegram_raw.get("owner_chat_ids", []) +
                telegram_raw.get("admin_chat_ids", [])
            ) if str(x).strip()
        ))

        chat_var = ctk.StringVar(value=all_chats[0] if all_chats else "")
        row = ctk.CTkFrame(self._content_area, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Target Chat ID", font=("Segoe UI", 10), width=100, anchor="w").pack(side="left")
        if all_chats:
            ctk.CTkComboBox(row, variable=chat_var, values=all_chats, font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True)
        else:
            ctk.CTkEntry(row, textvariable=chat_var, font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True)
        self._force_vars = {"chat_id": chat_var}

        msg_text = ctk.CTkTextbox(self._content_area, font=("Segoe UI", 10), height=120)
        msg_text.pack(fill="x", pady=4)
        self._force_vars["message"] = msg_text

        self._force_status = ctk.StringVar(value="")
        ctk.CTkLabel(self._content_area, textvariable=self._force_status,
                     font=("Segoe UI", 9), text_color=("#888888", "#888888")).pack(pady=4)

        btn_row = ctk.CTkFrame(self._content_area, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Send", command=self._send_force_message,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right")
        ctk.CTkButton(btn_row, text="Cancel", command=self._show_dashboard,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444")).pack(side="right", padx=(8, 0))

    def _send_force_message(self) -> None:
        chat_id = self._force_vars["chat_id"].get().strip()
        text = self._force_vars["message"].get("0.0", "end-1c").strip()
        if not chat_id or not text:
            self._force_status.set("chat_id and message are required")
            return
        t = threading.Thread(target=self._send_force_thread, args=(chat_id, text), daemon=True)
        t.start()

    def _send_force_thread(self, chat_id: str, text: str) -> None:
        result = send_forced_message(self.config_path, chat_id, text)
        if result.get("ok"):
            self.root.after(0, lambda: self._force_status.set("Message sent successfully"))
        else:
            self.root.after(0, lambda: self._force_status.set(f"Failed: {result.get('error', 'unknown')}"))

    def _on_config_saved(self) -> None:
        self._load_config()
        self._determine_status()
        self._refresh_all(reload_config=True)

    def _open_config(self) -> None:
        self._set_active_nav("config")
        def _on_test_ok(username: str, bot_id: str) -> None:
            self._bot_username = username
            self._bot_id = bot_id
            self._test_ok = True
            self._refresh_all()
        ConfigDialog(self.root, self.config_path, self._on_config_saved, on_test_ok=_on_test_ok)

    def _open_access(self) -> None:
        self._set_active_nav("access")
        AccessDialog(self.root, self.config_path, self._on_config_saved)

    def _open_tools(self) -> None:
        self._set_active_nav("tools")
        ToolsDialog(self.root, self.config_path, self._on_config_saved)

    def _open_force_message(self) -> None:
        self._set_active_nav("force_msg")
        ForceMessageDialog(self.root, self.config_path)

    def _open_logs(self) -> None:
        self._nav_stack.append("logs")
        self._set_active_nav("logs")
        self._show_panel("logs")

    def _open_details(self) -> None:
        self._nav_stack.append("details")
        self._set_active_nav("details")
        self._show_panel("details")

    def _go_back(self) -> None:
        if self._nav_stack:
            self._nav_stack.pop()
        if self._nav_stack:
            prev = self._nav_stack[-1]
            self._show_panel(prev)
        else:
            self._show_dashboard()

    def _create_card(self, card_data: dict, parent: ctk.CTkFrame | None = None) -> ctk.CTkFrame:
        if parent is None:
            parent = self.cards_area
        ok = card_data["ok"]
        card_bg = CARD_OK_BG if ok else CARD_ERR_BG
        accent_color = GREEN if ok else RED
        status_bg = "#123B2F" if ok else "#3B1C22"

        card = ctk.CTkFrame(
            parent,
            fg_color=card_bg,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
            height=LAYOUT["CARD_H"],
        )
        card.grid_propagate(False)

        card.columnconfigure(0, weight=0)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(2, weight=0)
        card.rowconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        card._accent = ctk.CTkFrame(card, fg_color=accent_color, width=5, corner_radius=0)
        card._accent.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 14))

        ctk.CTkLabel(card, text=card_data["title"],
                     font=("Segoe UI", 13, "bold"), text_color=TEXT,
                     anchor="w").grid(row=0, column=1, sticky="sw", pady=(14, 0))

        card._subtitle_lbl = ctk.CTkLabel(card, text=card_data.get("subtitle", ""),
                                          font=("Segoe UI", 10),
                                          text_color="#A7B0BC", anchor="w")
        card._subtitle_lbl.grid(row=1, column=1, sticky="nw", pady=(2, 14))

        card._status_lbl = ctk.CTkLabel(card, text=card_data.get("status", ""),
                                        font=("Segoe UI", 9, "bold"),
                                        fg_color=status_bg,
                                        text_color=accent_color,
                                        corner_radius=7,
                                        width=76,
                                        height=28)
        card._status_lbl.grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 16))

        return card

    def _update_cards(self) -> None:
        for widget, data in zip(self._card_widgets, self._card_data):
            ok = data["ok"]
            color = GREEN if ok else RED
            status_bg = "#123B2F" if ok else "#3B1C22"
            widget._subtitle_lbl.configure(text=data.get("subtitle", ""))
            widget._status_lbl.configure(text=data.get("status", ""), fg_color=status_bg, text_color=color)
            widget._accent.configure(fg_color=color)
            widget.configure(fg_color=CARD_OK_BG if ok else CARD_ERR_BG)

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
        if self.status == BotStatus.SETUP:
            return "Bot token is missing \u2014 open Config."
        if self.status == BotStatus.READY:
            return "Bot is ready."
        if self.status == BotStatus.RUNNING:
            return "Bot is running."
        if self.status == BotStatus.ERROR:
            return f"Bot error \u2014 {self._last_msg or 'open Logs'}"
        if self.status == BotStatus.STARTING:
            return "Starting bot\u2026"
        return ""

    def _get_badge_info(self) -> tuple[str, str]:
        s = self.status
        if s == BotStatus.ERROR:
            return "ERROR", RED
        if s in (BotStatus.SETUP,):
            return "SETUP", YELLOW
        if s == BotStatus.READY:
            return "READY", GREEN
        if s == BotStatus.STARTING:
            return "STARTING", YELLOW
        if s == BotStatus.RUNNING:
            return "RUNNING", GREEN
        return "READY", GREEN

    def _update_card_data(self) -> None:
        cfg = self._config
        telegram = cfg.telegram if cfg else None
        token_present = bool(telegram and telegram.bot_token and not telegram.bot_token.startswith("${"))
        api_ok = self._test_ok and bool(self._bot_username)
        model_present = bool(telegram and telegram.provider and telegram.model)
        model_label = f"{telegram.provider} / {telegram.model}" if model_present else ""
        allowed_count = len(telegram.allowed_chat_ids) if telegram else 0
        tp = telegram.tool_policy if telegram else None
        tool_count = (len(tp.ai_auto_tools) + len(tp.command_tools)) if tp else 0
        access_label = f"Allowed: {allowed_count} \u2022 Tools: {tool_count}"
        if telegram and telegram.allow_all_chats:
            access_label = "All chats allowed"
        access_ok = allowed_count > 0 or (telegram and telegram.allow_all_chats) if telegram else False

        self._card_data[0] = {"key": "token", "title": "Bot Token", "ok": token_present,
                              "subtitle": "Set" if token_present else "Missing",
                              "status": "OK" if token_present else "Missing"}
        self._card_data[1] = {"key": "api", "title": "Telegram API", "ok": api_ok,
                              "subtitle": self._bot_username if self._bot_username else "Not tested",
                              "status": "OK" if api_ok else "Test"}
        self._card_data[2] = {"key": "model", "title": "AI Model", "ok": model_present,
                              "subtitle": model_label if model_present else "Missing",
                              "status": "OK" if model_present else "Missing"}
        self._card_data[3] = {"key": "access", "title": "Access & Tools", "ok": access_ok,
                              "subtitle": access_label,
                              "status": "Safe" if access_ok else "Warn"}

    def _load_config(self) -> None:
        try:
            self._config = load_config(self.config_path)
        except Exception as exc:
            self._config = None
            self._append_log(f"[Config load error: {exc}]")

    def _determine_status(self) -> None:
        def _worker():
            try:
                status_info = telegram_bot_status(self.config_path)
                self.root.after(0, lambda: self._apply_status(status_info))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_status(self, status_info: dict) -> None:
        if status_info.get("running"):
            self.status = BotStatus.RUNNING
            return
        if not self._config:
            self.status = BotStatus.SETUP
            return
        telegram = self._config.telegram
        if not telegram.bot_token or telegram.bot_token.startswith("${"):
            self.status = BotStatus.SETUP
            return
        self.status = BotStatus.READY if self._test_ok else BotStatus.SETUP

    def _refresh_all(self, reload_config: bool = False) -> None:
        if reload_config:
            self._load_config()
        old_data = [d.copy() for d in self._card_data]
        self._update_card_data()
        if self._card_data != old_data:
            self._rebuild_cards()
        else:
            self._update_cards()
        self._update_header()
        self._update_dashboard_summary()

    def _append_log(self, line: str) -> None:
        self._log_lines.append(line)
        if len(self._log_lines) > MAX_LOG_LINES:
            self._log_lines = self._log_lines[-MAX_LOG_LINES:]

    def _poll_status(self) -> None:
        if self._stopped.is_set():
            return

        def _worker():
            try:
                status_info = telegram_bot_status(self.config_path)
                self.root.after(0, lambda: self._apply_poll(status_info))
            except Exception:
                pass
            if not self._stopped.is_set():
                self.root.after(5000, self._poll_status)

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_poll(self, status_info: dict) -> None:
        was_running = self.status == BotStatus.RUNNING
        now_running = status_info.get("running", False)
        if was_running and not now_running:
            self.status = BotStatus.ERROR
            self._last_msg = "Bot stopped unexpectedly"
            self._append_log("[Bot stopped unexpectedly]")
            self._refresh_all()
        elif not was_running and now_running:
            self.status = BotStatus.RUNNING
            self._refresh_all()

    def _auto_test_token(self) -> None:
        if self._test_ok:
            return
        token = None
        try:
            cfg = load_config(self.config_path)
            t = cfg.telegram
            if t.bot_token and not t.bot_token.startswith("${"):
                token = t.bot_token
        except Exception:
            pass
        if token:
            t = threading.Thread(target=self._auto_test_thread, args=(token,), daemon=True)
            t.start()

    def _auto_test_thread(self, token: str) -> None:
        result = test_telegram_token(token)
        if result.get("ok"):
            bot_info = result["result"]
            username = bot_info.get("username", "?")
            bot_id = bot_info.get("id", "?")
            self.root.after(0, lambda: self._apply_auto_test(username, str(bot_id)))

    def _apply_auto_test(self, username: str, bot_id: str) -> None:
        self._bot_username = username
        self._bot_id = bot_id
        self._test_ok = True
        self._determine_status()
        self._refresh_all()

    def _on_close(self) -> None:
        self._stopped.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()

    def _go_home(self) -> None:
        self._current_view = "home"
        self._nav_stack.clear()
        self._footer_frame.pack_forget()
        self.cards_area.pack(fill="x", expand=False, padx=PAD, pady=(8, 0))
        self._content_area.pack_forget()
        self.subtitle_label.configure(text=self._subtitle_text())
        self._set_active_nav("dashboard")


class ConfigDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path, on_save=None, on_test_ok=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.on_test_ok = on_test_ok
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Telegram Config")
        self.dialog.geometry("540x580")
        self.dialog.minsize(480, 420)
        self.dialog.resizable(True, True)

        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.telegram_raw = self.raw.setdefault("telegram", {})

    def _build(self) -> None:
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll._scrollbar.grid_remove()

        row = 0
        self.enabled_var = ctk.BooleanVar(value=bool(self.telegram_raw.get("enabled", False)))
        ctk.CTkCheckBox(scroll, text="Enable Telegram bot", variable=self.enabled_var,
                        font=("Segoe UI", 10)).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        self.allow_all_var = ctk.BooleanVar(value=bool(self.telegram_raw.get("allow_all_chats", False)))
        self.allow_all_cb = ctk.CTkCheckBox(
            scroll, text="Allow all chats (open to everyone)",
            variable=self.allow_all_var,
            font=("Segoe UI", 10),
            text_color=RED if self.allow_all_var.get() else None,
        )
        self.allow_all_cb.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        def _on_allow_toggle():
            self.allow_all_cb.configure(text_color=RED if self.allow_all_var.get() else None)
        self.allow_all_var.trace_add("write", lambda *_: _on_allow_toggle())
        row += 1

        self.auto_var = ctk.BooleanVar(value=bool(self.telegram_raw.get("autonomous_enabled", False)))
        ctk.CTkCheckBox(scroll, text="Autonomous mode (bot can initiate conversations)",
                        variable=self.auto_var, font=("Segoe UI", 10)).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        ctk.CTkLabel(scroll, text="Bot Token", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        token_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        token_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        self.token_entry = ctk.CTkEntry(token_frame, font=("Segoe UI", 10), show="*")
        self.token_entry.pack(side="left", fill="x", expand=True)
        self.token_entry.insert(0, str(self.telegram_raw.get("bot_token", "")))
        self.show_token_var = ctk.BooleanVar(value=False)
        def _toggle_token_show():
            self.token_entry.configure(show="" if self.show_token_var.get() else "*")
        ctk.CTkCheckBox(token_frame, text="Show", variable=self.show_token_var,
                        command=_toggle_token_show, font=("Segoe UI", 9)).pack(side="left", padx=(6, 0))
        row += 1

        ctk.CTkButton(scroll, text="Test Token", command=self._test_token,
                      font=("Segoe UI", 9, "bold"), width=100, height=28).grid(
            row=row, column=1, sticky="w", pady=2)
        self.test_result_var = ctk.StringVar(value="")
        ctk.CTkLabel(scroll, textvariable=self.test_result_var,
                     font=("Segoe UI", 9), text_color=("#888888", "#888888")).grid(
            row=row, column=2, sticky="w", padx=(8, 0))
        row += 1

        provider_names = list(self.raw.get("providers", {}).keys())
        ctk.CTkLabel(scroll, text="Provider", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.provider_var = ctk.StringVar(
            value=str(self.telegram_raw.get("provider", provider_names[0] if provider_names else "")))
        ctk.CTkComboBox(scroll, variable=self.provider_var, values=provider_names,
                        state="readonly", font=("Segoe UI", 10), width=200).grid(
            row=row, column=1, columnspan=2, sticky="w", pady=2)
        row += 1

        ctk.CTkLabel(scroll, text="Model", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.model_entry = ctk.CTkEntry(scroll, font=("Segoe UI", 10))
        self.model_entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        self.model_entry.insert(0, str(self.telegram_raw.get("model", "")))
        row += 1

        ctk.CTkLabel(scroll, text="System Prompt", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.prompt_text = ctk.CTkTextbox(scroll, font=("Segoe UI", 10), height=80)
        self.prompt_text.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        self.prompt_text.insert("0.0", str(self.telegram_raw.get("system_prompt", "")))
        row += 1

        ctk.CTkLabel(scroll, text="Max Input Chars", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.input_chars_var = ctk.StringVar(value=str(self.telegram_raw.get("max_input_chars", 4000)))
        ctk.CTkEntry(scroll, textvariable=self.input_chars_var, width=100,
                     font=("Segoe UI", 10)).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ctk.CTkLabel(scroll, text="Max Output Tokens", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.output_tokens_var = ctk.StringVar(value=str(self.telegram_raw.get("max_output_tokens", 512)))
        ctk.CTkEntry(scroll, textvariable=self.output_tokens_var, width=100,
                     font=("Segoe UI", 10)).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ctk.CTkLabel(scroll, text="Poll Interval (s)", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.poll_interval_var = ctk.StringVar(value=str(self.telegram_raw.get("poll_interval_seconds", 2.0)))
        ctk.CTkEntry(scroll, textvariable=self.poll_interval_var, width=100,
                     font=("Segoe UI", 10)).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ctk.CTkLabel(scroll, text="Response Timeout (s)", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.resp_timeout_var = ctk.StringVar(value=str(self.telegram_raw.get("response_timeout_seconds", 180.0)))
        ctk.CTkEntry(scroll, textvariable=self.resp_timeout_var, width=100,
                     font=("Segoe UI", 10)).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        scroll.columnconfigure(1, weight=1)

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

    def _test_token(self) -> None:
        token = self.token_entry.get().strip()
        if not token or token.startswith("${"):
            self.test_result_var.set("Token not configured")
            return
        t = threading.Thread(target=self._test_token_thread, args=(token,), daemon=True)
        t.start()

    def _test_token_thread(self, token: str) -> None:
        result = test_telegram_token(token)
        if result.get("ok"):
            bot_info = result["result"]
            username = bot_info.get("username", "?")
            bot_id = bot_info.get("id", "?")
            first_name = bot_info.get("first_name", "?")
            msg = f"OK @{username} (id={bot_id}, {first_name})"
            self.dialog.after(0, lambda: self.test_result_var.set(msg))
            if self.on_test_ok:
                self.dialog.after(0, lambda: self.on_test_ok(username, str(bot_id)))
        else:
            self.dialog.after(0, lambda: self.test_result_var.set(f"FAIL: {result.get('error', 'unknown')}"))

    def _save(self) -> None:
        self.telegram_raw["enabled"] = bool(self.enabled_var.get())
        self.telegram_raw["allow_all_chats"] = bool(self.allow_all_var.get())
        self.telegram_raw["autonomous_enabled"] = bool(self.auto_var.get())
        token = self.token_entry.get().strip()
        if token:
            self.telegram_raw["bot_token"] = token
        self.telegram_raw["provider"] = self.provider_var.get()
        self.telegram_raw["model"] = self.model_entry.get().strip()
        self.telegram_raw["system_prompt"] = self.prompt_text.get("0.0", "end-1c").strip()
        self.telegram_raw["max_input_chars"] = int(self.input_chars_var.get())
        self.telegram_raw["max_output_tokens"] = int(self.output_tokens_var.get())
        self.telegram_raw["poll_interval_seconds"] = float(self.poll_interval_var.get())
        self.telegram_raw["response_timeout_seconds"] = float(self.resp_timeout_var.get())
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class AccessDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Access Control")
        self.dialog.geometry("560x520")
        self.dialog.minsize(480, 420)
        self.dialog.resizable(True, True)

        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.telegram_raw = self.raw.setdefault("telegram", {})

    def _build(self) -> None:
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll._scrollbar.grid_remove()

        row = 0
        self.allow_all_var = ctk.BooleanVar(value=bool(self.telegram_raw.get("allow_all_chats", False)))
        ctk.CTkCheckBox(scroll, text="Allow all chats (WARNING: open to everyone)",
                        variable=self.allow_all_var, font=("Segoe UI", 10),
                        text_color=RED).grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        row += 1

        ctk.CTkLabel(scroll, text="Allowed Chat IDs", font=("Segoe UI", 10, "bold"),
                     anchor="w").grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        self.allowed_text = ctk.CTkTextbox(scroll, font=("Segoe UI", 10), height=60)
        self.allowed_text.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        self.allowed_text.insert("0.0", "\n".join(str(x) for x in (self.telegram_raw.get("allowed_chat_ids") or [])))
        row += 1

        ctk.CTkLabel(scroll, text="Owner Chat IDs", font=("Segoe UI", 10, "bold"),
                     anchor="w").grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        self.owner_text = ctk.CTkTextbox(scroll, font=("Segoe UI", 10), height=50)
        self.owner_text.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        self.owner_text.insert("0.0", "\n".join(str(x) for x in (self.telegram_raw.get("owner_chat_ids") or [])))
        row += 1

        ctk.CTkLabel(scroll, text="Admin Chat IDs", font=("Segoe UI", 10, "bold"),
                     anchor="w").grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        self.admin_text = ctk.CTkTextbox(scroll, font=("Segoe UI", 10), height=50)
        self.admin_text.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        self.admin_text.insert("0.0", "\n".join(str(x) for x in (self.telegram_raw.get("admin_chat_ids") or [])))
        row += 1

        ctk.CTkLabel(scroll, text="To get your chat ID, send /myid to the bot in Telegram.",
                     font=("Segoe UI", 9), text_color=("#888888", "#888888")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=4)
        row += 1

        self.core_edit_var = ctk.BooleanVar(value=bool(self.telegram_raw.get("core_editing_enabled", False)))
        ctk.CTkCheckBox(scroll, text="Core editing enabled", variable=self.core_edit_var,
                        font=("Segoe UI", 10)).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1

        scroll.columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

    def _save(self) -> None:
        self.telegram_raw["allow_all_chats"] = bool(self.allow_all_var.get())
        self.telegram_raw["allowed_chat_ids"] = [x.strip() for x in self.allowed_text.get("0.0", "end-1c").split("\n") if x.strip()]
        self.telegram_raw["owner_chat_ids"] = [x.strip() for x in self.owner_text.get("0.0", "end-1c").split("\n") if x.strip()]
        self.telegram_raw["admin_chat_ids"] = [x.strip() for x in self.admin_text.get("0.0", "end-1c").split("\n") if x.strip()]
        self.telegram_raw["core_editing_enabled"] = bool(self.core_edit_var.get())
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


_COMMAND_LIST = [
    "help", "status", "clear", "reload", "whoami", "memory", "remember",
    "docs", "editdoc", "image", "file", "schedule", "evolve", "poll",
    "web", "deep", "summarize", "explain", "myid", "allowlist",
    "allow", "admin", "owner", "core", "project", "tools",
]
_PERMISSION_LEVELS = ["everyone", "allowed", "admin", "owner"]


class ToolsDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Command & Tool Policy")
        self.dialog.geometry("700x580")
        self.dialog.minsize(580, 480)
        self.dialog.resizable(True, True)

        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.telegram_raw = self.raw.setdefault("telegram", {})
        self.cmd_policy_raw = self.telegram_raw.setdefault("command_policy", {})
        self.tool_policy_raw = self.telegram_raw.setdefault("tool_policy", {})

    def _build(self) -> None:
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        tabview = ctk.CTkTabview(main)
        tabview.pack(fill="both", expand=True)

        cmd_tab = tabview.add("Commands")
        tool_tab = tabview.add("AI Tools")

        self._build_commands(cmd_tab)
        self._build_tools(tool_tab)

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btn_row, text="Save", command=self._save,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

    def _build_commands(self, parent: ctk.CTkFrame) -> None:
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        header_row = ctk.CTkFrame(scroll, fg_color="transparent")
        header_row.pack(fill="x", padx=4, pady=(4, 0))
        for i, h in enumerate(["Command", "Enabled", "Visible", "Permission"]):
            ctk.CTkLabel(header_row, text=h, font=("Segoe UI", 9, "bold"),
                         anchor="w", width=90 if i == 0 else 60).pack(side="left", padx=4)

        self._cmd_widgets = {}
        for cmd in _COMMAND_LIST:
            policy = self.cmd_policy_raw.get(cmd, {})
            if not isinstance(policy, dict):
                policy = {}
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=1)

            ctk.CTkLabel(row, text=f"/{cmd}", font=("Segoe UI", 9), anchor="w",
                         width=90).pack(side="left", padx=4)

            en_var = ctk.BooleanVar(value=bool(policy.get("enabled", True)))
            ctk.CTkCheckBox(row, variable=en_var, text="", width=20).pack(side="left", padx=4)

            vis_var = ctk.BooleanVar(value=bool(policy.get("visible", True)))
            ctk.CTkCheckBox(row, variable=vis_var, text="", width=20).pack(side="left", padx=4)

            perm_var = ctk.StringVar(value=str(policy.get("permission", "everyone")))
            ctk.CTkComboBox(row, variable=perm_var, values=_PERMISSION_LEVELS,
                            state="readonly", font=("Segoe UI", 9), width=100).pack(side="left", padx=4)

            self._cmd_widgets[cmd] = (en_var, vis_var, perm_var)

    def _build_tools(self, parent: ctk.CTkFrame) -> None:
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        tp_default = ToolPolicy()
        self._ai_auto_var = ctk.StringVar(value=", ".join(self.tool_policy_raw.get("ai_auto_tools", tp_default.ai_auto_tools)))
        self._cmd_tools_var = ctk.StringVar(value=", ".join(self.tool_policy_raw.get("command_tools", tp_default.command_tools)))
        self._blocked_var = ctk.StringVar(value=", ".join(self.tool_policy_raw.get("blocked_tools", tp_default.blocked_tools)))
        self._user_vis_var = ctk.StringVar(value=", ".join(self.tool_policy_raw.get("user_visible_tools", tp_default.user_visible_tools)))
        self._req_admin_var = ctk.StringVar(value=", ".join(self.tool_policy_raw.get("require_admin_for", tp_default.require_admin_for)))
        self._req_owner_var = ctk.StringVar(value=", ".join(self.tool_policy_raw.get("require_owner_for", tp_default.require_owner_for)))

        sections = [
            ("AI Auto Tools", "Tools the AI can call automatically", self._ai_auto_var),
            ("Command-only Tools", "Tools callable only via explicit /tools", self._cmd_tools_var),
            ("Blocked Tools (dangerous)", "Never allowed in Telegram", self._blocked_var),
            ("User-visible Tools", "Shown in /tools list for non-admins", self._user_vis_var),
            ("Require Admin", "These tools need admin role", self._req_admin_var),
            ("Require Owner", "These tools need owner role", self._req_owner_var),
        ]
        for title, sub, var in sections:
            section = ctk.CTkFrame(scroll, fg_color="transparent")
            section.pack(fill="x", pady=4)
            ctk.CTkLabel(section, text=title, font=("Segoe UI", 10, "bold"),
                         anchor="w").pack(anchor="w")
            ctk.CTkLabel(section, text=sub, font=("Segoe UI", 8),
                         text_color=("#888888", "#888888"), anchor="w").pack(anchor="w")
            ctk.CTkEntry(section, textvariable=var, font=("Segoe UI", 9)).pack(fill="x", pady=(2, 0))

    def _save(self) -> None:
        cmd_policy = {}
        for cmd, (en_var, vis_var, perm_var) in self._cmd_widgets.items():
            cmd_policy[cmd] = {
                "enabled": bool(en_var.get()),
                "visible": bool(vis_var.get()),
                "permission": perm_var.get(),
            }
        self.telegram_raw["command_policy"] = cmd_policy
        self.tool_policy_raw["ai_auto_tools"] = [x.strip() for x in self._ai_auto_var.get().split(",") if x.strip()]
        self.tool_policy_raw["command_tools"] = [x.strip() for x in self._cmd_tools_var.get().split(",") if x.strip()]
        self.tool_policy_raw["blocked_tools"] = [x.strip() for x in self._blocked_var.get().split(",") if x.strip()]
        self.tool_policy_raw["user_visible_tools"] = [x.strip() for x in self._user_vis_var.get().split(",") if x.strip()]
        self.tool_policy_raw["require_admin_for"] = [x.strip() for x in self._req_admin_var.get().split(",") if x.strip()]
        self.tool_policy_raw["require_owner_for"] = [x.strip() for x in self._req_owner_var.get().split(",") if x.strip()]
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class ForceMessageDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path) -> None:
        self.config_path = config_path
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Force Message")
        self.dialog.geometry("540x420")
        self.dialog.minsize(480, 360)
        self.dialog.resizable(True, True)

        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.telegram_raw = self.raw.setdefault("telegram", {})

    def _build(self) -> None:
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        row = 0
        all_chats = list(set(
            str(x).strip() for x in (
                self.telegram_raw.get("allowed_chat_ids", []) +
                self.telegram_raw.get("owner_chat_ids", []) +
                self.telegram_raw.get("admin_chat_ids", [])
            ) if str(x).strip()
        ))

        ctk.CTkLabel(main, text="Target Chat ID", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.chat_var = ctk.StringVar(value=all_chats[0] if all_chats else "")
        if all_chats:
            ctk.CTkComboBox(main, variable=self.chat_var, values=all_chats,
                           font=("Segoe UI", 10), width=240).grid(
                row=row, column=1, sticky="ew", pady=2)
        else:
            ctk.CTkEntry(main, textvariable=self.chat_var, font=("Segoe UI", 10)).grid(
                row=row, column=1, sticky="ew", pady=2)
        row += 1

        ctk.CTkLabel(main, text="Message", font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.msg_text = ctk.CTkTextbox(main, font=("Segoe UI", 10), height=120)
        self.msg_text.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        main.columnconfigure(1, weight=1)

        self.status_var = ctk.StringVar(value="")
        ctk.CTkLabel(main, textvariable=self.status_var, font=("Segoe UI", 9),
                     text_color=("#888888", "#888888")).grid(
            row=row, column=0, columnspan=2, pady=(4, 0))
        row += 1

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.grid(row=row, column=0, columnspan=2, pady=(12, 0), sticky="e")
        ctk.CTkButton(btn_row, text="Send", command=self._send,
                      font=("Segoe UI", 10, "bold"), width=90).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_row, text="Cancel", command=self.dialog.destroy,
                      font=("Segoe UI", 10), width=90,
                      fg_color=("#555555", "#444444"),
                      hover_color=("#666666", "#555555")).pack(side="right")

    def _send(self) -> None:
        chat_id = self.chat_var.get().strip()
        text = self.msg_text.get("0.0", "end-1c").strip()
        if not chat_id or not text:
            self.status_var.set("chat_id and message are required")
            return
        t = threading.Thread(target=self._send_thread, args=(chat_id, text), daemon=True)
        t.start()

    def _send_thread(self, chat_id: str, text: str) -> None:
        result = send_forced_message(self.config_path, chat_id, text)
        if result.get("ok"):
            self.dialog.after(0, lambda: self.status_var.set("Message sent successfully"))
        else:
            self.dialog.after(0, lambda: self.status_var.set(f"Failed: {result.get('error', 'unknown')}"))


class LogsDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path, log_fetcher=None) -> None:
        self.config_path = config_path
        self.log_fetcher = log_fetcher or (lambda: follow_telegram_log(config_path))
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Telegram Bot Logs")
        self.dialog.geometry("740x540")
        self.dialog.minsize(580, 380)
        self.dialog.resizable(True, True)

        self._filter = "all"
        self._build()

    def _build(self) -> None:
        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        filter_row = ctk.CTkFrame(main, fg_color="transparent")
        filter_row.pack(fill="x", pady=(0, 8))
        for f in ["all", "errors", "access", "tools", "provider", "sends"]:
            btn = ctk.CTkButton(filter_row, text=f, command=lambda lbl=f: self._set_filter(lbl),
                                font=("Segoe UI", 9), height=28,
                                fg_color=("#e0e0e0", "#2a2a2a"),
                                text_color=("#333333", "#cccccc"),
                                hover_color=("#d0d0d0", "#3a3a3a"),
                                corner_radius=4)
            btn.pack(side="left", padx=(0, 4))

        self.text = ctk.CTkTextbox(main, font=("Consolas", 10), wrap="none",
                                   activate_scrollbars=True)
        self.text.pack(fill="both", expand=True)
        self.text.bind("<MouseWheel>", lambda e: self.text.yview_scroll(-1 * (e.delta // 120), "units"))

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", pady=(8, 0))
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
        self._refresh()

    def _set_filter(self, label: str) -> None:
        self._filter = label
        self._refresh()

    def _refresh(self) -> None:
        lines = self.log_fetcher() if self.log_fetcher else []
        self.text.delete("0.0", "end")
        for line in lines:
            lower = line.lower()
            if self._filter != "all":
                if self._filter == "errors" and "error" not in lower:
                    continue
                elif self._filter == "access" and "access" not in lower:
                    continue
                elif self._filter == "tools" and "tool" not in lower:
                    continue
                elif self._filter == "provider" and "provider" not in lower and "model" not in lower:
                    continue
                elif self._filter == "sends" and "send" not in lower and "message" not in lower:
                    continue
            self.text.insert("end", line + "\n")
        self.text.see("end")

    def _clear(self) -> None:
        from .telegram_launcher import _log_path
        try:
            log_path = _log_path(self.config_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")
        except Exception:
            pass
        self._refresh()

    def _close(self) -> None:
        self.dialog.destroy()


class DetailsDialog:
    def __init__(self, parent: ctk.CTk, config_path: Path) -> None:
        self.config_path = config_path
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Telegram Bot Details")
        self.dialog.geometry("780x540")
        self.dialog.minsize(640, 440)

        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)

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
        style.configure("Telegram.Treeview",
                        background=dark_bg, foreground=dark_fg,
                        fieldbackground=dark_bg,
                        font=("Segoe UI", 9), rowheight=28, borderwidth=0)
        style.configure("Telegram.Treeview.Heading",
                        background=heading_bg, foreground=heading_fg,
                        font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Telegram.Treeview",
                  background=[("selected", sel_bg)],
                  foreground=[("selected", dark_fg)])
        style.layout("Telegram.Treeview", [("Telegram.Treeview.treearea", {"sticky": "nswe"})])

        tree_frame = ctk.CTkFrame(main, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("field", "value"), show="headings",
                                 selectmode="browse", style="Telegram.Treeview")
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
        items: list[DetailsDialog._DetailItem] = []
        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}
        telegram_raw = raw.get("telegram", {}) or {}

        items.append(DetailsDialog._DetailItem("Config path", _compact_path(str(self.config_path)), str(self.config_path)))
        items.append(DetailsDialog._DetailItem("Bot Token", "set" if telegram_raw.get("bot_token") else "missing",
                                               "set" if telegram_raw.get("bot_token") else "missing"))
        items.append(DetailsDialog._DetailItem("Provider", telegram_raw.get("provider", "-"),
                                               telegram_raw.get("provider", "-")))
        items.append(DetailsDialog._DetailItem("Model", telegram_raw.get("model", "-"),
                                               telegram_raw.get("model", "-")))
        items.append(DetailsDialog._DetailItem("Allow all chats", str(telegram_raw.get("allow_all_chats", False)),
                                               str(telegram_raw.get("allow_all_chats", False))))
        allowed_ids = telegram_raw.get("allowed_chat_ids", [])
        items.append(DetailsDialog._DetailItem("Allowed IDs", str(len(allowed_ids)),
                                               ", ".join(str(x) for x in allowed_ids) if allowed_ids else "-"))
        owner_ids = telegram_raw.get("owner_chat_ids", [])
        items.append(DetailsDialog._DetailItem("Owner IDs", str(len(owner_ids)),
                                               ", ".join(str(x) for x in owner_ids) if owner_ids else "-"))
        admin_ids = telegram_raw.get("admin_chat_ids", [])
        items.append(DetailsDialog._DetailItem("Admin IDs", str(len(admin_ids)),
                                               ", ".join(str(x) for x in admin_ids) if admin_ids else "-"))
        cmd_policy = telegram_raw.get("command_policy", {})
        items.append(DetailsDialog._DetailItem("Command policies", str(len(cmd_policy)),
                                               ", ".join(cmd_policy.keys()) if cmd_policy else "-"))
        items.append(DetailsDialog._DetailItem("System prompt",
                                               _compact_path(telegram_raw.get("system_prompt", "-"), 60) if telegram_raw.get("system_prompt") else "-",
                                               telegram_raw.get("system_prompt", "-")))
        items.append(DetailsDialog._DetailItem("Max input chars", str(telegram_raw.get("max_input_chars", 4000)),
                                               str(telegram_raw.get("max_input_chars", 4000))))
        items.append(DetailsDialog._DetailItem("Max output tokens", str(telegram_raw.get("max_output_tokens", 512)),
                                               str(telegram_raw.get("max_output_tokens", 512))))
        items.append(DetailsDialog._DetailItem("Poll interval", f'{telegram_raw.get("poll_interval_seconds", 2.0)}s',
                                               str(telegram_raw.get("poll_interval_seconds", 2.0))))
        return items

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
            return self._build_items()[int(idx)].full_value
        except (IndexError, ValueError):
            return ""

    def _copy_selected(self) -> None:
        idx = self._selected_idx
        if idx is None:
            sel = self.tree.selection()
            idx = sel[0] if sel else None
        if idx is None:
            from tkinter import messagebox
            messagebox.showinfo("Copy", "No item selected.", parent=self.dialog)
            return
        full = self._get_full_value(idx)
        if full:
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(full)

    def _copy_all(self) -> None:
        parts: list[str] = []
        for item in self._build_items():
            parts.append(f"{item.label:<25} {item.full_value}")
        text = "\n".join(parts)
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(text)


def launch_gui(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    if not HAS_TK:
        print("customtkinter is not available.")
        sys.exit(1)
    app = TelegramSetupCenter(config_path=config_path)
    app.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Telegram Bot Setup GUI")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG_PATH
    launch_gui(config_path)
