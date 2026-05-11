from __future__ import annotations

import json
import os
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, load_config
from .telegram_launcher import (
    start_telegram_bot,
    stop_telegram_bot,
    restart_telegram_bot,
    telegram_bot_status,
    follow_telegram_log,
    test_telegram_token,
    send_forced_message,
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

GREEN = "#4FD1A1"
RED = "#FF6B6B"
YELLOW = "#F2C66D"

LAYOUT = {
    "PAD": 20,
    "CARD_H": 68,
    "CARD_GAP": 8,
    "WIN_W": 640,
    "WIN_H": 480,
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
        self.root.minsize(520, 400)
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
        self._refresh_all()
        self.root.after(100, self._auto_test_token)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(2000, self._poll_status)

    def _build_ui(self) -> None:
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True)

        self._build_header(main)

        sep = ctk.CTkFrame(main, fg_color=GREEN, height=2, corner_radius=0)
        sep.pack(fill="x", padx=PAD, pady=(4, 0))

        self.status_label = ctk.CTkLabel(main, text="", font=("Segoe UI", 9),
                                          text_color=("#888888", "#888888"), anchor="w")
        self.status_label.pack(fill="x", padx=PAD, pady=(8, 0))

        self._build_footer(main)

        self.cards_area = ctk.CTkFrame(main, fg_color="transparent")
        self.cards_area.pack(fill="x", expand=False, padx=PAD, pady=(8, 0))

    def _build_header(self, parent: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=PAD, pady=(PAD, 0))

        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(title_frame, text="Telegram Bot Setup Center",
                     font=("Segoe UI", 18, "bold"), anchor="w").pack(anchor="w")
        self.subtitle_label = ctk.CTkLabel(title_frame, text="",
                                           font=("Segoe UI", 10),
                                           text_color=("#888888", "#888888"), anchor="w")
        self.subtitle_label.pack(anchor="w", pady=(2, 0))

        self.badge = ctk.CTkLabel(header, text="  SETUP  ",
                                  font=("Segoe UI", 9, "bold"), corner_radius=4)
        self.badge.grid(row=0, column=1, sticky="ne", pady=(4, 0))

        self._update_header()

    def _update_header(self) -> None:
        self.subtitle_label.configure(text=self._subtitle_text())
        badge_text, badge_color = self._get_badge_info()
        badge_bg = {
            BotStatus.RUNNING: "#0a2a1a",
            BotStatus.READY: "#0a2a1a",
            BotStatus.SETUP: "#2a2a0a",
            BotStatus.STARTING: "#2a2a0a",
            BotStatus.ERROR: "#2a0a0a",
        }.get(self.status, "#0a2a1a")
        self.badge.configure(text=f"  {badge_text}  ",
                             fg_color=badge_bg, text_color=badge_color)

    def _build_footer(self, parent: ctk.CTkFrame) -> None:
        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.pack(fill="x", side="bottom", pady=(4, 12), padx=PAD)

        util = ctk.CTkFrame(footer, fg_color="transparent")
        util.pack(fill="x")

        util_items = [
            ("btn_config", "\u2699 Config", self._open_config),
            ("btn_access", "\U0001f511 Access", self._open_access),
            ("btn_tools", "\U0001f527 Tools", self._open_tools),
            ("btn_force", "\U0001f4e9 Force Msg", self._open_force_message),
            ("btn_logs", "\U0001f4cb Logs", self._open_logs),
            ("btn_details", "\u2139 Details", self._open_details),
        ]
        self.util_btns: dict[str, ctk.CTkButton] = {}
        for key, text, cmd in util_items:
            btn = ctk.CTkButton(
                util, text=text, command=cmd,
                font=("Segoe UI", 10), height=30,
                fg_color=("#e0e0e0", "#2a2a2a"),
                text_color=("#333333", "#cccccc"),
                hover_color=("#d0d0d0", "#3a3a3a"),
                corner_radius=4,
            )
            btn.pack(side="left", padx=(0, 8))
            self.util_btns[key] = btn
            tooltip_map = {
                "btn_config": "Open configuration dialog",
                "btn_access": "Manage chat access control",
                "btn_tools": "Configure commands and AI tool policy",
                "btn_force": "Force-send a message to a chat",
                "btn_logs": "View Telegram bot logs",
                "btn_details": "Show full technical details",
            }
            ToolTip(btn, tooltip_map[key])

        action = ctk.CTkFrame(footer, fg_color="transparent")
        action.pack(fill="x", pady=(6, 0))

        self.access_btn = ctk.CTkButton(
            action, text="Access: Private", command=self._toggle_access,
            font=("Segoe UI", 10, "bold"), height=34,
            fg_color=("#1a4030", "#1a4030"), text_color=GREEN,
            hover_color=("#153828", "#153828"), corner_radius=6,
        )
        self.access_btn.pack(side="left", padx=(0, 8))
        ToolTip(self.access_btn, "Toggle between private and all-chats access")

        self.auto_btn = ctk.CTkButton(
            action, text="Auto: Off", command=self._toggle_auto,
            font=("Segoe UI", 10, "bold"), height=34,
            fg_color=("#1a4030", "#1a4030"), text_color=GREEN,
            hover_color=("#153828", "#153828"), corner_radius=6,
        )
        self.auto_btn.pack(side="left")
        ToolTip(self.auto_btn, "Toggle autonomous mode on/off")

        self.primary_btn = ctk.CTkButton(
            action, text="Start Bot", command=self._primary_action,
            font=("Segoe UI", 11, "bold"), height=38,
            fg_color=("#1a4030", "#1a4030"), text_color=GREEN,
            hover_color=("#153828", "#153828"), corner_radius=6,
        )
        self.primary_btn.pack(side="right")
        ToolTip(self.primary_btn, "Start or stop the Telegram bot")

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
        if self.status == BotStatus.SETUP:
            return "Bot token is missing \u2014 open Config."
        if self.status == BotStatus.READY:
            return "Bot is ready. Choose access/tools, then click Start."
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
        owner_count = len(telegram.owner_chat_ids) if telegram else 0
        admin_count = len(telegram.admin_chat_ids) if telegram else 0
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
            self._log_lines.append(f"[Config load error: {exc}]")

    def _determine_status(self) -> None:
        status_info = telegram_bot_status(self.config_path)
        if status_info["running"]:
            self.status = BotStatus.RUNNING
            return
        if not self._config:
            self.status = BotStatus.SETUP
            return
        telegram = self._config.telegram
        if not telegram.bot_token or telegram.bot_token.startswith("${"):
            self.status = BotStatus.SETUP
            return
        if self._test_ok:
            self.status = BotStatus.READY
        else:
            self.status = BotStatus.SETUP

    def _refresh_all(self) -> None:
        self._load_config()
        self._update_card_data()
        self._update_header()
        self._rebuild_cards()
        self._update_buttons()

    def _update_buttons(self) -> None:
        if not self.primary_btn:
            return
        if self.status == BotStatus.SETUP:
            self.primary_btn.configure(text="Config", state="normal", fg_color="#1a4030", text_color=GREEN)
        elif self.status == BotStatus.READY:
            self.primary_btn.configure(text="Start Bot", state="normal", fg_color="#1a4030", text_color=GREEN)
        elif self.status == BotStatus.STARTING:
            self.primary_btn.configure(text="Starting\u2026", state="disabled", fg_color=("#444444", "#333333"), text_color=("#888888", "#888888"))
        elif self.status == BotStatus.RUNNING:
            self.primary_btn.configure(text="Stop Bot", state="normal", fg_color="#281414", text_color=RED)
        elif self.status == BotStatus.ERROR:
            self.primary_btn.configure(text="Restart Bot", state="normal", fg_color="#1a4030", text_color=GREEN)

        if self._config:
            tg = self._config.telegram
            self.access_btn.configure(text=f"Access: {'All Chats' if tg.allow_all_chats else 'Private'}")
            self.auto_btn.configure(text=f"Auto: {'On' if tg.autonomous_enabled else 'Off'}")

    def _primary_action(self) -> None:
        if self.status == BotStatus.SETUP:
            self._open_config()
        elif self.status == BotStatus.READY:
            self._action_start()
        elif self.status == BotStatus.RUNNING:
            self._action_stop()
        elif self.status == BotStatus.ERROR:
            self._action_restart()

    def _toggle_access(self) -> None:
        if not self._config:
            return
        self._config.telegram.allow_all_chats = not self._config.telegram.allow_all_chats
        self._save_raw_config()
        self._update_card_data()
        self._rebuild_cards()
        self._update_buttons()

    def _toggle_auto(self) -> None:
        if not self._config:
            return
        self._config.telegram.autonomous_enabled = not self._config.telegram.autonomous_enabled
        self._save_raw_config()
        self._update_buttons()

    def _save_raw_config(self) -> None:
        import yaml
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        telegram = raw.setdefault("telegram", {})
        if self._config:
            telegram["allow_all_chats"] = self._config.telegram.allow_all_chats
            telegram["autonomous_enabled"] = self._config.telegram.autonomous_enabled
        self.config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=False), encoding="utf-8")

    def _action_start(self) -> None:
        self.status = BotStatus.STARTING
        self._refresh_all()
        t = threading.Thread(target=self._start_thread, daemon=True)
        t.start()

    def _start_thread(self) -> None:
        try:
            start_telegram_bot(self.config_path)
            self.root.after(1500, self._on_start_ok)
        except RuntimeError as exc:
            self.root.after(0, lambda: self._on_start_failed(str(exc)))
        except Exception as exc:
            self.root.after(0, lambda: self._on_start_failed(str(exc)))

    def _on_start_ok(self) -> None:
        self._log_lines.append("[Telegram Bot started successfully]")
        self._test_ok = True
        self._determine_status()
        self._refresh_all()

    def _on_start_failed(self, msg: str) -> None:
        self._log_lines.append(f"[Start failed: {msg}]")
        self._last_msg = msg
        self.status = BotStatus.ERROR
        self._refresh_all()

    def _action_stop(self) -> None:
        t = threading.Thread(target=self._stop_thread, daemon=True)
        t.start()

    def _stop_thread(self) -> None:
        try:
            stop_telegram_bot(self.config_path)
            self.root.after(1000, self._on_stop_ok)
        except Exception as exc:
            self.root.after(0, lambda: self._on_stop_failed(str(exc)))

    def _on_stop_ok(self) -> None:
        self._log_lines.append("[Telegram Bot stopped]")
        self._determine_status()
        self._refresh_all()

    def _on_stop_failed(self, msg: str) -> None:
        self._log_lines.append(f"[Stop failed: {msg}]")
        self._last_msg = msg
        self.status = BotStatus.ERROR
        self._refresh_all()

    def _action_restart(self) -> None:
        self.status = BotStatus.STARTING
        self._refresh_all()
        t = threading.Thread(target=self._restart_thread, daemon=True)
        t.start()

    def _restart_thread(self) -> None:
        try:
            restart_telegram_bot(self.config_path)
            self.root.after(2000, self._on_start_ok)
        except Exception as exc:
            self.root.after(0, lambda: self._on_start_failed(str(exc)))

    def _poll_status(self) -> None:
        if self._stopped.is_set():
            return
        try:
            status_info = telegram_bot_status(self.config_path)
            was_running = self.status == BotStatus.RUNNING
            now_running = status_info["running"]
            if was_running and not now_running:
                self.status = BotStatus.ERROR
                self._last_msg = "Bot stopped unexpectedly"
                self._log_lines.append("[Bot stopped unexpectedly]")
                self._refresh_all()
            elif not was_running and now_running:
                self.status = BotStatus.RUNNING
                self._refresh_all()
        except Exception:
            pass
        self.root.after(5000, self._poll_status)

    def _auto_test_token(self) -> None:
        if self._test_ok:
            return
        token = None
        try:
            from .config import load_config
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

    def _open_config(self) -> None:
        def _on_test_ok(username: str, bot_id: str) -> None:
            self._bot_username = username
            self._bot_id = bot_id
            self._test_ok = True
            self._update_card_data()
            self._rebuild_cards()
            self._update_header()
            self._update_buttons()
        ConfigDialog(self.root, self.config_path, self._on_config_saved, on_test_ok=_on_test_ok)

    def _on_config_saved(self) -> None:
        self._load_config()
        self._determine_status()
        self._refresh_all()

    def _open_access(self) -> None:
        AccessDialog(self.root, self.config_path, self._on_config_saved)

    def _open_tools(self) -> None:
        ToolsDialog(self.root, self.config_path, self._on_config_saved)

    def _open_force_message(self) -> None:
        ForceMessageDialog(self.root, self.config_path)

    def _open_logs(self) -> None:
        LogsDialog(self.root, self.config_path, lambda: follow_telegram_log(self.config_path))

    def _open_details(self) -> None:
        DetailsDialog(self.root, self.config_path)


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
        import yaml
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
        import yaml
        self.telegram_raw["enabled"] = bool(self.enabled_var.get())
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
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
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
        import yaml
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
        import yaml
        self.telegram_raw["allow_all_chats"] = bool(self.allow_all_var.get())
        self.telegram_raw["allowed_chat_ids"] = [x.strip() for x in self.allowed_text.get("0.0", "end-1c").split("\n") if x.strip()]
        self.telegram_raw["owner_chat_ids"] = [x.strip() for x in self.owner_text.get("0.0", "end-1c").split("\n") if x.strip()]
        self.telegram_raw["admin_chat_ids"] = [x.strip() for x in self.admin_text.get("0.0", "end-1c").split("\n") if x.strip()]
        self.telegram_raw["core_editing_enabled"] = bool(self.core_edit_var.get())
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
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
        import yaml
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
        scroll._scrollbar.grid_remove()

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
        scroll._scrollbar.grid_remove()

        from .config import ToolPolicy as Tp
        tp_default = Tp()
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
        import yaml
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
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
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
        import yaml
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
        self.dialog.transient(parent)
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
                if self._filter == "access" and "rejected" not in lower and "allow" not in lower:
                    continue
                if self._filter == "tools" and "tool" not in lower:
                    continue
                if self._filter == "provider" and "provider" not in lower:
                    continue
                if self._filter == "sends" and "sent" not in lower:
                    continue
            self.text.insert("end", line + "\n")
        self.text.see("end")

    def _clear(self) -> None:
        self.text.delete("0.0", "end")

    def _close(self) -> None:
        self.dialog.destroy()


class _DetailItem:
    def __init__(self, label: str, display_value: str, full_value: str) -> None:
        self.label = label
        self.display_value = display_value
        self.full_value = full_value


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

        import tkinter as tk
        from tkinter import ttk

        main = ctk.CTkFrame(self.dialog, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        self.items = self._build_items()

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

    def _build_items(self) -> list[_DetailItem]:
        import yaml
        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}
        telegram_raw = raw.get("telegram", {}) or {}
        items: list[_DetailItem] = []

        from .telegram_launcher import telegram_bot_status
        status_info = telegram_bot_status(self.config_path)

        cfg_path = str(self.config_path)
        items.append(_DetailItem("Config path", _compact_path(cfg_path), cfg_path))

        items.append(_DetailItem("Enabled", str(telegram_raw.get("enabled", False)),
                                 str(telegram_raw.get("enabled", False))))

        token = telegram_raw.get("bot_token", "")
        token_status = "set" if (token and not str(token).startswith("${")) else "missing"
        items.append(_DetailItem("Bot token", token_status, token_status))

        items.append(_DetailItem("Provider", str(telegram_raw.get("provider", "-")),
                                 str(telegram_raw.get("provider", "-"))))
        items.append(_DetailItem("Model", str(telegram_raw.get("model", "-")),
                                 str(telegram_raw.get("model", "-"))))
        items.append(_DetailItem("Allow all chats", str(telegram_raw.get("allow_all_chats", False)),
                                 str(telegram_raw.get("allow_all_chats", False))))

        allowed = telegram_raw.get("allowed_chat_ids", [])
        items.append(_DetailItem("Allowed chats", str(len(allowed)),
                                 ", ".join(str(x) for x in allowed) if allowed else "none"))
        owners = telegram_raw.get("owner_chat_ids", [])
        items.append(_DetailItem("Owners", str(len(owners)),
                                 ", ".join(str(x) for x in owners) if owners else "none"))
        admins = telegram_raw.get("admin_chat_ids", [])
        items.append(_DetailItem("Admins", str(len(admins)),
                                 ", ".join(str(x) for x in admins) if admins else "none"))

        items.append(_DetailItem("Max input chars", str(telegram_raw.get("max_input_chars", 4000)),
                                 str(telegram_raw.get("max_input_chars", 4000))))
        items.append(_DetailItem("Max output tokens", str(telegram_raw.get("max_output_tokens", 512)),
                                 str(telegram_raw.get("max_output_tokens", 512))))
        items.append(_DetailItem("Poll interval", f"{telegram_raw.get('poll_interval_seconds', 2.0)}s",
                                 str(telegram_raw.get('poll_interval_seconds', 2.0))))
        items.append(_DetailItem("Response timeout", f"{telegram_raw.get('response_timeout_seconds', 180.0)}s",
                                 str(telegram_raw.get('response_timeout_seconds', 180.0))))
        items.append(_DetailItem("Auto enabled", str(telegram_raw.get("autonomous_enabled", True)),
                                 str(telegram_raw.get("autonomous_enabled", True))))
        items.append(_DetailItem("Self evolution", str(telegram_raw.get("self_evolution_enabled", True)),
                                 str(telegram_raw.get("self_evolution_enabled", True))))
        items.append(_DetailItem("Core editing", str(telegram_raw.get("core_editing_enabled", False)),
                                 str(telegram_raw.get("core_editing_enabled", False))))

        items.append(_DetailItem("Runtime running", str(status_info.get("running", False)),
                                 str(status_info.get("running", False))))
        pid = status_info.get("pid")
        items.append(_DetailItem("Runtime PID", str(pid) if pid else "none",
                                 str(pid) if pid else "none"))

        tp = telegram_raw.get("tool_policy", {})
        ai_auto = tp.get("ai_auto_tools", [])
        cmd_tools = tp.get("command_tools", [])
        blocked = tp.get("blocked_tools", [])
        items.append(_DetailItem("AI auto tools", str(len(ai_auto)),
                                 ", ".join(ai_auto) if ai_auto else "none"))
        items.append(_DetailItem("Command tools", str(len(cmd_tools)),
                                 ", ".join(cmd_tools) if cmd_tools else "none"))
        items.append(_DetailItem("Blocked tools", str(len(blocked)),
                                 ", ".join(blocked) if blocked else "none"))

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
