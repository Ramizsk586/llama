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
    "WIN_W": 608,
    "WIN_H": 620,
}
PAD = LAYOUT["PAD"]


class BotStatus(Enum):
    SETUP = auto()
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
        self.config_path = config_path
        self._stopped = threading.Event()

        W, H = LAYOUT["WIN_W"], LAYOUT["WIN_H"]
        self.root = tk.Tk()
        self.root.title("Llama Bridge - Telegram Bot Setup")
        _center_window(self.root, W, H)
        self.root.minsize(560, 520)
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        _set_dark_titlebar(self.root)

        self.status = BotStatus.SETUP
        self._config = None
        self._last_msg = ""
        self._log_lines: list[str] = []
        self._bot_username: str | None = None
        self._bot_id: str | None = None
        self._test_ok = False

        self.header_canvas: tk.Canvas | None = None
        self.cards_canvas: tk.Canvas | None = None
        self.status_label: tk.Label | None = None
        self.primary_btn: tk.Button | None = None
        self.access_btn: tk.Button | None = None
        self.auto_btn: tk.Button | None = None
        self.util_btns: dict[str, tk.Button] = {}

        self._card_data = [
            {"key": "token", "title": "Bot Token", "ok": False, "subtitle": "Missing", "status": "Missing"},
            {"key": "api", "title": "Telegram API", "ok": False, "subtitle": "Not tested", "status": "Test"},
            {"key": "model", "title": "AI Model", "ok": False, "subtitle": "Missing", "status": "Missing"},
            {"key": "access", "title": "Access & Tools", "ok": False, "subtitle": "Not configured", "status": "Warn"},
        ]

        self._build_ui()
        self._load_config()
        self._refresh_all()
        self.root.after(100, self._auto_test_token)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(2000, self._poll_status)

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        self.header_canvas = tk.Canvas(main, bg=BG, highlightthickness=0)
        self.header_canvas.pack(fill="x", side="top")
        self.header_canvas.configure(height=LAYOUT["HEADER_H"])

        self.cards_canvas = tk.Canvas(main, bg=BG, highlightthickness=0)
        self.cards_canvas.pack(fill="both", expand=True, side="top")

        self.status_label = tk.Label(main, text="", bg=BG, fg=MUTED, font=("Segoe UI", 9), anchor="w", padx=PAD)
        self.status_label.pack(fill="x", side="top")

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
            ("btn_config", "\u2699 Config", self._open_config),
            ("btn_access", "\U0001f511 Access", self._open_access),
            ("btn_tools", "\U0001f527 Tools", self._open_tools),
            ("btn_force", "\U0001f4e9 Force Msg", self._open_force_message),
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
            tooltip_map = {
                "btn_config": "Open configuration dialog",
                "btn_access": "Manage chat access control",
                "btn_tools": "Configure commands and AI tool policy",
                "btn_force": "Force-send a message to a chat",
                "btn_logs": "View Telegram bot logs",
                "btn_details": "Show full technical details",
            }
            ToolTip(btn, tooltip_map[key])

        action = tk.Frame(footer, bg=BG)
        action.pack(fill="x", padx=PAD, pady=(4, 0))

        self.access_btn = tk.Button(
            action, text="Access: Private", command=self._toggle_access,
            bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=16, pady=8,
            activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
            cursor="hand2",
        )
        self.access_btn.pack(side="left", padx=(0, 8))
        ToolTip(self.access_btn, "Toggle between private and all-chats access")

        self.auto_btn = tk.Button(
            action, text="Auto: Off", command=self._toggle_auto,
            bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=16, pady=8,
            activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
            cursor="hand2",
        )
        self.auto_btn.pack(side="left")
        ToolTip(self.auto_btn, "Toggle autonomous mode on/off")

        self.primary_btn = tk.Button(
            action, text="Start Bot", command=self._primary_action,
            bg="#1A4030", fg=GREEN, font=("Segoe UI", 11, "bold"),
            relief="flat", bd=0, padx=24, pady=8,
            activebackground="#153828", activeforeground=GREEN,
            cursor="hand2",
        )
        self.primary_btn.pack(side="right")
        ToolTip(self.primary_btn, "Start or stop the Telegram bot")

    def _draw_header(self) -> None:
        cv = self.header_canvas
        if not cv:
            return
        cv.delete("all")
        w = cv.winfo_width() or LAYOUT["WIN_W"]
        h = LAYOUT["HEADER_H"]
        p = PAD

        cv.create_text(p, 16, anchor="nw", text="Telegram Bot Setup Center",
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

    def _get_separator_color(self) -> str:
        s = self.status
        if s == BotStatus.ERROR:
            return RED
        if s == BotStatus.RUNNING:
            return GREEN
        if s == BotStatus.READY:
            return GREEN
        return BORDER

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
        self._draw_header()
        self._draw_cards()
        self._update_buttons()

    def _update_buttons(self) -> None:
        if not self.primary_btn:
            return
        if self.status == BotStatus.SETUP:
            self.primary_btn.configure(text="Config", state="normal", bg="#1A4030", fg=GREEN)
        elif self.status == BotStatus.READY:
            self.primary_btn.configure(text="Start Bot", state="normal", bg="#1A4030", fg=GREEN)
        elif self.status == BotStatus.STARTING:
            self.primary_btn.configure(text="Starting\u2026", state="disabled", bg=SURFACE3, fg=MUTED)
        elif self.status == BotStatus.RUNNING:
            self.primary_btn.configure(text="Stop Bot", state="normal", bg=CARD_RED, fg=RED)
        elif self.status == BotStatus.ERROR:
            self.primary_btn.configure(text="Restart Bot", state="normal", bg="#1A4030", fg=GREEN)

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
        self._draw_cards()
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
            self._draw_cards()
            self._draw_header()
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
    def __init__(self, parent: tk.Widget, config_path: Path, on_save=None, on_test_ok=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.on_test_ok = on_test_ok
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Telegram Config")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("520x560")
        self.dialog.minsize(460, 400)
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self._load()
        self._build()

    def _load(self) -> None:
        import yaml
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.telegram_raw = self.raw.setdefault("telegram", {})

    def _add_label(self, parent, row, text):
        tk.Label(parent, text=text, bg=BG, fg=TEXT, font=("Segoe UI", 9), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))

    def _build(self) -> None:
        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        row = 0
        self.enabled_var = tk.BooleanVar(value=bool(self.telegram_raw.get("enabled", False)))
        tk.Checkbutton(main, text="Enable Telegram bot", variable=self.enabled_var,
                       bg=BG, fg=TEXT, selectcolor=SURFACE2, activebackground=BG).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        self._add_label(main, row, "Bot Token")
        token_frame = tk.Frame(main, bg=BG)
        token_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        self.token_entry = tk.Entry(token_frame, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                     relief="flat", bd=0, highlightbackground=BORDER, highlightthickness=1,
                                     show="*")
        self.token_entry.pack(side="left", fill="x", expand=True)
        self.token_entry.insert(0, str(self.telegram_raw.get("bot_token", "")))
        self.show_token_var = tk.BooleanVar(value=False)
        show_cb = tk.Checkbutton(token_frame, text="Show", variable=self.show_token_var,
                                  bg=BG, fg=MUTED, selectcolor=SURFACE2, activebackground=BG,
                                  command=self._toggle_token_show)
        show_cb.pack(side="left", padx=(4, 0))
        row += 1

        test_btn = tk.Button(main, text="Test Token", command=self._test_token,
                             bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 9, "bold"),
                             relief="flat", bd=0, padx=12, pady=4,
                             activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
                             cursor="hand2")
        test_btn.grid(row=row, column=1, sticky="w", pady=2)
        self.test_result_var = tk.StringVar(value="")
        tk.Label(main, textvariable=self.test_result_var, bg=BG, fg=MUTED, font=("Segoe UI", 9)).grid(
            row=row, column=2, sticky="w", padx=(8, 0))
        row += 1

        provider_names = list(self.raw.get("providers", {}).keys())
        self._add_label(main, row, "Provider")
        self.provider_var = tk.StringVar(
            value=str(self.telegram_raw.get("provider", provider_names[0] if provider_names else "")))
        ttk.Combobox(main, textvariable=self.provider_var, values=provider_names,
                     state="readonly", width=28).grid(row=row, column=1, columnspan=2, sticky="w", pady=2)
        row += 1

        self._add_label(main, row, "Model")
        self.model_entry = tk.Entry(main, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                     relief="flat", bd=0, highlightbackground=BORDER, highlightthickness=1)
        self.model_entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        self.model_entry.insert(0, str(self.telegram_raw.get("model", "")))
        row += 1

        self._add_label(main, row, "System Prompt")
        self.prompt_text = tk.Text(main, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                    relief="flat", bd=0, highlightbackground=BORDER, highlightthickness=1,
                                    height=4)
        self.prompt_text.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        self.prompt_text.insert("1.0", str(self.telegram_raw.get("system_prompt", "")))
        row += 1

        self._add_label(main, row, "Max Input Chars")
        self.input_chars_var = tk.StringVar(value=str(self.telegram_raw.get("max_input_chars", 4000)))
        tk.Spinbox(main, from_=200, to=16000, textvariable=self.input_chars_var,
                   bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE2, relief="flat",
                   highlightbackground=BORDER, highlightthickness=1, width=10).grid(
            row=row, column=1, sticky="w", pady=2)
        row += 1

        self._add_label(main, row, "Max Output Tokens")
        self.output_tokens_var = tk.StringVar(value=str(self.telegram_raw.get("max_output_tokens", 512)))
        tk.Spinbox(main, from_=64, to=8192, textvariable=self.output_tokens_var,
                   bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE2, relief="flat",
                   highlightbackground=BORDER, highlightthickness=1, width=10).grid(
            row=row, column=1, sticky="w", pady=2)
        row += 1

        self._add_label(main, row, "Poll Interval (s)")
        self.poll_interval_var = tk.StringVar(value=str(self.telegram_raw.get("poll_interval_seconds", 2.0)))
        tk.Spinbox(main, from_=0.5, to=30, increment=0.5, textvariable=self.poll_interval_var,
                   bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE2, relief="flat",
                   highlightbackground=BORDER, highlightthickness=1, width=10).grid(
            row=row, column=1, sticky="w", pady=2)
        row += 1

        self._add_label(main, row, "Response Timeout (s)")
        self.resp_timeout_var = tk.StringVar(value=str(self.telegram_raw.get("response_timeout_seconds", 180.0)))
        tk.Spinbox(main, from_=10, to=600, increment=10, textvariable=self.resp_timeout_var,
                   bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE2, relief="flat",
                   highlightbackground=BORDER, highlightthickness=1, width=10).grid(
            row=row, column=1, sticky="w", pady=2)
        row += 1

        btn_row = tk.Frame(main, bg=BG)
        btn_row.grid(row=row + 1, column=0, columnspan=3, pady=(12, 0))
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

    def _toggle_token_show(self) -> None:
        self.token_entry.configure(show="" if self.show_token_var.get() else "*")

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
        self.telegram_raw["system_prompt"] = self.prompt_text.get("1.0", "end-1c").strip()
        self.telegram_raw["max_input_chars"] = int(self.input_chars_var.get())
        self.telegram_raw["max_output_tokens"] = int(self.output_tokens_var.get())
        self.telegram_raw["poll_interval_seconds"] = float(self.poll_interval_var.get())
        self.telegram_raw["response_timeout_seconds"] = float(self.resp_timeout_var.get())
        self.config_path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
        if self.on_save:
            self.on_save()
        self.dialog.destroy()


class AccessDialog:
    def __init__(self, parent: tk.Widget, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Access Control")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("540x500")
        self.dialog.minsize(460, 400)
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
        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        row = 0
        self.allow_all_var = tk.BooleanVar(value=bool(self.telegram_raw.get("allow_all_chats", False)))
        tk.Checkbutton(main, text="Allow all chats (WARNING: open to everyone)", variable=self.allow_all_var,
                       bg=BG, fg=RED, selectcolor=SURFACE2, activebackground=BG).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        tk.Label(main, text="Allowed Chat IDs", bg=BG, fg=TEXT_BRIGHT,
                 font=("Segoe UI", 10, "bold"), anchor="w").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        self.allowed_text = tk.Text(main, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                     height=4, relief="flat", bd=0,
                                     highlightbackground=BORDER, highlightthickness=1)
        self.allowed_text.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        self.allowed_text.insert("1.0", "\n".join(str(x) for x in (self.telegram_raw.get("allowed_chat_ids") or [])))
        row += 1

        tk.Label(main, text="Owner Chat IDs", bg=BG, fg=TEXT_BRIGHT,
                 font=("Segoe UI", 10, "bold"), anchor="w").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        self.owner_text = tk.Text(main, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                   height=3, relief="flat", bd=0,
                                   highlightbackground=BORDER, highlightthickness=1)
        self.owner_text.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        self.owner_text.insert("1.0", "\n".join(str(x) for x in (self.telegram_raw.get("owner_chat_ids") or [])))
        row += 1

        tk.Label(main, text="Admin Chat IDs", bg=BG, fg=TEXT_BRIGHT,
                 font=("Segoe UI", 10, "bold"), anchor="w").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        self.admin_text = tk.Text(main, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                   height=3, relief="flat", bd=0,
                                   highlightbackground=BORDER, highlightthickness=1)
        self.admin_text.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        self.admin_text.insert("1.0", "\n".join(str(x) for x in (self.telegram_raw.get("admin_chat_ids") or [])))
        row += 1

        tk.Label(main, text="To get your chat ID, send /myid to the bot in Telegram.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        self.core_edit_var = tk.BooleanVar(value=bool(self.telegram_raw.get("core_editing_enabled", False)))
        tk.Checkbutton(main, text="Core editing enabled", variable=self.core_edit_var,
                       bg=BG, fg=TEXT, selectcolor=SURFACE2, activebackground=BG).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        btn_row = tk.Frame(main, bg=BG)
        btn_row.grid(row=row, column=0, columnspan=3, pady=(12, 0))
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

        main.columnconfigure(0, weight=1)

    def _save(self) -> None:
        import yaml
        self.telegram_raw["allow_all_chats"] = bool(self.allow_all_var.get())
        self.telegram_raw["allowed_chat_ids"] = [x.strip() for x in self.allowed_text.get("1.0", "end-1c").split("\n") if x.strip()]
        self.telegram_raw["owner_chat_ids"] = [x.strip() for x in self.owner_text.get("1.0", "end-1c").split("\n") if x.strip()]
        self.telegram_raw["admin_chat_ids"] = [x.strip() for x in self.admin_text.get("1.0", "end-1c").split("\n") if x.strip()]
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
    def __init__(self, parent: tk.Widget, config_path: Path, on_save=None) -> None:
        self.config_path = config_path
        self.on_save = on_save
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Command & Tool Policy")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("660x560")
        self.dialog.minsize(560, 480)
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
        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        nb = ttk.Notebook(main)
        nb.pack(fill="both", expand=True)

        cmd_frame = tk.Frame(nb, bg=BG)
        nb.add(cmd_frame, text="Commands")
        self._build_commands(cmd_frame)

        tool_frame = tk.Frame(nb, bg=BG)
        nb.add(tool_frame, text="AI Tools")
        self._build_tools(tool_frame)

        btn_row = tk.Frame(main, bg=BG)
        btn_row.pack(fill="x", pady=(8, 0))
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

    def _build_commands(self, parent: tk.Widget) -> None:
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for i, h in enumerate(["Command", "Enabled", "Visible", "Permission"]):
            tk.Label(scroll_frame, text=h, bg=BG, fg=TEXT_BRIGHT, font=("Segoe UI", 9, "bold")).grid(
                row=0, column=i, padx=4, pady=2)

        self._cmd_widgets = {}
        for idx, cmd in enumerate(_COMMAND_LIST, start=1):
            policy = self.cmd_policy_raw.get(cmd, {})
            if not isinstance(policy, dict):
                policy = {}
            tk.Label(scroll_frame, text=f"/{cmd}", bg=BG, fg=TEXT, font=("Segoe UI", 9)).grid(
                row=idx, column=0, sticky="w", padx=4, pady=1)

            en_var = tk.BooleanVar(value=bool(policy.get("enabled", True)))
            tk.Checkbutton(scroll_frame, variable=en_var, bg=BG, selectcolor=SURFACE2).grid(
                row=idx, column=1, padx=4, pady=1)

            vis_var = tk.BooleanVar(value=bool(policy.get("visible", True)))
            tk.Checkbutton(scroll_frame, variable=vis_var, bg=BG, selectcolor=SURFACE2).grid(
                row=idx, column=2, padx=4, pady=1)

            perm_var = tk.StringVar(value=str(policy.get("permission", "everyone")))
            ttk.Combobox(scroll_frame, textvariable=perm_var, values=_PERMISSION_LEVELS,
                         state="readonly", width=12).grid(row=idx, column=3, padx=4, pady=1)

            self._cmd_widgets[cmd] = (en_var, vis_var, perm_var)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_tools(self, parent: tk.Widget) -> None:
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        from .config import ToolPolicy as Tp
        tp_default = Tp()
        self._ai_auto_var = tk.StringVar(value=", ".join(self.tool_policy_raw.get("ai_auto_tools", tp_default.ai_auto_tools)))
        self._cmd_tools_var = tk.StringVar(value=", ".join(self.tool_policy_raw.get("command_tools", tp_default.command_tools)))
        self._blocked_var = tk.StringVar(value=", ".join(self.tool_policy_raw.get("blocked_tools", tp_default.blocked_tools)))
        self._user_vis_var = tk.StringVar(value=", ".join(self.tool_policy_raw.get("user_visible_tools", tp_default.user_visible_tools)))
        self._req_admin_var = tk.StringVar(value=", ".join(self.tool_policy_raw.get("require_admin_for", tp_default.require_admin_for)))
        self._req_owner_var = tk.StringVar(value=", ".join(self.tool_policy_raw.get("require_owner_for", tp_default.require_owner_for)))

        sections = [
            ("AI Auto Tools", "Tools the AI can call automatically", self._ai_auto_var),
            ("Command-only Tools", "Tools callable only via explicit /tools", self._cmd_tools_var),
            ("Blocked Tools (dangerous)", "Never allowed in Telegram", self._blocked_var),
            ("User-visible Tools", "Shown in /tools list for non-admins", self._user_vis_var),
            ("Require Admin", "These tools need admin role", self._req_admin_var),
            ("Require Owner", "These tools need owner role", self._req_owner_var),
        ]
        for i, (title, sub, var) in enumerate(sections):
            r = i * 2
            tk.Label(scroll_frame, text=title, bg=BG, fg=TEXT_BRIGHT,
                     font=("Segoe UI", 10, "bold"), anchor="w").grid(
                row=r, column=0, sticky="w", pady=(8, 0))
            tk.Label(scroll_frame, text=sub, bg=BG, fg=MUTED, font=("Segoe UI", 8), anchor="w").grid(
                row=r + 1, column=0, sticky="w")
            tk.Entry(scroll_frame, textvariable=var, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                      relief="flat", bd=0, highlightbackground=BORDER, highlightthickness=1).grid(
                row=r + 1, column=0, sticky="ew", padx=(0, 8))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

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
    def __init__(self, parent: tk.Widget, config_path: Path) -> None:
        self.config_path = config_path
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Force Message")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("520x400")
        self.dialog.minsize(460, 340)
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
        main = tk.Frame(self.dialog, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        row = 0
        all_chats = list(set(
            str(x).strip() for x in (
                self.telegram_raw.get("allowed_chat_ids", []) +
                self.telegram_raw.get("owner_chat_ids", []) +
                self.telegram_raw.get("admin_chat_ids", [])
            ) if str(x).strip()
        ))

        tk.Label(main, text="Target Chat ID", bg=BG, fg=TEXT, font=("Segoe UI", 9), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.chat_var = tk.StringVar(value=all_chats[0] if all_chats else "")
        if all_chats:
            ttk.Combobox(main, textvariable=self.chat_var, values=all_chats, width=28).grid(
                row=row, column=1, sticky="ew", pady=2)
        else:
            tk.Entry(main, textvariable=self.chat_var, bg=SURFACE2, fg=TEXT,
                     insertbackground=TEXT, relief="flat", bd=0,
                     highlightbackground=BORDER, highlightthickness=1).grid(
                row=row, column=1, sticky="ew", pady=2)
        row += 1

        tk.Label(main, text="Message", bg=BG, fg=TEXT, font=("Segoe UI", 9), anchor="w").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        self.msg_text = tk.Text(main, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                 height=6, relief="flat", bd=0,
                                 highlightbackground=BORDER, highlightthickness=1)
        self.msg_text.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        btn_row = tk.Frame(main, bg=BG)
        btn_row.grid(row=row, column=0, columnspan=2, pady=(12, 0))
        tk.Button(btn_row, text="Send", command=self._send,
                  bg=BUTTON_TOP, fg=TEXT_BRIGHT, font=("Segoe UI", 10, "bold"),
                  relief="flat", bd=0, padx=16, pady=6,
                  activebackground=BUTTON_BOTTOM, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="Cancel", command=self.dialog.destroy,
                  bg=BG_ALT, fg=MUTED, font=("Segoe UI", 10),
                  relief="flat", bd=0, padx=16, pady=6,
                  activebackground=SURFACE3, activeforeground=TEXT_BRIGHT,
                  cursor="hand2").pack(side="left")

        self.status_var = tk.StringVar(value="")
        tk.Label(main, textvariable=self.status_var, bg=BG, fg=MUTED, font=("Segoe UI", 9)).grid(
            row=row + 1, column=0, columnspan=2, pady=(4, 0))
        main.columnconfigure(1, weight=1)

    def _send(self) -> None:
        chat_id = self.chat_var.get().strip()
        text = self.msg_text.get("1.0", "end-1c").strip()
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
    def __init__(self, parent: tk.Widget, config_path: Path, log_fetcher=None) -> None:
        self.config_path = config_path
        self.log_fetcher = log_fetcher or (lambda: follow_telegram_log(config_path))
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Telegram Bot Logs")
        self.dialog.configure(bg=BG)
        self.dialog.geometry("720x520")
        self.dialog.minsize(560, 360)
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self._filter = "all"
        self._build()

    def _build(self) -> None:
        filter_row = tk.Frame(self.dialog, bg=BG)
        filter_row.pack(fill="x", padx=12, pady=(8, 4))
        for f in ["all", "errors", "access", "tools", "provider", "sends"]:
            btn = tk.Button(filter_row, text=f, command=lambda lbl=f: self._set_filter(lbl),
                            bg=BG_ALT, fg=MUTED, font=("Segoe UI", 9),
                            relief="flat", bd=0, padx=8, pady=2,
                            activebackground=SURFACE3, activeforeground=TEXT_BRIGHT,
                            cursor="hand2")
            btn.pack(side="left", padx=(0, 4))

        self.text = tk.Text(self.dialog, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                             font=("Consolas", 9), relief="flat", bd=0,
                             highlightbackground=BORDER, highlightthickness=1)
        self.text.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.text.config(state="disabled")
        self.text.bind("<MouseWheel>", lambda e: self.text.yview_scroll(-1 * (e.delta // 120), "units"))
        self.text.tag_configure("error", foreground=RED)
        self.text.tag_configure("warn", foreground=YELLOW)
        self.text.tag_configure("info", foreground=GREEN)
        self.text.tag_configure("muted", foreground=MUTED)
        self._refresh()

    def _set_filter(self, label: str) -> None:
        self._filter = label
        self._refresh()

    def _refresh(self) -> None:
        lines = self.log_fetcher() if self.log_fetcher else []
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
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
        self.dialog.after(3000, self._refresh)


class _DetailItem:
    def __init__(self, label: str, display_value: str, full_value: str) -> None:
        self.label = label
        self.display_value = display_value
        self.full_value = full_value


class DetailsDialog:
    def __init__(self, parent: tk.Widget, config_path: Path) -> None:
        self.config_path = config_path
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Telegram Bot Details")
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

        # Runtime status
        items.append(_DetailItem("Runtime running", str(status_info.get("running", False)),
                                 str(status_info.get("running", False))))
        pid = status_info.get("pid")
        items.append(_DetailItem("Runtime PID", str(pid) if pid else "none",
                                 str(pid) if pid else "none"))

        # Available tools
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
