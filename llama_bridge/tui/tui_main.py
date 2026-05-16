"""
Llama Bridge Main TUI - Redesigned dark theme dashboard
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Static

from .widgets import PanelHeader, StatusBar


class ServicePanel(Container):
    """Reusable panel for bot and server status."""

    DEFAULT_CSS = """
    ServicePanel {
        border: solid #1f2535;
        background: #0e1118;
        padding: 0;
        min-width: 40;
        height: 14;
    }
    """

    def __init__(self, service_name: str, **kwargs):
        super().__init__(**kwargs)
        self.service_name = service_name

    is_running = reactive(False)
    status_detail = reactive("")
    uptime = reactive("")
    messages = reactive(0)
    model = reactive("")

    def compose(self) -> ComposeResult:
        safe_name = self.service_name.replace(" ", "_").lower()
        with Vertical():
            yield Static(
                f"[#a6e3a1]●[/#a6e3a1] {self.service_name.upper()}",
                classes="panel-header",
                markup=True
            )
            yield Static("─" * 40, classes="panel-divider")
            with Vertical(classes="panel-content"):
                yield Static("STATUS  ", id=f"{safe_name}-status")
                yield Static("UPTIME  ", id=f"{safe_name}-uptime")
                if self.service_name == "Telegram Bot":
                    yield Static("MESSAGES", id=f"{safe_name}-messages")
                else:
                    yield Static("MODEL   ", id=f"{safe_name}-model")
                    yield Static("URL     ", id=f"{safe_name}-url")
                with Horizontal(classes="service-buttons"):
                    yield Button("Start", id=f"btn-{safe_name}-start", variant="primary")
                    yield Button("Stop", id=f"btn-{safe_name}-stop", variant="error")

    def watch_is_running(self, running: bool) -> None:
        """Update display when running state changes."""
        safe_name = self.service_name.replace(" ", "_").lower()
        status = self.query_one(f"#{safe_name}-status", Static)
        dot = f"[#a6e3a1]●[/#a6e3a1]" if running else f"[#f38ba8]○[/#f38ba8]"
        label = "Running" if running else "Stopped"
        status.update(f"{dot}  {label}", markup=True)


class ActivityLogPanel(Container):
    """Panel showing recent activity log entries."""

    DEFAULT_CSS = """
    ActivityLogPanel {
        border: solid #1f2535;
        background: #0e1118;
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📋 ACTIVITY LOG", classes="panel-header", markup=True)
            yield Static("─" * 120, classes="panel-divider")
            yield Static("", classes="log-content", id="activity-log-content")

    def set_log_lines(self, lines: list[str]) -> None:
        """Update the log display with new lines."""
        content = self.query_one("#activity-log-content", Static)
        if not lines:
            content.update("  No log entries found.")
            return

        # Show last 30 lines with simple formatting
        recent_lines = lines[-30:] if len(lines) > 30 else lines
        formatted = []
        for line in recent_lines[-15:]:  # Show last 15 in the panel
            if len(line) > 80:
                line = line[:77] + "..."
            formatted.append(line)

        content.update("\n".join(formatted))


class MainTUIApp(App):
    """Main TUI application for Llama Bridge dashboard."""

    TITLE = "Llama Bridge"
    CSS_PATH = "styles/main.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("b", "toggle_bot", "Toggle Bot", show=True),
        Binding("s", "toggle_server", "Toggle Server", show=True),
        Binding("l", "open_logs", "Logs", show=True),
        Binding("c", "show_config", "Config", show=True),
        Binding("?", "toggle_help", "Help", show=True),
    ]

    def __init__(self, config_path: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path or self._get_default_config_path()
        self.bot_running = False
        self.server_running = False
        self.server_url = "http://localhost:8080"
        self.bot_start_time: Optional[datetime] = None
        self.server_start_time: Optional[datetime] = None
        self.bot_messages = 0
        self.model = "llama3.2:3b"

    def _get_default_config_path(self) -> Path:
        """Get the default config path."""
        return Path.home() / ".llama" / "config.yml"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)

        with Vertical():
            # Title bar with Rich markup
            yield Static(
                "[bold #89b4fa]🦙 LLAMA BRIDGE[/]  [dim]│[/]  [white]v0.1.0[/]  [dim]│[/]  [dim]{}[/]".format(self.config_path),
                classes="title-bar",
                markup=True
            )

            # Main panels
            with Horizontal():
                yield ServicePanel("Telegram Bot", id="telegram-panel")
                yield ServicePanel("Llama Server", id="llama-panel")

            # Activity log
            yield ActivityLogPanel(id="activity-panel")

            # Key binding bar with highlighted key letters
            yield Static(
                " [bold #89b4fa]q[/]:Quit  [bold #89b4fa]r[/]:Refresh  [bold #89b4fa]b[/]:Bot  "
                "[bold #89b4fa]s[/]:Server  [bold #89b4fa]l[/]:Logs  [bold #89b4fa]c[/]:Config  "
                "[bold #89b4fa]?[/]:Help",
                classes="key-binding-bar",
                markup=True
            )

        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.set_interval(5, self._check_status)

    async def _check_status(self) -> None:
        """Check the status of bot and server."""
        try:
            loop = asyncio.get_event_loop()
            from ..cli import _server_is_running

            def _check():
                if self.config_path and self.config_path.exists():
                    config_dir = self.config_path.parent
                else:
                    config_dir = Path.home() / ".llama"
                pid_path = config_dir / "llama.pid"
                return _server_is_running(self.config_path, pid_path)

            running, url = await loop.run_in_executor(None, _check)
            self.server_running = running
            if url:
                self.server_url = url
        except ImportError:
            # Graceful degradation if CLI functions not available
            pass
        except Exception:
            pass

        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current status."""
        # Update Telegram panel
        telegram = self.query_one("#telegram-panel", ServicePanel)
        telegram.is_running = self.bot_running

        # Update uptime
        if self.bot_start_time:
            delta = datetime.now() - self.bot_start_time
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            telegram_uptime = f"{hours}h {minutes}m"
        else:
            telegram_uptime = "--"

        telegram_uptime_static = self.query_one("#telegram_bot-uptime", Static)
        telegram_uptime_static.update(f"{telegram_uptime}", markup=True)

        telegram_messages = self.query_one("#telegram_bot-messages", Static)
        telegram_messages.update(f"{self.bot_messages}", markup=True)

        # Update Llama panel
        llama = self.query_one("#llama-panel", ServicePanel)
        llama.is_running = self.server_running

        # Update uptime
        if self.server_start_time:
            delta = datetime.now() - self.server_start_time
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            llama_uptime = f"{hours}h {minutes}m"
        else:
            llama_uptime = "--"

        llama_uptime_static = self.query_one("#llama_server-uptime", Static)
        llama_uptime_static.update(f"{llama_uptime}", markup=True)

        llama_model = self.query_one("#llama_server-model", Static)
        llama_model.update(f"{self.model}", markup=True)

        llama_url = self.query_one("#llama_server-url", Static)
        llama_url.update(f"{self.server_url}" if self.server_running else "stopped", markup=True)

        # Update activity log
        self._load_activity_log()

    def _load_activity_log(self) -> None:
        """Load recent activity from log file."""
        try:
            if self.config_path and self.config_path.exists():
                log_path = self.config_path.parent / "llama.log"
            else:
                log_path = Path.home() / ".llama" / "llama.log"

            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-50:]
                activity = self.query_one("#activity-panel", ActivityLogPanel)
                activity.set_log_lines(lines)
        except Exception:
            pass

    async def _load_activity_log_async(self) -> None:
        """Load recent activity from log file (async)."""
        try:
            loop = asyncio.get_event_loop()

            def _read():
                if self.config_path and self.config_path.exists():
                    log_path = self.config_path.parent / "llama.log"
                else:
                    log_path = Path.home() / ".llama" / "llama.log"

                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8") as f:
                        return f.readlines()[-50:]
                return []

            lines = await loop.run_in_executor(None, _read)
            activity = self.query_one("#activity-panel", ActivityLogPanel)
            activity.set_log_lines(lines)
        except Exception:
            pass

    async def action_refresh(self) -> None:
        """Refresh the status."""
        await self._check_status()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_toggle_bot(self) -> None:
        """Toggle bot running state."""
        self.bot_running = not self.bot_running
        if self.bot_running:
            self.bot_start_time = datetime.now()
        else:
            self.bot_start_time = None
        self._update_display()

    def action_toggle_server(self) -> None:
        """Toggle server running state."""
        self.server_running = not self.server_running
        if self.server_running:
            self.server_start_time = datetime.now()
        else:
            self.server_start_time = None
        self._update_display()

    def action_show_config(self) -> None:
        """Show config path."""
        self.notify(f"Config: {self.config_path}")

    def action_toggle_help(self) -> None:
        """Toggle help overlay."""
        self.notify("Keys: q=Quit, r=Refresh, b=Bot, s=Server, l=Logs, c=Config, ?=Help")

    def action_open_logs(self) -> None:
        """Open logs TUI."""
        try:
            from .. import tui as tui_module
            self.exit()
            tui_module.run_logs_tui(config_path=self.config_path)
        except Exception as e:
            self.notify(f"Error opening logs: {e}", severity="error")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-telegram_bot-start":
            self.bot_running = True
            self.bot_start_time = datetime.now()
            self.notify("Starting Telegram bot...")
        elif button_id == "btn-telegram_bot-stop":
            self.bot_running = False
            self.bot_start_time = None
            self.notify("Stopping Telegram bot...")
        elif button_id == "btn-llama_server-start":
            await self._start_server()
        elif button_id == "btn-llama_server-stop":
            await self._stop_server()

        self._update_display()

    async def _start_server(self) -> None:
        """Start the Llama server."""
        try:
            from ..cli import _cmd_start

            def _do_start():
                if self.config_path and self.config_path.exists():
                    config_dir = self.config_path.parent
                else:
                    config_dir = Path.home() / ".llama"

                pid_path = config_dir / "llama.pid"
                log_path = config_dir / "llama.log"
                _cmd_start(self.config_path, pid_path, log_path, 0)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do_start)
            self.server_start_time = datetime.now()
            self.server_running = True
            self.notify("Server started")
        except Exception as e:
            self.notify(f"Error starting server: {e}", severity="error")

    async def _stop_server(self) -> None:
        """Stop the Llama server."""
        try:
            from ..cli import _cmd_stop

            def _do_stop():
                if self.config_path and self.config_path.exists():
                    config_dir = self.config_path.parent
                else:
                    config_dir = Path.home() / ".llama"

                pid_path = config_dir / "llama.pid"
                _cmd_stop(pid_path)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do_stop)
            self.server_running = False
            self.server_start_time = None
            self.notify("Server stopped")
        except Exception as e:
            self.notify(f"Error stopping server: {e}", severity="error")


def run_main_tui(config_path: Path | None = None) -> None:
    """Run the main TUI application."""
    app = MainTUIApp(config_path=config_path)
    app.run()