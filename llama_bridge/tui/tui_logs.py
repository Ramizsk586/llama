"""
Llama Bridge Logs TUI - Redesigned dark theme log viewer
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Static, TextInput

from .widgets import StatusBar, parse_log_line
from rich.text import Text


class FilterBar(Container):
    """Inline filter tabs for log filtering."""

    current_filter = reactive("all")

    DEFAULT_CSS = """
    FilterBar {
        height: 1;
        background: #12151c;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("All", id="filter-all", variant="primary")
            yield Button("Errors", id="filter-errors")
            yield Button("200 OK", id="filter-200")
            yield Button("400+", id="filter-400")

    def watch_current_filter(self, filter_name: str) -> None:
        """Update button states when filter changes."""
        for btn_id, variant in [
            ("filter-all", "primary" if filter_name == "all" else "default"),
            ("filter-errors", "primary" if filter_name == "errors" else "default"),
            ("filter-200", "primary" if filter_name == "200" else "default"),
            ("filter-400", "primary" if filter_name == "400" else "default"),
        ]:
            try:
                btn = self.query_one(f"#{btn_id}", Button)
                btn.variant = variant
            except Exception:
                pass


class SearchBar(Container):
    """Search bar for filtering log lines."""

    DEFAULT_CSS = """
    SearchBar {
        height: 1;
        background: #1a1e28;
        border-top: solid #2a2f3d;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static("find:", classes="search-prompt")
            yield TextInput(placeholder="search logs...", classes="search-input", id="search-input")
            yield Button("Clear", id="btn-search-clear", variant="default")


class LogsTUIApp(App):
    """Logs TUI application for viewing Llama Bridge logs."""

    TITLE = "Llama Bridge - Logs"
    CSS_PATH = "styles/logs.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("d", "toggle_dev", "Dev", show=True),
        Binding("n", "toggle_normal", "Normal", show=True),
        Binding("1", "filter_all", "All", show=True),
        Binding("2", "filter_errors", "Errors", show=True),
        Binding("3", "filter_200", "200 OK", show=True),
        Binding("4", "filter_400", "400+", show=True),
        Binding("f", "toggle_search", "Find", show=True),
        Binding("a", "toggle_auto_scroll", "Auto", show=True),
        Binding("g", "scroll_top", "Top", show=True),
        Binding("G", "scroll_bottom", "Bottom", show=True),
        Binding("escape", "clear_filter", "Clear", show=True),
    ]

    def __init__(
        self,
        log_path: Path | None = None,
        dev: bool = False,
        config_path: Path | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.log_path = log_path
        self.dev = dev
        self.config_path = config_path or self._get_default_config_path()
        self.current_panel = "dev" if dev else "normal"
        self.current_filter = "all"
        self.auto_scroll = True
        self.search_active = False
        self.search_term = ""

        # Determine log paths
        if config_path:
            self.dev_log_path = config_path.parent / "llama.dev.log"
            self.normal_log_path = config_path.parent / "llama.log"
        else:
            default_dir = Path.home() / ".llama"
            self.dev_log_path = default_dir / "llama.dev.log"
            self.normal_log_path = default_dir / "llama.log"

        if log_path:
            if dev:
                self.dev_log_path = log_path
            else:
                self.normal_log_path = log_path

        self.log_lines: list[str] = []
        self.filtered_lines: list[str] = []

    def _get_default_config_path(self) -> Path:
        """Get the default config path."""
        return Path.home() / ".llama" / "config.yml"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)

        with Vertical():
            # Title bar with panel switcher
            yield Static(
                f"🦙 LLAMA BRIDGE  ›  LOGS  |  [d] Dev  |  [n] Normal",
                classes="title-bar"
            )

            # Dev logs panel
            with Vertical(id="dev-panel"):
                yield Static("▶ DEV LOGS", classes="panel-header")
                yield FilterBar(id="dev-filter-bar")
                yield Static("", classes="log-content", id="dev-log-content")

            # Normal logs panel (hidden by default)
            with Vertical(id="normal-panel"):
                yield Static("▶ NORMAL LOGS", classes="panel-header")
                yield FilterBar(id="normal-filter-bar")
                yield Static("", classes="log-content", id="normal-log-content")

            # Search bar (hidden by default)
            yield SearchBar(id="search-bar")

            # Stats bar
            yield StatusBar(id="stats-bar")

            # Key binding bar
            yield Static(
                " q:Quit  r:Refresh  d:Dev  n:Normal  1:All  2:Errors  3:200  4:400+  f:Find  a:Auto  g/G:Top/Bottom",
                classes="key-binding-bar"
            )

        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self._load_logs()
        self.set_interval(2, self._load_logs)
        self._update_panel_visibility()

    def _update_panel_visibility(self) -> None:
        """Show/hide panels based on current_panel."""
        dev_panel = self.query_one("#dev-panel", Vertical)
        normal_panel = self.query_one("#normal-panel", Vertical)

        if self.current_panel == "dev":
            dev_panel.display = "block"
            normal_panel.display = "none"
        else:
            dev_panel.display = "none"
            normal_panel.display = "block"

    def _load_logs(self) -> None:
        """Load the log files."""
        if self.current_panel == "dev":
            log_path = self.dev_log_path
        else:
            log_path = self.normal_log_path

        try:
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    self.log_lines = f.readlines()[-1000:]  # Last 1000 lines
            else:
                self.log_lines = []
        except Exception as e:
            self.log_lines = [f"Error reading log: {e}"]

        self._apply_filter()

    def _apply_filter(self) -> None:
        """Apply the current filter to the log lines."""
        if self.current_filter == "all":
            self.filtered_lines = self.log_lines
        elif self.current_filter == "errors":
            self.filtered_lines = [line for line in self.log_lines if self._is_error(line)]
        elif self.current_filter == "200":
            self.filtered_lines = [line for line in self.log_lines if self._is_200(line)]
        elif self.current_filter == "400":
            self.filtered_lines = [line for line in self.log_lines if self._is_400(line)]

        self._update_display()
        self._update_stats()

    def _is_error(self, line: str) -> bool:
        """Check if a log line is an error."""
        line_lower = line.lower()
        return any(
            pattern in line_lower
            for pattern in ["error", "exception", "traceback", "failed", "fatal"]
        )

    def _is_200(self, line: str) -> bool:
        """Check if a log line contains a 200 OK response."""
        # More precise pattern matching
        return bool(re.search(r'\b200\b', line)) and (
            "OK" in line or "status" in line.lower() or re.search(r'HTTP/\S+\s+200', line)
        )

    def _is_400(self, line: str) -> bool:
        """Check if a log line contains a 400+ bad request."""
        match = re.search(r'\b4\d{2}\b', line)
        return match is not None

    def _update_display(self) -> None:
        """Update the log display."""
        content_id = "dev-log-content" if self.current_panel == "dev" else "normal-log-content"
        text_area = self.query_one(f"#{content_id}", Static)

        if not self.filtered_lines:
            log_path = self.dev_log_path if self.current_panel == "dev" else self.normal_log_path
            text_area.update(f"  No log file found.\n  Expected: {log_path}")
            return

        # Format lines with rich text
        full_text = Text()
        for i, line in enumerate(self.filtered_lines):
            if i > 0:
                full_text.append("\n")
            full_text.append(parse_log_line(line, self.search_term))

        text_area.update(full_text)

        if self.auto_scroll:
            text_area.scroll_end(animate=False)

    def _update_stats(self) -> None:
        """Update the statistics display."""
        total = len(self.log_lines)
        errors = sum(1 for line in self.log_lines if self._is_error(line))
        ok_200 = sum(1 for line in self.log_lines if self._is_200(line))
        bad_400 = sum(1 for line in self.log_lines if self._is_400(line))

        stats_bar = self.query_one("#stats-bar", StatusBar)
        stats_bar.update(total, errors, ok_200, bad_400, self.auto_scroll)

    def action_refresh(self) -> None:
        """Refresh the logs."""
        self._load_logs()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_toggle_dev(self) -> None:
        """Switch to Dev logs."""
        self.current_panel = "dev"
        self._load_logs()
        self._update_panel_visibility()

    def action_toggle_normal(self) -> None:
        """Switch to Normal logs."""
        self.current_panel = "normal"
        self._load_logs()
        self._update_panel_visibility()

    def action_filter_all(self) -> None:
        """Show all logs."""
        self.current_filter = "all"
        self._apply_filter()
        self._update_filter_buttons()

    def action_filter_errors(self) -> None:
        """Show error logs."""
        self.current_filter = "errors"
        self._apply_filter()
        self._update_filter_buttons()

    def action_filter_200(self) -> None:
        """Show 200 OK logs."""
        self.current_filter = "200"
        self._apply_filter()
        self._update_filter_buttons()

    def action_filter_400(self) -> None:
        """Show 400+ logs."""
        self.current_filter = "400"
        self._apply_filter()
        self._update_filter_buttons()

    def _update_filter_buttons(self) -> None:
        """Update filter button states."""
        filter_bar_id = "dev-filter-bar" if self.current_panel == "dev" else "normal-filter-bar"
        try:
            filter_bar = self.query_one(f"#{filter_bar_id}", FilterBar)
            filter_bar.current_filter = self.current_filter
        except Exception:
            pass

    def action_toggle_search(self) -> None:
        """Toggle search bar."""
        self.search_active = not self.search_active
        search_bar = self.query_one("#search-bar", SearchBar)
        search_bar.display = "block" if self.search_active else "none"
        if self.search_active:
            self.query_one("#search-input", TextInput).focus()

    def action_toggle_auto_scroll(self) -> None:
        """Toggle auto-scroll."""
        self.auto_scroll = not self.auto_scroll
        self._update_stats()

    def action_scroll_top(self) -> None:
        """Scroll to top."""
        content_id = "dev-log-content" if self.current_panel == "dev" else "normal-log-content"
        self.query_one(f"#{content_id}", Static).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll to bottom."""
        content_id = "dev-log-content" if self.current_panel == "dev" else "normal-log-content"
        self.query_one(f"#{content_id}", Static).scroll_end(animate=False)

    def action_clear_filter(self) -> None:
        """Clear search filter."""
        self.search_term = ""
        self.search_active = False
        search_bar = self.query_one("#search-bar", SearchBar)
        search_bar.display = "none"
        self._apply_filter()

    def on_text_input_changed(self, event: TextInput.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search-input":
            self.search_term = event.input.value
            self._apply_filter()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "filter-all":
            self.action_filter_all()
        elif button_id == "filter-errors":
            self.action_filter_errors()
        elif button_id == "filter-200":
            self.action_filter_200()
        elif button_id == "filter-400":
            self.action_filter_400()
        elif button_id == "btn-search-clear":
            self.action_clear_filter()


def run_logs_tui(
    log_path: Path | None = None,
    dev: bool = False,
    config_path: Path | None = None,
) -> None:
    """Run the logs TUI application."""
    app = LogsTUIApp(log_path=log_path, dev=dev, config_path=config_path)
    app.run()