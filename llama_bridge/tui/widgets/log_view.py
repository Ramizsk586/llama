from __future__ import annotations

import re
from typing import Iterable

from textual.widget import Widget
from textual.widgets import Static
from rich.text import Text
from rich.style import Style


LEVEL_STYLES = {
    "ERROR": Style(color="#f38ba8", bold=True),
    "WARN": Style(color="#f9e2af"),
    "WARNING": Style(color="#f9e2af"),
    "INFO": Style(color="#89b4fa"),
    "DEBUG": Style(color="#585b70"),
    "TRACE": Style(color="#313244"),
}


def parse_log_line(line: str, search_term: str = "") -> Text:
    """
    Parse a raw log line into a Rich Text object with level-based coloring.
    Expected formats:
      2024-01-15 14:32:01,442 INFO  message...
      14:32:01.442 DEBUG message...
      [INFO] message...
    """
    text = Text()

    # Extract timestamp (first token if it looks like time)
    # Common patterns: "2024-01-15 14:32:01,442" or "14:32:01.442"
    timestamp_match = re.match(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,\.]\d{3}|\d{2}:\d{2}:\d{2}[,\.]\d{3})",
        line
    )

    if timestamp_match:
        timestamp = timestamp_match.group(1)
        text.append(timestamp + "  ", Style(color="#585b70"))
        remaining = line[timestamp_match.end():]
    else:
        remaining = line

    # Extract level keyword
    level_match = re.match(r"\s*(ERROR|WARN|WARNING|INFO|DEBUG|TRACE)", remaining, re.IGNORECASE)

    if level_match:
        level = level_match.group(1).upper()
        level_style = LEVEL_STYLES.get(level, Style())
        text.append(f"{level:8}", level_style)
        # Add dim separator after level
        text.append(" │ ", Style(color="#1f2535"))
        message = remaining[level_match.end():]
    else:
        message = remaining

    # Check for HTTP status codes
    http_match = re.search(r"\b(200|201|204)\b", message)
    if http_match:
        text.append(message[:http_match.start()], Style(color="#cdd6f4"))
        text.append(http_match.group(1), Style(color="#a6e3a1", bold=True))
        text.append(message[http_match.end():], Style(color="#cdd6f4"))
    else:
        http_400_match = re.search(r"\b(4\d{2}|5\d{2})\b", message)
        if http_400_match:
            text.append(message[:http_400_match.start()], Style(color="#cdd6f4"))
            text.append(http_400_match.group(1), Style(color="#f9e2af", bold=True))
            text.append(message[http_400_match.end():], Style(color="#cdd6f4"))
        else:
            text.append(message, Style(color="#cdd6f4"))

    # Highlight search term with cyan + underline
    if search_term:
        text = text.highlight_regex(re.compile(re.escape(search_term), re.IGNORECASE), Style(bold=True, color="#74c7ec", underline=True))

    return text


class LogLineView(Widget):
    """Widget for displaying log lines with syntax highlighting."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_term = ""
        self._lines: list[str] = []

    def compose(self):
        yield Static("", classes="log-content")

    def set_lines(self, lines: list[str]) -> None:
        """Set log lines to display."""
        self._lines = lines
        container = self.query_one(".log-content", Static)
        if not lines:
            container.update("  No log file found.")
            return

        full_text = Text()
        for i, line in enumerate(lines):
            if i > 0:
                full_text.append("\n")
            full_text.append(parse_log_line(line, self.search_term))

        container.update(full_text)

    def filter_lines(self, search_term: str) -> None:
        """Filter lines by search term."""
        self.search_term = search_term
        self.set_lines(self._lines)