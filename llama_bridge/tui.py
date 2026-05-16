"""
Llama Bridge TUI Entry Point

This module provides the entry points for the Textual-based TUI interfaces.
Keeps backward compatibility with the original CLI integration.
"""
from __future__ import annotations

import sys
from pathlib import Path


def check_textual() -> bool:
    """Check if textual is available and meets minimum version requirement."""
    try:
        import textual
        from packaging.version import Version

        if Version(textual.__version__) < Version("0.47.0"):
            print(
                "Error: textual>=0.47.0 required. Run: pip install 'textual>=0.47.0'",
                file=sys.stderr
            )
            return False
        return True
    except ImportError:
        print(
            "Error: Textual is not installed. Run: pip install 'llama-bridge[tui]' or 'textual>=0.47.0'",
            file=sys.stderr
        )
        return False


def run_main_tui(config_path: Path | None = None) -> None:
    """Run the main TUI dashboard."""
    if not check_textual():
        sys.exit(1)

    from .tui.tui_main import MainTUIApp
    app = MainTUIApp(config_path=config_path)
    app.run()


def run_logs_tui(
    log_path: Path | None = None,
    dev: bool = False,
    config_path: Path | None = None,
) -> None:
    """Run the logs TUI viewer."""
    if not check_textual():
        sys.exit(1)

    from .tui.tui_logs import LogsTUIApp
    app = LogsTUIApp(log_path=log_path, dev=dev, config_path=config_path)
    app.run()