"""Llama Bridge TUI package."""

from pathlib import Path


def run_main_tui(config_path: Path | None = None) -> None:
    """Run the main TUI dashboard."""
    from .tui_main import MainTUIApp

    app = MainTUIApp(config_path=config_path)
    app.run()


def run_logs_tui(
    log_path: Path | None = None,
    dev: bool = False,
    config_path: Path | None = None,
) -> None:
    """Run the logs TUI viewer."""
    from .tui_logs import LogsTUIApp

    app = LogsTUIApp(log_path=log_path, dev=dev, config_path=config_path)
    app.run()