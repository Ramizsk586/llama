from textual.widget import Widget
from textual.widgets import Static


class PanelHeader(Static):
    """
    A single-line panel header with:
      - Left: colored dot + service name
      - Right: filepath or URL (dim)
    """

    DEFAULT_CSS = """
    PanelHeader {
        height: 1;
        background: $bg-elevated;
        color: $accent;
        text-style: bold;
        padding: 0 1;
    }
    """

    def __init__(self, title: str, running: bool = False, detail: str = "", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.running = running
        self.detail = detail

    def render(self) -> str:
        dot = "●" if self.running else "○"
        dot_color = "#a6e3a1" if self.running else "#f38ba8"
        detail = f"  [dim]{self.detail}[/dim]" if self.detail else ""
        return f"[{dot_color}]{dot}[/{dot_color}] [bold]{self.title.upper()}[/bold]{detail}"