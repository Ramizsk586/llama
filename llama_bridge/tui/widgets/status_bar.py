from textual.widget import Widget
from textual.widgets import Static
from rich.text import Text


class StatusBar(Static):
    """
    Bottom status bar showing:
      total | errors (%) | 200s | 400s | auto-refresh indicator
    Updates reactively from parent app state.
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $bg-panel;
        color: $text-muted;
        padding: 0 1;
        border-top: solid $border;
    }
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.total = 0
        self.errors = 0
        self.ok_200 = 0
        self.bad_400 = 0
        self.auto_refresh = True

    def update(self, total: int, errors: int, ok_200: int, bad_400: int, auto_refresh: bool = True) -> None:
        """Update the status bar with new values."""
        self.total = total
        self.errors = errors
        self.ok_200 = ok_200
        self.bad_400 = bad_400
        self.auto_refresh = auto_refresh
        self.refresh()

    def render(self) -> str:
        error_pct = (self.errors / self.total * 100) if self.total > 0 else 0
        auto_dot = "[#a6e3a1]●[/#a6e3a1]" if self.auto_refresh else "[#f38ba8]○[/#f38ba8]"
        error_color = "#f38ba8" if error_pct > 5 else "#585b70"

        return (
            f" [dim]Total:[/dim] [white]{self.total}[/white]"
            f"  [dim]│[/dim]  [dim]Errors:[/dim] [{error_color}]{self.errors} ({error_pct:.1f}%)[/{error_color}]"
            f"  [dim]│[/dim]  [dim]200 OK:[/dim] [#a6e3a1]{self.ok_200}[/#a6e3a1]"
            f"  [dim]│[/dim]  [dim]400+:[/dim] [#f9e2af]{self.bad_400}[/#f9e2af]"
            f"  [dim]│[/dim]  Auto {auto_dot}"
        )