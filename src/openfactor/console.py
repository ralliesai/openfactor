import logging

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.traceback import install


console = Console()


def setup_logging(level=logging.INFO):
    """Install a Rich handler so build logs are colorized and timed.

    Example:
        setup_logging() turns plain factory logs into leveled, colored output.
    """
    install(show_locals=False)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def print_table(title, frame, index=True):
    """Render one report table with Rich.

    Example:
        print_table("style", report["style"]) prints a colored, aligned table.
    """
    frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    table = Table(title=title, title_style="bold cyan", title_justify="left", header_style="bold magenta")
    if index:
        table.add_column(str(frame.index.name or ""), style="cyan")
    for column in frame.columns:
        table.add_column(str(column), justify="right")
    for key, row in frame.iterrows():
        cells = [cell(value) for value in row]
        table.add_row(*([str(key), *cells] if index else cells))
    console.print(table)


def cell(value):
    """Return a compact, sign-colored cell string.

    Example:
        cell(-0.12) returns red text; cell(None) returns a dim dash.
    """
    if pd.isna(value):
        return "[dim]—[/dim]"
    if isinstance(value, (int, float, np.integer, np.floating)):
        text = f"{value:,.4f}" if isinstance(value, (float, np.floating)) else f"{int(value):,}"
        return f"[red]{text}[/red]" if value < 0 else text
    return str(value)
