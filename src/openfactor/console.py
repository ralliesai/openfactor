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


def echo(message=""):
    """Print one log line through the Rich console.

    Example:
        echo("semantic candidate factors") prints a styled section header.
    """
    text = str(message)
    if text.startswith("semantic "):
        console.print(text, markup=False, highlight=False, style="bold cyan")
    else:
        console.print(text, markup=False, highlight=True)


def print_risk_summary(summary):
    """Print the portfolio vs active risk summary box.

    Example:
        portfolio total risk and tracking error appear side by side.
    """
    table = Table(title="Risk Summary", title_style="bold cyan", title_justify="left", header_style="bold magenta")
    table.add_column("", style="cyan")
    for name in ["Total Risk", "Common Factor", "Specific"]:
        table.add_column(name, justify="right")
    table.add_row("Portfolio", pct(summary["total"]), pct(summary["common_factor"]), pct(summary["specific"]))
    table.add_row("Active", pct(summary["tracking_error"]), pct(summary["active_factor"]), pct(summary["active_specific"]))
    console.print(table)


def print_risk_table(rows):
    """Print the nested factor risk decomposition.

    Example:
        Common Factor, Style, Industry, Specific, and Total rows nest by indent.
    """
    table = Table(title="Factor Risk Decomposition", title_style="bold cyan", title_justify="left", header_style="bold magenta")
    table.add_column("Factor")
    for name in ["Exposure", "Active", "Volatility", "% Risk"]:
        table.add_column(name, justify="right")
    styles = {"section": "bold", "group": "bold cyan", "total": "bold"}
    for row in rows:
        if row["kind"] == "total":
            table.add_section()
        cells = [row["label"], num(row["exposure"]), num(row["active"]), pct(row["volatility"]), pct(row["pct"])]
        table.add_row(*cells, style=styles.get(row["kind"]))
    console.print(table)


def pct(value):
    """Return a percent string or dash.

    Example:
        pct(0.1234) returns "12.3%".
    """
    return "[dim]—[/dim]" if value is None or pd.isna(value) else f"{value * 100:.1f}%"


def num(value):
    """Return a signed two-decimal string or dash.

    Example:
        num(-0.16) returns red "-0.16".
    """
    if value is None or pd.isna(value):
        return "[dim]—[/dim]"
    text = f"{value:.2f}"
    return f"[red]{text}[/red]" if value < 0 else text
