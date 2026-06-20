import argparse

import numpy as np
import pandas as pd

from openfactor.console import console, print_risk_summary, print_risk_table
from openfactor.io.snapshot import load_snapshot
from openfactor.llm.cache import DEFAULT_SEMANTIC_CACHE
from openfactor.portfolio.report import missing_holdings
from openfactor.portfolio.summary import risk_decomposition


def main():
    """Run the OpenFactor portfolio report CLI.

    Example:
        openfactor --universe openfactor-us1000 --portfolio portfolio.csv
    """
    args = parse_args()
    try:
        portfolio = load_portfolio(args.portfolio)
        snapshot = load_snapshot(args.universe, args.snapshot)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    summary, rows = risk_decomposition(portfolio, snapshot)
    console.rule(
        f"[bold]OpenFactor[/bold] · {snapshot.universe_name} · "
        f"as of {snapshot.as_of_date} · {len(snapshot.universe)} tickers"
    )
    print_risk_summary(summary)
    print_risk_table(rows)
    missing = missing_holdings(portfolio, snapshot.universe)["ticker"].tolist()
    if missing:
        console.print(f"[dim]missing holdings (not in universe): {', '.join(missing)}[/dim]")
    if args.semantic_discovery:
        from openfactor.llm import discover_semantic_factors

        discover_semantic_factors(
            portfolio,
            snapshot,
            threshold=args.semantic_threshold,
            window=args.semantic_window,
            batch_size=args.semantic_batch_size,
            semantic_cache=args.semantic_cache,
        )


def parse_args():
    """Return CLI arguments.

    Example:
        --snapshot latest is used unless a date is supplied.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True)
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--snapshot", default="latest")
    parser.add_argument("--semantic-discovery", action="store_true")
    parser.add_argument("--semantic-threshold", type=float, default=0.10)
    parser.add_argument("--semantic-window", type=int, default=63)
    parser.add_argument("--semantic-batch-size", type=int, default=50)
    parser.add_argument("--semantic-cache", default=DEFAULT_SEMANTIC_CACHE)
    return parser.parse_args()


def load_portfolio(path):
    """Load ticker allocations from a CSV file.

    Example input:
        ticker,allocation
        AAPL,0.30

    Example output:
        one row with ticker="AAPL" and allocation=0.30.
    """
    portfolio = pd.read_csv(path)
    missing = {"ticker", "allocation"} - set(portfolio.columns)
    if missing:
        raise ValueError(f"portfolio missing columns: {sorted(missing)}")

    portfolio = portfolio[["ticker", "allocation"]].copy()
    portfolio["ticker"] = portfolio["ticker"].astype(str)
    portfolio["allocation"] = pd.to_numeric(portfolio["allocation"])
    if not np.isclose(portfolio["allocation"].sum(), 1.0):
        raise ValueError("portfolio allocation must sum to 1.0")
    return portfolio


if __name__ == "__main__":
    main()
