import argparse

import numpy as np
import pandas as pd

from openfactor.io.snapshot import load_snapshot
from openfactor.llm import discover_semantic_factors
from openfactor.portfolio.report import portfolio_report


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

    report = portfolio_report(portfolio, snapshot)
    print_frame("portfolio holdings", portfolio, rows=12)
    print(
        f"snapshot as_of_date={snapshot.as_of_date} "
        f"universe={snapshot.universe_name} "
        f"tickers={len(snapshot.universe)}"
    )
    print_frame("missing holdings", report["missing_holdings"])
    print_frame("style factor exposures", report["style"])
    print_frame("sector allocation", report["sector"])
    print_frame("stock-specific risk", report["specific_risk"])
    print_frame("factor risk contribution", report["factor_risk"])
    print_frame("factor vs residual risk share", report["risk_share"])
    print_frame("portfolio total risk", report["total_risk"])
    if args.semantic_discovery:
        discover_semantic_factors(
            portfolio,
            snapshot,
            threshold=args.semantic_threshold,
            window=args.semantic_window,
            batch_size=args.semantic_batch_size,
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


def print_frame(title, frame, rows=None):
    """Print one report table.

    Example:
        print_frame("portfolio", portfolio) prints the full table.
    """
    preview = frame if rows is None else frame.head(rows)
    print(f"{title}\n{preview.to_string()}")
