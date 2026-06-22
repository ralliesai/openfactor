import argparse

import numpy as np
import pandas as pd

from openfactor.io.snapshot import load_snapshot
from openfactor.tui.app import OpenFactorApp
from openfactor.tui.report import tui_report


def main():
    """Launch the OpenFactor interactive portfolio risk terminal.

    Example:
        openfactor --universe openfactor-us1000 --portfolio portfolio.csv
    """
    args = parse_args()
    try:
        portfolio = load_portfolio(args.portfolio)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"loading {args.universe} @ {args.snapshot} (with exposure panel) …")
    snapshot = load_snapshot(args.universe, args.snapshot, include_exposures_panel=True)
    report = tui_report(portfolio, snapshot)
    if args.track:
        from openfactor.tui.track import realized_attribution, realized_stats, realized_windows, update_track

        track = update_track(args.track, report)
        report["track"] = realized_stats(track)
        report["realized"] = realized_attribution(track)
        report["realized_windows"] = realized_windows(track)
        print(f"recorded {report['meta']['as_of_date']} -> {args.track} ({report['track']['days']} day(s) stored)")
    OpenFactorApp(report, snapshot=snapshot).run()


def parse_args():
    """Return CLI arguments.

    Example:
        --snapshot latest is used unless a date is supplied.
    """
    parser = argparse.ArgumentParser(prog="openfactor")
    parser.add_argument("--universe", default="openfactor-us1000")
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--snapshot", default="latest")
    parser.add_argument("--track", help="Local folder to accumulate detailed daily portfolio history")
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
