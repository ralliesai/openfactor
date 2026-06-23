import argparse

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
    """Load a dollar-valued portfolio from a CSV file.

    The file lists the dollar value held in each name (negative for a short).
    OpenFactor normalizes by gross exposure (sum of absolute values) into the
    signed weights the model uses, so the absolute book size does not change
    percentage risk — a $1M and a $10M book of the same shape report alike.

    Example input:
        ticker,value
        AAPL,400000
        MSFT,300000
        NVDA,300000

    Example output:
        rows with allocation 0.40, 0.30, 0.30 (gross-normalized weights).
    """
    portfolio = pd.read_csv(path)
    missing = {"ticker", "value"} - set(portfolio.columns)
    if missing:
        raise ValueError(f"portfolio missing columns: {sorted(missing)}")

    portfolio = portfolio[["ticker", "value"]].copy()
    portfolio["ticker"] = portfolio["ticker"].astype(str)
    portfolio["value"] = pd.to_numeric(portfolio["value"])
    gross = portfolio["value"].abs().sum()
    if not gross > 0:
        raise ValueError("portfolio gross value must be positive")
    portfolio["allocation"] = portfolio["value"] / gross
    return portfolio[["ticker", "allocation"]]


if __name__ == "__main__":
    main()
