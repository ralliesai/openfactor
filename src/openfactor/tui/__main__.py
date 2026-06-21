import argparse

import pandas as pd

from openfactor.cli import load_portfolio
from openfactor.io.snapshot import load_snapshot
from openfactor.tui.app import OpenFactorTUI
from openfactor.tui.report import tui_report


def main():
    """Load a snapshot and launch the interactive risk terminal.

    Example:
        python3.10 -m openfactor.tui --universe openfactor-us1000 --portfolio portfolio.csv
    """
    args = parse_args()
    portfolio = load_portfolio(args.portfolio)
    print(f"loading {args.universe} @ {args.snapshot} (with exposure panel) …")
    snapshot = load_snapshot(args.universe, args.snapshot, include_exposures_panel=True)
    report = tui_report(portfolio, snapshot)
    OpenFactorTUI(report).run()


def parse_args():
    parser = argparse.ArgumentParser(prog="openfactor-tui")
    parser.add_argument("--universe", default="openfactor-us1000")
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--snapshot", default="latest")
    return parser.parse_args()


if __name__ == "__main__":
    main()
