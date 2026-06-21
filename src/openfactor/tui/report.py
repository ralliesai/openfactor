import re

from openfactor.portfolio.active_risk import (
    active_risk_report,
    benchmark_profile,
    specific_by_name,
    tail_metrics,
)
from openfactor.portfolio.attribution import attribution_index
from openfactor.portfolio.report import missing_holdings
from openfactor.portfolio.summary import risk_decomposition


HORIZONS = ["1 Day", "1 Week"]
WINDOWS = [1, 5]


def tui_report(portfolio, snapshot):
    """Assemble every panel the terminal UI renders for one portfolio.

    Example:
        tui_report(meme_book, us1000) returns headline numbers, the active-risk
        rows, idiosyncratic name risk, and the tail metrics.
    """
    rows = risk_decomposition(portfolio, snapshot)
    total = find(rows, "Total")
    common = find(rows, "Common Factor")
    specific = find(rows, "Idiosyncratic")
    active = active_risk_report(portfolio, snapshot)
    index = attribution_index(portfolio, snapshot)
    attach_returns(active["rows"], index)
    names = specific_by_name(portfolio, snapshot)
    tail = tail_metrics(portfolio, snapshot, total["volatility"], active["tracking_error"])
    blank = [None] * len(HORIZONS)
    portfolio_ret = index["total"] if index else blank
    benchmark_ret = (index["factor"].get("market") if index else None) or blank
    active_ret = [diff(p, b) for p, b in zip(portfolio_ret, benchmark_ret)]
    return {
        "today": today_contributions(index),
        "meta": meta(portfolio, snapshot),
        "risk_rows": rows,
        "summary": {
            "total_risk": total["volatility"],
            "tracking_error": active["tracking_error"],
            "factor_share": common["pct"],
            "specific_share_total": specific["pct"],
            "specific_share_te": active["specific_share"],
            "specific_te_contribution": active["specific_contribution"],
            "predicted_beta": tail["beta"],
            "beta": tail["beta"],
            "var": tail["var"],
            "return": dict(zip(HORIZONS, portfolio_ret)),
        },
        "active_rows": active["rows"],
        "specific_te_share": active["specific_share"],
        "specific_te_contribution": active["specific_contribution"],
        "specific_ret": index["specific"] if index else blank,
        "portfolio_ret": portfolio_ret,
        "benchmark_ret": benchmark_ret,
        "active_ret": active_ret,
        "names": names,
        "horizons": HORIZONS,
        "horizon_dates": horizon_dates(snapshot),
        "track": None,
        "realized": None,
    }


def today_contributions(index):
    """Return this snapshot's realized 1-day contribution per factor and idiosyncratic return.

    Example:
        --track stores these so a real holding path can be summed later instead
        of running today's weights backward.
    """
    if not index:
        return {"factor": {}, "specific": 0.0}
    return {
        "factor": {factor: num(values[0]) for factor, values in index["factor"].items()},
        "specific": num(index["specific"][0]),
    }


def num(value):
    """Return a finite float, mapping None and NaN to 0.0."""
    return 0.0 if value is None or value != value else float(value)


def horizon_dates(snapshot):
    """Return a display date or date range for each return horizon.

    Example:
        1 Day shows the latest date; 1 Quarter shows "start → end".
    """
    dates = sorted(str(date) for date in snapshot.factor_returns.index)
    labels = []
    for window in WINDOWS:
        if not dates:
            labels.append("—")
        elif window <= 1:
            labels.append(dates[-1])
        else:
            labels.append(f"{dates[max(0, len(dates) - window)]} → {dates[-1]}")
    return labels


def diff(left, right):
    """Return left - right, or None when either side is missing."""
    return None if left is None or right is None else left - right


def attach_returns(rows, index):
    """Attach each factor's horizon return contributions to its active row."""
    for row in rows:
        row["ret"] = index["factor"].get(row["factor"]) if index else None


def meta(portfolio, snapshot):
    """Return universe context, dropped holdings, and the benchmark proxy."""
    missing = missing_holdings(portfolio, snapshot.universe)["ticker"].astype(str).tolist()
    return {
        "universe": snapshot.universe_name,
        "as_of_date": str(snapshot.as_of_date),
        "tickers": int(len(snapshot.universe)),
        "held": int(len(portfolio)),
        "holdings": [{"ticker": str(ticker), "weight": float(weight)}
                     for ticker, weight in zip(portfolio["ticker"], portfolio["allocation"])],
        "missing": missing,
        "benchmark": {
            "name": benchmark_label(snapshot.universe_name),
            "tagline": "cap-weighted model benchmark",
            **benchmark_profile(snapshot),
        },
    }


def benchmark_label(universe):
    """Return a display name for the cap-weighted benchmark proxy.

    Example:
        "openfactor-us1000" becomes "US 1000".
    """
    core = universe.split("-", 1)[-1]
    return re.sub(r"([a-zA-Z])(\d)", r"\1 \2", core).upper()


def find(rows, label):
    """Return the first decomposition row matching a stripped label."""
    return next(row for row in rows if row["label"].strip() == label)
