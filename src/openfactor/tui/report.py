from openfactor.portfolio.active_risk import active_risk_report, specific_by_name, tail_metrics
from openfactor.portfolio.attribution import attribution_index
from openfactor.portfolio.report import missing_holdings
from openfactor.portfolio.summary import risk_decomposition


HORIZONS = ["1 Day", "1 Month", "1 Quarter"]


def tui_report(portfolio, snapshot):
    """Assemble every panel the terminal UI renders for one portfolio.

    Example:
        tui_report(meme_book, us1000) returns headline numbers, the active-risk
        rows, stock-specific names, and the tail metrics.
    """
    rows = risk_decomposition(portfolio, snapshot)
    total = find(rows, "Total")
    common = find(rows, "Common Factor")
    specific = find(rows, "Specific")
    active = active_risk_report(portfolio, snapshot)
    index = attribution_index(portfolio, snapshot)
    attach_returns(active["rows"], index)
    names = specific_by_name(portfolio, snapshot)
    tail = tail_metrics(portfolio, snapshot, total["volatility"], active["tracking_error"])
    return {
        "meta": meta(portfolio, snapshot),
        "summary": {
            "total_risk": total["volatility"],
            "tracking_error": active["tracking_error"],
            "factor_share": common["pct"],
            "specific_share_total": specific["pct"],
            "specific_share_te": active["specific_share"],
            "beta": tail["beta"],
            "var": tail["var"],
            "return": dict(zip(HORIZONS, index["total"] if index else [None, None, None])),
        },
        "active_rows": active["rows"],
        "specific_te_share": active["specific_share"],
        "specific_ret": index["specific"] if index else [None, None, None],
        "total_ret": index["total"] if index else [None, None, None],
        "names": names,
        "horizons": HORIZONS,
    }


def attach_returns(rows, index):
    """Attach each factor's horizon return contributions to its active row."""
    for row in rows:
        row["ret"] = index["factor"].get(row["factor"]) if index else None


def meta(portfolio, snapshot):
    """Return universe context plus dropped holdings."""
    missing = missing_holdings(portfolio, snapshot.universe)["ticker"].astype(str).tolist()
    return {
        "universe": snapshot.universe_name,
        "as_of_date": str(snapshot.as_of_date),
        "tickers": int(len(snapshot.universe)),
        "held": int(len(portfolio)),
        "missing": missing,
    }


def find(rows, label):
    """Return the first decomposition row matching a stripped label."""
    return next(row for row in rows if row["label"].strip() == label)
