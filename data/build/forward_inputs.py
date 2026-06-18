import numpy as np
import pandas as pd


FORWARD_COLUMNS = [
    "ticker",
    "as_of_date",
    "forward_earnings_yield",
    "forward_earnings_yield_observations",
    "forward_growth",
    "forward_growth_observations",
]


def forward_estimate_inputs(frame, estimates):
    """Return current forward estimate factor inputs.

    Example:
        estimated net income 10 and market cap 100 gives forward_earnings_yield=0.10.
    """
    if frame.empty:
        return pd.DataFrame(columns=FORWARD_COLUMNS)

    day = frame["as_of_date"].max()
    current = frame[frame["as_of_date"] == day].drop_duplicates("ticker", keep="last")
    groups = {ticker: rows for ticker, rows in clean_estimates(estimates).groupby("ticker")}
    rows = []
    for item in current.itertuples(index=False):
        estimate = next_estimate(groups.get(item.ticker), day)
        rows.append(forward_row(item, estimate, day))
    return pd.DataFrame(rows, columns=FORWARD_COLUMNS)


def clean_estimates(estimates):
    """Return FMP estimate rows with comparable dates and numbers.

    Example:
        estimate_date strings become timestamps and estimate values become floats.
    """
    if estimates.empty:
        return pd.DataFrame(columns=["ticker", "estimate_date"])
    frame = estimates.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["estimate_date"] = pd.to_datetime(frame["estimate_date"], errors="coerce")
    for column in ["revenue", "net_income", "eps", "revenue_analysts", "eps_analysts"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["estimate_date"]).sort_values("estimate_date")


def next_estimate(estimates, as_of_date):
    """Return the first fiscal estimate after as_of_date.

    Example:
        as_of_date 2026-06-17 chooses the next future annual estimate.
    """
    if estimates is None or estimates.empty:
        return None
    rows = estimates[estimates["estimate_date"] > pd.to_datetime(as_of_date)]
    return None if rows.empty else rows.iloc[0]


def forward_row(item, estimate, as_of_date):
    """Return one ticker's forward estimate inputs.

    Example:
        missing estimates return NaN values with zero observations.
    """
    if estimate is None:
        return {
            "ticker": item.ticker,
            "as_of_date": as_of_date,
            "forward_earnings_yield": np.nan,
            "forward_earnings_yield_observations": 0,
            "forward_growth": np.nan,
            "forward_growth_observations": 0,
        }

    count = max_count([estimate.revenue_analysts, estimate.eps_analysts])
    return {
        "ticker": item.ticker,
        "as_of_date": as_of_date,
        "forward_earnings_yield": safe_divide(number(estimate.net_income), number(item.market_cap)),
        "forward_earnings_yield_observations": count,
        "forward_growth": average_finite(
            [
                positive_growth(estimate.revenue, item.revenue),
                positive_growth(estimate.net_income, item.net_income),
            ]
        ),
        "forward_growth_observations": count,
    }


def positive_growth(current, prior):
    """Return growth only when both values are positive.

    Example:
        120 over 100 returns 0.20; loss-making bases return NaN.
    """
    current = number(current)
    prior = number(prior)
    if current <= 0 or prior <= 0 or not np.isfinite(current) or not np.isfinite(prior):
        return np.nan
    return current / prior - 1


def safe_divide(top, bottom):
    """Return top / bottom or NaN.

    Example:
        10 / 100 returns 0.10.
    """
    if not np.isfinite(top) or not np.isfinite(bottom) or bottom == 0:
        return np.nan
    return top / bottom


def average_finite(values):
    """Return the average of finite values.

    Example:
        average_finite([1, nan, 3]) returns 2.
    """
    values = [number(value) for value in values]
    values = [value for value in values if np.isfinite(value)]
    return np.nan if not values else float(np.mean(values))


def max_count(values):
    """Return the largest finite observation count.

    Example:
        max_count([nan, 12]) returns 12.
    """
    values = [number(value) for value in values]
    values = [value for value in values if np.isfinite(value)]
    return 0 if not values else int(max(values))


def number(value):
    """Return a float or NaN.

    Example:
        number("10") returns 10.0.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
