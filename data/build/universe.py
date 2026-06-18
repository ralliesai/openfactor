import pandas as pd

from data.providers.massive.stocks import all_tickers


def us_candidates(client, as_of_date=None):
    """Return active US common stock ticker candidates.

    Example:
        if Massive returns AAPL as type CS, us_candidates(client) includes AAPL.
    """
    frame = all_tickers(client, date=as_of_date, active=True)
    if frame.empty:
        return []

    keep = pd.Series(True, index=frame.index)
    for column, value in [("market", "stocks"), ("locale", "us"), ("active", True)]:
        if column in frame:
            keep &= frame[column] == value
    if "type" in frame:
        keep &= frame["type"] == "CS"

    return sorted(frame.loc[keep, "ticker"].astype(str).str.upper().unique())


def top_market_cap_tickers(reference, limit=1000):
    """Return the largest tickers by market cap.

    Example:
        if AAPL market_cap > MSFT market_cap and limit=1, this returns ["AAPL"].
    """
    if reference.empty:
        return []

    frame = reference[["ticker", "market_cap"]].copy()
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame = frame.dropna().sort_values("market_cap", ascending=False)
    return frame["ticker"].astype(str).head(limit).to_list()
