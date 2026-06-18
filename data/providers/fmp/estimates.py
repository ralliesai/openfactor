import pandas as pd


def analyst_estimates(client, ticker):
    """Download annual analyst estimate rows.

    Example:
        analyst_estimates(client, "AAPL")
        returns future revenue and net-income estimate rows.
    """
    rows = client.get(
        "analyst-estimates",
        {"symbol": ticker.upper(), "period": "annual"},
    )
    return estimates_to_frame(rows, ticker.upper())


def estimates_to_frame(rows, ticker):
    """Turn FMP analyst estimate JSON into rows.

    Example:
        revenueAvg 120 and numAnalystsRevenue 10 become one estimate row.
    """
    frame = pd.DataFrame(rows if isinstance(rows, list) else [])
    columns = [
        "ticker",
        "estimate_date",
        "revenue",
        "net_income",
        "eps",
        "revenue_analysts",
        "eps_analysts",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        {
            "ticker": ticker,
            "estimate_date": frame.get("date"),
            "revenue": pd.to_numeric(frame.get("revenueAvg"), errors="coerce"),
            "net_income": pd.to_numeric(frame.get("netIncomeAvg"), errors="coerce"),
            "eps": pd.to_numeric(frame.get("epsAvg"), errors="coerce"),
            "revenue_analysts": pd.to_numeric(frame.get("numAnalystsRevenue"), errors="coerce"),
            "eps_analysts": pd.to_numeric(frame.get("numAnalystsEps"), errors="coerce"),
        }
    )
