import pandas as pd


def compute(reference, as_of_date="latest"):
    """Create one-hot readable industry exposures.

    Example input row:
        ticker  industry
        NVDA    Semiconductors

    Example output row:
        NVDA industry:Semiconductors 1.0
    """
    if "ticker" not in reference:
        raise ValueError("missing required columns: ['ticker']")

    latest = reference.drop_duplicates("ticker", keep="last")
    rows = []
    for row in latest.itertuples(index=False):
        industry = getattr(row, "industry", None)
        if pd.isna(industry):
            continue

        rows.append(
            {
                "as_of_date": str(as_of_date),
                "ticker": row.ticker,
                "factor": f"industry:{industry}",
                "group": "industry",
                "value": 1.0,
                "observations": 1,
            }
        )
    return pd.DataFrame(rows)
