import pandas as pd

from openfactor.core.sic import sector_from_sic


def compute(reference, as_of_date="latest"):
    """Create one-hot sector exposures.

    Example input row:
        ticker  sector
        AAPL    Technology

    Example output row:
        AAPL sector:Technology 1.0
    """
    if "ticker" not in reference:
        raise ValueError("missing required columns: ['ticker']")

    latest = reference.drop_duplicates("ticker", keep="last")
    rows = []
    for row in latest.itertuples(index=False):
        sector = sector_for(row)
        if pd.isna(sector):
            continue

        rows.append(
            {
                "as_of_date": str(as_of_date),
                "ticker": row.ticker,
                "factor": f"sector:{sector}",
                "group": "sector",
                "value": 1.0,
                "observations": 1,
            }
        )
    return pd.DataFrame(rows)


def sector_for(row):
    """Return sector from row.sector or row.sic.

    Example:
        SIC 3571 returns Manufacturing.
    """
    sector = getattr(row, "sector", None)
    if pd.notna(sector):
        return str(sector)

    sic = getattr(row, "sic", None)
    if pd.isna(sic):
        return None
    return sector_from_sic(sic)
