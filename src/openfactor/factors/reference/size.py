import numpy as np

from openfactor.core.checks import require_columns
from openfactor.factors.output import make_exposure_rows


def compute(reference, as_of_date="latest"):
    """Compute log market capitalization exposure.

    Example input row:
        ticker  market_cap
        AAPL    100.0

    Example output row:
        ticker  factor  value           observations
        AAPL    size    np.log(100.0)   1
    """
    require_columns(reference, ["ticker", "market_cap"])
    latest = reference.drop_duplicates("ticker", keep="last").set_index("ticker")
    values = np.log(latest["market_cap"].where(latest["market_cap"] > 0))
    observations = np.isfinite(values).astype(int)
    return make_exposure_rows(
        tickers=latest.index,
        values=values,
        factor="size",
        group="reference",
        as_of_date=as_of_date,
        observations=observations,
    )
