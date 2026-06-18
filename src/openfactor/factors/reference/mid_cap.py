import numpy as np

from openfactor.core.checks import require_columns
from openfactor.factors.output import make_exposure_rows


def compute(reference, as_of_date="latest"):
    """Compute nonlinear mid-cap exposure from market cap.

    Example input rows:
        ticker  market_cap
        BIG     1000
        MID     100
        SMALL   10

    Example output:
        MID receives the highest raw mid_cap value before normalization.
    """
    require_columns(reference, ["ticker", "market_cap"])
    latest = reference.drop_duplicates("ticker", keep="last").set_index("ticker")
    size = np.log(latest["market_cap"].where(latest["market_cap"] > 0).astype(float))
    values = mid_cap_values(size.to_numpy(dtype=float))
    return make_exposure_rows(
        tickers=latest.index,
        values=values,
        factor="mid_cap",
        group="reference",
        as_of_date=as_of_date,
    )


def mid_cap_values(size):
    """Return nonlinear size after removing linear size exposure.

    Example:
        very large and very small stocks score lower than middle-sized stocks.
    """
    values = np.full(len(size), np.nan)
    good = np.isfinite(size)
    if good.sum() < 3:
        return values

    scale = size[good].std()
    if scale == 0:
        return values

    z = (size[good] - size[good].mean()) / scale
    raw = -(z * z)
    slope, intercept = np.polyfit(z, raw, 1)
    values[good] = raw - (slope * z + intercept)
    return values
