import numpy as np

from openfactor.core.checks import require_columns
from openfactor.factors.output import make_exposure_rows


def latest(reference):
    """Return one latest reference row per ticker.

    Example:
        two AAPL rows return the last AAPL row.
    """
    require_columns(reference, ["ticker"])
    frame = reference.drop_duplicates("ticker", keep="last").copy()
    return frame.set_index("ticker")


def column(frame, name):
    """Return a numeric column or NaNs if the column is unavailable.

    Example:
        column(frame, "revenue") returns revenue values aligned to tickers.
    """
    if name not in frame:
        return np.full(len(frame), np.nan)
    return np.asarray(frame[name], dtype=float)


def ratio(reference, factor, numerator, denominator, as_of_date):
    """Return numerator / denominator factor rows.

    Example:
        stockholders_equity=50 and market_cap=100 returns value=0.5.
    """
    frame = latest(reference)
    top = column(frame, numerator)
    bottom = column(frame, denominator)
    values = np.divide(
        top,
        bottom,
        out=np.full(len(frame), np.nan),
        where=np.isfinite(top) & np.isfinite(bottom) & (bottom != 0),
    )
    return make_exposure_rows(
        tickers=frame.index,
        values=values,
        factor=factor,
        group="reference",
        as_of_date=as_of_date,
    )


def scalar(reference, factor, metric, as_of_date):
    """Return one metric as factor rows.

    Example:
        asset_growth=0.10 returns investment value=0.10.
    """
    frame = latest(reference)
    return make_exposure_rows(
        tickers=frame.index,
        values=column(frame, metric),
        factor=factor,
        group="reference",
        as_of_date=as_of_date,
    )
