import numpy as np

from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def short_reversal_for_stock(close, lookback=21):
    """Return negative one-month return.

    Example:
        close = np.array([100, 105])
        short_reversal_for_stock(close, lookback=1)
        returns FactorResult(value=-0.05, observations=2)
    """
    observations = int(np.isfinite(close).sum())
    if len(close) <= lookback:
        return FactorResult(np.nan, observations)

    start = close[-1 - lookback]
    end = close[-1]
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return FactorResult(np.nan, observations)
    return FactorResult(-(end / start - 1), observations)


def compute(matrix, lookback=21):
    """Compute short-term reversal rows from a PriceMatrix.

    Example:
        compute(matrix) returns rows like:
        ticker  factor               value
        AAPL    short_term_reversal -0.05
    """
    results = [short_reversal_for_stock(close, lookback) for close in matrix.close.T]
    return make_price_factor_rows(matrix, results, "short_term_reversal")
