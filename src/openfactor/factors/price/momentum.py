import numpy as np

from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def momentum_for_stock(close, lookback=252, skip=21):
    """Return 12 month momentum, skipping the most recent month.

    Example:
        close = np.array([100, 110, 120, 130])
        momentum_for_stock(close, lookback=2, skip=1)
        returns FactorResult(value=0.20, observations=4)
    """
    observations = int(np.isfinite(close).sum())
    if len(close) < lookback + skip + 1:
        return FactorResult(value=np.nan, observations=observations)

    # Use the price from one month ago, then compare it to 12 months before that.
    end = close[-1 - skip]
    start = close[-1 - skip - lookback]
    if not np.isfinite(end) or not np.isfinite(start) or start <= 0:
        return FactorResult(value=np.nan, observations=observations)

    return FactorResult(value=end / start - 1, observations=observations)


def compute(matrix, lookback=252, skip=21):
    """Compute medium momentum rows from a PriceMatrix.

    Example:
        matrix = price_matrix(price_rows)
        compute(matrix) returns rows like:
        ticker  factor    value  observations
        AAPL    momentum  0.20   274
    """
    results = [momentum_for_stock(close, lookback, skip) for close in matrix.close.T]
    return make_price_factor_rows(matrix, results, "momentum")
