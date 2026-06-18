import numpy as np

from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def liquidity_for_stock(close, volume):
    """Return log average dollar volume.

    Example:
        close = np.array([10, 11])
        volume = np.array([100, 200])
        liquidity_for_stock(close, volume)
        returns FactorResult(value=np.log1p(1600.0), observations=2)
    """
    dollar_volume = close * volume
    good = np.isfinite(dollar_volume)
    observations = int(good.sum())
    if observations == 0:
        return FactorResult(value=np.nan, observations=observations)

    return FactorResult(value=np.log1p(dollar_volume[good].mean()), observations=observations)


def compute(matrix, window=63):
    """Compute liquidity rows from a PriceMatrix with volume.

    Example:
        matrix = price_matrix(price_rows, require_volume=True)
        compute(matrix) returns rows like:
        ticker  factor     value  observations
        AAPL    liquidity  18.4   63
    """
    if matrix.volume is None:
        raise ValueError("liquidity requires PriceMatrix.volume")

    pairs = zip(matrix.close[-window:].T, matrix.volume[-window:].T)
    results = [liquidity_for_stock(close, volume) for close, volume in pairs]
    return make_price_factor_rows(matrix, results, "liquidity")
