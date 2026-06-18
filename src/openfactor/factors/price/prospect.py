import numpy as np
import pandas as pd

from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def prospect_for_stock(close, window=756, drawdown_window=252):
    """Return lottery-like upside skew plus recent drawdown.

    Example:
        prices with jumpy positive returns and a 20% drawdown get a higher value
        than a smooth, low-skew price path.
    """
    observations = int(np.isfinite(close).sum())
    prices = pd.Series(close).dropna().tail(window + 1)
    if len(prices) < 30:
        return FactorResult(np.nan, observations)

    returns = prices.pct_change().dropna()
    return FactorResult(
        skew(returns.to_numpy(dtype=float)) + max_drawdown(prices.tail(drawdown_window)),
        observations,
    )


def skew(values):
    """Return return-distribution skew.

    Example:
        one large positive return among small returns gives positive skew.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan
    move = values - values.mean()
    scale = move.std()
    if scale == 0:
        return np.nan
    return float(np.mean((move / scale) ** 3))


def max_drawdown(close):
    """Return positive maximum drawdown.

    Example:
        prices 100, 80, 90 return 0.20.
    """
    prices = pd.Series(close).dropna()
    if len(prices) < 2:
        return np.nan
    return float((1 - prices / prices.cummax()).max())


def compute(matrix):
    """Compute prospect exposure rows from a PriceMatrix.

    Example:
        compute(matrix) returns rows like:
        ticker  factor    value  observations
        AAPL    prospect  0.40   756
    """
    results = [prospect_for_stock(close) for close in matrix.close.T]
    return make_price_factor_rows(matrix, results, "prospect")
