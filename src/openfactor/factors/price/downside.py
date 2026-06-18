import numpy as np

from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def downside_for_stock(returns, minimum=20):
    """Return annualized volatility of negative daily returns.

    Example:
        negative returns [-1%, -2%] drive downside risk; positive days are ignored.
    """
    downside = returns[np.isfinite(returns) & (returns < 0)]
    observations = int(len(downside))
    if observations < minimum:
        return FactorResult(np.nan, observations)
    return FactorResult(float(downside.std(ddof=1) * np.sqrt(252)), observations)


def compute(matrix, window=252):
    """Compute downside-risk rows from recent daily returns.

    Example:
        compute(matrix) returns rows like:
        ticker  factor         value
        AAPL    downside_risk  0.20
    """
    returns = matrix.returns[-window:]
    results = [downside_for_stock(stock) for stock in returns.T]
    return make_price_factor_rows(matrix, results, "downside_risk")
