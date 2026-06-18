import numpy as np
from scipy.stats import linregress

from openfactor.core.returns import market_returns_for
from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def beta_for_stock(stock_returns, market_returns):
    """Estimate one stock's market beta.

    Example:
        stock_returns = np.array([0.02, 0.01, -0.01])
        market_returns = np.array([0.01, 0.00, -0.01])
        beta_for_stock(stock_returns, market_returns)
        returns FactorResult(value=1.5, observations=3)
    """
    good = np.isfinite(stock_returns) & np.isfinite(market_returns)
    observations = int(good.sum())
    if observations < 2:
        return FactorResult(value=np.nan, observations=observations)

    market = market_returns[good]
    if np.ptp(market) == 0:
        return FactorResult(value=np.nan, observations=observations)

    # beta = how much the stock moves when the market moves.
    return FactorResult(
        value=linregress(market, stock_returns[good]).slope,
        observations=observations,
    )


def compute(matrix, window=252, market_returns=None):
    """Compute market beta rows from a PriceMatrix.

    Example:
        matrix = price_matrix(price_rows)
        compute(matrix) returns rows like:
        ticker  factor  value  observations
        AAPL    beta    1.5    252
    """
    returns = matrix.returns[-window:]
    market = market_returns_for(returns, market_returns)

    results = [beta_for_stock(stock, market) for stock in returns.T]
    return make_price_factor_rows(matrix, results, "beta")
