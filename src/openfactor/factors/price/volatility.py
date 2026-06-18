import numpy as np
from scipy.stats import linregress

from openfactor.core.returns import market_returns_for
from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def residual_for_stock(stock_returns, market_returns):
    """Return annualized volatility left after market beta.

    Example:
        stock_returns = np.array([0.01, 0.03, 0.05])
        market_returns = np.array([0.00, 0.01, 0.02])
        residual_for_stock(stock_returns, market_returns)
        returns FactorResult(value=0.0, observations=3)
    """
    good = np.isfinite(stock_returns) & np.isfinite(market_returns)
    observations = int(good.sum())
    if observations < 3:
        return FactorResult(value=np.nan, observations=observations)

    market = market_returns[good]
    if np.ptp(market) == 0:
        return FactorResult(value=np.nan, observations=observations)

    # Fit stock = alpha + beta * market, then measure leftover volatility.
    fit = linregress(market, stock_returns[good])
    leftover = stock_returns[good] - (fit.intercept + fit.slope * market)
    return FactorResult(
        value=leftover.std(ddof=1) * np.sqrt(252),
        observations=observations,
    )


def residual(matrix, window=252, market_returns=None):
    """Compute residual volatility rows from a PriceMatrix.

    Example:
        matrix = price_matrix(price_rows)
        residual(matrix) returns rows like:
        ticker  factor               value  observations
        AAPL    residual_volatility  0.22   252
    """
    returns = matrix.returns[-window:]
    market = market_returns_for(returns, market_returns)

    results = [residual_for_stock(stock, market) for stock in returns.T]
    return make_price_factor_rows(matrix, results, "residual_volatility")
