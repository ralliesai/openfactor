import numpy as np


def price_returns(close):
    """Return close-to-close returns from a close matrix.

    Example:
        close = np.array([[100.0], [105.0]])
        price_returns(close) returns [[0.05]]
    """
    # close shape: dates x tickers. Returns have one fewer date row.
    previous = close[:-1]
    current = close[1:]
    returns = np.full(current.shape, np.nan, dtype=float)
    good = np.isfinite(current) & np.isfinite(previous) & (previous > 0)
    returns[good] = current[good] / previous[good] - 1
    return returns


def market_returns_for(returns, market_returns=None):
    """Return the market return series used by beta-style factors.

    Example:
        returns = np.array([[0.01, 0.03], [-0.02, 0.00]])
        market_returns_for(returns) returns [0.02, -0.01]
    """
    if market_returns is not None:
        market = np.asarray(market_returns, dtype=float)
    else:
        market = np.full(len(returns), np.nan)
        has_returns = np.isfinite(returns).any(axis=1)
        market[has_returns] = np.nanmean(returns[has_returns], axis=1)

    if len(market) != len(returns):
        raise ValueError("market_returns must match the number of return rows")

    return market
