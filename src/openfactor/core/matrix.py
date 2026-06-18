from dataclasses import dataclass

import numpy as np
import pandas as pd

from openfactor.core.checks import require_columns
from openfactor.core.returns import price_returns


@dataclass(frozen=True)
class PriceMatrix:
    """Price arrays used by factor math.

    Example:
        dates = ["2024-01-02", "2024-01-03"]
        tickers = ["AAPL"]
        close shape is dates x tickers: [[100.0], [101.0]]
        returns shape is one fewer date x tickers: [[0.01]]
    """

    dates: np.ndarray
    tickers: np.ndarray
    close: np.ndarray
    returns: np.ndarray
    volume: np.ndarray | None = None


def price_matrix(prices, require_volume=False):
    """Turn price rows into arrays.

    Example input rows:
        date        ticker  close  volume
        2024-01-02  AAPL    185.0  10
        2024-01-02  MSFT    370.0  20
        2024-01-03  AAPL    184.0  11
        2024-01-03  MSFT    369.0  21

    Example output:
        dates = ["2024-01-02", "2024-01-03"]
        tickers = ["AAPL", "MSFT"]
        close = [[185.0, 370.0], [184.0, 369.0]]
        returns = [[-0.0054, -0.0027]]

    Shape:
        close[date_index, ticker_index]
        returns[return_date_index, ticker_index]
        volume is only built when require_volume=True

    Duplicate date/ticker rows raise an error.
    Invalid close or volume observations stay np.nan.
    """
    columns = ["date", "ticker", "close"]
    if require_volume:
        columns.append("volume")
    require_columns(prices, columns)
    if prices.empty:
        raise ValueError("prices is empty")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate price rows for date/ticker")

    dates = np.array(sorted(frame["date"].unique()))
    tickers = np.array(sorted(frame["ticker"].unique()))
    date_index = {date: row for row, date in enumerate(dates)}
    ticker_index = {ticker: col for col, ticker in enumerate(tickers)}

    # Each matrix is dates x tickers. Missing observations stay np.nan.
    close = np.full((len(dates), len(tickers)), np.nan)
    volume = np.full_like(close, np.nan) if require_volume else None

    for row in frame[columns].itertuples(index=False):
        i = date_index[row.date]
        j = ticker_index[row.ticker]
        close_value = float(row.close)
        if np.isfinite(close_value) and close_value > 0:
            close[i, j] = close_value

        if require_volume:
            volume_value = float(row.volume)
            if np.isfinite(volume_value) and volume_value >= 0:
                volume[i, j] = volume_value

    return PriceMatrix(
        dates=dates,
        tickers=tickers,
        close=close,
        returns=price_returns(close),
        volume=volume,
    )
