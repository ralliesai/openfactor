import numpy as np
import pandas as pd

from openfactor.core.checks import require_columns


def exposure_matrix(exposures):
    """Turn exposure rows into a ticker x factor table.

    Example input rows:
        ticker  factor  value
        AAPL    beta    1.2
        AAPL    size    10.0
        MSFT    beta    0.9

    Example output:
               beta  size
        AAPL   1.2   10.0
        MSFT   0.9   nan
    """
    require_columns(exposures, ["ticker", "factor", "value"])

    frame = exposures.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["factor"] = frame["factor"].astype(str)
    if frame.duplicated(["ticker", "factor"]).any():
        raise ValueError("duplicate exposure rows for ticker/factor")

    tickers = np.array(sorted(frame["ticker"].unique()))
    factors = np.array(sorted(frame["factor"].unique()))
    ticker_index = {ticker: row for row, ticker in enumerate(tickers)}
    factor_index = {factor: col for col, factor in enumerate(factors)}

    # matrix shape: tickers x factors. Missing exposures stay np.nan.
    matrix = np.full((len(tickers), len(factors)), np.nan)
    for row in frame[["ticker", "factor", "value"]].itertuples(index=False):
        matrix[ticker_index[row.ticker], factor_index[row.factor]] = float(row.value)

    return pd.DataFrame(matrix, index=tickers, columns=factors)


def model_exposure_matrix(exposures):
    """Return exposures ready for factor regression.

    Example input rows:
        ticker  factor         group     value
        AAPL    size           price     1.2
        AAPL    sector:Technology  sector  1.0

    Example output:
        missing exposures become 0.0, which is neutral after normalization.
    """
    require_columns(exposures, ["factor", "group"])
    matrix = exposure_matrix(exposures)
    one_hot = exposures.loc[exposures["group"].isin(["sector", "industry"]), "factor"]
    one_hot = sorted(set(one_hot.astype(str)) & set(matrix.columns))
    if one_hot:
        matrix[one_hot] = matrix[one_hot].fillna(0.0)
    return matrix.fillna(0.0)
