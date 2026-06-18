import numpy as np
import pandas as pd


def make_exposure_rows(tickers, values, factor, group, as_of_date, observations=None):
    """Return one exposure row per ticker value.

    Example:
        tickers = ["AAPL", "MSFT"]
        values = [1.2, np.nan]
        observations = [252, 12]

        output:
        as_of_date   ticker  factor  group  value  observations
        2024-01-31   AAPL    beta    price  1.2    252
        2024-01-31   MSFT    beta    price  NaN    12
    """
    tickers = np.asarray(tickers, dtype=str)
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    if observations is None:
        observations = np.isfinite(values).astype(int)

    return pd.DataFrame(
        {
            "as_of_date": str(as_of_date),
            "ticker": tickers,
            "factor": factor,
            "group": group,
            "value": values,
            "observations": np.asarray(observations, dtype=int),
        }
    )


def make_factor_rows(tickers, results, factor, group, as_of_date):
    """Return exposure rows from one FactorResult per ticker.

    Example:
        tickers = ["AAPL"]
        results = [FactorResult(value=1.2, observations=252)]

        output:
        ticker  factor  value  observations
        AAPL    beta    1.2    252
    """
    return make_exposure_rows(
        tickers=tickers,
        values=[result.value for result in results],
        factor=factor,
        group=group,
        as_of_date=as_of_date,
        observations=[result.observations for result in results],
    )


def make_price_factor_rows(matrix, results, factor):
    """Return price factor rows from one FactorResult per ticker.

    Example:
        matrix.tickers = ["AAPL"]
        matrix.dates[-1] = "2024-01-31"
        results = [FactorResult(value=1.2, observations=252)]

        output:
        ticker  factor       group  value  observations
        AAPL    beta         price  1.2    252
    """
    return make_factor_rows(
        tickers=matrix.tickers,
        results=results,
        factor=factor,
        group="price",
        as_of_date=matrix.dates[-1],
    )
