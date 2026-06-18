import numpy as np
import pandas as pd

from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def seasonality_for_stock(dates, close, years=3):
    """Return average same-month return in prior years.

    Example:
        prior June returns of 1%, 2%, and 3% return 0.02.
    """
    observations = int(np.isfinite(close).sum())
    date_values = pd.to_datetime(dates)
    as_of = date_values[-1]
    values = []
    for year in range(as_of.year - years, as_of.year):
        same_month = (date_values.year == year) & (date_values.month == as_of.month)
        value = month_return(close[same_month])
        if np.isfinite(value):
            values.append(value)
    return FactorResult(np.nan if not values else float(np.mean(values)), observations)


def month_return(close):
    """Return one calendar month's price return.

    Example:
        prices from 100 to 102 return 0.02.
    """
    values = close[np.isfinite(close)]
    if len(values) < 2 or values[0] <= 0:
        return np.nan
    return values[-1] / values[0] - 1


def compute(matrix, years=3):
    """Compute seasonality rows from a PriceMatrix.

    Example:
        compute(matrix) returns rows like:
        ticker  factor       value
        AAPL    seasonality  0.02
    """
    results = [seasonality_for_stock(matrix.dates, close, years) for close in matrix.close.T]
    return make_price_factor_rows(matrix, results, "seasonality")
