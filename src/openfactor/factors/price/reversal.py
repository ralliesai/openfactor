import numpy as np
import pandas as pd

from openfactor.factors.output import make_price_factor_rows
from openfactor.factors.result import FactorResult


def long_term_for_stock(dates, close, start_years=3, end_years=1):
    """Return negative return from three years ago to one year ago.

    Example:
        price 100 three years ago and 150 one year ago returns -0.50.
    """
    observations = int(np.isfinite(close).sum())
    start = close_on_or_before(dates, close, years_ago(dates[-1], start_years))
    end = close_on_or_before(dates, close, years_ago(dates[-1], end_years))
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return FactorResult(np.nan, observations)
    return FactorResult(-(end / start - 1), observations)


def close_on_or_before(dates, close, target):
    """Return latest finite close on or before a target date.

    Example:
        target on a weekend uses the prior trading day's close.
    """
    target = np.datetime64(target)
    date_values = pd.to_datetime(dates).to_numpy(dtype="datetime64[D]")
    index = np.searchsorted(date_values, target, side="right") - 1
    while index >= 0:
        value = close[index]
        if np.isfinite(value):
            return value
        index -= 1
    return np.nan


def years_ago(value, years):
    """Return the same date a number of years earlier.

    Example:
        years_ago("2026-06-16", 1) returns "2025-06-16".
    """
    day = pd.to_datetime(value).date()
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, month=2, day=28)


def compute(matrix):
    """Compute long-term reversal rows from a PriceMatrix.

    Example:
        compute(matrix) returns rows like:
        ticker  factor              value
        AAPL    long_term_reversal -0.50
    """
    results = [long_term_for_stock(matrix.dates, close) for close in matrix.close.T]
    return make_price_factor_rows(matrix, results, "long_term_reversal")
