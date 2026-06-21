import re

import numpy as np
import pandas as pd


ALLOWED_SHARE_SOURCES = {
    "EntityCommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding_sum",
    "CommonStockSharesOutstanding",
    "CommonStockSharesOutstanding_sum",
}
CLASS_EQUIVALENT_SOURCE = re.compile(
    r"^CommonStockSharesOutstanding_class_equivalent_[a-z0-9]+_to_[a-z0-9]+$"
)


def validate_snapshot(snapshot):
    """Raise if a factor snapshot has broken core tables.

    Example:
        duplicate AAPL/beta exposure rows raise ValueError.
    """
    require_columns(snapshot.universe, ["ticker"], "universe")
    require_columns(snapshot.exposures, ["ticker", "factor", "value"], "exposures")
    require_columns(snapshot.idiosyncratic_risk, ["ticker", "idiosyncratic_risk"], "idiosyncratic_risk")

    if snapshot.universe.empty:
        raise ValueError("universe is empty")
    if snapshot.universe["ticker"].duplicated().any():
        raise ValueError("universe has duplicate tickers")
    if snapshot.exposures.duplicated(["ticker", "factor"]).any():
        raise ValueError("exposures has duplicate ticker/factor rows")
    if snapshot.idiosyncratic_risk["ticker"].duplicated().any():
        raise ValueError("idiosyncratic_risk has duplicate tickers")

    factors = set(snapshot.factor_returns.columns)
    covariance = snapshot.factor_covariance
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("factor_covariance index and columns must match")
    if not set(covariance.columns).issubset(factors):
        raise ValueError("factor_covariance has unknown factors")

    exposure_factors = set(snapshot.exposures["factor"].astype(str))
    missing_exposures = sorted(set(covariance.columns) - exposure_factors - {"market"})
    if missing_exposures:
        raise ValueError(f"factor_covariance has factors missing exposures: {missing_exposures[:10]}")


def validate_private_inputs(result):
    """Raise if private model inputs can produce invalid public outputs.

    Example:
        weighted-average EPS shares in fundamentals raise before publish.
    """
    validate_fundamental_share_sources(result.fundamentals)
    validate_market_cap_formula(result.fundamentals, result.prices)


def validate_fundamental_share_sources(fundamentals):
    """Raise when shares_outstanding does not come from instant share facts.

    Example:
        WeightedAverageNumberOfSharesOutstandingBasic is a period EPS denominator,
        so it cannot be used for point-in-time market cap.
    """
    if fundamentals is None or fundamentals.empty:
        return
    require_columns(
        fundamentals,
        ["ticker", "as_of_date", "shares_outstanding", "shares_outstanding_source"],
        "fundamentals",
    )

    frame = fundamentals[["ticker", "as_of_date", "shares_outstanding", "shares_outstanding_source"]].copy()
    frame["shares_outstanding"] = pd.to_numeric(frame["shares_outstanding"], errors="coerce")
    finite = frame["shares_outstanding"].notna()
    valid = frame["shares_outstanding_source"].map(valid_share_source)
    bad = frame[finite & ~valid]
    if not bad.empty:
        sample = bad.head(10).to_dict("records")
        raise ValueError(f"fundamentals has invalid shares_outstanding_source rows: {sample}")


def valid_share_source(source):
    """Return True for point-in-time share-count source labels.

    Example:
        EntityCommonStockSharesOutstanding is valid; WeightedAverage... is not.
    """
    source = str(source)
    return source in ALLOWED_SHARE_SOURCES or bool(CLASS_EQUIVALENT_SOURCE.fullmatch(source))


def validate_market_cap_formula(fundamentals, prices):
    """Raise when stored market cap is not shares times unadjusted close.

    Example:
        WAT market_cap must equal WAT shares_outstanding times same-day raw close.
    """
    if fundamentals is None or fundamentals.empty:
        return
    require_columns(fundamentals, ["ticker", "as_of_date", "shares_outstanding", "market_cap"], "fundamentals")
    require_columns(prices, ["ticker", "date", "unadjusted_close"], "prices")

    frame = fundamentals[["ticker", "as_of_date", "shares_outstanding", "market_cap"]].copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date.astype(str)
    frame["shares_outstanding"] = pd.to_numeric(frame["shares_outstanding"], errors="coerce")
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")

    price = prices[["ticker", "date", "unadjusted_close"]].copy()
    price["date"] = pd.to_datetime(price["date"]).dt.date.astype(str)
    price["unadjusted_close"] = pd.to_numeric(price["unadjusted_close"], errors="coerce")
    price = price.rename(columns={"date": "as_of_date", "unadjusted_close": "market_cap_close"})

    joined = frame.merge(price, on=["ticker", "as_of_date"], how="left")
    stored = joined["market_cap"]
    expected = joined["shares_outstanding"] * joined["market_cap_close"]
    missing_price = stored.notna() & joined["market_cap_close"].isna()
    if missing_price.any():
        sample = joined.loc[missing_price, ["ticker", "as_of_date"]].head(10).to_dict("records")
        raise ValueError(f"fundamentals market_cap rows missing unadjusted close: {sample}")

    checked = stored.notna() & expected.notna()
    diff = (stored - expected).abs()
    tolerance = np.maximum(1.0, expected.abs() * 1e-8)
    bad = joined[checked & (diff > tolerance)]
    if not bad.empty:
        sample = bad.head(10)[["ticker", "as_of_date", "shares_outstanding", "market_cap"]].to_dict("records")
        raise ValueError(f"fundamentals market_cap is not shares_outstanding * unadjusted_close: {sample}")


def require_columns(frame, columns, name):
    """Raise if a table is missing required columns.

    Example:
        require_columns(frame, ["ticker"], "universe") checks for ticker.
    """
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def clean_tickers(tickers):
    """Return uppercase unique tickers in input order.

    Example:
        ["sndk", "SNDK", "AMC"] returns ["SNDK", "AMC"].
    """
    clean = []
    seen = set()
    for ticker in tickers or []:
        value = str(ticker).strip().upper()
        if value and value not in seen:
            clean.append(value)
            seen.add(value)
    return clean
