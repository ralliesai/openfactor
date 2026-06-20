def validate_snapshot(snapshot):
    """Raise if a factor snapshot has broken core tables.

    Example:
        duplicate AAPL/beta exposure rows raise ValueError.
    """
    require_columns(snapshot.universe, ["ticker"], "universe")
    require_columns(snapshot.exposures, ["ticker", "factor", "value"], "exposures")
    require_columns(snapshot.specific_risk, ["ticker", "specific_risk"], "specific_risk")

    if snapshot.universe.empty:
        raise ValueError("universe is empty")
    if snapshot.universe["ticker"].duplicated().any():
        raise ValueError("universe has duplicate tickers")
    if snapshot.exposures.duplicated(["ticker", "factor"]).any():
        raise ValueError("exposures has duplicate ticker/factor rows")
    if snapshot.specific_risk["ticker"].duplicated().any():
        raise ValueError("specific_risk has duplicate tickers")

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
