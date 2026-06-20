import numpy as np
import pandas as pd
from tqdm import tqdm

from openfactor.core.matrix import PriceMatrix
from openfactor.factors.defaults import default_price_factors, default_reference_factors
from openfactor.model.exposures import exposure_matrix, model_exposure_matrix
from openfactor.model.normalize import normalize_exposures, winsorize


MARKET_FACTOR = "market"


def as_of_price_matrix(matrix, close_rows):
    """Return a PriceMatrix ending before the modeled return date.

    Example:
        close_rows=10 gives dates[:10], close[:10], and returns[:9].
    """
    volume = None if matrix.volume is None else matrix.volume[:close_rows]
    return PriceMatrix(
        dates=matrix.dates[:close_rows],
        tickers=matrix.tickers,
        close=matrix.close[:close_rows],
        returns=matrix.returns[: close_rows - 1],
        volume=volume,
    )


def rolling_exposures(
    matrix,
    static_exposures,
    return_row,
    price_factors,
    reference_history=None,
    reference_factors=None,
):
    """Return normalized exposures known before one return row.

    Example:
        return_row=10 uses prices through date index 10, then models return 10.
    """
    as_of = as_of_price_matrix(matrix, return_row + 1)
    frames = [factor.compute(as_of) for factor in price_factors]
    if reference_history is None:
        if static_exposures is not None and not static_exposures.empty:
            frames.append(static_exposures)
    else:
        date = str(matrix.dates[return_row])
        rows = reference_history[reference_history["as_of_date"] == date]
        frames += [factor.compute(rows, date) for factor in reference_factors]
        if static_exposures is not None and not static_exposures.empty:
            frames.append(static_exposures)
    weights = None if reference_history is None else market_cap_weights(rows)
    return normalize_exposures(pd.concat(frames, ignore_index=True), weights)


def common_exposure_matrix(exposures, tickers, min_group_members=5):
    """Return factors broad enough for cross-sectional regression.

    Example:
        sectors are always kept; tiny industry groups need min_group_members.
    """
    raw = exposure_matrix(exposures).reindex(tickers)
    x_frame = model_exposure_matrix(exposures).reindex(tickers)
    groups = factor_groups(exposures)
    one_hot = exposures.loc[exposures["group"].isin(["sector", "industry"]), "factor"]
    one_hot = set(one_hot.astype(str))

    keep = []
    for factor in x_frame.columns:
        if factor not in raw or not raw[factor].notna().any():
            keep.append(False)
        elif groups.get(factor) == "sector":
            keep.append(True)
        elif factor in one_hot:
            keep.append((x_frame[factor] > 0).sum() >= min_group_members)
        else:
            keep.append(True)
    return x_frame.loc[:, keep]


def factor_groups(exposures):
    """Return factor group names keyed by factor id.

    Example:
        beta is price and sector:Technology is sector.
    """
    return exposures.drop_duplicates("factor").set_index("factor")["group"].to_dict()


def market_caps_for(reference_history, date, tickers):
    """Return same-day market caps aligned to tickers.

    Example:
        AAPL then MSFT rows become [AAPL market cap, MSFT market cap].
    """
    if reference_history is None or "market_cap" not in reference_history:
        return None
    rows = reference_history[reference_history["as_of_date"].astype(str) == str(date)]
    caps = rows.drop_duplicates("ticker").set_index("ticker")["market_cap"]
    return caps.reindex(tickers).to_numpy(dtype=float)


def market_cap_weights(reference):
    """Return market-cap weights indexed by ticker.

    Example:
        ticker rows with market_cap become normalization weights.
    """
    if reference is None or reference.empty or "market_cap" not in reference:
        return None
    return reference.drop_duplicates("ticker").set_index("ticker")["market_cap"]


def weighted_lstsq(x, y, weights):
    """Solve a market-cap weighted least squares fit.

    Example:
        a stock with 4x market cap gets 2x row weight after sqrt weighting.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    root = np.sqrt(weights)
    return np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]


def constrained_lstsq(x, y, weights, constraints):
    """Solve weighted least squares with exact zero-sum constraints.

    Example:
        constraints=[[0, 1, 1]] forces the last two coefficients to sum to 0.
    """
    if len(constraints) == 0:
        return weighted_lstsq(x, y, weights)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    root = np.sqrt(weights)
    x_weighted = x * root[:, None]
    y_weighted = y * root
    zero = np.zeros((len(constraints), len(constraints)))
    left = np.block(
        [
            [x_weighted.T @ x_weighted, constraints.T],
            [constraints, zero],
        ]
    )
    right = np.r_[x_weighted.T @ y_weighted, np.zeros(len(constraints))]
    return np.linalg.lstsq(left, right, rcond=None)[0][: x.shape[1]]


def group_columns(columns, groups, group):
    """Return factor columns for one group.

    Example:
        group_columns(["beta", "sector:Tech"], groups, "sector")
        returns ["sector:Tech"].
    """
    return [column for column in columns if groups.get(column) == group]


def classification_constraints(x_frame, groups, weights):
    """Return market-relative sector and industry constraints.

    Example:
        sector returns sum to zero across the market.
        industry returns sum to zero inside their parent sector.
    """
    columns = list(x_frame.columns)
    sectors = group_columns(columns, groups, "sector")
    industries = group_columns(columns, groups, "industry")
    rows = []

    if sectors:
        rows.append(constraint_row(x_frame, sectors, weights))

    if sectors and industries:
        parents = industry_parent_sectors(x_frame, sectors, industries, weights)
        if not parents.empty:
            for _, group in parents.groupby("sector"):
                rows.append(constraint_row(x_frame, list(group["industry"]), weights))

    return np.asarray([row for row in rows if np.any(row)], dtype=float)


def constraint_row(x_frame, columns, weights):
    """Return one cap-weighted zero-sum row for selected columns.

    Example:
        columns=["sector:Tech", "sector:Bank"] constrains their weighted return.
    """
    row = np.zeros(len(x_frame.columns))
    totals = x_frame[columns].multiply(weights, axis=0).sum()
    for column, total in totals.items():
        row[x_frame.columns.get_loc(column)] = total
    scale = np.abs(row).sum()
    return row if scale == 0 else row / scale


def industry_parent_sectors(x_frame, sectors, industries, weights):
    """Return each industry's dominant sector.

    Example:
        industry:Semiconductors maps to sector:Technology.
    """
    rows = []
    for industry in industries:
        members = x_frame[industry].to_numpy(dtype=float) > 0
        if not members.any():
            continue
        sector_weights = x_frame.loc[members, sectors].multiply(weights[members], axis=0).sum()
        if sector_weights.sum() > 0:
            rows.append({"industry": industry, "sector": sector_weights.idxmax()})
    return pd.DataFrame(rows)


def fit_cross_section(x_frame, returns, groups=None, market_caps=None, return_limit=5.0):
    """Fit one day's Barra-like factor returns and residuals.

    Example:
        two sectors plus beta produce market, sector, and beta returns.
        Sector returns sum to zero across the market.
        Industry returns sum to zero inside each parent sector.
    """
    groups = groups or {}
    x_frame = pd.DataFrame(x_frame).astype(float)
    returns = np.asarray(returns, dtype=float)
    returns = winsorize(returns, return_limit)
    weights = np.ones(len(returns)) if market_caps is None else np.asarray(market_caps, dtype=float)

    columns = list(x_frame.columns)
    sector_cols = group_columns(columns, groups, "sector")
    factor_cols = sector_cols + [column for column in columns if column not in sector_cols]
    design = pd.concat([pd.Series(1.0, index=x_frame.index, name=MARKET_FACTOR), x_frame[factor_cols]], axis=1)
    sectors = x_frame[sector_cols].to_numpy(dtype=float) if sector_cols else np.empty((len(x_frame), 0))
    sector_ok = sectors.sum(axis=1) > 0 if sector_cols else np.ones(len(x_frame), dtype=bool)
    good = (
        np.isfinite(returns)
        & np.isfinite(weights)
        & (weights > 0)
        & sector_ok
        & np.isfinite(design.to_numpy(dtype=float)).all(axis=1)
    )

    names = [MARKET_FACTOR] + factor_cols
    factor_returns = pd.Series(np.nan, index=names)
    residuals = pd.Series(np.nan, index=x_frame.index)
    constraints = classification_constraints(x_frame.loc[good, factor_cols], groups, weights[good])
    constraints = np.column_stack([np.zeros(len(constraints)), constraints])
    if good.sum() <= design.shape[1] - len(constraints):
        return factor_returns, residuals

    fitted = constrained_lstsq(
        design.loc[good].to_numpy(dtype=float),
        returns[good],
        weights[good],
        constraints,
    )
    factor_returns.loc[names] = fitted
    predicted = (
        factor_returns[MARKET_FACTOR]
        + x_frame[factor_cols].to_numpy(dtype=float) @ factor_returns.loc[factor_cols].fillna(0.0).to_numpy()
    )
    residuals.iloc[good] = returns[good] - predicted[good]
    return factor_returns, residuals


def factor_model_history(
    matrix,
    exposures,
    window=None,
    min_group_members=5,
    price_factors=None,
    reference_history=None,
    reference_factors=None,
    progress_label=None,
):
    """Estimate daily factor returns and stock residuals.

    Example:
        factor_model_history(matrix, exposures, window=252)
        returns factor_return_rows and residual_return_rows.
    """
    price_factors = price_factors or default_price_factors()
    reference_factors = reference_factors or default_reference_factors()
    static = exposures.loc[exposures["group"] != "price"].copy()
    if reference_history is not None:
        reference_history = reference_history.copy()
        reference_history["as_of_date"] = reference_history["as_of_date"].astype(str)
        static = static.loc[~static["group"].isin(["reference", "sector", "industry"])]
    start = 0 if window is None else max(0, len(matrix.returns) - window)

    factor_rows = []
    residual_rows = []
    rows = range(start, len(matrix.returns))
    if progress_label:
        rows = tqdm(rows, desc=progress_label, unit="day", dynamic_ncols=True)

    for row in rows:
        daily_exposures = rolling_exposures(
            matrix,
            static,
            row,
            price_factors,
            reference_history,
            reference_factors,
        )
        x_frame = common_exposure_matrix(
            daily_exposures,
            matrix.tickers,
            min_group_members,
        )
        factors, residuals = fit_cross_section(
            x_frame,
            matrix.returns[row],
            factor_groups(daily_exposures),
            market_caps_for(reference_history, matrix.dates[row], matrix.tickers),
        )
        factor_rows.append(factors)
        residual_rows.append(residuals.reindex(matrix.tickers))

    return (
        pd.DataFrame(factor_rows, index=matrix.dates[1:][start:]),
        pd.DataFrame(residual_rows, index=matrix.dates[1:][start:]),
    )
