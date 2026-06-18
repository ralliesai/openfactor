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


def sector_weights(sectors, weights):
    """Return market-cap weights for sector constraints.

    Example:
        if Technology is 80% of fitted market cap, its sector weight is 0.80.
    """
    totals = sectors.T @ weights
    if totals.sum() == 0:
        return np.full(sectors.shape[1], 1 / sectors.shape[1])
    return totals / totals.sum()


def fit_cross_section(x_frame, returns, groups=None, market_caps=None, return_limit=5.0):
    """Fit one day's Barra-like factor returns and residuals.

    Example:
        two sectors plus beta produce market, sector, and beta returns.
        Sector returns are market-cap weighted to sum to zero.
    """
    groups = groups or {}
    x_frame = pd.DataFrame(x_frame).astype(float)
    returns = np.asarray(returns, dtype=float)
    returns = winsorize(returns, return_limit)
    weights = np.ones(len(returns)) if market_caps is None else np.asarray(market_caps, dtype=float)

    sector_cols = [col for col in x_frame.columns if groups.get(col) == "sector"]
    style_cols = [col for col in x_frame.columns if col not in sector_cols]
    sectors = x_frame[sector_cols].to_numpy(dtype=float) if sector_cols else np.empty((len(x_frame), 0))
    styles = x_frame[style_cols].to_numpy(dtype=float) if style_cols else np.empty((len(x_frame), 0))

    sector_ok = sectors.sum(axis=1) > 0 if sector_cols else np.ones(len(x_frame), dtype=bool)
    x = np.column_stack([np.ones(len(x_frame)), sectors[:, :-1], styles])
    good = (
        np.isfinite(returns)
        & np.isfinite(weights)
        & (weights > 0)
        & sector_ok
        & np.isfinite(x).all(axis=1)
    )

    names = [MARKET_FACTOR] + sector_cols + style_cols
    factor_returns = pd.Series(np.nan, index=names)
    residuals = pd.Series(np.nan, index=x_frame.index)
    if good.sum() <= x.shape[1]:
        return factor_returns, residuals

    fitted = weighted_lstsq(x[good], returns[good], weights[good])
    style_start = 1 + max(len(sector_cols) - 1, 0)

    if sector_cols:
        raw_sector = np.r_[fitted[1:style_start], 0.0]
        shift = sector_weights(sectors[good], weights[good]) @ raw_sector
        factor_returns[MARKET_FACTOR] = fitted[0] + shift
        factor_returns.loc[sector_cols] = raw_sector - shift
    else:
        factor_returns[MARKET_FACTOR] = fitted[0]

    factor_returns.loc[style_cols] = fitted[style_start:]
    predicted = (
        factor_returns[MARKET_FACTOR]
        + sectors @ factor_returns.loc[sector_cols].fillna(0.0).to_numpy()
        + styles @ factor_returns.loc[style_cols].fillna(0.0).to_numpy()
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
