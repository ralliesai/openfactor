import numpy as np

from openfactor.core.checks import require_columns


def winsorize(values, limit=3.0):
    """Clip extreme values around the cross-sectional median.

    Example:
        winsorize([1.0, 2.0, 100.0], limit=1.0)
        returns values with 100.0 clipped toward the median.
    """
    values = np.asarray(values, dtype=float).copy()
    good = np.isfinite(values)
    if good.sum() < 2:
        return values

    center = np.median(values[good])
    scale = np.median(np.abs(values[good] - center)) * 1.4826
    if not np.isfinite(scale) or scale == 0:
        scale = values[good].std()
    if not np.isfinite(scale) or scale == 0:
        return values

    return np.clip(values, center - limit * scale, center + limit * scale)


def standardize(values, weights=None):
    """Turn values into cross-sectional z-scores.

    Example:
        weights=None uses equal-weight mean and volatility.
        weights=[90, 10] uses the cap-weighted mean and equal-weighted volatility.
    """
    values = np.asarray(values, dtype=float).copy()
    good = np.isfinite(values)
    if good.sum() < 2:
        values[good] = 0.0
        return values

    if weights is None:
        center = values[good].mean()
        scale = values[good].std()
    else:
        weights = np.asarray(weights, dtype=float)
        fit = good & np.isfinite(weights) & (weights > 0)
        if fit.sum() < 2:
            return standardize(values)
        weights = weights[fit] / weights[fit].sum()
        center = (values[fit] * weights).sum()  # cap-weighted mean (USE4 §2.3)
        scale = values[fit].std()  # equal-weighted std, so mega-caps don't set the scale

    if scale == 0:
        values[good] = 0.0
    else:
        values[good] = (values[good] - center) / scale
    return values


def normalize_exposures(exposures, weights=None, limit=3.0):
    """Winsorize and standardize scalar factor exposures.

    Example:
        beta values [1.0, 2.0, 100.0]
        become model-ready exposures, with raw_value preserved.
    """
    require_columns(exposures, ["ticker", "factor", "group", "value"])
    frame = exposures.copy()
    if "raw_value" not in frame:
        frame["raw_value"] = frame["value"]
    else:
        frame["raw_value"] = frame["raw_value"].fillna(frame["value"])

    weights = exposure_weights(frame, weights)
    for factor in frame["factor"].unique():
        rows = frame["factor"] == factor
        group = frame.loc[rows, "group"].iloc[0]
        if group in ["sector", "industry"]:
            continue

        raw = frame.loc[rows, "raw_value"]
        factor_weights = None if weights is None else weights[rows]
        frame.loc[rows, "value"] = standardize(winsorize(raw, limit), factor_weights)

    return frame


def exposure_weights(exposures, weights):
    """Return one numeric weight per exposure row.

    Example:
        ticker-indexed market caps become exposure-row weights.
    """
    if weights is None:
        return None
    if hasattr(weights, "set_index"):
        weights = weights.set_index("ticker")["market_cap"]
    weights = weights.copy()
    weights.index = weights.index.astype(str)
    weights = weights.reindex(exposures["ticker"].astype(str))
    return weights.to_numpy(dtype=float)
