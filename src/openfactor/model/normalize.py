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


def standardize(values):
    """Turn values into cross-sectional z-scores.

    Example:
        standardize([1.0, 2.0, 3.0]) returns roughly [-1.22, 0.0, 1.22].
    """
    values = np.asarray(values, dtype=float).copy()
    good = np.isfinite(values)
    if good.sum() < 2:
        values[good] = 0.0
        return values

    scale = values[good].std()
    if scale == 0:
        values[good] = 0.0
    else:
        values[good] = (values[good] - values[good].mean()) / scale
    return values


def normalize_exposures(exposures, limit=3.0):
    """Winsorize and standardize scalar factor exposures.

    Example:
        beta values [1.0, 2.0, 100.0]
        become model-ready z-scores, with raw_value preserved.
    """
    require_columns(exposures, ["ticker", "factor", "group", "value"])
    frame = exposures.copy()
    if "raw_value" not in frame:
        frame["raw_value"] = frame["value"]
    else:
        frame["raw_value"] = frame["raw_value"].fillna(frame["value"])

    for factor in frame["factor"].unique():
        rows = frame["factor"] == factor
        group = frame.loc[rows, "group"].iloc[0]
        if group in ["sector", "industry"]:
            continue

        raw = frame.loc[rows, "raw_value"]
        frame.loc[rows, "value"] = standardize(winsorize(raw, limit))

    return frame
