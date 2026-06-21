import numpy as np
import pandas as pd

from openfactor.core.checks import require_columns
from openfactor.portfolio.report import weights_for
from openfactor.portfolio.summary import clean_label, family


HORIZONS = [("1 Day", 1), ("1 Week", 5)]


def return_attribution(portfolio, snapshot):
    """Return contribution-to-return rows for the current positioning.

    Example:
        return_attribution(portfolio, snapshot) decomposes 1-day, 1-month, and
        1-quarter return into common-factor and idiosyncratic contributions.
    """
    panel = getattr(snapshot, "exposures_panel", None)
    if panel is None or panel.empty:
        return []
    require_columns(panel, ["factor", "group"])
    weights = weights_for(portfolio)
    contrib = daily_contributions(factor_exposure_panel(panel, weights), snapshot.factor_returns)
    specific = specific_contributions(snapshot.residual_returns, weights)
    groups = panel.drop_duplicates("factor").set_index("factor")["group"].to_dict()
    return attribution_rows(contrib, specific, groups)


def attribution_index(portfolio, snapshot):
    """Return horizon contributions keyed by factor, family, and section.

    Example:
        attribution_index(portfolio, snapshot)["factor"]["beta"] is beta's
        1-day, 1-month, and 1-quarter return contributions.
    """
    panel = getattr(snapshot, "exposures_panel", None)
    if panel is None or panel.empty:
        return None
    require_columns(panel, ["factor", "group"])
    weights = weights_for(portfolio)
    contrib = daily_contributions(factor_exposure_panel(panel, weights), snapshot.factor_returns)
    if contrib.empty:
        return None
    specific = specific_contributions(snapshot.residual_returns, weights).reindex(contrib.index)
    groups = panel.drop_duplicates("factor").set_index("factor")["group"].to_dict()
    dates = sorted(contrib.dropna(how="all").index)
    factor_h = trailing_sums(contrib, dates)
    spec_h = [float(value) for value in trailing_sums(specific, dates)]
    families = {factor: family(factor, groups) for factor in contrib.columns}

    def family_sum(name):
        members = [f for f, fam in families.items() if fam == name]
        return [float(series.reindex(members).sum()) for series in factor_h]

    common = [float(series.sum()) for series in factor_h]
    return {
        "factor": {f: [float(series.get(f, np.nan)) for series in factor_h] for f in contrib.columns},
        "family": {name: family_sum(name) for name in ["Market", "Style", "Sector", "Industry"]},
        "common": common,
        "specific": spec_h,
        "total": [c + s for c, s in zip(common, spec_h)],
    }


def factor_exposure_panel(panel, weights):
    """Return the portfolio's factor exposure on every panel date.

    Example:
        weighting each stock's daily exposures gives a date x factor table.
    """
    require_columns(panel, ["as_of_date", "factor", "ticker", "value"])
    frame = panel[["as_of_date", "factor", "ticker", "value"]].copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date.astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["weighted"] = frame["value"] * frame["ticker"].map(weights).fillna(0.0)
    exposure = frame.groupby(["as_of_date", "factor"])["weighted"].sum().unstack(fill_value=0.0)
    held = frame[frame["ticker"].isin(weights.index)].drop_duplicates(["as_of_date", "ticker"])
    market = held.groupby("as_of_date")["ticker"].apply(lambda names: weights.reindex(names).sum())
    exposure["market"] = market.reindex(exposure.index).fillna(0.0)
    return exposure


def daily_contributions(exposure, factor_returns):
    """Return each factor's daily return contribution, lagging exposure by one day.

    Example:
        yesterday's exposure times today's factor return is today's contribution.
    """
    if exposure.empty or factor_returns.empty:
        return pd.DataFrame()
    returns = factor_returns.copy()
    returns.index = pd.to_datetime(returns.index).date.astype(str)
    returns = returns.apply(pd.to_numeric, errors="coerce")
    exposure = exposure.copy()
    exposure.index = pd.to_datetime(exposure.index).date.astype(str)
    dates = sorted(set(exposure.index) | set(returns.index))
    lagged = exposure.reindex(dates).ffill().shift(1)
    factors = exposure.columns.intersection(returns.columns)
    return lagged[factors].reindex(returns.index) * returns[factors]


def specific_contributions(residual_returns, weights):
    """Return the portfolio's idiosyncratic return on every date.

    Example:
        weighting each stock's residual return gives one idiosyncratic return per day.
    """
    require_columns(residual_returns, ["date", "ticker", "residual_return"])
    frame = residual_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    frame["residual_return"] = pd.to_numeric(frame["residual_return"], errors="coerce")
    frame["weighted"] = frame["residual_return"] * frame["ticker"].map(weights).fillna(0.0)
    return frame.groupby("date")["weighted"].sum()


def trailing_sums(series, dates):
    """Return trailing-window sums for each reporting horizon.

    Example:
        trailing_sums(daily, dates) returns 1-day, 1-month, and 1-quarter sums.
    """
    return [series.reindex(dates[-window:]).sum() for _, window in HORIZONS]


def attribution_rows(contrib, specific, groups):
    """Return nested contribution rows ending in a reconciling total.

    Example:
        Common Factor, Market, Style, Sector, Industry, Idiosyncratic, then Total Return.
    """
    if contrib.empty:
        return []
    dates = sorted(contrib.dropna(how="all").index)
    if not dates:
        return []
    specific = specific.reindex(contrib.index)
    factor_h = trailing_sums(contrib, dates)
    spec_h = [float(value) for value in trailing_sums(specific, dates)]
    families = {factor: family(factor, groups) for factor in contrib.columns}
    active = set(contrib.columns[contrib.abs().sum() > 1e-9])

    rows = [totals_row("Common Factor", "section", [series.sum() for series in factor_h])]
    if "market" in contrib.columns:
        rows.append(factor_row("  Market", factor_h, "market"))
    for name in ["Style", "Sector", "Industry"]:
        members = [f for f, fam in families.items() if fam == name and f != "market" and f in active]
        if not members:
            continue
        rows.append(totals_row(f"  {name}", "group", [series.reindex(members).sum() for series in factor_h]))
        for factor in sorted(members, key=lambda f: -abs(factor_h[-1].get(f, 0.0))):
            rows.append(factor_row("    " + clean_label(factor), factor_h, factor))
    rows.append(totals_row("Idiosyncratic", "section", spec_h))
    rows.append(totals_row("Total Return", "total", [series.sum() + spec for series, spec in zip(factor_h, spec_h)]))
    return rows


def factor_row(label, factor_horizons, factor):
    """Return one factor's contribution across horizons.

    Example:
        Beta contributing +0.4% over a month becomes one row.
    """
    return {"label": label, "kind": "factor",
            "values": [float(series.get(factor, np.nan)) for series in factor_horizons]}


def totals_row(label, kind, values):
    """Return one subtotal or total contribution row.

    Example:
        Common Factor sums every factor contribution per horizon.
    """
    return {"label": label, "kind": kind, "values": [float(value) for value in values]}
