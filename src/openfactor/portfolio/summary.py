import numpy as np
import pandas as pd

from openfactor.model.risk import (
    active_holdings,
    factor_risk_report_from_covariance,
    portfolio_specific_risk,
)
from openfactor.portfolio.report import display_factor_name


GROUP_LABELS = {"price": "Style", "reference": "Style", "sector": "Industry", "industry": "Industry"}


def risk_decomposition(portfolio, snapshot):
    """Return a Barra-style nested risk decomposition and summary.

    Example:
        risk_decomposition(portfolio, snapshot) returns (summary, rows)
        with common-factor, style, industry, specific, and total rows.
    """
    exposures, cov = snapshot.exposures, snapshot.factor_covariance
    fr = factor_risk_report_from_covariance(exposures, portfolio, cov)
    active = active_holdings(portfolio, exposures)
    ar = factor_risk_report_from_covariance(exposures, active, cov)
    groups = exposures.drop_duplicates("factor").set_index("factor")["group"].to_dict()

    factor_var = float(fr["variance_contribution"].sum())
    specific = portfolio_specific_risk(portfolio, snapshot.specific_risk)
    specific_var = 0.0 if pd.isna(specific) else float(specific) ** 2
    total_var = (factor_var + specific_var) or np.nan

    fr = fr.assign(
        active=ar["exposure"].reindex(fr.index).fillna(0.0),
        family=[family(factor, groups) for factor in fr.index],
        pct=fr["variance_contribution"] / total_var,
    )
    rows = decomposition_rows(fr, factor_var / total_var, specific_var / total_var)
    summary = risk_summary(factor_var, specific, ar, active, snapshot)
    return summary, rows


def family(factor, groups):
    """Return the report family for one factor.

    Example:
        beta is Style, sector:Technology is Industry, market is Market.
    """
    if factor == "market":
        return "Market"
    return GROUP_LABELS.get(str(groups.get(factor)), "Style")


def clean_label(factor):
    """Return a factor label without a redundant group prefix.

    Example:
        sector:Technology becomes Technology under the Industry group.
    """
    name = display_factor_name(factor)
    return name.split(": ", 1)[1] if name.startswith(("Sector: ", "Industry: ")) else name


def decomposition_rows(fr, common_share, specific_share):
    """Return nested decomposition rows from enriched factor risk.

    Example:
        a Style group row precedes its individual style factor rows.
    """
    rows = [node("Common Factor", "section", pct=common_share)]
    if "market" in fr.index:
        rows.append(leaf(fr.loc["market"], "  ", "Market"))
    for name in ["Style", "Industry"]:
        sub = fr[fr["family"] == name]
        if sub.empty:
            continue
        rows.append(node(f"  {name}", "group", pct=float(sub["pct"].sum())))
        for factor, row in sub.sort_values("pct", ascending=False).iterrows():
            rows.append(leaf(row, "    ", clean_label(factor)))
    rows.append(node("Specific", "section", pct=specific_share))
    rows.append(node("Total Risk", "total", pct=1.0))
    return rows


def leaf(row, indent, label):
    """Return one factor row.

    Example:
        Beta exposure 0.32 at 22% volatility becomes a leaf row.
    """
    return node(
        indent + label,
        "factor",
        exposure=float(row["exposure"]),
        active=float(row["active"]),
        volatility=float(row["factor_volatility"]),
        pct=float(row["pct"]),
    )


def node(label, kind, exposure=np.nan, active=np.nan, volatility=np.nan, pct=np.nan):
    """Return one decomposition row.

    Example:
        node("Specific", "section", pct=0.36) is a bold subtotal row.
    """
    return {"label": label, "kind": kind, "exposure": exposure, "active": active, "volatility": volatility, "pct": pct}


def risk_summary(factor_var, specific, ar, active, snapshot):
    """Return standalone portfolio and active (tracking-error) risks.

    Example:
        total, common-factor, specific, and tracking-error volatilities.
    """
    active_factor = np.sqrt(max(float(ar["variance_contribution"].sum()), 0.0))
    active_specific = portfolio_specific_risk(active, snapshot.specific_risk, strict=False)
    common = np.sqrt(max(factor_var, 0.0))
    spec = 0.0 if pd.isna(specific) else float(specific)
    a_spec = 0.0 if pd.isna(active_specific) else float(active_specific)
    return {
        "total": float(np.sqrt(common**2 + spec**2)),
        "common_factor": common,
        "specific": specific,
        "tracking_error": float(np.sqrt(active_factor**2 + a_spec**2)),
        "active_factor": active_factor,
        "active_specific": active_specific,
    }
