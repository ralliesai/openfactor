import numpy as np
import pandas as pd

from openfactor.model.risk import (
    active_holdings,
    benchmark_weights,
    factor_risk_report_from_covariance,
    portfolio_factor_exposure,
)
from openfactor.model.specific_risk import portfolio_specific_risk
from openfactor.portfolio.summary import clean_label, family


TRADING_DAYS = 252
CONFIDENCE = {"95%": 1.645, "99%": 2.326}


def active_risk_report(portfolio, snapshot):
    """Return the factor decomposition in active (tracking-error) space.

    Example:
        each factor's active exposure and its share of the tracking-error budget,
        measured against the cap-weighted universe benchmark.
    """
    exposures, cov = snapshot.exposures, snapshot.factor_covariance
    active = active_holdings(portfolio, exposures)
    report = factor_risk_report_from_covariance(exposures, active, cov)
    active_specific = portfolio_specific_risk(active, snapshot.specific_risk, strict=False)
    specific_var = 0.0 if pd.isna(active_specific) else float(active_specific) ** 2
    te2 = float(report["variance_contribution"].sum()) + specific_var
    tracking_error = float(np.sqrt(max(te2, 0.0)))
    groups = exposures.drop_duplicates("factor").set_index("factor")["group"].to_dict()
    rows = [
        {
            "factor": factor,
            "label": clean_label(factor),
            "family": family(factor, groups),
            "active_exposure": float(row["exposure"]),
            "factor_volatility": float(row["factor_volatility"]),
            "te_contribution": contribution(float(row["variance_contribution"]), tracking_error),
            "te_share": share(float(row["variance_contribution"]), te2),
        }
        for factor, row in report.iterrows()
        if factor != "market"
    ]
    return {
        "rows": rows,
        "tracking_error": tracking_error,
        "specific_contribution": contribution(specific_var, tracking_error),
        "specific_share": share(specific_var, te2),
    }


def benchmark_profile(snapshot):
    """Describe the cap-weighted benchmark so it reads as a credible proxy.

    Example:
        1000 constituents, top-10 weight, effective names, and the largest style
        tilt (near zero, since the benchmark defines the market).
    """
    weights = benchmark_weights(snapshot.exposures).sort_values(ascending=False)
    bench = weights.rename("allocation").reset_index()
    exposure = portfolio_factor_exposure(snapshot.exposures, bench)
    groups = snapshot.exposures.drop_duplicates("factor").set_index("factor")["group"].to_dict()
    style = [abs(float(value)) for factor, value in exposure.items()
             if groups.get(factor) in ("price", "reference")]
    return {
        "constituents": int(len(weights)),
        "top10_weight": float(weights.head(10).sum()),
        "effective_names": 1.0 / float((weights ** 2).sum()),
        "max_style_tilt": max(style) if style else None,
    }


def specific_by_name(portfolio, snapshot):
    """Return each holding's share of the portfolio's idiosyncratic risk.

    Example:
        a concentrated name with high residual volatility dominates the
        idiosyncratic risk and surfaces at the top.
    """
    weights = portfolio.set_index("ticker")["allocation"]
    risks = snapshot.specific_risk.set_index("ticker")["specific_risk"].reindex(weights.index)
    variance = (weights * risks.fillna(0.0)) ** 2
    total = float(variance.sum())
    shares = variance / total if total > 0 else variance * 0.0
    names = [
        {
            "ticker": str(ticker),
            "weight": float(weights[ticker]),
            "specific_vol": none_if_nan(risks.get(ticker)),
            "share": float(shares[ticker]),
        }
        for ticker in shares.sort_values(ascending=False).index
    ]
    return {
        "names": names,
        "total_specific": float(np.sqrt(total)),
        "top_share": names[0]["share"] if names else None,
        "effective_names": 1.0 / float((shares ** 2).sum()) if total > 0 else None,
    }


def tail_metrics(portfolio, snapshot, total_risk, tracking_error):
    """Return parametric VaR and predicted beta to the benchmark.

    Example:
        a 33% annual volatility implies about a 3.4% one-day 95% VaR.
    """
    var = {
        name: {
            "total_1d": z * daily(total_risk),
            "active_1d": z * daily(tracking_error),
        }
        for name, z in CONFIDENCE.items()
    }
    return {"var": var, "beta": predicted_beta(portfolio, snapshot)}


def predicted_beta(portfolio, snapshot):
    """Return the model-predicted beta to the cap-weighted benchmark.

    Example:
        beta = covariance(portfolio, benchmark) / variance(benchmark).
    """
    exposures, cov = snapshot.exposures, snapshot.factor_covariance
    benchmark = benchmark_weights(exposures).rename("allocation").reset_index()
    port_x = portfolio_factor_exposure(exposures, portfolio)
    bench_x = portfolio_factor_exposure(exposures, benchmark)
    factors = port_x.index.intersection(bench_x.index).intersection(cov.index)
    port_x = port_x.reindex(factors).fillna(0.0).to_numpy(dtype=float)
    bench_x = bench_x.reindex(factors).fillna(0.0).to_numpy(dtype=float)
    matrix = cov.reindex(index=factors, columns=factors).fillna(0.0).to_numpy(dtype=float)
    covariance = port_x @ matrix @ bench_x + specific_overlap(portfolio, benchmark, snapshot)
    variance = bench_x @ matrix @ bench_x + specific_overlap(benchmark, benchmark, snapshot)
    return float(covariance / variance) if variance > 0 else None


def specific_overlap(left, right, snapshot):
    """Return the shared idiosyncratic variance between two weight sets.

    Example:
        names held in both books contribute weight*weight*specific-variance.
    """
    risks = snapshot.specific_risk.set_index("ticker")["specific_risk"] ** 2
    a = left.set_index("ticker")["allocation"]
    b = right.set_index("ticker")["allocation"]
    shared = a.index.intersection(b.index).intersection(risks.index)
    return float((a.reindex(shared) * b.reindex(shared) * risks.reindex(shared)).sum())


def share(value, total):
    """Return value/total, or None when the total is zero."""
    return value / total if total else None


def contribution(variance, risk):
    """Return variance contribution in annualized risk units."""
    return variance / risk if risk else None


def daily(annual):
    """Return the one-day figure from an annualized volatility."""
    return annual / np.sqrt(TRADING_DAYS)


def none_if_nan(value):
    """Return a float or None for a possibly-missing value."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value
