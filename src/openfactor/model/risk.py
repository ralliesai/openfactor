import numpy as np
import pandas as pd

from openfactor.core.checks import require_columns
from openfactor.model.exposures import model_exposure_matrix
from openfactor.model.specific_risk import portfolio_specific_risk


def factor_covariance(factor_returns):
    """Return annualized covariance for factors with full return history.

    Example:
        a factor with missing return days stays out of covariance.
    """
    return factor_returns.dropna(axis=1).cov() * 252


def portfolio_factor_exposure(exposures, portfolio):
    """Return portfolio exposure to every model factor.

    Example:
        AAPL weight 0.50 and size exposure 2.0 gives size exposure 1.0.
        market exposure is the portfolio's net allocation.
    """
    require_columns(portfolio, ["ticker", "allocation"])
    weights = portfolio.set_index("ticker")["allocation"]
    matrix = model_exposure_matrix(exposures).reindex(weights.index)
    exposure = matrix.multiply(weights, axis=0).sum(min_count=len(weights))
    exposure.loc["market"] = weights.sum()
    return exposure


def benchmark_weights(exposures):
    """Return cap-weighted universe benchmark weights by ticker.

    Example:
        a stock with 2x the market cap gets 2x the benchmark weight.
    """
    size = exposures[exposures["factor"] == "size"].drop_duplicates("ticker")
    caps = np.exp(pd.to_numeric(size.set_index("ticker")["raw_value"], errors="coerce"))
    caps = caps[np.isfinite(caps) & (caps > 0)]
    return (caps / caps.sum()).rename("weight")


def active_holdings(portfolio, exposures):
    """Return portfolio-minus-benchmark active weights over the universe.

    Example:
        30% in Technology against a 25% benchmark gives +5% active Technology.
    """
    benchmark = benchmark_weights(exposures)
    weights = portfolio.set_index("ticker")["allocation"].reindex(benchmark.index).fillna(0.0)
    return (weights - benchmark).rename("allocation").reset_index()


def factor_risk_report(exposures, portfolio, factor_returns):
    """Return factor risk contributions for one portfolio.

    Example:
        beta exposure 1.0 and beta variance 0.04 gives 20% factor risk.
    """
    return factor_risk_report_from_covariance(
        exposures,
        portfolio,
        factor_covariance(factor_returns),
    )


def factor_risk_report_from_covariance(exposures, portfolio, covariance):
    """Return factor risk contributions from a covariance matrix.

    Example:
        covariance.loc["beta", "beta"] = 0.04 and beta exposure is 1.0
        gives about 20% factor risk.
    """
    missing = missing_model_exposures(exposures, covariance)
    if missing:
        raise ValueError(f"factor_covariance has factors missing exposures: {missing[:10]}")

    exposure = portfolio_factor_exposure(exposures, portfolio).dropna()
    factors = exposure.index.intersection(covariance.index)
    exposure = exposure.reindex(factors)
    covariance = covariance.reindex(index=factors, columns=factors)
    keep = np.isfinite(np.diag(covariance.to_numpy(dtype=float)))
    covariance = covariance.loc[keep, keep]
    covariance = covariance.fillna(0.0)
    exposure = exposure.reindex(covariance.index)

    values = exposure.to_numpy(dtype=float)
    cov = covariance.to_numpy(dtype=float)
    marginal = cov @ values
    variance = values * marginal
    factor_risk = np.sqrt(max(variance.sum(), 0.0))
    contribution = variance / factor_risk if factor_risk > 0 else variance

    report = pd.DataFrame(
        {
            "exposure": values,
            "factor_volatility": np.sqrt(np.maximum(np.diag(cov), 0.0)),
            "risk_contribution": contribution,
            "variance_contribution": variance,
        },
        index=exposure.index,
    )
    report = report[report["variance_contribution"] != 0]
    return report.sort_values("risk_contribution", key=np.abs, ascending=False)


def missing_model_exposures(exposures, covariance):
    """Return covariance factors absent from exposure rows.

    Example:
        market is allowed because portfolio_factor_exposure adds it.
    """
    modeled = set(covariance.index.astype(str)) - {"market"}
    available = set(model_exposure_matrix(exposures).columns.astype(str))
    return sorted(modeled - available)


def portfolio_risk_report(factor_report, specific_risks, portfolio, strict=True):
    """Return total, factor, and stock-specific portfolio risk.

    Example:
        12% factor risk and 5% specific risk combine to 13% total risk.
        strict=False skips unmodeled names (benchmark-relative tracking error).
    """
    factor_variance = factor_report["variance_contribution"].sum()
    specific = portfolio_specific_risk(portfolio, specific_risks, strict)
    total = np.sqrt(max(factor_variance, 0.0) + specific**2)
    rows = [
        ("factor", np.sqrt(max(factor_variance, 0.0))),
        ("stock_specific", specific),
        ("total", total),
    ]
    return pd.DataFrame(rows, columns=["component", "risk"]).set_index("component")


def risk_explanation_report(factor_report, specific_risks, portfolio):
    """Return factor versus residual risk as variance percentages.

    Example:
        if factor variance is 60% of total variance,
        residual_unexplained_percent is 40.
    """
    factor_variance = max(factor_report["variance_contribution"].sum(), 0.0)
    specific = portfolio_specific_risk(portfolio, specific_risks)
    residual_variance = specific**2
    total = factor_variance + residual_variance
    if total == 0:
        total = np.nan

    rows = [
        ("factor_explained_percent", factor_variance / total * 100),
        ("residual_unexplained_percent", residual_variance / total * 100),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
