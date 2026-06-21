import pandas as pd

from openfactor.model.exposures import exposure_matrix, model_exposure_matrix
from openfactor.model.risk import (
    active_holdings,
    factor_risk_report_from_covariance,
    portfolio_risk_report,
    risk_explanation_report,
)


FACTOR_DISPLAY_NAMES = {
    "market": "Market",
    "beta": "Beta",
    "momentum": "Momentum",
    "prospect": "Prospect",
    "long_term_reversal": "Long-Term Reversal",
    "short_term_reversal": "Short-Term Reversal",
    "seasonality": "Seasonality",
    "residual_volatility": "Residual Volatility",
    "downside_risk": "Downside Risk",
    "liquidity": "Liquidity",
    "size": "Size",
    "mid_cap": "Mid-Cap",
    "value": "Value",
    "earnings_yield": "Earnings Yield",
    "forward_earnings_yield": "Forward Earnings Yield",
    "dividend_yield": "Dividend Yield",
    "growth": "Growth",
    "forward_growth": "Forward Growth",
    "sentiment": "Analyst Sentiment",
    "industry_momentum": "Industry Momentum",
    "profitability": "Profitability",
    "gross_profitability": "Gross Profitability",
    "leverage": "Leverage",
    "investment": "Asset Growth",
    "investment_quality": "Capital Discipline",
    "earnings_quality": "Earnings Quality",
    "earnings_variability": "Earnings Variability",
    "short_interest": "Short Interest",
}


def portfolio_report(portfolio, snapshot):
    """Return the standard OpenFactor portfolio report tables.

    Example:
        portfolio_report(portfolio, snapshot)["total_risk"]
        returns factor, idiosyncratic, and total risk rows.
    """
    factor_risk = with_display_index(
        factor_risk_report_from_covariance(snapshot.exposures, portfolio, snapshot.factor_covariance)
    )
    active = active_holdings(portfolio, snapshot.exposures)
    active_risk = with_display_index(
        factor_risk_report_from_covariance(snapshot.exposures, active, snapshot.factor_covariance)
    )
    return {
        "missing_holdings": missing_holdings(portfolio, snapshot.universe),
        "style": with_display_index(style_report(snapshot.exposures, portfolio)),
        "sector": sector_report(snapshot.exposures, portfolio),
        "specific_risk": idiosyncratic_report(snapshot.specific_risk, portfolio),
        "factor_risk": factor_risk,
        "active_risk": active_risk,
        "risk_share": risk_explanation_report(factor_risk, snapshot.specific_risk, portfolio),
        "total_risk": portfolio_risk_report(factor_risk, snapshot.specific_risk, portfolio),
        "tracking_error": portfolio_risk_report(active_risk, snapshot.specific_risk, active, strict=False),
    }


def style_report(exposures, portfolio):
    """Return portfolio scalar factor exposures.

    Example:
        beta 1.2 at 50% weight contributes 0.6 to portfolio beta exposure.
    """
    exposures = exposures[~exposures["group"].isin(["sector", "industry"])]
    weights = weights_for(portfolio)
    matrix = model_exposure_matrix(exposures).reindex(weights.index)
    values = matrix.multiply(weights, axis=0).sum(min_count=len(weights))
    return values.rename("exposure").to_frame()


def sector_report(exposures, portfolio):
    """Return portfolio allocation by sector factor.

    Example:
        AAPL at 30% in sector:Technology returns Technology=0.30.
    """
    sectors = exposures[exposures["group"] == "sector"].copy()
    sectors["factor"] = sectors["factor"].str.replace("sector:", "", regex=False)
    matrix = exposure_matrix(sectors)
    weights = weights_for(portfolio).reindex(matrix.index)
    allocation = matrix.multiply(weights, axis=0).sum()
    return allocation[allocation != 0].rename("allocation").to_frame()


def with_display_index(frame):
    """Return a report table with display factor names.

    Example:
        investment becomes Asset Growth and sector:Technology becomes Sector: Technology.
    """
    rows = frame.copy()
    rows.index = [display_factor_name(name) for name in rows.index]
    return rows


def display_factor_name(name):
    """Return the report label for one factor id.

    Example:
        display_factor_name("investment_quality") returns "Capital Discipline".
    """
    name = str(name)
    if name.startswith("sector:"):
        return "Sector: " + name.removeprefix("sector:")
    if name.startswith("industry:"):
        return "Industry: " + name.removeprefix("industry:")
    return FACTOR_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def idiosyncratic_report(risks, portfolio):
    """Return idiosyncratic risk for portfolio tickers.

    Example:
        AAPL specific_risk 0.20 returns one AAPL row with 0.20.
    """
    tickers = portfolio["ticker"].astype(str)
    return risks.set_index("ticker").reindex(tickers)[["specific_risk"]]


def missing_holdings(portfolio, universe):
    """Return portfolio tickers absent from the snapshot universe.

    Example:
        if TSLA is not in universe, the output contains one TSLA row.
    """
    tickers = set(universe["ticker"].astype(str))
    missing = portfolio.loc[~portfolio["ticker"].astype(str).isin(tickers), "ticker"]
    return missing.astype(str).rename("ticker").to_frame()


def weights_for(portfolio):
    """Return portfolio weights indexed by ticker.

    Example:
        ticker=AAPL, allocation=0.30 becomes weights["AAPL"] == 0.30.
    """
    frame = portfolio[["ticker", "allocation"]].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["allocation"] = pd.to_numeric(frame["allocation"])
    return frame.set_index("ticker")["allocation"]
