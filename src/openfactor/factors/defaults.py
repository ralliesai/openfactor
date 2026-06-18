from openfactor.factors.price.beta import compute as beta
from openfactor.factors.price.downside import compute as downside_risk
from openfactor.factors.price.liquidity import compute as liquidity
from openfactor.factors.price.momentum import compute as momentum
from openfactor.factors.price.prospect import compute as prospect
from openfactor.factors.price.reversal import compute as long_term_reversal
from openfactor.factors.price.seasonality import compute as seasonality
from openfactor.factors.price.short_reversal import compute as short_term_reversal
from openfactor.factors.price.volatility import residual
from openfactor.factors.reference.dividend import compute as dividend_yield
from openfactor.factors.reference.earnings_yield import compute as earnings_yield
from openfactor.factors.reference.earnings import earnings_quality, earnings_variability
from openfactor.factors.reference.forward_earnings import forward_earnings_yield, forward_growth
from openfactor.factors.reference.gross_profitability import compute as gross_profitability
from openfactor.factors.reference.industry import compute as industry
from openfactor.factors.reference.industry_momentum import compute as industry_momentum
from openfactor.factors.reference.investment import compute as investment
from openfactor.factors.reference.investment_quality import compute as investment_quality
from openfactor.factors.reference.leverage import compute as leverage
from openfactor.factors.reference.growth import compute as growth
from openfactor.factors.reference.mid_cap import compute as mid_cap
from openfactor.factors.reference.profitability import compute as profitability
from openfactor.factors.reference.sector import compute as sector
from openfactor.factors.reference.sentiment import compute as sentiment
from openfactor.factors.reference.short_interest import compute as short_interest
from openfactor.factors.reference.size import compute as size
from openfactor.factors.reference.value import compute as value
from openfactor.factors.factor import Factor


def default_price_factors():
    """Return built-in factors that consume a PriceMatrix.

    Example:
        factors = default_price_factors()
        factors[0].name == "beta"
    """
    return [
        Factor("beta", "price", "Sensitivity to the broad market.", beta),
        Factor("momentum", "price", "12 month return skipping the last month.", momentum),
        Factor(
            "prospect",
            "price",
            "Lottery-like upside skew plus recent drawdown.",
            prospect,
        ),
        Factor(
            "long_term_reversal",
            "price",
            "Negative return from three years ago to one year ago.",
            long_term_reversal,
        ),
        Factor(
            "short_term_reversal",
            "price",
            "Negative recent one month return.",
            short_term_reversal,
        ),
        Factor(
            "seasonality",
            "price",
            "Average same-month return from prior years.",
            seasonality,
        ),
        Factor(
            "residual_volatility",
            "price",
            "Annualized volatility after market beta.",
            residual,
        ),
        Factor(
            "downside_risk",
            "price",
            "Annualized volatility of negative daily returns.",
            downside_risk,
        ),
        Factor("liquidity", "price", "Log average dollar volume.", liquidity),
    ]


def default_reference_factors():
    """Return built-in factors that consume reference rows.

    Example:
        factors = default_reference_factors()
        factors[0].name == "sector"
    """
    return [
        Factor("sector", "sector", "Broad SIC sector exposure.", sector),
        Factor("industry", "industry", "Readable SEC-API industry exposure.", industry),
        Factor("size", "reference", "Log market capitalization.", size),
        Factor("mid_cap", "reference", "Nonlinear middle-cap exposure.", mid_cap),
        Factor("value", "reference", "Book equity divided by market value.", value),
        Factor("earnings_yield", "reference", "Net income divided by market value.", earnings_yield),
        Factor(
            "forward_earnings_yield",
            "reference",
            "Forward net-income estimate divided by market value.",
            forward_earnings_yield,
        ),
        Factor("dividend_yield", "reference", "Trailing dividends divided by price.", dividend_yield),
        Factor("growth", "reference", "Average revenue and earnings growth.", growth),
        Factor("forward_growth", "reference", "Forward revenue and earnings growth.", forward_growth),
        Factor("sentiment", "reference", "Time-decayed analyst recommendation score.", sentiment),
        Factor(
            "industry_momentum",
            "reference",
            "Recent momentum of stocks in the same industry.",
            industry_momentum,
        ),
        Factor(
            "profitability",
            "reference",
            "Net income divided by assets.",
            profitability,
        ),
        Factor(
            "gross_profitability",
            "reference",
            "Gross profit divided by assets.",
            gross_profitability,
        ),
        Factor("leverage", "reference", "Liabilities divided by assets.", leverage),
        Factor("investment", "reference", "Asset growth from latest filing.", investment),
        Factor(
            "investment_quality",
            "reference",
            "Asset growth, capex, buyback, and issuance quality.",
            investment_quality,
        ),
        Factor(
            "earnings_quality",
            "reference",
            "Cash flow from operations minus net income, scaled by assets.",
            earnings_quality,
        ),
        Factor(
            "earnings_variability",
            "reference",
            "Variability of recent quarterly earnings scaled by assets.",
            earnings_variability,
        ),
        Factor("short_interest", "reference", "Short interest divided by shares.", short_interest),
    ]


def default_factors():
    """Return the full built-in deterministic factor list.

    Example:
        factors = default_factors()
        factors[0].name == "beta"
    """
    return default_price_factors() + default_reference_factors()
