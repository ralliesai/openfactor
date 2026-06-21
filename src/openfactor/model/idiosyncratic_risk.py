import numpy as np
import pandas as pd

from openfactor.core.checks import require_columns
from openfactor.model.factor_returns import factor_model_history


def idiosyncratic_risk_from_residuals(residuals):
    """Return annualized idiosyncratic risk from residual returns.

    Example:
        daily residuals with 1% volatility become about 15.9% annual risk.
    """
    risk = residuals.std(skipna=True) * np.sqrt(252)
    observations = residuals.notna().sum()

    return pd.DataFrame(
        {
            "ticker": risk.index,
            "idiosyncratic_risk": risk.values,
            "observations": observations.values,
        }
    )


def idiosyncratic_risk(matrix, exposures, window=252, price_factors=None):
    """Return annualized idiosyncratic risk after all common factors.

    Example:
        idiosyncratic_risk(matrix, exposures)
        returns one row per ticker with annualized residual volatility.
    """
    _, residuals = factor_model_history(
        matrix,
        exposures,
        window,
        price_factors=price_factors,
    )
    return idiosyncratic_risk_from_residuals(residuals)


def portfolio_idiosyncratic_risk(portfolio, risks, strict=True):
    """Return portfolio annualized idiosyncratic risk.

    Example:
        AAPL has 20% idiosyncratic risk and a 50% weight.
        Its variance contribution is (0.50 * 0.20) ** 2.
        strict=False skips names without a modeled risk (benchmark weights).
    """
    require_columns(portfolio, ["ticker", "allocation"])
    require_columns(risks, ["ticker", "idiosyncratic_risk"])

    weights = portfolio.set_index("ticker")["allocation"]
    stock_risk = risks.set_index("ticker")["idiosyncratic_risk"].reindex(weights.index)
    if strict and stock_risk.isna().any():
        return np.nan

    return np.sqrt(((weights * stock_risk) ** 2).sum())
