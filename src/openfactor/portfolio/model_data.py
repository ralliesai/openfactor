from dataclasses import dataclass

import numpy as np
import pandas as pd

from openfactor.core.checks import require_columns
from openfactor.model.exposures import model_exposure_matrix
from openfactor.model.risk import benchmark_weights, missing_model_exposures


__all__ = ["FactorModelData", "factor_model_data"]


@dataclass(frozen=True)
class FactorModelData:
    """Aligned factor-model data for portfolio analytics.

    Shape:
        exposures is assets x factors.
        factor_covariance is factors x factors.
        idiosyncratic_variance and benchmark_weights are length assets.
    """

    tickers: pd.Index
    exposures: pd.DataFrame
    factor_covariance: pd.DataFrame
    idiosyncratic_variance: pd.Series
    benchmark_weights: pd.Series
    factor_groups: pd.Series

    @property
    def n_assets(self):
        return len(self.tickers)

    @property
    def n_factors(self):
        return len(self.factor_names)

    @property
    def factor_names(self):
        return self.exposures.columns

    def weight_vector(self, weights):
        """Return weights as a vector aligned to tickers."""
        if isinstance(weights, pd.DataFrame):
            require_columns(weights, ["ticker", "allocation"])
            values = weights.set_index("ticker")["allocation"].reindex(self.tickers).fillna(0.0)
            vector = values.to_numpy(dtype=float)
        elif isinstance(weights, pd.Series):
            vector = weights.reindex(self.tickers).fillna(0.0).to_numpy(dtype=float)
        else:
            vector = np.asarray(weights, dtype=float)

        if vector.shape != (self.n_assets,):
            raise ValueError(f"weights must have shape ({self.n_assets},)")
        if not np.isfinite(vector).all():
            raise ValueError("weights contain non-finite values")
        return vector

    def portfolio_frame(self, weights, min_abs=0.0):
        """Return a portfolio DataFrame from an aligned weight vector."""
        values = self.weight_vector(weights)
        frame = pd.DataFrame({"ticker": self.tickers, "allocation": values})
        if min_abs:
            frame = frame[frame["allocation"].abs() >= float(min_abs)]
        return frame.reset_index(drop=True)

    def factor_exposure(self, weights):
        """Return factor exposures for an aligned weight vector."""
        values = self.weight_vector(weights)
        exposure = self.exposures.to_numpy(dtype=float).T @ values
        return pd.Series(exposure, index=self.factor_names, name="exposure")

    def active_weights(self, weights):
        """Return portfolio weights minus benchmark weights."""
        return self.weight_vector(weights) - self.benchmark_weights.to_numpy(dtype=float)

    def active_factor_exposure(self, weights):
        """Return benchmark-relative factor exposures."""
        return self.factor_exposure(self.active_weights(weights)).rename("active_exposure")

    def factor_variance(self, weights):
        """Return common-factor variance for aligned weights."""
        exposure = self.factor_exposure(weights).to_numpy(dtype=float)
        covariance = self.factor_covariance.to_numpy(dtype=float)
        return float(exposure @ covariance @ exposure)

    def idiosyncratic_variance_for(self, weights):
        """Return idiosyncratic variance for aligned weights."""
        values = self.weight_vector(weights)
        idiosyncratic = self.idiosyncratic_variance.to_numpy(dtype=float)
        return float(np.sum((values**2) * idiosyncratic))

    def variance(self, weights):
        """Return total model variance for aligned weights."""
        return self.factor_variance(weights) + self.idiosyncratic_variance_for(weights)

    def risk(self, weights):
        """Return annualized model risk for aligned weights."""
        return float(np.sqrt(max(self.variance(weights), 0.0)))

    def tracking_error(self, weights):
        """Return annualized tracking error versus the benchmark."""
        return self.risk(self.active_weights(weights))

    def beta_loadings(self):
        """Return each asset's linear contribution to beta versus benchmark.

        Example:
            data.beta(weights) == data.beta_loadings() @ weights.
        """
        x = self.exposures.to_numpy(dtype=float)
        f = self.factor_covariance.to_numpy(dtype=float)
        b = self.benchmark_weights.to_numpy(dtype=float)
        d = self.idiosyncratic_variance.to_numpy(dtype=float)
        benchmark_exposure = x.T @ b
        benchmark_variance = float(benchmark_exposure @ f @ benchmark_exposure + np.sum((b**2) * d))
        if benchmark_variance <= 0:
            return pd.Series(np.nan, index=self.tickers, name="beta")
        loadings = (x @ (f @ benchmark_exposure) + b * d) / benchmark_variance
        return pd.Series(loadings, index=self.tickers, name="beta")

    def beta(self, weights):
        """Return model beta versus the benchmark."""
        return float(self.beta_loadings().to_numpy(dtype=float) @ self.weight_vector(weights))


def factor_model_data(snapshot, tickers=None, strict=True):
    """Return factor-model arrays aligned for portfolio analytics.

    Example:
        data = factor_model_data(snapshot)
        data.exposures.to_numpy() is assets x factors.

    Note:
        Use the full universe for exact benchmark-relative tracking error. A
        ticker subset is useful for research, but omitted benchmark names do not
        contribute to active risk inside the returned object.
    """
    covariance = clean_covariance(snapshot.factor_covariance)
    missing = missing_model_exposures(snapshot.exposures, covariance)
    if missing:
        raise ValueError(f"factor_covariance has factors missing exposures: {missing[:10]}")

    benchmark = benchmark_weights(snapshot.exposures)
    tickers = ticker_index(tickers, benchmark.index)
    exposures = aligned_exposures(snapshot.exposures, covariance.index, tickers, strict)
    idiosyncratic = aligned_idiosyncratic_variance(snapshot.idiosyncratic_risk, tickers, strict)
    benchmark = benchmark.reindex(tickers).fillna(0.0)
    covariance = covariance.reindex(index=exposures.columns, columns=exposures.columns)
    groups = factor_groups(snapshot.exposures, exposures.columns)

    return FactorModelData(
        tickers=tickers,
        exposures=exposures,
        factor_covariance=covariance,
        idiosyncratic_variance=idiosyncratic,
        benchmark_weights=benchmark,
        factor_groups=groups,
    )


def ticker_index(tickers, default):
    """Return a unique ticker index."""
    values = default if tickers is None else tickers
    index = pd.Index([str(ticker) for ticker in values], name="ticker")
    if index.has_duplicates:
        raise ValueError("tickers contains duplicates")
    return index


def clean_covariance(covariance):
    """Return a numeric square factor covariance matrix."""
    matrix = covariance.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    factors = pd.Index(matrix.index, name="factor")
    matrix = matrix.reindex(index=factors, columns=factors).apply(pd.to_numeric, errors="coerce")
    bad = factors[~np.isfinite(np.diag(matrix.to_numpy(dtype=float)))]
    if len(bad):
        raise ValueError(f"factor_covariance has missing diagonal values: {list(bad[:10])}")
    return matrix.fillna(0.0)


def aligned_exposures(exposures, factors, tickers, strict):
    """Return an assets x factors exposure matrix."""
    matrix = model_exposure_matrix(exposures)
    market = pd.Series(1.0, index=matrix.index, name="market")
    matrix = pd.concat([matrix, market], axis=1).copy()
    missing = sorted(set(tickers) - set(matrix.index.astype(str)))
    if strict and missing:
        raise ValueError(f"tickers missing exposures: {missing[:10]}")
    matrix.index = matrix.index.astype(str)
    return matrix.reindex(tickers).reindex(columns=factors).fillna(0.0)


def aligned_idiosyncratic_variance(idiosyncratic_risk, tickers, strict):
    """Return idiosyncratic variance aligned to tickers."""
    require_columns(idiosyncratic_risk, ["ticker", "idiosyncratic_risk"])
    risks = idiosyncratic_risk.drop_duplicates("ticker").set_index("ticker")["idiosyncratic_risk"]
    risks.index = risks.index.astype(str)
    risks = pd.to_numeric(risks, errors="coerce").reindex(tickers)
    missing = list(risks[risks.isna()].index)
    if strict and missing:
        raise ValueError(f"tickers missing idiosyncratic risk: {missing[:10]}")
    return (risks.fillna(0.0) ** 2).rename("idiosyncratic_variance")


def factor_groups(exposures, factors):
    """Return factor group labels aligned to factors."""
    groups = exposures.drop_duplicates("factor").set_index("factor")["group"]
    groups.index = groups.index.astype(str)
    groups = groups.reindex(factors)
    if "market" in groups.index:
        groups.loc["market"] = "market"
    return groups.fillna("unknown").rename("group")
