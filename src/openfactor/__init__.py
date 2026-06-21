from openfactor.factors.defaults import (
    default_factors,
    default_price_factors,
    default_reference_factors,
)
from openfactor.factors.factor import Factor
from openfactor.factors.result import FactorResult
from openfactor.io.snapshot import Snapshot, load_snapshot
from openfactor.llm import (
    DEFAULT_RESIDUAL_THRESHOLD,
    SemanticCandidate,
    SemanticDiscoveryResult,
    discover_semantic_factors,
    semantic_factor_members,
)
from openfactor.portfolio.model_data import FactorModelData, factor_model_data
from openfactor.portfolio.report import portfolio_report

__all__ = [
    "DEFAULT_RESIDUAL_THRESHOLD",
    "Factor",
    "FactorModelData",
    "FactorResult",
    "SemanticCandidate",
    "SemanticDiscoveryResult",
    "SemanticLLMClient",
    "Snapshot",
    "default_factors",
    "default_price_factors",
    "default_reference_factors",
    "discover_semantic_factors",
    "load_snapshot",
    "factor_model_data",
    "portfolio_report",
    "semantic_factor_members",
]


def __getattr__(name):
    """Load optional LLM client only when requested.

    Example:
        openfactor.SemanticLLMClient imports the optional client.
    """
    if name == "SemanticLLMClient":
        from openfactor.llm import SemanticLLMClient

        return SemanticLLMClient
    raise AttributeError(name)
