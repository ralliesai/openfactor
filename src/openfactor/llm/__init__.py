from openfactor.llm.cache import DEFAULT_SEMANTIC_CACHE, semantic_factor_members
from openfactor.llm.semantic import (
    DEFAULT_RESIDUAL_THRESHOLD,
    SemanticCandidate,
    SemanticDiscoveryResult,
    discover_semantic_factors,
)

__all__ = [
    "DEFAULT_RESIDUAL_THRESHOLD",
    "DEFAULT_SEMANTIC_CACHE",
    "SemanticCandidate",
    "SemanticDiscoveryResult",
    "SemanticLLMClient",
    "ReportChat",
    "discover_semantic_factors",
    "report_chat_enabled",
    "semantic_factor_members",
]


def __getattr__(name):
    """Load optional LLM client only when requested.

    Example:
        from openfactor.llm import SemanticLLMClient imports the optional client.
    """
    if name == "SemanticLLMClient":
        from openfactor.llm.client import SemanticLLMClient

        return SemanticLLMClient
    if name in {"ReportChat", "report_chat_enabled"}:
        from openfactor.llm.report_chat import ReportChat, report_chat_enabled

        return {"ReportChat": ReportChat, "report_chat_enabled": report_chat_enabled}[name]
    raise AttributeError(name)
