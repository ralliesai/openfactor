from dataclasses import dataclass


@dataclass(frozen=True)
class FactorResult:
    """One ticker's result for one factor.

    Example:
        FactorResult(value=1.2, observations=252)
        means the factor exposure is 1.2 using 252 valid data points.
    """

    value: float
    observations: int
