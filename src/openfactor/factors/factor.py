from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Factor:
    """Metadata for one factor.

    Example:
        Factor("beta", "price", "Sensitivity to the broad market.", beta)
        means beta(matrix) computes the beta rows.
    """

    name: str
    group: str
    description: str
    compute: Callable
