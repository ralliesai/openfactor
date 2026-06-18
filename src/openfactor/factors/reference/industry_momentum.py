import numpy as np

from openfactor.factors.output import make_exposure_rows
from openfactor.factors.reference.metrics import column, latest


def compute(reference, as_of_date="latest"):
    """Compute same-industry momentum exposure.

    Example input row:
        ticker  industry_momentum
        NVDA    0.25

    Example output row:
        NVDA industry_momentum 0.25
    """
    frame = latest(reference)
    values = column(frame, "industry_momentum")
    observations = column(frame, "industry_momentum_observations")
    observations = np.where(np.isfinite(observations), observations, np.isfinite(values))
    return make_exposure_rows(
        frame.index,
        values,
        "industry_momentum",
        "reference",
        as_of_date,
        observations.astype(int),
    )
