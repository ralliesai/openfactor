import numpy as np

from openfactor.factors.output import make_exposure_rows
from openfactor.factors.reference.metrics import column, latest


def compute(reference, as_of_date="latest"):
    """Compute analyst sentiment exposure.

    Example input row:
        ticker  analyst_sentiment
        AAPL    0.80

    Example output row:
        AAPL sentiment 0.80
    """
    frame = latest(reference)
    values = column(frame, "analyst_sentiment")
    observations = column(frame, "analyst_sentiment_observations")
    observations = np.where(np.isfinite(observations), observations, np.isfinite(values))
    return make_exposure_rows(
        frame.index,
        values,
        "sentiment",
        "reference",
        as_of_date,
        observations.astype(int),
    )
