import numpy as np

from openfactor.factors.output import make_exposure_rows
from openfactor.factors.reference.metrics import column, latest


def forward_earnings_yield(reference, as_of_date="latest"):
    """Compute forward earnings-yield exposure.

    Example input row:
        ticker  forward_earnings_yield
        AAPL    0.04

    Example output row:
        AAPL forward_earnings_yield 0.04
    """
    return observed_scalar(
        reference,
        "forward_earnings_yield",
        "forward_earnings_yield",
        "forward_earnings_yield_observations",
        as_of_date,
    )


def forward_growth(reference, as_of_date="latest"):
    """Compute forward growth exposure.

    Example input row:
        ticker  forward_growth
        AAPL    0.12

    Example output row:
        AAPL forward_growth 0.12
    """
    return observed_scalar(
        reference,
        "forward_growth",
        "forward_growth",
        "forward_growth_observations",
        as_of_date,
    )


def observed_scalar(reference, factor, metric, observations_metric, as_of_date):
    """Return one observed metric as exposure rows.

    Example:
        value 0.12 with 15 analyst observations returns one factor row.
    """
    frame = latest(reference)
    values = column(frame, metric)
    observations = column(frame, observations_metric)
    observations = np.where(np.isfinite(observations), observations, np.isfinite(values))
    return make_exposure_rows(
        frame.index,
        values,
        factor,
        "reference",
        as_of_date,
        observations.astype(int),
    )
