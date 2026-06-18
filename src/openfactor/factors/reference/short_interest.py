from openfactor.factors.reference.metrics import scalar


def compute(reference, as_of_date="latest"):
    """Compute short-interest exposure.

    Example input row:
        ticker  short_interest
        AAPL    0.01

    Example output row:
        AAPL short_interest 0.01
    """
    return scalar(reference, "short_interest", "short_interest", as_of_date)
