from openfactor.factors.reference.metrics import scalar


def compute(reference, as_of_date="latest"):
    """Compute fundamental growth exposure.

    Example input row:
        ticker  growth
        AAPL    0.12

    Example output row:
        AAPL growth 0.12
    """
    return scalar(reference, "growth", "growth", as_of_date)
