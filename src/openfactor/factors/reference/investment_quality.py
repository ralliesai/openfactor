from openfactor.factors.reference.metrics import scalar


def compute(reference, as_of_date="latest"):
    """Compute investment quality exposure.

    Example input row:
        ticker  investment_quality
        AAPL    -0.04

    Example output row:
        AAPL investment_quality -0.04
    """
    return scalar(reference, "investment_quality", "investment_quality", as_of_date)
