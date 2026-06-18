from openfactor.factors.reference.metrics import scalar


def compute(reference, as_of_date="latest"):
    """Compute management-quality exposure.

    Example input row:
        ticker  management_quality
        AAPL    0.20

    Example output row:
        AAPL management_quality 0.20
    """
    return scalar(reference, "management_quality", "management_quality", as_of_date)
