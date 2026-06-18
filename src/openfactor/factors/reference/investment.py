from openfactor.factors.reference.metrics import scalar


def compute(reference, as_of_date="latest"):
    """Compute asset-growth investment exposure.

    Example input row:
        ticker  asset_growth
        AAPL    0.10

    Example output row:
        AAPL investment 0.10
    """
    return scalar(reference, "investment", "asset_growth", as_of_date)
