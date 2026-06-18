from openfactor.factors.reference.metrics import ratio


def compute(reference, as_of_date="latest"):
    """Compute book-to-price value exposure.

    Example input row:
        ticker  stockholders_equity  market_cap
        AAPL    50.0                 100.0

    Example output row:
        AAPL value 0.5
    """
    return ratio(
        reference,
        "value",
        "stockholders_equity",
        "market_cap",
        as_of_date,
    )
