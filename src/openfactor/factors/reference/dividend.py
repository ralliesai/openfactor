from openfactor.factors.reference.metrics import scalar


def compute(reference, as_of_date="latest"):
    """Compute trailing dividend yield exposure.

    Example input row:
        ticker  dividend_yield
        AAPL    0.004

    Example output row:
        AAPL dividend_yield 0.004
    """
    return scalar(reference, "dividend_yield", "dividend_yield", as_of_date)
