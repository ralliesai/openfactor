from openfactor.factors.reference.metrics import ratio


def compute(reference, as_of_date="latest"):
    """Compute earnings yield exposure.

    Example input row:
        ticker  net_income  market_cap
        AAPL    10.0        100.0

    Example output row:
        AAPL earnings_yield 0.10
    """
    return ratio(reference, "earnings_yield", "net_income", "market_cap", as_of_date)
