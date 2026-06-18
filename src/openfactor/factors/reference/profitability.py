from openfactor.factors.reference.metrics import ratio


def compute(reference, as_of_date="latest"):
    """Compute return-on-assets profitability exposure.

    Example input row:
        ticker  net_income  total_assets
        AAPL    20.0        100.0

    Example output row:
        AAPL profitability 0.2
    """
    return ratio(reference, "profitability", "net_income", "total_assets", as_of_date)
