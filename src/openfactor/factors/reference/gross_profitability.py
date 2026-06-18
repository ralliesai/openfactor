from openfactor.factors.reference.metrics import ratio


def compute(reference, as_of_date="latest"):
    """Compute gross-profitability exposure.

    Example input row:
        ticker  gross_profit  total_assets
        AAPL    40            100

    Example output row:
        AAPL gross_profitability 0.40
    """
    return ratio(reference, "gross_profitability", "gross_profit", "total_assets", as_of_date)
