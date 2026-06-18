from openfactor.factors.reference.metrics import ratio


def compute(reference, as_of_date="latest"):
    """Compute liabilities-to-assets leverage exposure.

    Example input row:
        ticker  total_liabilities  total_assets
        AAPL    40.0               100.0

    Example output row:
        AAPL leverage 0.4
    """
    return ratio(reference, "leverage", "total_liabilities", "total_assets", as_of_date)
