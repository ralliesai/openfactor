from openfactor.factors.reference.metrics import scalar


def earnings_quality(reference, as_of_date="latest"):
    """Compute cash-vs-accrual earnings quality exposure.

    Example input row:
        ticker  earnings_quality
        AAPL    0.05

    Example output row:
        AAPL earnings_quality 0.05
    """
    return scalar(reference, "earnings_quality", "earnings_quality", as_of_date)


def earnings_variability(reference, as_of_date="latest"):
    """Compute scaled earnings variability exposure.

    Example input row:
        ticker  earnings_variability
        AAPL    0.01

    Example output row:
        AAPL earnings_variability 0.01
    """
    return scalar(reference, "earnings_variability", "earnings_variability", as_of_date)
