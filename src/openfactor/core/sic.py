import pandas as pd


def sector_from_sic(sic):
    """Return a broad SIC sector label.

    Example:
        sector_from_sic("3571") returns "Manufacturing".
    """
    if pd.isna(sic):
        return None

    code = int(float(sic))
    if code < 1000:
        return "Agriculture"
    if code < 1500:
        return "Mining"
    if code < 1800:
        return "Construction"
    if code < 4000:
        return "Manufacturing"
    if code < 5000:
        return "TransportUtilities"
    if code < 5200:
        return "Wholesale"
    if code < 6000:
        return "Retail"
    if code < 6800:
        return "Financials"
    if code < 9000:
        return "Services"
    return "PublicAdministration"
