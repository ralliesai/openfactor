import numpy as np
import pandas as pd


FIELDS = {
    "net_income": ("ic", ["NetIncomeLoss", "ProfitLoss"]),
    "operating_cash_flow": ("cf", ["NetCashProvidedByUsedInOperatingActivities"]),
    "total_assets": ("bs", ["Assets"]),
}
FINNHUB_TICKER_ALIAS = {"BRK.B": "BRK-B", "GOOG": "GOOGL"}


def reported_financials(client, ticker):
    """Download reported financial rows from Finnhub.

    Example:
        reported_financials(client, "AAPL")
        returns 10-Q and 10-K rows with accepted_date and key metrics.
    """
    rows = []
    provider_ticker = FINNHUB_TICKER_ALIAS.get(ticker.upper(), ticker.upper())
    for freq in ["quarterly", "annual"]:
        data = client.get(
            "stock/financials-reported",
            {"symbol": provider_ticker, "freq": freq},
        )
        rows += filing_rows(ticker.upper(), data.get("data", []))
    return clean_rows(rows)


def filing_rows(ticker, filings):
    """Return slim filing rows from Finnhub JSON.

    Example:
        one filing with NetIncomeLoss becomes one net_income row value.
    """
    rows = []
    for filing in filings:
        rows.append(
            {
                "ticker": ticker,
                "accession_no": filing.get("accessNumber"),
                "form_type": filing.get("form"),
                "accepted_date": date_text(filing.get("acceptedDate")),
                "filed_at": date_text(filing.get("filedDate")),
                "start_date": date_text(filing.get("startDate")),
                "end_date": date_text(filing.get("endDate")),
                **metric_values(filing),
            }
        )
    return rows


def metric_values(filing):
    """Return the financial values OpenFactor needs.

    Example:
        if Assets is present in report.bs, total_assets is numeric.
    """
    return {name: metric_value(filing, name) for name in FIELDS}


def metric_value(filing, name):
    """Return one metric from a Finnhub filing report.

    Example:
        metric_value(filing, "net_income") reads report.ic concepts.
    """
    statement, suffixes = FIELDS[name]
    for suffix in suffixes:
        for row in filing.get("report", {}).get(statement, []):
            if concept_name(row.get("concept")) == suffix:
                return number(row.get("value"))
    return np.nan


def concept_name(value):
    """Return the XBRL concept name without namespace.

    Example:
        us-gaap_Assets returns Assets, while us-gaap_OtherAssets returns OtherAssets.
    """
    return str(value or "").replace(":", "_").split("_")[-1]


def clean_rows(rows):
    """Return de-duplicated reported-financial rows.

    Example:
        duplicate accessions keep the last downloaded row.
    """
    columns = [
        "ticker",
        "accession_no",
        "form_type",
        "accepted_date",
        "filed_at",
        "start_date",
        "end_date",
        "net_income",
        "operating_cash_flow",
        "total_assets",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.drop_duplicates(["ticker", "accession_no"], keep="last")


def date_text(value):
    """Return the ISO date part of a provider timestamp.

    Example:
        "2026-05-01 18:30:00" becomes "2026-05-01".
    """
    return str(value)[:10] if value else None


def number(value):
    """Return a float or NaN.

    Example:
        number("10") returns 10.0.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
