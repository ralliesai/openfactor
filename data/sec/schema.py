"""Shared SEC data columns.

Example:
    METRIC_COLUMNS is the row shape for canonical rows like revenue and assets.
"""

REFERENCE_COLUMNS = [
    "ticker",
    "name",
    "cik",
    "cusip",
    "exchange",
    "is_delisted",
    "sector",
    "industry",
    "sic",
    "sic_sector",
    "sic_industry",
    "fama_sector",
    "fama_industry",
    "currency",
    "location",
]

FILING_COLUMNS = [
    "ticker",
    "accession_no",
    "form_type",
    "filed_at",
    "period_of_report",
    "link_to_html",
    "link_to_details",
]

FACT_COLUMNS = [
    "ticker",
    "accession_no",
    "statement",
    "concept",
    "start_date",
    "end_date",
    "instant_date",
    "unit",
    "value",
    "has_segment",
]

METRIC_COLUMNS = [
    "ticker",
    "accession_no",
    "metric",
    "source_concept",
    "period_end",
    "unit",
    "value",
]

DAILY_FUNDAMENTAL_COLUMNS = [
    "ticker",
    "as_of_date",
    "accession_no",
    "form_type",
    "filed_at",
    "period_of_report",
    "sector",
    "industry",
    "sic",
    "sic_industry",
    "fama_industry",
    "revenue",
    "revenue_method",
    "gross_profit",
    "gross_profit_method",
    "operating_income",
    "operating_income_method",
    "net_income",
    "net_income_method",
    "capex",
    "buybacks",
    "share_issuance",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "shares_outstanding",
    "shares_outstanding_date",
    "shares_outstanding_source",
    "asset_growth",
]
