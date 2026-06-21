from io import StringIO
import gzip
import json

import numpy as np
import pandas as pd

from openfactor.io.snapshot import SNAPSHOT_FILES
from openfactor.io.indexes import INDEX_COLUMNS, INDEX_PRICE_COLUMNS, INDEX_RETURN_COLUMNS
from openfactor.model.exposures import exposure_matrix


TTM_NAMES = {
    "revenue": "revenue_ttm",
    "gross_profit": "gross_profit_ttm",
    "operating_income": "operating_income_ttm",
    "net_income": "net_income_ttm",
}

FUNDAMENTAL_COLUMNS = [
    "ticker",
    "as_of_date",
    "sector",
    "industry",
    "revenue_ttm",
    "gross_profit_ttm",
    "operating_income_ttm",
    "net_income_ttm",
    "growth",
    "forward_earnings_yield",
    "forward_growth",
    "industry_momentum",
    "analyst_sentiment",
    "dividend_yield",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "shares_outstanding",
    "asset_growth",
    "capex",
    "buybacks",
    "share_issuance",
    "investment_quality",
    "earnings_quality",
    "earnings_variability",
    "short_interest",
    "market_cap",
]

FUNDAMENTAL_AUDIT_COLUMNS = [
    "ticker",
    "as_of_date",
    "accession_no",
    "form_type",
    "filed_at",
    "period_of_report",
    "sic",
    "sic_industry",
    "fama_industry",
    "revenue_method",
    "gross_profit_method",
    "operating_income_method",
    "net_income_method",
    "shares_outstanding_date",
    "shares_outstanding_source",
]


def snapshot_csvs(snapshot):
    """Return public snapshot CSV files as text.

    Example:
        snapshot_csvs(snapshot) includes the wide exposures.csv.
    """
    files = [
        (SNAPSHOT_FILES["exposures"], spreadsheet_csv(wide_exposures(snapshot.exposures))),
        (SNAPSHOT_FILES["exposures_detail"], spreadsheet_csv(detail_exposures(snapshot.exposures))),
        (SNAPSHOT_FILES["factor_returns"], spreadsheet_csv(sorted_columns(snapshot.factor_returns), True, "date")),
        (SNAPSHOT_FILES["residual_returns"], spreadsheet_csv(residual_returns_file(snapshot.residual_returns))),
        (
            SNAPSHOT_FILES["factor_covariance"],
            spreadsheet_csv(sorted_covariance(snapshot.factor_covariance), True, "factor"),
        ),
        (SNAPSHOT_FILES["specific_risk"], spreadsheet_csv(sort_tickers(snapshot.specific_risk))),
        (SNAPSHOT_FILES["universe"], spreadsheet_csv(sort_tickers(snapshot.universe))),
    ]
    files += index_csvs(snapshot)
    return files


def index_csvs(snapshot):
    """Return public index files when the snapshot carries them."""
    files = []
    if present(snapshot, "indexes"):
        files.append((SNAPSHOT_FILES["indexes"], spreadsheet_csv(index_file(snapshot.indexes))))
    if present(snapshot, "index_prices"):
        files.append((SNAPSHOT_FILES["index_prices"], spreadsheet_csv(index_prices_file(snapshot.index_prices))))
    if present(snapshot, "index_returns"):
        files.append((SNAPSHOT_FILES["index_returns"], spreadsheet_csv(index_returns_file(snapshot.index_returns))))
    return files


def present(snapshot, name):
    """Return True when a snapshot frame exists and has rows."""
    frame = getattr(snapshot, name, None)
    return frame is not None and not frame.empty


def index_file(indexes):
    """Return the public index metadata table."""
    return keep_columns(indexes, INDEX_COLUMNS).sort_values("ticker").reset_index(drop=True)


def index_prices_file(prices):
    """Return public index prices sorted by date and ticker."""
    return keep_columns(prices, INDEX_PRICE_COLUMNS).sort_values(["date", "ticker"]).reset_index(drop=True)


def index_returns_file(returns):
    """Return public index returns sorted by date and ticker."""
    return keep_columns(returns, INDEX_RETURN_COLUMNS).sort_values(["date", "ticker"]).reset_index(drop=True)


def panel_gzip(snapshot):
    """Return the gzipped exposure-history file, or None when absent.

    Example:
        panel_gzip(snapshot) compresses the multi-day exposure panel for upload.
    """
    panel = getattr(snapshot, "exposures_panel", None)
    if panel is None or panel.empty:
        return None
    text = spreadsheet_csv(detail_exposures(panel))
    return SNAPSHOT_FILES["exposures_panel"], gzip.compress(text.encode("utf-8"))


def wide_exposures(exposures):
    """Return one model exposure row per ticker.

    Example:
        beta and size long rows become columns beta and size.
    """
    matrix = exposure_matrix(exposures)
    for factor in one_hot_factors(exposures, matrix.columns):
        matrix[factor] = matrix[factor].fillna(0.0)
    table = matrix.reset_index().rename(columns={"index": "ticker"})
    dates = exposures.drop_duplicates("ticker").set_index("ticker")["as_of_date"]
    table.insert(0, "as_of_date", table["ticker"].map(dates))
    return table[["as_of_date", "ticker"] + exposure_columns(matrix.columns)]


def exposure_columns(columns):
    """Return spreadsheet-friendly exposure column order.

    Example:
        beta, size, sector:Technology, then industry:Software.
    """
    columns = [str(column) for column in columns]
    scalar = [column for column in columns if ":" not in column]
    sector = [column for column in columns if column.startswith("sector:")]
    industry = [column for column in columns if column.startswith("industry:")]
    other = [column for column in columns if column not in scalar + sector + industry]
    return sorted(scalar) + sorted(sector) + sorted(industry) + sorted(other)


def one_hot_factors(exposures, columns):
    """Return sector and industry columns that should be zero-filled.

    Example:
        sector:Technology is 0.0 for non-Technology tickers.
    """
    factors = exposures.loc[exposures["group"].isin(["sector", "industry"]), "factor"]
    return sorted(set(factors.astype(str)) & set(columns))


def detail_exposures(exposures):
    """Return long exposure rows for audits and loaders.

    Example:
        detail_exposures(rows) keeps factor, group, value, raw_value, and observations.
    """
    columns = ["as_of_date", "ticker", "factor", "group", "value", "raw_value", "observations"]
    frame = exposures.copy()
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns].sort_values(["as_of_date", "ticker", "factor"]).reset_index(drop=True)


def sorted_columns(frame):
    """Return a table with alphabetically sorted columns.

    Example:
        columns ["size", "beta"] become ["beta", "size"].
    """
    return frame.reindex(sorted(frame.columns), axis=1)


def sorted_covariance(frame):
    """Return covariance with matching sorted rows and columns.

    Example:
        beta appears before size in both axes.
    """
    factors = sorted(set(frame.index) & set(frame.columns))
    return frame.reindex(index=factors, columns=factors)


def residual_returns_file(residuals):
    """Return residual returns as one row per date and ticker.

    Example:
        residuals has dates as rows and tickers as columns.
        AAPL=0.01 and MSFT=NaN on 2026-06-18 returns one AAPL row.
    """
    frame = residuals.copy()
    frame.index.name = "date"
    return (
        frame.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="residual_return")
        .dropna(subset=["residual_return"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def sort_tickers(frame):
    """Return rows sorted by ticker when ticker exists.

    Example:
        MSFT then AAPL becomes AAPL then MSFT.
    """
    if "ticker" not in frame:
        return frame
    return frame.sort_values("ticker").reset_index(drop=True)


def fundamentals_file(fundamentals):
    """Return the model-ready PIT fundamentals table.

    Example:
        revenue becomes revenue_ttm and audit columns are removed.
    """
    frame = fundamentals.rename(columns=TTM_NAMES).copy()
    return keep_columns(frame, FUNDAMENTAL_COLUMNS).sort_values(["ticker", "as_of_date"])


def fundamentals_audit_file(fundamentals):
    """Return PIT filing provenance.

    Example:
        accession_no and revenue_method live here, not in fundamentals.csv.
    """
    frame = fundamentals.copy()
    return keep_columns(frame, FUNDAMENTAL_AUDIT_COLUMNS).sort_values(["ticker", "as_of_date"])


def keep_columns(frame, columns):
    """Return columns in order, creating missing ones as blanks.

    Example:
        keep_columns(frame, ["ticker", "market_cap"]) keeps only those columns.
    """
    frame = frame.copy()
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame[columns]


def spreadsheet_csv(frame, index=False, index_label=None):
    """Return a human-readable rounded CSV string.

    Example:
        spreadsheet_csv(frame) returns CSV text with compact floats.
    """
    output = StringIO()
    frame.to_csv(
        output,
        index=index,
        index_label=index_label,
        float_format=format_float,
    )
    return output.getvalue()


def json_text(data):
    """Return readable JSON text.

    Example:
        json_text({"tickers": 50}) returns indented JSON with a trailing newline.
    """
    return json.dumps(data, indent=2) + "\n"


def format_float(value):
    """Return a compact spreadsheet-friendly float string.

    Example:
        format_float(1.2300001) returns "1.23".
    """
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.6f}".rstrip("0").rstrip(".")
