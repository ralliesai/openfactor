from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from data.sec.schema import (
    FACT_COLUMNS,
    FILING_COLUMNS,
    METRIC_COLUMNS,
    REFERENCE_COLUMNS,
)
from openfactor.core.sic import sector_from_sic


INCOME_METRICS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
        "InterestIncomeExpenseNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": [
        "OperatingIncomeLoss",
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"
            "NoncontrollingInterest"
        ),
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
}

BALANCE_METRICS = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}

SHARE_METRICS = {
    "shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
    ],
    "weighted_basic": ["WeightedAverageNumberOfSharesOutstandingBasic"],
}


@dataclass(frozen=True)
class SecFundamentals:
    """SEC fundamentals for one ticker as of one date.

    Example:
        result = load_fundamentals(client, "AAPL", "2026-06-16")
        result.metrics contains revenue, assets, liabilities, and income rows.
    """

    reference: pd.DataFrame
    filing: pd.DataFrame
    facts: pd.DataFrame
    metrics: pd.DataFrame


class SecFundamentalsBuilder:
    """Load SEC-API reference, filing, facts, and canonical metrics.

    Example:
        SecFundamentalsBuilder(client).load("AAPL", "2026-06-16")
        returns SecFundamentals(reference, filing, facts, metrics).
    """

    def __init__(self, client):
        self.client = client

    def load(self, ticker, as_of_date):
        """Load all SEC fundamentals for one ticker and as-of date.

        Example:
            builder.load("AAPL", "2026-06-16")
            returns reference rows plus the latest available filing metrics.
        """
        reference = self.company_reference(ticker)
        filing = self.latest_filing(ticker, as_of_date)
        facts = self.statement_facts(filing)
        metrics = self.canonical_metrics(filing, facts)
        return SecFundamentals(reference, filing, facts, metrics)

    def company_reference(self, ticker):
        """Return SEC-API company reference rows.

        Example:
            company_reference("AAPL") returns CIK, SIC, sector, and industry.
        """
        return company_reference(self.client, ticker)

    def latest_filing(self, ticker, as_of_date):
        """Return the latest 10-K or 10-Q filed by an as-of date.

        Example:
            latest_filing("AAPL", "2024-12-31")
            returns the newest filing available by that date.
        """
        return latest_filing(self.client, ticker, as_of_date)

    def statement_facts(self, filing):
        """Return raw XBRL facts for one filing row.

        Example:
            statement_facts(filing) returns Assets and Revenue fact rows.
        """
        return statement_facts(self.client, filing)

    def canonical_metrics(self, filing, facts):
        """Return OpenFactor metric rows from one filing.

        Example:
            canonical_metrics(filing, facts) returns value and leverage inputs.
        """
        return canonical_metrics(self.client, filing, facts)


def company_reference(client, ticker):
    """Return SEC-API company reference rows.

    Example:
        company_reference(client, "AAPL") returns CIK, SIC, sector, and industry.
    """
    ticker = str(ticker).upper()
    rows = client.mapping(ticker)
    if not rows:
        return pd.DataFrame(columns=REFERENCE_COLUMNS)

    row = best_mapping(rows, ticker)
    sic = row.get("sic")
    sector = row.get("sector") or sector_from_sic(sic)
    data = {
        "ticker": ticker,
        "name": row.get("name"),
        "cik": row.get("cik"),
        "cusip": row.get("cusip"),
        "exchange": row.get("exchange"),
        "is_delisted": row.get("isDelisted"),
        "sector": sector,
        "industry": row.get("industry"),
        "sic": sic,
        "sic_sector": row.get("sicSector"),
        "sic_industry": row.get("sicIndustry"),
        "fama_sector": np.nan,
        "fama_industry": row.get("famaIndustry"),
        "currency": row.get("currency"),
        "location": row.get("location"),
    }
    return pd.DataFrame([data], columns=REFERENCE_COLUMNS)


def best_mapping(rows, ticker):
    """Return the best SEC-API mapping row for one ticker.

    Example:
        BAC fuzzy results return BANK OF AMERICA CORP, not ABACAN RESOURCE.
    """
    ticker = str(ticker).upper()
    exact = [row for row in rows if str(row.get("ticker", "")).upper() == ticker]
    choices = exact or rows
    return sorted(choices, key=mapping_rank)[0]


def mapping_rank(row):
    """Return a sort key for active common-stock mappings.

    Example:
        active NYSE primary common stock ranks before delisted matches.
    """
    category = str(row.get("category", "")).lower()
    exchange = str(row.get("exchange", "")).upper()
    return (
        bool(row.get("isDelisted")),
        "primary class" not in category,
        "common stock" not in category,
        exchange not in {"NYSE", "NASDAQ"},
    )


def latest_filing(client, ticker, as_of_date):
    """Return the latest 10-K or 10-Q filed by an as-of date.

    Example:
        latest_filing(client, "AAPL", "2024-12-31")
        returns the newest SEC-API filing available by that date.
    """
    filings = client.filings(ticker, as_of_date)
    if filings.empty:
        return pd.DataFrame(columns=FILING_COLUMNS)
    return filings.iloc[[0]].reset_index(drop=True)


def statement_facts(client, filing):
    """Return raw XBRL facts for one filing row.

    Example:
        statement_facts(client, filing)
        returns rows like RevenueFromContractWithCustomerExcludingAssessedTax.
    """
    if filing.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)

    row = filing.iloc[0]
    xbrl = client.xbrl(row["accession_no"])
    return fact_rows(row["ticker"], row["accession_no"], xbrl)


def fact_rows(ticker, accession_no, xbrl):
    """Turn SEC-API XBRL JSON into OpenFactor fact rows.

    Example:
        BalanceSheets.Assets with value 100 becomes one Assets fact row.
    """
    rows = []
    for statement, concepts in xbrl.items():
        if not isinstance(concepts, dict):
            continue
        for concept, items in concepts.items():
            for item in as_list(items):
                if not isinstance(item, dict):
                    continue
                value = number(item.get("value"))
                if not np.isfinite(value):
                    continue
                period = item.get("period", {})
                rows.append(
                    {
                        "ticker": ticker,
                        "accession_no": accession_no,
                        "statement": statement,
                        "concept": concept,
                        "start_date": period.get("startDate"),
                        "end_date": period.get("endDate"),
                        "instant_date": period.get("instant"),
                        "unit": item.get("unitRef"),
                        "value": value,
                        "has_segment": has_segment(item),
                    }
                )
    return pd.DataFrame(rows, columns=FACT_COLUMNS)


def canonical_metrics(client, filing, facts):
    """Return OpenFactor metric rows from one filing.

    Example:
        canonical_metrics(client, filing, facts)
        returns net_income, total_assets, leverage inputs, and asset_growth.
    """
    if filing.empty or facts.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    filing = filing.iloc[0]
    rows = []
    values = {}
    for metric, concepts in INCOME_METRICS.items():
        fact = best_income_fact(facts, concepts, filing)
        values[metric] = value_from(fact)
        if fact is not None:
            rows.append(metric_row(filing, metric, fact["concept"], fact["value"]))

    for metric, concepts in BALANCE_METRICS.items():
        fact = best_instant_fact(facts, concepts, filing)
        values[metric] = value_from(fact)
        if fact is not None:
            rows.append(metric_row(filing, metric, fact["concept"], fact["value"]))

    if not np.isfinite(values.get("total_liabilities", np.nan)):
        assets = values.get("total_assets")
        equity = values.get("stockholders_equity")
        if np.isfinite(assets) and np.isfinite(equity):
            rows.append(
                metric_row(
                    filing,
                    "total_liabilities",
                    "assets_minus_equity",
                    assets - equity,
                )
            )

    growth = asset_growth_row(filing, facts)
    if growth is not None:
        rows.append(growth)
    shares = shares_outstanding_row(filing, facts)
    if shares is not None:
        rows.append(shares)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def best_income_fact(facts, concepts, filing):
    """Return the best unsegmented income fact for one filing.

    Example:
        10-Q rows prefer the shortest period ending on period_of_report.
    """
    rows = concept_rows(facts, concepts)
    rows = rows[rows["end_date"].astype(str) == str(filing["period_of_report"])]
    if rows.empty:
        return None

    rows = rows.copy()
    rows["_days"] = rows.apply(duration_days, axis=1)
    if filing["form_type"] == "10-K":
        return rows.sort_values("_days").iloc[-1]
    return rows.sort_values("_days").iloc[0]


def best_instant_fact(facts, concepts, filing):
    """Return the best balance-sheet fact for one filing.

    Example:
        Assets dated 2026-03-31 is used for a 2026-03-31 10-Q.
    """
    rows = concept_rows(facts, concepts)
    rows = rows[rows["instant_date"].astype(str) == str(filing["period_of_report"])]
    if rows.empty:
        return None
    return rows.iloc[0]


def concept_rows(facts, concepts):
    """Return unsegmented USD facts for prioritized concepts.

    Example:
        concept_rows(facts, ["Assets"]) returns consolidated USD asset rows.
    """
    rows = facts[
        facts["concept"].isin(concepts)
        & (~facts["has_segment"])
        & is_usd_unit(facts["unit"])
    ].copy()
    rows["_order"] = rows["concept"].map({concept: i for i, concept in enumerate(concepts)})
    return rows.sort_values("_order")


def is_usd_unit(units):
    """Return True for SEC-API dollar unit strings.

    Example:
        "USD" and "Unit_Standard_USD_x" are both dollar units.
    """
    return units.astype(str).str.lower().str.contains("usd", na=False)


def is_share_unit(units):
    """Return True for SEC-API share unit strings.

    Example:
        "shares" and "Unit_Standard_shares_x" are both share units.
    """
    return units.astype(str).str.lower().str.contains("shares", na=False)


def asset_growth_row(filing, facts):
    """Return asset growth from current and prior asset facts.

    Example:
        assets move from 100 to 110, so asset_growth is 0.10.
    """
    rows = concept_rows(facts, ["Assets"])
    rows = rows[rows["instant_date"].astype(str) <= str(filing["period_of_report"])]
    rows = rows.sort_values("instant_date").drop_duplicates("instant_date", keep="last")
    if len(rows) < 2:
        return None

    current = rows.iloc[-1]["value"]
    prior = rows.iloc[-2]["value"]
    if prior == 0:
        return None
    return metric_row(filing, "asset_growth", "Assets", (current - prior) / abs(prior))


def shares_outstanding_row(filing, facts):
    """Return latest common shares outstanding from SEC XBRL.

    Example:
        EntityCommonStockSharesOutstanding becomes the shares_outstanding metric.
    """
    rows = share_rows(facts, SHARE_METRICS["shares_outstanding"])
    if rows.empty:
        return None

    order = rows["_order"].min()
    rows = rows[rows["_order"] == order]
    latest = rows["instant_date"].max()
    rows = rows[rows["instant_date"] == latest]
    source = rows["concept"].iloc[0]
    if len(rows) > 1:
        source = f"{source}_sum"
    value = rows["value"].sum()

    weighted_basic = weighted_basic_share_count(filing, facts)
    if weighted_basic is not None and weighted_basic["value"] > value * 1.2:
        return metric_row(
            filing,
            "shares_outstanding",
            f"{weighted_basic['concept']}_weighted_basic",
            weighted_basic["value"],
            weighted_basic["period_end"],
        )

    return metric_row(
        filing,
        "shares_outstanding",
        source,
        value,
        latest,
    )


def weighted_basic_share_count(filing, facts):
    """Return latest weighted basic shares when it fixes class conversion.

    Example:
        BRK.B cover-page A/B counts can use B-equivalent weighted shares.
    """
    rows = facts[
        facts["concept"].isin(SHARE_METRICS["weighted_basic"])
        & is_share_unit(facts["unit"])
        & (facts["end_date"].astype(str) == str(filing["period_of_report"]))
    ].copy()
    if rows.empty:
        return None

    rows["_days"] = rows.apply(duration_days, axis=1)
    days = rows["_days"].max() if filing["form_type"] == "10-K" else rows["_days"].min()
    rows = rows[rows["_days"] == days]
    row = rows.sort_values("value").iloc[-1]
    return {
        "concept": row["concept"],
        "period_end": row["end_date"],
        "value": row["value"],
    }


def share_rows(facts, concepts):
    """Return share-count facts for prioritized concepts.

    Example:
        share_rows(facts, ["EntityCommonStockSharesOutstanding"]) returns share facts.
    """
    rows = facts[
        facts["concept"].isin(concepts)
        & is_share_unit(facts["unit"])
    ].copy()
    if rows.empty:
        return rows

    rows["_order"] = rows["concept"].map({concept: i for i, concept in enumerate(concepts)})
    rows["instant_date"] = pd.to_datetime(rows["instant_date"]).dt.date
    return rows.dropna(subset=["instant_date"])


def metric_row(filing, metric, source, value, period_end=None):
    """Return one canonical metric row.

    Example:
        metric_row(filing, "net_income", "NetIncomeLoss", 10)
        returns one net_income row with value 10.
    """
    return {
        "ticker": filing["ticker"],
        "accession_no": filing["accession_no"],
        "metric": metric,
        "source_concept": source,
        "period_end": period_end or filing["period_of_report"],
        "unit": unit_for(metric),
        "value": value,
    }


def as_list(value):
    """Return a list for scalar or list XBRL values.

    Example:
        as_list({"value": "1"}) returns [{"value": "1"}].
    """
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def has_segment(item):
    """Return True when an XBRL fact has a segment.

    Example:
        product revenue rows return True; consolidated rows return False.
    """
    segment = item.get("segment")
    return segment not in (None, {})


def duration_days(row):
    """Return duration length for a period fact.

    Example:
        2026-01-01 to 2026-03-31 returns about 89 days.
    """
    start = clean_date(row["start_date"])
    end = clean_date(row["end_date"])
    if start is None or end is None:
        return 10**9
    return (end - start).days


def clean_date(value):
    """Return a date or None.

    Example:
        clean_date("2026-03-31") returns date(2026, 3, 31).
    """
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def value_from(row):
    """Return a float value from a fact row or NaN.

    Example:
        value_from(None) returns np.nan.
    """
    if row is None:
        return np.nan
    return float(row["value"])


def number(value):
    """Return a float or NaN.

    Example:
        number("10") returns 10.0; number("bad") returns NaN.
    """
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def unit_for(metric):
    """Return the natural unit for a metric.

    Example:
        unit_for("asset_growth") returns None.
    """
    if metric == "asset_growth":
        return None
    if metric == "shares_outstanding":
        return "shares"
    return "USD"


def load_fundamentals(client, ticker, as_of_date):
    """Load SEC reference, filing, raw facts, and canonical metrics.

    Example:
        load_fundamentals(client, "AAPL", "2026-06-16")
        returns SecFundamentals(reference, filing, facts, metrics).
    """
    return SecFundamentalsBuilder(client).load(ticker, as_of_date)
