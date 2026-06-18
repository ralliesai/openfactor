from datetime import timedelta

import numpy as np
import pandas as pd

from data.providers.sec_api.fundamentals import (
    INCOME_METRICS,
    duration_days,
    is_usd_unit,
)
from data.sec.schema import METRIC_COLUMNS


INCOME_TTM_METRICS = dict(INCOME_METRICS)
INCOME_TTM_METRICS["operating_income"] = ["OperatingIncomeLoss"]
FLOW_TTM_METRICS = {
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "buybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "StockRepurchasedAndRetiredDuringPeriodValue",
    ],
    "share_issuance": [
        "ProceedsFromIssuanceOfCommonStock",
        "StockIssuedDuringPeriodValueNewIssues",
    ],
}
MAX_STALE_DAYS = 455


def ttm_metric_rows(facts, filing):
    """Return TTM flow metric rows known at one filing.

    Example:
        annual revenue 100 returns one revenue row; annual capex 10 returns capex 10.
    """
    rows = []
    for metric, concepts in all_ttm_metrics().items():
        periods = income_periods(facts, metric, concepts)
        value, method = ttm_value(periods, filing["period_of_report"])
        if np.isfinite(value):
            rows.append(metric_row(filing, metric, value, method))

    columns = METRIC_COLUMNS + ["method"]
    return pd.DataFrame(rows, columns=columns)


def all_ttm_metrics():
    """Return TTM metrics built from period facts.

    Example:
        revenue and capex are both trailing-twelve-month metrics.
    """
    return {**INCOME_TTM_METRICS, **FLOW_TTM_METRICS}


def income_periods(facts, metric, concepts):
    """Return clean income periods for one metric.

    Example:
        a Q1 revenue fact becomes one 70-110 day revenue period.
    """
    rows = income_rows(facts, concepts)
    rows = rows.dropna(subset=["start_date", "end_date"]).copy()
    if rows.empty:
        return rows

    rows["metric"] = metric
    rows["start_date"] = pd.to_datetime(rows["start_date"], errors="coerce").dt.date
    rows["end_date"] = pd.to_datetime(rows["end_date"], errors="coerce").dt.date
    rows = rows.dropna(subset=["start_date", "end_date"])
    rows["days"] = rows.apply(duration_days, axis=1)
    rows = rows[(rows["days"] >= 70) & (rows["days"] <= 390)]
    rows = rows.sort_values(
        ["start_date", "end_date", "filed_at", "_order"],
        ascending=[True, True, False, True],
    )
    return rows.drop_duplicates(["start_date", "end_date"], keep="first")


def income_rows(facts, concepts):
    """Return USD income rows, preferring unsegmented facts.

    Example:
        COST revenue uses one segmented fact per period, so it is accepted.
    """
    rows = usd_concept_rows(facts, concepts)
    clean = rows[~rows["has_segment"]]
    if not clean.empty:
        return clean

    rows["_count"] = rows.groupby(["concept", "start_date", "end_date"])["value"].transform("count")
    return rows[rows["_count"] == 1].drop(columns=["_count"])


def usd_concept_rows(facts, concepts):
    """Return prioritized USD rows for candidate concepts.

    Example:
        revenue concept rows get _order=0 when it is the first choice.
    """
    rows = facts[
        facts["concept"].isin(concepts)
        & is_usd_unit(facts["unit"])
    ].copy()
    rows["_order"] = rows["concept"].map({concept: i for i, concept in enumerate(concepts)})
    return rows.sort_values("_order")


def ttm_value(periods, report_date):
    """Return TTM value and method for a filing period.

    Example:
        a 10-K annual fact returns (annual_value, "annual").
    """
    report_date = clean_date(report_date)
    if periods.empty or report_date is None:
        return np.nan, "no_income_facts"

    annual = periods[(periods["end_date"] == report_date) & periods["days"].between(330, 390)]
    if not annual.empty:
        return float(annual.sort_values("days").iloc[-1]["value"]), "annual"

    last4 = last_four_quarters(quarter_values(periods), report_date)
    if len(last4) == 4:
        return float(last4["value"].sum()), "sum_last_4_quarters"
    return np.nan, "not_enough_contiguous_quarters"


def quarter_values(periods):
    """Return direct and derived quarter values.

    Example:
        six-month YTD 230 and Q1 100 derives Q2 as 130.
    """
    rows = []
    for row in periods.itertuples(index=False):
        if is_quarter_length(row.start_date, row.end_date):
            rows.append(period_row(row.start_date, row.end_date, row.value, "direct_quarter"))

    ytd = periods[periods["days"].between(120, 390)].sort_values(["start_date", "end_date"])
    for start, group in ytd.groupby("start_date"):
        add_cumulative_quarters(rows, group)
        for row in group.itertuples(index=False):
            known = [item for item in rows if start <= item["end_date"] <= row.end_date]
            if any(item["end_date"] == row.end_date for item in known):
                add_missing_start_quarter(rows, start, row, known)
            elif known:
                add_trailing_quarter(rows, row, known)

    if not rows:
        return pd.DataFrame(columns=["start_date", "end_date", "value", "source"])
    return pd.DataFrame(rows).drop_duplicates("end_date", keep="last")


def add_cumulative_quarters(rows, group):
    """Derive quarters from two cumulative rows with the same fiscal start.

    Example:
        nine-month YTD 300 and six-month YTD 180 derives Q3 as 120.
    """
    group = group.sort_values("end_date")
    previous = None
    for row in group.itertuples(index=False):
        if previous is not None:
            start = previous.end_date + timedelta(days=1)
            if is_quarter_length(start, row.end_date):
                rows.append(
                    period_row(
                        start,
                        row.end_date,
                        row.value - previous.value,
                        "derived_cumulative_quarter",
                    )
                )
        previous = row


def add_trailing_quarter(rows, ytd_row, known):
    """Derive the final quarter inside one YTD row.

    Example:
        nine-month YTD minus Q1 and Q2 gives Q3.
    """
    start = max(item["end_date"] for item in known) + timedelta(days=1)
    if is_quarter_length(start, ytd_row.end_date):
        rows.append(
            period_row(
                start,
                ytd_row.end_date,
                ytd_row.value - sum_values(known),
                "derived_quarter",
            )
        )


def add_missing_start_quarter(rows, start, ytd_row, known):
    """Derive the first missing quarter inside one YTD row.

    Example:
        six-month YTD minus Q2 gives Q1.
    """
    end = min(item["start_date"] for item in known) - timedelta(days=1)
    if end > start and is_quarter_length(start, end):
        rows.append(
            period_row(
                start,
                end,
                ytd_row.value - sum_values(known),
                "derived_missing_start",
            )
        )


def last_four_quarters(quarters, report_date):
    """Return four contiguous quarters ending on report_date.

    Example:
        Q1, Q2, Q3, Q4 returns all four for a Q4 report date.
    """
    report_date = clean_date(report_date)
    if report_date is None:
        return pd.DataFrame(columns=quarters.columns)
    rows = quarters[quarters["end_date"] <= report_date].sort_values("end_date")
    current = rows[rows["end_date"] == report_date]
    if current.empty:
        return pd.DataFrame(columns=quarters.columns)

    block = [current.iloc[-1].to_dict()]
    while len(block) < 4:
        previous = previous_quarter(rows, block[-1]["start_date"])
        if previous is None:
            break
        block.append(previous)

    if len(block) < 4:
        return pd.DataFrame(columns=quarters.columns)
    return pd.DataFrame(reversed(block))


def previous_quarter(rows, start):
    """Return the quarter immediately before start.

    Example:
        previous quarter for Apr 1 ends around Mar 31.
    """
    candidates = rows[rows["end_date"] < start].sort_values("end_date", ascending=False)
    for row in candidates.itertuples(index=False):
        gap = (start - row.end_date).days
        if 1 <= gap <= 10:
            return row._asdict()
        if gap > 10:
            return None
    return None


def metric_row(filing, metric, value, method):
    """Return one TTM metric row.

    Example:
        revenue TTM 100 becomes a USD revenue metric row.
    """
    if metric in FLOW_TTM_METRICS:
        value = abs(value)
    return {
        "ticker": filing["ticker"],
        "accession_no": filing["accession_no"],
        "metric": metric,
        "source_concept": f"ttm_{method}",
        "period_end": filing["period_of_report"],
        "unit": "USD",
        "value": value,
        "method": method,
    }


def period_row(start_date, end_date, value, source):
    """Return one quarter row.

    Example:
        Jan-Mar revenue 100 becomes one period row with source direct_quarter.
    """
    return {
        "start_date": start_date,
        "end_date": end_date,
        "value": float(value),
        "source": source,
    }


def is_quarter_length(start_date, end_date):
    """Return True for normal fiscal-quarter periods.

    Example:
        Jan 1 to Mar 31 returns True.
    """
    return 70 <= (end_date - start_date).days <= 120


def sum_values(rows):
    """Return the sum of row values.

    Example:
        two rows with values 100 and 130 return 230.
    """
    return sum(item["value"] for item in rows)


def clean_date(value):
    """Return a date or None.

    Example:
        clean_date("2026-06-16") returns a date object.
    """
    value = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(value) else value.date()
