import numpy as np
import pandas as pd
from data.providers.sec_api.fundamentals import SecFundamentalsBuilder
from data.sec.schema import DAILY_FUNDAMENTAL_COLUMNS, FACT_COLUMNS, METRIC_COLUMNS
from data.sec.ttm import INCOME_TTM_METRICS, MAX_STALE_DAYS, ttm_metric_rows


SEC_TICKER_ALIAS = {"GOOGL": "GOOG"}


class SecHistoryBuilder:
    """Build point-in-time daily SEC fundamentals.

    Example:
        SecHistoryBuilder(client).rows("AAPL", ["2026-06-16"])
        returns the filing metrics known on that date.
    """

    def __init__(self, client):
        self.client = client
        self.fundamentals = SecFundamentalsBuilder(client)

    def rows(self, ticker, dates):
        """Build daily rows using the latest filing available on each date.

        Example:
            after a 10-Q lands, later dates reuse it until a newer filing appears.
        """
        ticker = str(ticker).upper()
        sec_ticker = SEC_TICKER_ALIAS.get(ticker, ticker)
        dates = sorted({pd.to_datetime(date).date() for date in dates})
        reference = self.fundamentals.company_reference(sec_ticker)
        as_of_date = max(dates)
        filings = self.client.filings(
            sec_ticker,
            as_of_date,
            start_date=filing_start_date(as_of_date),
        )
        if reference.empty or filings.empty:
            return self.empty_frame()

        filings = filings.copy()
        filings["filed_at"] = pd.to_datetime(filings["filed_at"]).dt.date
        filings["period_of_report"] = pd.to_datetime(
            filings["period_of_report"],
            errors="coerce",
        ).dt.date
        filings = filings.dropna(subset=["accession_no", "filed_at", "period_of_report"])
        filings = filings.sort_values("filed_at")
        if filings.empty:
            return self.empty_frame()

        facts = self.filing_facts(filings)
        metrics_by_accession = self.filing_metrics(filings, facts)
        rows = []
        for day in dates:
            filing = self.latest_filing_row(filings, day)
            if filing is None or stale(filing, day):
                continue

            rows.append(
                self.daily_row(
                    reference,
                    filing,
                    metrics_by_accession.get(filing["accession_no"], self.empty_metrics()),
                    day,
                    ticker,
                )
            )

        return pd.DataFrame(rows, columns=DAILY_FUNDAMENTAL_COLUMNS)

    def filing_facts(self, filings):
        """Return XBRL facts for every filing.

        Example:
            AAPL 10-K and 10-Q filings become one fact table.
        """
        frames = []
        for filing in filings.to_dict("records"):
            frames.append(self.facts_for(filing))
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not frames:
            return self.empty_facts()
        return pd.concat(frames, ignore_index=True)

    def facts_for(self, filing):
        """Return one filing's facts with filing metadata.

        Example:
            AAPL 10-Q facts get filed_at and form_type columns.
        """
        facts = self.fundamentals.statement_facts(pd.DataFrame([filing]))
        facts["filed_at"] = filing["filed_at"]
        facts["form_type"] = filing["form_type"]
        return facts

    def filing_metrics(self, filings, facts):
        """Return point-in-time metrics keyed by accession number.

        Example:
            raw 10-Q income rows are replaced by TTM income rows.
        """
        metrics = {}
        for filing in filings.to_dict("records"):
            current = pd.DataFrame([filing])
            current_facts = facts[facts["accession_no"] == filing["accession_no"]]
            point = self.fundamentals.canonical_metrics(current, current_facts)
            point = point[~point["metric"].isin(INCOME_TTM_METRICS)]
            available = facts[facts["filed_at"] <= filing["filed_at"]]
            metrics[filing["accession_no"]] = concat_metrics([
                point,
                ttm_metric_rows(available, filing),
            ])
        return metrics

    def empty_metrics(self):
        """Return an empty metric frame.

        Example:
            missing filing facts produce no metric values.
        """
        return pd.DataFrame(columns=METRIC_COLUMNS + ["method"])

    def empty_facts(self):
        """Return an empty fact frame with filing metadata columns.

        Example:
            a skipped XBRL call still has the expected fact columns.
        """
        return pd.DataFrame(columns=FACT_COLUMNS + ["filed_at", "form_type"])

    def latest_filing_row(self, filings, as_of_date):
        """Return the newest fiscal period available by one date.

        Example:
            a late 10-K/A does not replace a newer 10-Q period.
        """
        rows = filings[filings["filed_at"] <= as_of_date]
        if rows.empty:
            return None
        rows = rows.sort_values(["period_of_report", "filed_at"])
        return rows.iloc[-1].to_dict()

    def daily_row(self, reference, filing, metrics, as_of_date, ticker):
        """Return one daily wide fundamental row.

        Example:
            revenue and assets metric rows become revenue/assets columns.
        """
        reference = reference.iloc[0]
        values = {row.metric: row.value for row in metrics.itertuples(index=False)}
        dates = {row.metric: row.period_end for row in metrics.itertuples(index=False)}
        sources = {row.metric: row.source_concept for row in metrics.itertuples(index=False)}
        methods = metric_methods(metrics)
        row = {
            "ticker": ticker,
            "as_of_date": as_of_date,
            "accession_no": filing["accession_no"],
            "form_type": filing["form_type"],
            "filed_at": filing["filed_at"],
            "period_of_report": filing["period_of_report"],
            "sector": reference.get("sector"),
            "industry": reference.get("industry"),
            "sic": reference.get("sic"),
            "sic_industry": reference.get("sic_industry"),
            "fama_industry": reference.get("fama_industry"),
            "shares_outstanding": values.get("shares_outstanding", np.nan),
            "shares_outstanding_date": dates.get("shares_outstanding"),
            "shares_outstanding_source": sources.get("shares_outstanding"),
        }
        for metric in INCOME_TTM_METRICS:
            row[f"{metric}_method"] = methods.get(metric)
        for column in DAILY_FUNDAMENTAL_COLUMNS:
            row.setdefault(column, values.get(column, np.nan))
        return row

    def empty_frame(self):
        """Return an empty daily SEC frame.

        Example:
            empty_frame() has the daily fundamentals columns and zero rows.
        """
        return pd.DataFrame(columns=DAILY_FUNDAMENTAL_COLUMNS)


def fetch_daily_rows(client, ticker, dates):
    """Build daily point-in-time SEC rows without a local cache.

    Example:
        fetch_daily_rows(client, "AAPL", ["2026-06-16"])
        returns the filing metrics known on that date.
    """
    return SecHistoryBuilder(client).rows(ticker, dates)


def stale(filing, as_of_date):
    """Return True when a filing is too old for daily PIT use.

    Example:
        a 2016 filing is stale for a 2025 model date.
    """
    return (as_of_date - filing["filed_at"]).days > MAX_STALE_DAYS


def filing_start_date(as_of_date):
    """Return the SEC filing start date for one build.

    Example:
        2026-06-16 returns 2022-01-01.
    """
    as_of_date = pd.to_datetime(as_of_date).date()
    return as_of_date.replace(year=as_of_date.year - 4, month=1, day=1)


def metric_methods(metrics):
    """Return TTM method labels keyed by metric.

    Example:
        revenue row with method annual returns {"revenue": "annual"}.
    """
    if "method" not in metrics:
        return {}
    rows = metrics.dropna(subset=["method"])
    return {row.metric: row.method for row in rows.itertuples(index=False)}


def concat_metrics(frames):
    """Return non-empty metric frames as one table.

    Example:
        empty raw rows plus TTM rows returns only the TTM rows.
    """
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=METRIC_COLUMNS + ["method"])
    return pd.concat(frames, ignore_index=True)
