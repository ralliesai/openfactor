from dataclasses import dataclass
from datetime import timedelta
import logging

import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger("openfactor.build")
MAX_FILING_RANGE_DAYS = 14
TTM_TO_INTERNAL = {
    "revenue_ttm": "revenue",
    "gross_profit_ttm": "gross_profit",
    "operating_income_ttm": "operating_income",
    "net_income_ttm": "net_income",
}


@dataclass(frozen=True)
class FundamentalHistory:
    """Build or reuse point-in-time fundamentals.

    Example:
        cached rows through 2026-06-15 become 2026-06-16 rows unless a new filing landed.
    """

    downloader: object
    previous: pd.DataFrame | None = None

    def rows(self, tickers, dates):
        """Return daily PIT rows for tickers and model dates.

        Example:
            rows(["AAPL"], ["2026-06-15", "2026-06-16"]) returns both dates.
        """
        tickers = [str(ticker).upper() for ticker in tickers]
        dates = sorted({clean_date(date) for date in dates})
        if self.previous is None or self.previous.empty:
            return self.downloader.sec_history(tickers, dates)

        target = max(dates)
        frame = clean_frame(self.previous)
        frame = frame[frame["as_of_date"] <= target].copy()
        cached = requested_rows(frame, tickers, dates)
        missing = missing_dates_by_ticker(cached, tickers, dates)
        if not missing:
            LOGGER.info("SEC daily cache used tickers=%s missing=0", len(tickers))
            return cached

        carried = self.carried_missing_rows(frame, missing)
        if not carried.empty:
            frame = pd.concat([frame, carried], ignore_index=True)
            cached = requested_rows(frame, tickers, dates)
            missing = missing_dates_by_ticker(cached, tickers, dates)
            LOGGER.info(
                "SEC daily cache carried rows=%s tickers=%s missing_tickers=%s",
                len(carried),
                carried["ticker"].nunique(),
                len(missing),
            )
            if not missing:
                LOGGER.info("SEC daily cache carried tickers=%s missing=0", len(tickers))
                return cached

        LOGGER.info("SEC daily cache miss tickers=%s", len(missing))
        fresh = self.downloader.sec_history_by_ticker_dates(missing, allow_empty=True)
        return requested_rows(pd.concat([cached, fresh], ignore_index=True), tickers, dates)

    def carried_missing_rows(self, frame, missing):
        """Carry every missing PIT date when no newer filing exists.

        Example:
            a Friday AAPL row becomes Monday and Tuesday rows when no filing landed.
        """
        candidates = carry_candidates(frame, missing)
        if candidates.empty:
            return frame.iloc[0:0].copy()

        groups = list(candidates.groupby(["_source_date", "_target_date"]))
        filing_ranges = carry_filing_ranges(groups)
        filing_dates = self.carry_filing_dates(candidates["ticker"].unique(), filing_ranges, len(groups))
        carried = []
        for (source_date, target_date), group in groups:
            changed = changed_tickers(
                filing_dates,
                group["ticker"],
                next_day(source_date),
                target_date,
            )
            rows = group[~group["ticker"].isin(changed)].copy()
            if rows.empty:
                continue
            rows["as_of_date"] = rows["_target_date"]
            carried.append(rows.drop(columns=["_source_date", "_target_date"]))

        LOGGER.info("SEC daily carry check finished")
        if not carried:
            return frame.iloc[0:0].copy()
        return pd.concat(carried, ignore_index=True)

    def carry_filing_dates(self, tickers, filing_ranges, window_count):
        """Fetch filing dates once per carry range for local window checks.

        Example:
            one Friday-to-Tuesday range covers every ticker in that gap.
        """
        LOGGER.info(
            "SEC daily carry check started windows=%s ranges=%s tickers=%s",
            window_count,
            len(filing_ranges),
            len(tickers),
        )
        filing_dates = {}
        for start_date, end_date in tqdm(
            filing_ranges,
            desc="SEC carry filings",
            unit="range",
            dynamic_ncols=True,
        ):
            rows = self.downloader.sec_filing_dates_by_ticker(tickers, start_date, end_date)
            for ticker, dates in rows.items():
                filing_dates.setdefault(ticker, set()).update(dates)
        return filing_dates


def clean_frame(frame):
    """Return normalized cached fundamentals.

    Example:
        ticker aapl becomes AAPL and dates become strings.
    """
    frame = frame.rename(columns=TTM_TO_INTERNAL).copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date.astype(str)
    return frame


def requested_rows(frame, tickers, dates):
    """Return cached rows needed by the model.

    Example:
        AAPL rows outside the requested dates are ignored.
    """
    rows = frame[frame["ticker"].isin(tickers) & frame["as_of_date"].isin(dates)].copy()
    return rows.drop_duplicates(["ticker", "as_of_date"], keep="last")


def carry_candidates(frame, missing):
    """Return prior cached rows that could be carried to missing dates.

    Example:
        missing AAPL Monday and Tuesday rows use AAPL's latest cached Friday row.
    """
    rows = []
    for ticker, dates in missing.items():
        history = frame[frame["ticker"] == ticker].sort_values("as_of_date")
        if history.empty:
            continue
        for target_date in dates:
            prior = history[history["as_of_date"] < target_date]
            if prior.empty:
                continue
            row = prior.iloc[-1].copy()
            row["_source_date"] = row["as_of_date"]
            row["_target_date"] = target_date
            rows.append(row)
    if not rows:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(rows)


def carry_filing_ranges(groups):
    """Return batched SEC filing ranges needed for carry decisions.

    Example:
        many one-day windows across a quarter become a few two-week API ranges.
    """
    intervals = []
    for (source_date, target_date), _ in groups:
        start = pd.to_datetime(next_day(source_date)).date()
        end = pd.to_datetime(target_date).date()
        if start <= end:
            intervals.append((start, end))
    return split_ranges(merged_intervals(intervals), MAX_FILING_RANGE_DAYS)


def merged_intervals(intervals):
    """Merge overlapping filing-date intervals.

    Example:
        Monday-Wednesday and Wednesday-Friday become Monday-Friday.
    """
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + timedelta(days=1):
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def split_ranges(intervals, days):
    """Split long SEC filing intervals into bounded API ranges.

    Example:
        a 40-day interval becomes 14-day, 14-day, and 12-day ranges.
    """
    ranges = []
    for start, end in intervals:
        current = start
        while current <= end:
            chunk_end = min(end, current + timedelta(days=days - 1))
            ranges.append((current.isoformat(), chunk_end.isoformat()))
            current = chunk_end + timedelta(days=1)
    return ranges


def changed_tickers(filing_dates, tickers, start_date, end_date):
    """Return tickers with any filing date inside one carry window.

    Example:
        AAPL filed Monday, so a Friday-to-Tuesday carry window marks AAPL changed.
    """
    return {
        ticker
        for ticker in tickers
        if any(start_date <= filed_at <= end_date for filed_at in filing_dates.get(ticker, ()))
    }


def missing_dates_by_ticker(frame, tickers, dates):
    """Return requested ticker/date pairs absent from cache.

    Example:
        cached AAPL 2026-06-16 leaves MSFT 2026-06-16 missing.
    """
    present = set(zip(frame["ticker"], frame["as_of_date"]))
    return {
        ticker: [date for date in dates if (ticker, date) not in present]
        for ticker in tickers
        if any((ticker, date) not in present for date in dates)
    }


def next_day(value):
    """Return the day after one ISO date.

    Example:
        next_day("2026-06-15") returns "2026-06-16".
    """
    return (pd.to_datetime(value).date() + timedelta(days=1)).isoformat()


def clean_date(value):
    """Return one ISO date string.

    Example:
        clean_date("2026-06-16 00:00:00") returns "2026-06-16".
    """
    return pd.to_datetime(value).date().isoformat()
