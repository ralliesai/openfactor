from dataclasses import dataclass
from datetime import timedelta
import logging

import pandas as pd


LOGGER = logging.getLogger("openfactor.build")
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
        dates = [clean_date(date) for date in dates]
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

        if any(target in missing_dates for missing_dates in missing.values()):
            frame = pd.concat([frame, self.carried_today_rows(frame, tickers, target)])
            cached = requested_rows(frame, tickers, dates)
            missing = missing_dates_by_ticker(cached, tickers, dates)
            if not missing:
                LOGGER.info("SEC daily cache carried tickers=%s missing=0", len(tickers))
                return cached

        fresh = []
        for wanted_dates, wanted_tickers in grouped_missing(missing):
            LOGGER.info("SEC daily cache miss tickers=%s dates=%s", len(wanted_tickers), len(wanted_dates))
            fresh.append(self.downloader.sec_history(wanted_tickers, wanted_dates))
        return requested_rows(pd.concat([cached] + fresh, ignore_index=True), tickers, dates)

    def carried_today_rows(self, frame, tickers, today):
        """Carry latest rows to today when no newer filing exists.

        Example:
            a 2026-06-15 AAPL row becomes 2026-06-16 when AAPL had no new filing.
        """
        latest = latest_before_today(frame, tickers, today)
        if latest.empty:
            return latest

        start = next_day(latest["as_of_date"].min())
        changed = self.downloader.new_sec_filing_tickers(latest["ticker"], start, today)
        carried = latest[~latest["ticker"].isin(changed)].copy()
        carried["as_of_date"] = today
        return carried


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


def latest_before_today(frame, tickers, today):
    """Return latest cached rows before today.

    Example:
        latest_before_today(rows, ["AAPL"], "2026-06-16") returns AAPL's prior row.
    """
    rows = frame[frame["ticker"].isin(tickers) & (frame["as_of_date"] < today)].copy()
    return rows.sort_values("as_of_date").drop_duplicates("ticker", keep="last")


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


def grouped_missing(missing):
    """Group tickers that need the same missing dates.

    Example:
        two tickers missing the same year-ago dates are downloaded together.
    """
    groups = {}
    for ticker, dates in missing.items():
        groups.setdefault(tuple(dates), []).append(ticker)
    return [(list(dates), tickers) for dates, tickers in groups.items()]


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
