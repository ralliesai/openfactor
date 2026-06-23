from dataclasses import dataclass
from datetime import timedelta
import logging

import pandas as pd
from tqdm import tqdm


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

        fresh = []
        for wanted_dates, wanted_tickers in grouped_missing(missing):
            LOGGER.info("SEC daily cache miss tickers=%s dates=%s", len(wanted_tickers), len(wanted_dates))
            fresh.append(self.downloader.sec_history(wanted_tickers, wanted_dates))
        return requested_rows(pd.concat([cached] + fresh, ignore_index=True), tickers, dates)

    def carried_missing_rows(self, frame, missing):
        """Carry every missing PIT date when no newer filing exists.

        Example:
            a Friday AAPL row becomes Monday and Tuesday rows when no filing landed.
        """
        candidates = carry_candidates(frame, missing)
        if candidates.empty:
            return frame.iloc[0:0].copy()

        groups = list(candidates.groupby(["_source_date", "_target_date"]))
        LOGGER.info(
            "SEC daily carry check started count=%s tickers=%s",
            len(groups),
            candidates["ticker"].nunique(),
        )
        carried = []
        for (source_date, target_date), group in tqdm(
            groups,
            desc="SEC carry check",
            unit="window",
            dynamic_ncols=True,
        ):
            changed = set(
                self.downloader.new_sec_filing_tickers(
                    group["ticker"],
                    next_day(source_date),
                    target_date,
                )
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
