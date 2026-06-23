from datetime import timedelta
import logging

import pandas as pd


LOGGER = logging.getLogger("openfactor.build")


def cached_price_history(previous, tickers, start_date, end_date, fetch, label):
    """Return price rows by extending cached ticker histories.

    Example:
        an existing AAPL cache through Friday fetches only Monday onward.
    """
    tickers = clean_tickers(tickers)
    previous = filter_dated_rows(previous, tickers, start_date, end_date, "date")
    if previous.empty:
        LOGGER.info("%s cache miss tickers=%s reason=no_previous", label, len(tickers))
        return fetch(tickers, start_date, end_date)

    fresh = []
    groups = price_refresh_groups(previous, tickers, start_date, end_date)
    for (fetch_start, fetch_end), group_tickers in groups:
        LOGGER.info("%s cache miss tickers=%s dates=%s..%s", label, len(group_tickers), fetch_start, fetch_end)
        fresh.append(fetch(group_tickers, fetch_start, fetch_end))
    rows = concat_cached_rows(previous, concat_frames(fresh), ["ticker", "date"])
    LOGGER.info("%s cache used rows=%s tickers=%s refresh_groups=%s", label, len(rows), len(tickers), len(groups))
    return filter_dated_rows(rows, tickers, start_date, end_date, "date")


def cached_event_history(
    previous,
    cached_tickers,
    cache_date,
    tickers,
    start_date,
    end_date,
    date_column,
    fetch,
    keys,
    label,
    min_existing_ticker_coverage=None,
):
    """Return dated event rows by fetching only unchecked dates.

    Example:
        dividend cache checked through Friday fetches Saturday-Tuesday only.
    """
    tickers = clean_tickers(tickers)
    if previous is None:
        LOGGER.info("%s cache miss tickers=%s reason=no_previous", label, len(tickers))
        return fetch(tickers, start_date, end_date)

    previous = filter_dated_rows(previous, tickers, start_date, end_date, date_column)
    if cache_date is None:
        LOGGER.info("%s cache miss tickers=%s reason=no_watermark", label, len(tickers))
        return fetch(tickers, start_date, end_date)

    cached_tickers = set(cached_tickers)
    if sparse_event_cache(previous, cached_tickers, min_existing_ticker_coverage):
        LOGGER.info(
            "%s cache miss tickers=%s reason=sparse_previous rows=%s",
            label,
            len(tickers),
            len(previous),
        )
        return fetch(tickers, start_date, end_date)

    existing = [ticker for ticker in tickers if ticker in cached_tickers]
    new = [ticker for ticker in tickers if ticker not in cached_tickers]
    fresh = []
    refresh_start = next_date(cache_date)
    if existing and refresh_start <= end_date:
        LOGGER.info("%s cache miss tickers=%s dates=%s..%s", label, len(existing), refresh_start, end_date)
        fresh.append(fetch(existing, refresh_start, end_date))
    if new:
        LOGGER.info("%s cache miss tickers=%s dates=%s..%s reason=new_ticker", label, len(new), start_date, end_date)
        fresh.append(fetch(new, start_date, end_date))

    rows = concat_cached_rows(previous, concat_frames(fresh), keys)
    LOGGER.info("%s cache used rows=%s tickers=%s", label, len(rows), len(tickers))
    return filter_dated_rows(rows, tickers, start_date, end_date, date_column)


def sparse_event_cache(previous, cached_tickers, minimum):
    """Return True when an existing raw event cache is clearly under-seeded."""
    if minimum is None or not cached_tickers:
        return False
    covered = ticker_set(previous)
    return len(covered) / len(cached_tickers) < minimum


def missing_earnings_tickers(fundamentals, tickers, dates, previous=None, reported=None):
    """Return tickers whose requested rows lack carried earnings inputs.

    Example:
        a fresh SEC row without earnings_quality asks Finnhub for that ticker.
    """
    tickers = clean_tickers(tickers)
    if fundamentals.empty:
        return []
    checked_keys = previous_earnings_keys(previous) | reported_accession_keys(reported)
    rows = fundamentals.copy()
    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    rows["as_of_date"] = pd.to_datetime(rows["as_of_date"]).dt.date.astype(str)
    requested_dates = {pd.to_datetime(date).date().isoformat() for date in dates}
    rows = rows[rows["as_of_date"].isin(requested_dates)]
    if "accession_no" not in rows:
        return sorted(set(rows["ticker"]) & set(tickers))

    missing = []
    for ticker, group in rows.groupby("ticker"):
        if ticker not in tickers:
            continue
        if "earnings_quality" not in group or "earnings_variability" not in group:
            missing.append(ticker)
            continue
        quality = pd.to_numeric(group["earnings_quality"], errors="coerce")
        variability = pd.to_numeric(group["earnings_variability"], errors="coerce")
        blank = group[quality.isna() & variability.isna()]
        if any((row.ticker, str(row.accession_no)) not in checked_keys for row in blank.itertuples(index=False)):
            missing.append(ticker)
    return sorted(set(missing))


def previous_earnings_keys(previous):
    """Return ticker/accession pairs with carried earnings inputs."""
    if (
        previous is None
        or previous.empty
        or "ticker" not in previous
        or "accession_no" not in previous
        or "earnings_quality" not in previous
        or "earnings_variability" not in previous
    ):
        return set()
    rows = previous[["ticker", "accession_no", "earnings_quality", "earnings_variability"]].dropna(
        subset=["ticker", "accession_no"]
    ).copy()
    quality = pd.to_numeric(rows["earnings_quality"], errors="coerce")
    variability = pd.to_numeric(rows["earnings_variability"], errors="coerce")
    rows = rows[quality.notna() | variability.notna()]
    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    rows["accession_no"] = rows["accession_no"].astype(str)
    return set(rows[["ticker", "accession_no"]].itertuples(index=False, name=None))


def reported_accession_keys(reported):
    """Return ticker/accession pairs present in raw reported-financial rows."""
    if reported is None or reported.empty or "ticker" not in reported or "accession_no" not in reported:
        return set()
    rows = reported[["ticker", "accession_no"]].dropna().copy()
    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    rows["accession_no"] = rows["accession_no"].astype(str)
    return set(rows.itertuples(index=False, name=None))


def price_refresh_groups(previous, tickers, start_date, end_date):
    """Return grouped missing price windows by ticker.

    Example:
        two tickers cached through Friday share one Monday-Tuesday fetch.
    """
    groups = {}
    by_ticker = {
        ticker: max_date(rows, "date")
        for ticker, rows in previous.groupby("ticker")
    }
    for ticker in tickers:
        last = by_ticker.get(ticker)
        fetch_start = start_date if last is None else next_date(last)
        if fetch_start <= end_date:
            groups.setdefault((fetch_start, end_date), []).append(ticker)
    return list(groups.items())


def filter_ticker_rows(frame, tickers):
    """Return rows for requested tickers with normalized ticker symbols."""
    if frame is None or frame.empty or "ticker" not in frame:
        return pd.DataFrame()
    rows = frame.copy()
    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    return rows[rows["ticker"].isin(clean_tickers(tickers))].copy()


def filter_dated_rows(frame, tickers, start_date, end_date, date_column):
    """Return cached rows inside one ticker/date window."""
    rows = filter_ticker_rows(frame, tickers)
    if rows.empty or date_column not in rows:
        return pd.DataFrame()
    rows[date_column] = pd.to_datetime(rows[date_column], errors="coerce").dt.date.astype(str)
    return rows[(rows[date_column] >= start_date) & (rows[date_column] <= end_date)].copy()


def concat_cached_rows(previous, fresh, keys):
    """Return cached and fresh rows de-duplicated by stable keys."""
    rows = concat_frames([previous, fresh])
    if rows.empty:
        return rows
    keys = [key for key in keys if key in rows]
    if not keys:
        return rows
    return rows.drop_duplicates(keys, keep="last").reset_index(drop=True)


def concat_frames(frames):
    """Return non-empty frames as one table."""
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def max_date(frame, column):
    """Return the max ISO date in a frame column."""
    if frame is None or frame.empty or column not in frame:
        return None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return values.max().date().isoformat()


def next_date(value):
    """Return the day after one ISO date."""
    return (pd.to_datetime(value).date() + timedelta(days=1)).isoformat()


def ticker_set(frame):
    """Return uppercase tickers present in a frame."""
    if frame is None or frame.empty or "ticker" not in frame:
        return set()
    return set(frame["ticker"].astype(str).str.upper())


def clean_tickers(tickers):
    """Return uppercase ticker strings in input order."""
    return [str(ticker).upper() for ticker in tickers]
