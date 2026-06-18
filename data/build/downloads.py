from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

import pandas as pd
from tqdm import tqdm

from data.providers.finnhub import FinnhubClient, reported_financials
from data.providers.fmp import FmpClient, analyst_estimates
from data.providers.massive import MassiveClient
from data.providers.massive.stocks import (
    stock_bars,
    stock_dividends,
    stock_short_interest,
    ticker_overview,
)
from data.providers.sec_api import SecApiClient
from data.providers.tipranks import TipRanksClient, analyst_ratings
from data.sec.history import SEC_TICKER_ALIAS, fetch_daily_rows


LOGGER = logging.getLogger("openfactor.factory")


@dataclass(frozen=True)
class ProviderDownloader:
    """Download OpenFactor provider inputs.

    Example:
        downloader = ProviderDownloader(workers=8, sec_workers=1)
        downloader.prices(["AAPL"], "2024-01-01", "2024-01-31")
        returns daily Massive price rows.
    """

    workers: int = 8
    sec_workers: int = 1

    def prices(self, tickers, start_date, end_date):
        """Download daily price rows from Massive.

        Example:
            downloader.prices(["AAPL"], "2024-01-01", "2024-01-31")
            returns date, ticker, close, and volume rows.
        """
        rows = self.threaded_map(
            "prices",
            tickers,
            lambda ticker: self.price_rows(ticker, start_date, end_date),
            self.workers,
        )
        self.log_missing(rows, "prices")
        return self.concat_or_raise([frame for _, frame in rows], "no usable price rows")

    def reference(self, tickers, as_of_date):
        """Download market reference rows from Massive.

        Example:
            downloader.reference(["AAPL"], "2026-06-16")
            returns one AAPL reference row when Massive has it.
        """
        rows = self.threaded_map(
            "reference",
            tickers,
            lambda ticker: self.reference_row(ticker, as_of_date),
            self.workers,
        )
        self.log_missing(rows, "reference")
        return self.clean_reference(self.concat_or_raise([frame for _, frame in rows], "no reference rows"))

    def dividends(self, tickers, start_date, end_date):
        """Download cash dividend rows from Massive.

        Example:
            downloader.dividends(["AAPL"], "2025-01-01", "2026-01-01")
            returns ex-date cash rows.
        """
        rows = self.threaded_map(
            "dividends",
            tickers,
            lambda ticker: self.dividend_rows(ticker, start_date, end_date),
            self.workers,
        )
        self.log_missing(rows, "dividends")
        return self.concat_or_empty([frame for _, frame in rows])

    def short_interest(self, tickers, start_date, end_date):
        """Download short interest rows from Massive.

        Example:
            downloader.short_interest(["AAPL"], "2025-01-01", "2026-01-01")
            returns settlement-date short-interest rows.
        """
        rows = self.threaded_map(
            "short interest",
            tickers,
            lambda ticker: self.short_interest_rows(ticker, start_date, end_date),
            self.workers,
        )
        self.log_missing(rows, "short interest")
        return self.concat_or_empty([frame for _, frame in rows])

    def finnhub_reported(self, tickers):
        """Download reported financial rows from Finnhub.

        Example:
            downloader.finnhub_reported(["AAPL"]) returns PIT filing metric rows.
        """
        rows = self.threaded_map(
            "Finnhub reported financials",
            tickers,
            self.finnhub_rows,
            1,
            progress_every=10,
        )
        self.log_missing(rows, "Finnhub reported financials")
        return self.concat_or_empty([frame for _, frame in rows])

    def analyst_ratings(self, tickers):
        """Download TipRanks analyst-rating rows.

        Example:
            downloader.analyst_ratings(["AAPL"]) returns dated Buy/Hold/Sell events.
        """
        rows = self.threaded_map(
            "TipRanks analyst ratings",
            tickers,
            self.analyst_rating_rows,
            1,
            progress_every=10,
        )
        self.log_missing(rows, "TipRanks analyst ratings")
        return self.concat_or_empty([frame for _, frame in rows])

    def analyst_estimates(self, tickers):
        """Download FMP annual analyst-estimate rows.

        Example:
            downloader.analyst_estimates(["AAPL"]) returns future annual estimate rows.
        """
        rows = self.threaded_map(
            "FMP analyst estimates",
            tickers,
            self.analyst_estimate_rows,
            1,
            progress_every=10,
        )
        self.log_missing(rows, "FMP analyst estimates")
        return self.concat_or_empty([frame for _, frame in rows])

    def sec_history(self, tickers, dates):
        """Download daily point-in-time SEC rows from SEC-API.

        Example:
            downloader.sec_history(["AAPL"], ["2026-06-16"])
            returns the filing metrics known on that date.
        """
        dates = sorted({pd.to_datetime(day).date() for day in dates})
        rows = self.threaded_map(
            "SEC daily",
            tickers,
            lambda ticker: self.sec_rows(ticker, dates),
            self.sec_workers,
            progress_every=10,
        )
        self.log_missing(rows, "SEC daily")
        frame = self.concat_or_empty([frame for _, frame in rows])
        if frame.empty:
            raise ValueError("no SEC daily rows")
        return frame

    def new_sec_filing_tickers(self, tickers, start_date, end_date):
        """Return requested tickers with new 10-K/10-Q filings.

        Example:
            if AAPL filed today, new_sec_filing_tickers(["AAPL"], today, today) returns ["AAPL"].
        """
        if pd.to_datetime(start_date).date() > pd.to_datetime(end_date).date():
            return []

        filings = SecApiClient().filings_between(start_date, end_date)
        if filings.empty:
            return []
        requested = {
            str(ticker).upper(): SEC_TICKER_ALIAS.get(str(ticker).upper(), str(ticker).upper())
            for ticker in tickers
        }
        filed = set(filings["ticker"].astype(str).str.upper())
        return sorted(ticker for ticker, sec_ticker in requested.items() if sec_ticker in filed)

    def threaded_map(self, label, items, function, workers, progress_every=50):
        """Run provider calls with simple progress logging.

        Example:
            downloader.threaded_map("prices", ["AAPL"], lambda ticker: ticker, 1)
            returns [("AAPL", "AAPL")].
        """
        items = list(items)
        if not items:
            LOGGER.info("%s skipped count=0", label)
            return []

        LOGGER.info("%s started count=%s workers=%s", label, len(items), workers)
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(function, item): item for item in items}
            completed = as_completed(futures)
            progress = tqdm(completed, total=len(items), desc=label, unit="ticker")
            for done, future in enumerate(progress, start=1):
                item = futures[future]
                results.append((item, future.result()))
                if done == len(items) or done % progress_every == 0:
                    LOGGER.info("%s progress=%s/%s", label, done, len(items))

        LOGGER.info("%s finished", label)
        return results

    def price_rows(self, ticker, start_date, end_date):
        """Download one ticker's daily prices from Massive.

        Example:
            downloader.price_rows("AAPL", "2024-01-01", "2024-01-31")
            returns AAPL bars.
        """
        client = MassiveClient()
        try:
            return stock_bars(client, ticker, start_date, end_date)
        finally:
            client.close()

    def reference_row(self, ticker, as_of_date):
        """Download one ticker's reference row from Massive.

        Example:
            downloader.reference_row("AAPL", "2026-06-16")
            returns market_cap when available.
        """
        client = MassiveClient()
        try:
            return ticker_overview(client, ticker, as_of_date)
        finally:
            client.close()

    def dividend_rows(self, ticker, start_date, end_date):
        """Download one ticker's dividends.

        Example:
            dividend_rows("AAPL", "2025-01-01", "2026-01-01") returns cash rows.
        """
        client = MassiveClient()
        try:
            return stock_dividends(client, ticker, start_date, end_date)
        finally:
            client.close()

    def short_interest_rows(self, ticker, start_date, end_date):
        """Download one ticker's short-interest rows.

        Example:
            short_interest_rows("AAPL", "2025-01-01", "2026-01-01") returns rows.
        """
        client = MassiveClient()
        try:
            return stock_short_interest(client, ticker, start_date, end_date)
        finally:
            client.close()

    def finnhub_rows(self, ticker):
        """Download one ticker's Finnhub reported financial rows.

        Example:
            finnhub_rows("AAPL") returns accepted-date filing rows.
        """
        return reported_financials(FinnhubClient(), ticker)

    def analyst_rating_rows(self, ticker):
        """Download one ticker's TipRanks analyst ratings.

        Example:
            analyst_rating_rows("AAPL") returns recent dated ratings.
        """
        return analyst_ratings(TipRanksClient(), ticker)

    def analyst_estimate_rows(self, ticker):
        """Download one ticker's FMP analyst estimates.

        Example:
            analyst_estimate_rows("AAPL") returns annual estimate rows.
        """
        return analyst_estimates(FmpClient(), ticker)

    def sec_rows(self, ticker, dates):
        """Download one ticker's daily SEC history from SEC-API.

        Example:
            downloader.sec_rows("AAPL", ["2026-06-16"])
            returns point-in-time metrics.
        """
        return fetch_daily_rows(SecApiClient(), ticker, dates)

    def clean_reference(self, reference):
        """Return ticker and numeric market cap columns.

        Example:
            market_cap strings become numeric values.
        """
        frame = reference.copy()
        for column in ["ticker", "market_cap"]:
            if column not in frame:
                frame[column] = None
        frame = frame[frame["ticker"].notna()]
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
        return frame[["ticker", "market_cap"]].drop_duplicates("ticker", keep="last")

    def log_missing(self, rows, label):
        """Log optional ticker rows that were skipped.

        Example:
            two empty optional price frames logs prices skipped=2.
        """
        missing = self.missing_tickers(rows)
        if missing:
            LOGGER.info("%s skipped=%s", label, len(missing))

    def missing_tickers(self, rows):
        """Return tickers with no usable frame.

        Example:
            [("AAPL", empty_frame)] returns ["AAPL"].
        """
        return [ticker for ticker, frame in rows if frame is None or frame.empty]

    def concat_or_raise(self, frames, message):
        """Concatenate frames or raise a clear error.

        Example:
            downloader.concat_or_raise([], "missing") raises ValueError("missing").
        """
        frame = self.concat_or_empty(frames)
        if frame.empty:
            raise ValueError(message)
        return frame

    def concat_or_empty(self, frames):
        """Concatenate frames or return an empty DataFrame.

        Example:
            downloader.concat_or_empty([]) returns an empty DataFrame.
        """
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame()

        columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
        trimmed = [frame.dropna(axis=1, how="all") for frame in frames]
        return pd.concat(trimmed, ignore_index=True).reindex(columns=columns)

def three_year_start(today):
    """Return the start date for three years of daily prices.

    Example:
        three_year_start(date(2026, 6, 16)) returns "2023-06-16".
    """
    return years_start(today, 3)


def years_start(today, years):
    """Return the same calendar date a number of years earlier.

    Example:
        years_start(date(2026, 6, 16), 4) returns "2022-06-16".
    """
    try:
        return today.replace(year=today.year - years).isoformat()
    except ValueError:
        return today.replace(year=today.year - years, month=2, day=28).isoformat()
