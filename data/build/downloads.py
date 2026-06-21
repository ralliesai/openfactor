from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

import pandas as pd
from tqdm import tqdm

from data.providers.finnhub import FinnhubClient, reported_financials
from data.providers.fmp import FmpClient, analyst_estimates
from data.providers.massive import MassiveClient
from data.providers.massive.stocks import (
    stock_bars_with_unadjusted_close,
    stock_dividends,
    stock_short_interest,
    ticker_overview,
)
from data.providers.sec_api import SecApiClient
from data.providers.tipranks import TipRanksClient, analyst_ratings
from data.sec.history import SEC_TICKER_ALIAS, fetch_daily_rows


LOGGER = logging.getLogger("openfactor.factory")
REQUIRED_COVERAGE = 0.95


@dataclass(frozen=True)
class ProviderDownloader:
    """Download OpenFactor provider inputs.

    Example:
        downloader = ProviderDownloader(workers=8, sec_workers=5)
        downloader.prices(["AAPL"], "2024-01-01", "2024-01-31")
        returns daily Massive price rows.
    """

    workers: int = 8
    sec_workers: int = 5

    def prices(self, tickers, start_date, end_date):
        """Download daily price rows from Massive.

        Example:
            downloader.prices(["AAPL"], "2024-01-01", "2024-01-31")
            returns adjusted close plus raw close for market caps.
        """
        rows = self.threaded_map(
            "prices",
            tickers,
            lambda ticker: self.price_rows(ticker, start_date, end_date),
            self.workers,
        )
        rows = self.retry_missing(
            "prices retry",
            rows,
            lambda ticker: self.price_rows(ticker, start_date, end_date),
        )
        return self.required_table(rows, "prices", 1.0, "no usable price rows")

    def index_prices(self, tickers, start_date, end_date):
        """Download public index proxy prices from Massive.

        Example:
            downloader.index_prices(["SPY"], "2024-01-01", "2024-01-31")
            returns adjusted ETF proxy bars without adding SPY to the stock universe.
        """
        rows = self.threaded_map(
            "index prices",
            tickers,
            lambda ticker: self.price_rows(ticker, start_date, end_date),
            self.workers,
        )
        rows = self.retry_missing(
            "index prices retry",
            rows,
            lambda ticker: self.price_rows(ticker, start_date, end_date),
        )
        return self.required_table(rows, "index prices", 1.0, "no usable index price rows")

    def reference(self, tickers, as_of_date, min_coverage=REQUIRED_COVERAGE):
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
        rows = self.retry_missing(
            "reference retry",
            rows,
            lambda ticker: self.reference_row(ticker, as_of_date),
        )
        return self.clean_reference(self.required_table(rows, "reference", min_coverage, "no reference rows"))

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
        self.log_report(rows, "dividends")
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
        self.log_report(rows, "short interest")
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
        )
        self.log_report(rows, "Finnhub reported financials")
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
        )
        self.log_report(rows, "TipRanks analyst ratings")
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
        )
        self.log_report(rows, "FMP analyst estimates")
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
        )
        self.log_report(rows, "SEC daily")
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

    def threaded_map(self, label, items, function, workers):
        """Run provider calls with one clean progress bar.

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
            progress = tqdm(completed, total=len(items), desc=label, unit="ticker", dynamic_ncols=True)
            errors = 0
            for future in progress:
                item = futures[future]
                try:
                    frame = future.result()
                except Exception as error:
                    frame = error_frame(error)
                if frame is not None and "download_error" in frame.attrs:
                    errors += 1
                results.append((item, frame))
                if errors:
                    progress.set_postfix(errors=errors)

        LOGGER.info("%s finished", label)
        return results

    def retry_missing(self, label, rows, function):
        """Retry empty required provider rows once before judging coverage.

        Example:
            if JPM timed out in the first price batch, retry_missing asks for JPM again.
        """
        missing = self.missing_tickers(rows)
        if not missing:
            return rows

        retry = dict(self.threaded_map(label, missing, function, self.workers))
        return [(ticker, retry.get(ticker, frame)) for ticker, frame in rows]

    def price_rows(self, ticker, start_date, end_date):
        """Download one ticker's daily prices from Massive.

        Example:
            downloader.price_rows("AAPL", "2024-01-01", "2024-01-31")
            returns AAPL bars.
        """
        client = MassiveClient()
        try:
            return stock_bars_with_unadjusted_close(client, ticker, start_date, end_date)
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

    def required_table(self, rows, label, min_coverage, message):
        """Return a required batch or raise after all tickers finish.

        Example:
            95 usable price rows out of 100 passes at 95% coverage.
        """
        self.log_report(rows, label)
        frame = self.concat_or_empty([frame for _, frame in rows])
        if frame.empty:
            raise ValueError(message)

        usable = len(rows) - len(self.missing_tickers(rows))
        coverage = usable / len(rows) if rows else 0
        if coverage < min_coverage:
            raise ValueError(
                f"{label} coverage {usable}/{len(rows)} "
                f"below required {min_coverage:.0%}"
            )
        return frame

    def log_report(self, rows, label):
        """Log one completed provider batch.

        Example:
            one timeout in 100 rows logs errors=1 and a small sample.
        """
        missing = self.missing_tickers(rows)
        errors = self.download_errors(rows)
        LOGGER.info(
            "%s summary total=%s usable=%s missing=%s errors=%s",
            label,
            len(rows),
            len(rows) - len(missing),
            len(missing),
            len(errors),
        )
        if missing:
            LOGGER.info("%s missing_sample=%s", label, ", ".join(missing[:10]))
        if errors:
            sample = "; ".join(f"{ticker}: {error}" for ticker, error in errors[:5])
            LOGGER.warning("%s error_sample=%s", label, sample)

    def missing_tickers(self, rows):
        """Return tickers with no usable frame.

        Example:
            [("AAPL", empty_frame)] returns ["AAPL"].
        """
        return [ticker for ticker, frame in rows if frame is None or frame.empty]

    def download_errors(self, rows):
        """Return per-ticker provider errors.

        Example:
            an empty frame with download_error becomes one error row.
        """
        return [
            (ticker, frame.attrs["download_error"])
            for ticker, frame in rows
            if frame is not None and "download_error" in frame.attrs
        ]

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


def error_frame(error):
    """Return an empty frame tagged with one provider error.

    Example:
        a ReadTimeout becomes a missing ticker instead of killing the batch.
    """
    frame = pd.DataFrame()
    frame.attrs["download_error"] = clean_error(error)
    return frame


def clean_error(error):
    """Return a short provider error without query strings.

    Example:
        RuntimeError with a provider URL keeps only the URL path.
    """
    message = str(error).split("?", 1)[0]
    return f"{type(error).__name__}: {message}"


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
