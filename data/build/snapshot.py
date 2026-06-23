from dataclasses import dataclass
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.build.calendar import default_as_of_date
from data.build.downloads import ProviderDownloader, years_start
from data.build.factor_inputs import add_factor_inputs, year_ago
from data.build.fundamentals import FundamentalHistory
from data.build.input_cache import (
    cached_event_history,
    cached_price_history,
    concat_cached_rows,
    filter_ticker_rows,
    max_date,
    missing_earnings_tickers,
    ticker_set,
)
from data.build.quality import (
    validate_fundamental_share_sources,
    validate_market_cap_formula,
    validate_private_inputs,
)
from data.build.serialize import (
    fundamentals_audit_file,
    fundamentals_file,
    json_text,
    panel_gzip,
    snapshot_csvs,
    spreadsheet_csv,
)
from data.build.universe import top_market_cap_tickers, us_candidates
from data.providers.massive import MassiveClient
from data.publish.r2 import R2Client
from openfactor import default_price_factors, default_reference_factors
from openfactor.core.matrix import price_matrix
from openfactor.io.indexes import (
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_INDEX_TICKERS,
    index_metadata,
    index_return_series,
    index_returns_from_prices,
)
from openfactor.io.snapshot import Snapshot
from openfactor.model.factor_returns import factor_model_history
from openfactor.model.normalize import normalize_exposures
from openfactor.model.risk import factor_covariance
from openfactor.model.idiosyncratic_risk import idiosyncratic_risk_from_residuals


LOGGER = logging.getLogger("openfactor.build")
MODEL_VERSION = "0.2.0"
UNIVERSE_NAME = "openfactor-us1000"
RISK_WINDOW = 252
REFERENCE_COLUMNS = [
    "ticker",
    "market_cap",
    "sector",
    "industry",
    "sic",
    "sic_industry",
    "fama_industry",
]


@dataclass(frozen=True)
class BuildResult:
    """One completed dataset build.

    Example:
        result = DatasetBuilder(universe_limit=25).build()
        result.snapshot.exposures contains public factor rows.
    """

    snapshot: Snapshot
    prices: pd.DataFrame
    reference: pd.DataFrame
    fundamentals: pd.DataFrame
    index_prices: pd.DataFrame = None
    dividends: pd.DataFrame = None
    short_interest: pd.DataFrame = None
    finnhub: pd.DataFrame = None


class DatasetBuilder:
    """Build OpenFactor public outputs and private input tables.

    Example:
        DatasetBuilder(universe_limit=25).build()
        returns snapshot, prices, reference, and point-in-time fundamentals.
    """

    def __init__(
        self,
        as_of_date=None,
        universe_limit=1000,
        workers=8,
        sec_workers=5,
        tickers=None,
        universe_name=UNIVERSE_NAME,
        downloader=None,
        previous_fundamentals=None,
        previous_prices=None,
        previous_index_prices=None,
        previous_dividends=None,
        previous_short_interest=None,
        previous_finnhub=None,
    ):
        self.as_of_date = as_of_date or default_as_of_date()
        self.universe_limit = universe_limit
        self.workers = workers
        self.sec_workers = sec_workers
        self.tickers = tickers
        self.universe_name = universe_name
        self.downloader = downloader or ProviderDownloader(workers, sec_workers)
        self.previous_fundamentals = previous_fundamentals
        self.previous_prices = previous_prices
        self.previous_index_prices = previous_index_prices
        self.previous_dividends = previous_dividends
        self.previous_short_interest = previous_short_interest
        self.previous_finnhub = previous_finnhub

    def build(self):
        """Build one complete dataset.

        Example:
            DatasetBuilder(tickers=["AAPL"], universe_name="sample").build()
            returns a BuildResult for that one-ticker universe.
        """
        LOGGER.info(
            "build started as_of_date=%s universe=%s limit=%s",
            self.as_of_date,
            self.universe_name,
            self.universe_limit,
        )
        start_date = price_start_date(self.as_of_date)
        tickers = self.tickers or self.model_universe()
        prices = self.cached_prices(tickers, start_date, self.as_of_date)
        index_prices = self.cached_index_prices(DEFAULT_INDEX_TICKERS, start_date, self.as_of_date)
        matrix = price_matrix(prices, require_volume=True)
        LOGGER.info(
            "prices ready rows=%s tickers=%s dates=%s",
            len(prices),
            len(matrix.tickers),
            len(matrix.dates),
        )
        model_dates = self.reference_dates(matrix)
        sec_dates = self.sec_dates(model_dates)
        reference = self.downloader.reference(matrix.tickers, self.as_of_date)
        LOGGER.info("reference ready rows=%s", len(reference))
        fundamentals = FundamentalHistory(
            self.downloader,
            self.previous_fundamentals,
        ).rows(matrix.tickers, sec_dates)
        validate_fundamental_share_sources(fundamentals)
        fundamentals = self.with_daily_market_caps(fundamentals, prices)
        dividends = self.cached_dividends(matrix.tickers, start_date, self.as_of_date)
        short_interest = self.cached_short_interest(matrix.tickers, start_date, self.as_of_date)
        finnhub = self.cached_finnhub(matrix.tickers, fundamentals, sec_dates)
        analyst_ratings = self.cached_analyst_ratings(matrix.tickers, start_date, self.as_of_date)
        analyst_estimates = self.cached_analyst_estimates(matrix.tickers)
        fundamentals = add_factor_inputs(
            fundamentals,
            matrix,
            dividends,
            short_interest,
            finnhub,
            analyst_ratings,
            analyst_estimates,
            sec_dates,
        )
        LOGGER.info(
            "factor inputs ready rows=%s columns=%s",
            len(fundamentals),
            len(fundamentals.columns),
        )
        current_reference = self.merge_reference(
            reference,
            self.reference_as_of(fundamentals, matrix.dates[-1]),
        )
        return self.result_from_ready_inputs(
            matrix,
            prices,
            current_reference,
            fundamentals,
            index_prices=index_prices,
            dividends=dividends,
            short_interest=short_interest,
            finnhub=finnhub,
        )

    def result_from_ready_inputs(
        self,
        matrix,
        prices,
        current_reference,
        fundamentals,
        metadata=None,
        index_prices=None,
        dividends=None,
        short_interest=None,
        finnhub=None,
    ):
        """Return a complete BuildResult from model-ready inputs.

        Example:
            result_from_ready_inputs(matrix, prices, reference, fundamentals)
            computes exposures, factor returns, covariance, and idiosyncratic risk.
        """
        validate_fundamental_share_sources(fundamentals)
        validate_market_cap_formula(fundamentals, prices)
        LOGGER.info("computing current exposures")
        exposures = normalize_exposures(
            self.compute_exposures(matrix, current_reference),
            self.market_cap_weights(current_reference),
        )
        index_returns = index_returns_from_prices(index_prices) if index_prices is not None else None
        benchmark_market = (
            None if index_returns is None or index_returns.empty
            else index_return_series(index_returns, DEFAULT_BENCHMARK_TICKER)
        )
        LOGGER.info(
            "current exposures rows=%s factors=%s",
            len(exposures),
            exposures["factor"].nunique(),
        )
        LOGGER.info("estimating factor returns window=%s", RISK_WINDOW)
        factor_returns, residuals, panel = factor_model_history(
            matrix,
            exposures,
            window=RISK_WINDOW,
            reference_history=fundamentals,
            market_returns=benchmark_market,
            progress_label="factor returns",
            collect_panel=True,
            panel_days=1,
        )
        exposures_panel = exposure_panel(panel, exposures)
        LOGGER.info(
            "factor returns ready rows=%s factors=%s residual_rows=%s",
            len(factor_returns),
            len(factor_returns.columns),
            len(residuals),
        )
        LOGGER.info("building risk snapshot")
        snapshot = Snapshot(
            as_of_date=str(matrix.dates[-1]),
            universe_name=self.universe_name,
            exposures=exposures,
            factor_returns=factor_returns,
            residual_returns=residuals,
            factor_covariance=factor_covariance(factor_returns),
            idiosyncratic_risk=idiosyncratic_risk_from_residuals(residuals),
            universe=self.universe_frame(matrix.tickers, matrix.dates[-1]),
            metadata=self.metadata(matrix, prices, factor_returns, metadata, index_prices),
            exposures_panel=exposures_panel,
            indexes=index_metadata(DEFAULT_INDEX_TICKERS),
            index_prices=index_prices,
            index_returns=index_returns,
        )
        LOGGER.info(
            "snapshot ready as_of_date=%s tickers=%s factors=%s",
            snapshot.as_of_date,
            len(snapshot.universe),
            len(snapshot.factor_returns.columns),
        )
        return BuildResult(
            snapshot,
            prices,
            self.reference_file(current_reference),
            fundamentals,
            index_prices,
            dividends,
            short_interest,
            finnhub,
        )

    def model_universe(self):
        """Return top US common stocks by current market cap.

        Example:
            universe_limit=1000 returns the OpenFactor US 1000 universe.
        """
        client = MassiveClient()
        try:
            candidates = us_candidates(client, self.as_of_date)
        finally:
            client.close()

        reference = self.downloader.reference(candidates, self.as_of_date, min_coverage=0.80)
        tickers = top_market_cap_tickers(reference, self.universe_limit)
        if len(tickers) < self.universe_limit:
            raise ValueError(
                f"universe selected {len(tickers)} "
                f"below requested {self.universe_limit}"
            )
        LOGGER.info(
            "universe=%s candidates=%s selected=%s",
            self.universe_name,
            len(candidates),
            len(tickers),
        )
        return tickers

    def cached_prices(self, tickers, start_date, end_date):
        """Return cached price history extended through the build date.

        Example:
            Friday cache plus Tuesday build downloads only Monday-Tuesday bars.
        """
        return cached_price_history(
            self.previous_prices,
            tickers,
            start_date,
            end_date,
            self.downloader.prices,
            "prices",
        )

    def cached_index_prices(self, tickers, start_date, end_date):
        """Return cached index price history extended through the build date."""
        return cached_price_history(
            self.previous_index_prices,
            tickers,
            start_date,
            end_date,
            self.downloader.index_prices,
            "index prices",
        )

    def cached_dividends(self, tickers, start_date, end_date):
        """Return dividend history extended from the prior checked date."""
        return cached_event_history(
            self.previous_dividends,
            self.previous_price_tickers(),
            self.previous_cache_date(),
            tickers,
            start_date,
            end_date,
            "ex_dividend_date",
            self.downloader.dividends,
            ["ticker", "ex_dividend_date"],
            "dividends",
        )

    def cached_short_interest(self, tickers, start_date, end_date):
        """Return short-interest history extended from the prior checked date."""
        return cached_event_history(
            self.previous_short_interest,
            self.previous_price_tickers(),
            self.previous_cache_date(),
            tickers,
            start_date,
            end_date,
            "settlement_date",
            self.downloader.short_interest,
            ["ticker", "settlement_date"],
            "short interest",
        )

    def cached_finnhub(self, tickers, fundamentals, dates):
        """Return Finnhub rows only for tickers missing earnings inputs.

        Example:
            carried AAPL earnings inputs skip Finnhub; new MSFT filing refreshes MSFT.
        """
        previous = filter_ticker_rows(self.previous_finnhub, tickers)
        refresh = missing_earnings_tickers(fundamentals, tickers, dates, self.previous_fundamentals, previous)
        if not refresh:
            LOGGER.info("Finnhub reported financials cache used tickers=%s refresh=0", len(tickers))
            return previous
        LOGGER.info("Finnhub reported financials cache miss tickers=%s", len(refresh))
        fresh = self.downloader.finnhub_reported(refresh)
        return concat_cached_rows(previous, fresh, ["ticker", "accession_no"])

    def cached_analyst_ratings(self, tickers, start_date, end_date):
        """Return analyst-rating events.

        Example:
            current TipRanks endpoint has no global since cursor, so this remains a full pull.
        """
        LOGGER.info("TipRanks analyst ratings full refresh tickers=%s reason=no_incremental_cursor", len(tickers))
        return self.downloader.analyst_ratings(tickers)

    def cached_analyst_estimates(self, tickers):
        """Return analyst-estimate rows.

        Example:
            current FMP endpoint has no update cursor, so this remains a full pull.
        """
        LOGGER.info("FMP analyst estimates full refresh tickers=%s reason=no_incremental_cursor", len(tickers))
        return self.downloader.analyst_estimates(tickers)

    def previous_cache_date(self):
        """Return the last private price date used as provider-cache watermark."""
        return max_date(self.previous_prices, "date")

    def previous_price_tickers(self):
        """Return tickers covered by the previous private price cache."""
        return ticker_set(self.previous_prices)

    def reference_dates(self, matrix):
        """Return model dates that need point-in-time fundamentals.

        Example:
            a 252-day risk model needs 252 return dates plus the report date.
        """
        start = max(0, len(matrix.returns) - RISK_WINDOW)
        dates = list(matrix.dates[start : len(matrix.returns)])
        dates.append(matrix.dates[-1])
        return dates

    def sec_dates(self, model_dates):
        """Return SEC dates needed for model rows and YoY growth.

        Example:
            2026-06-16 also requests 2025-06-16 for growth.
        """
        dates = {str(date) for date in model_dates}
        dates.update(year_ago(date) for date in model_dates)
        return sorted(dates)

    def with_daily_market_caps(self, fundamentals, prices):
        """Attach daily PIT market caps to SEC rows.

        Example:
            AAPL market_cap equals shares_outstanding times same-day raw close.
        """
        if fundamentals.empty:
            return fundamentals

        frame = fundamentals.copy()
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date.astype(str)
        frame = frame.merge(self.market_cap_price_frame(prices), on=["as_of_date", "ticker"], how="left")
        frame["market_cap"] = frame["shares_outstanding"] * frame["market_cap_close"]
        return frame.drop(columns=["market_cap_close"])

    def market_cap_price_frame(self, prices):
        """Return raw closes keyed by date and ticker for market-cap math.

        Example:
            market_cap_price_frame(prices) returns as_of_date, ticker, and market_cap_close.
        """
        required = {"date", "ticker", "close", "unadjusted_close"}
        missing = sorted(required - set(prices.columns))
        if missing:
            raise ValueError(f"prices missing columns for market-cap math: {missing}")

        frame = prices[["date", "ticker", "close", "unadjusted_close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
        frame["ticker"] = frame["ticker"].astype(str)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["unadjusted_close"] = pd.to_numeric(frame["unadjusted_close"], errors="coerce")
        missing_raw = frame["close"].notna() & frame["unadjusted_close"].isna()
        if missing_raw.any():
            sample = frame.loc[missing_raw, ["ticker", "date"]].head(10).to_dict("records")
            raise ValueError(f"prices has adjusted close rows missing unadjusted_close: {sample}")

        return frame.rename(columns={"date": "as_of_date", "unadjusted_close": "market_cap_close"})[
            ["as_of_date", "ticker", "market_cap_close"]
        ]

    def reference_as_of(self, fundamentals, as_of_date):
        """Return reference rows for one date.

        Example:
            reference_as_of(history, "2026-06-16") returns rows for that model date.
        """
        if fundamentals.empty:
            return fundamentals
        return fundamentals[fundamentals["as_of_date"].astype(str) == str(as_of_date)].copy()

    def merge_reference(self, reference, sec_reference):
        """Merge market reference rows with SEC factor inputs.

        Example:
            market_cap plus total_assets becomes one reference row per ticker.
        """
        if sec_reference.empty:
            return reference
        shared = [column for column in reference.columns if column in sec_reference.columns and column != "ticker"]
        frame = reference.merge(sec_reference, on="ticker", how="left", suffixes=("", "_sec"))
        for column in shared:
            sec_column = f"{column}_sec"
            frame[column] = frame[sec_column].combine_first(frame[column])
        return frame.drop(columns=[f"{column}_sec" for column in shared], errors="ignore")

    def market_cap_weights(self, reference):
        """Return market-cap weights indexed by ticker.

        Example:
            AAPL market_cap becomes the normalization weight for AAPL exposures.
        """
        return reference.drop_duplicates("ticker").set_index("ticker")["market_cap"]

    def reference_file(self, reference):
        """Return the current reference columns published for audits.

        Example:
            reference_file(rows) keeps ticker, market cap, sector, and industry.
        """
        frame = reference.copy()
        for column in REFERENCE_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        return frame[REFERENCE_COLUMNS].drop_duplicates("ticker", keep="last")

    def compute_exposures(self, matrix, reference):
        """Compute every default factor exposure.

        Example:
            compute_exposures(matrix, reference) returns ticker/factor/value rows.
        """
        frames = [factor.compute(matrix) for factor in default_price_factors()]
        frames += [
            factor.compute(reference, as_of_date=matrix.dates[-1])
            for factor in default_reference_factors()
        ]
        return pd.concat(frames, ignore_index=True)

    def universe_frame(self, tickers, as_of_date):
        """Return snapshot universe rows.

        Example:
            ["AAPL"] becomes one row with ticker AAPL and the snapshot date.
        """
        return pd.DataFrame({"as_of_date": str(as_of_date), "ticker": tickers})

    def metadata(self, matrix, prices, factor_returns, extra=None, index_prices=None):
        """Return snapshot metadata used by loaders and audits.

        Example:
            metadata records the model date, universe size, and factor count.
        """
        data = {
            "as_of_date": str(matrix.dates[-1]),
            "universe": self.universe_name,
            "model_version": MODEL_VERSION,
            "tickers": int(len(matrix.tickers)),
            "price_rows": int(len(prices)),
            "factor_count": int(len(factor_returns.columns)),
            "risk_window": RISK_WINDOW,
            "market_cap_source": "sec_shares_outstanding_x_daily_unadjusted_close",
        }
        if index_prices is not None:
            data.update(
                {
                    "benchmark_return_ticker": DEFAULT_BENCHMARK_TICKER,
                    "benchmark_return_source": "index_returns.csv",
                    "index_tickers": list(DEFAULT_INDEX_TICKERS),
                    "index_price_rows": int(len(index_prices)),
                }
            )
        if extra:
            data.update(extra)
        return data


def exposure_panel(panel, current):
    """Return per-date exposures, the rolling history plus today's snapshot.

    Example:
        the 252-day rolling panel and the current-date exposures, one block per date.
    """
    frames = [frame for frame in (panel, current) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["as_of_date", "ticker", "factor"], keep="last")


def price_start_date(as_of_date):
    """Return the price start date with a small lookback buffer.

    Example:
        price_start_date("2026-06-16") returns a date before 2022-06-16.
    """
    day = pd.to_datetime(years_start(pd.to_datetime(as_of_date).date(), 4))
    return (day - pd.Timedelta(days=10)).date().isoformat()


def publish_dataset(result, public_bucket, private_bucket, r2=None):
    """Upload one complete dataset to R2.

    Example:
        publish_dataset(result, "openfactor-public", "openfactor-private")
        uploads public factors and private input CSVs.
    """
    validate_private_inputs(result)
    r2 = r2 or R2Client.from_env()
    snapshot = result.snapshot
    dated = f"factors/{snapshot.universe_name}/date={snapshot.as_of_date}"
    latest = f"factors/{snapshot.universe_name}/latest"
    LOGGER.info("publish public started bucket=%s", public_bucket)
    upload_snapshot_files(r2, public_bucket, dated, snapshot)
    upload_snapshot_files(r2, public_bucket, latest, snapshot)
    r2.upload_text(
        json_text(
            {
                "latest": snapshot.as_of_date,
                "universe": snapshot.universe_name,
                "model_version": snapshot.metadata["model_version"],
            }
        ),
        public_bucket,
        f"factors/{snapshot.universe_name}/latest.json",
        "application/json; charset=utf-8",
    )
    LOGGER.info("publish public finished")
    LOGGER.info("publish private started bucket=%s", private_bucket)
    for folder, frame, filename in private_tables(result):
        LOGGER.info("publish private table=%s file=%s rows=%s", folder, filename, len(frame))
        text = spreadsheet_csv(frame)
        r2.upload_text(
            text,
            private_bucket,
            f"inputs/{snapshot.universe_name}/{folder}/date={snapshot.as_of_date}/{filename}",
            "text/csv; charset=utf-8",
        )
        r2.upload_text(
            text,
            private_bucket,
            f"inputs/{snapshot.universe_name}/{folder}/latest/{filename}",
            "text/csv; charset=utf-8",
        )
    LOGGER.info("publish private finished")


def upload_snapshot_files(r2, bucket, prefix, snapshot):
    """Upload public snapshot files under one R2 prefix.

    Example:
        upload_snapshot_files(r2, "openfactor-public", "factors/x/latest", snapshot)
        uploads CSVs and metadata.json under that prefix.
    """
    for filename, text in snapshot_csvs(snapshot):
        r2.upload_text(text, bucket, f"{prefix}/{filename}", "text/csv; charset=utf-8")
    panel = panel_gzip(snapshot)
    if panel:
        filename, data = panel
        r2.upload_bytes(data, bucket, f"{prefix}/{filename}", "application/gzip")
    r2.upload_text(
        json_text(snapshot.metadata),
        bucket,
        f"{prefix}/metadata.json",
        "application/json; charset=utf-8",
    )


def private_tables(result):
    """Return private input tables with their object names.

    Example:
        private_tables(result) includes fundamentals_pit/fundamentals.csv.
    """
    return [
        ("prices", sort_rows(result.prices, ["ticker", "date"]), "prices.csv"),
        ("index_prices", sort_rows(result.index_prices, ["ticker", "date"]), "index_prices.csv"),
        ("reference", sort_rows(result.reference, ["ticker"]), "reference.csv"),
        ("dividends", sort_rows(result.dividends, ["ticker", "ex_dividend_date"]), "dividends.csv"),
        (
            "short_interest",
            sort_rows(result.short_interest, ["ticker", "settlement_date"]),
            "short_interest.csv",
        ),
        ("finnhub_reported", sort_rows(result.finnhub, ["ticker", "accepted_date"]), "reported.csv"),
        ("fundamentals_pit", fundamentals_file(result.fundamentals), "fundamentals.csv"),
        ("fundamentals_pit", fundamentals_audit_file(result.fundamentals), "audit.csv"),
    ]


def sort_rows(frame, columns):
    """Return rows sorted by available columns.

    Example:
        sort_rows(prices, ["ticker", "date"]) groups each ticker's prices together.
    """
    if frame is None:
        return pd.DataFrame()
    columns = [column for column in columns if column in frame]
    if not columns:
        return frame
    return frame.sort_values(columns).reset_index(drop=True)
