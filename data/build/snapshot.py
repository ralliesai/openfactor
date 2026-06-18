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
from data.build.serialize import (
    fundamentals_audit_file,
    fundamentals_file,
    json_text,
    snapshot_csvs,
    spreadsheet_csv,
)
from data.build.universe import top_market_cap_tickers, us_candidates
from data.providers.massive import MassiveClient
from data.publish.r2 import R2Client
from openfactor import default_price_factors, default_reference_factors
from openfactor.core.matrix import price_matrix
from openfactor.io.snapshot import Snapshot
from openfactor.model.factor_returns import factor_model_history
from openfactor.model.normalize import normalize_exposures
from openfactor.model.risk import factor_covariance
from openfactor.model.specific_risk import specific_risk_from_residuals


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
    ):
        self.as_of_date = as_of_date or default_as_of_date()
        self.universe_limit = universe_limit
        self.workers = workers
        self.sec_workers = sec_workers
        self.tickers = tickers
        self.universe_name = universe_name
        self.downloader = downloader or ProviderDownloader(workers, sec_workers)
        self.previous_fundamentals = previous_fundamentals

    def build(self):
        """Build one complete dataset.

        Example:
            DatasetBuilder(tickers=["AAPL"], universe_name="sample").build()
            returns a BuildResult for that one-ticker universe.
        """
        start_date = price_start_date(self.as_of_date)
        tickers = self.tickers or self.model_universe()
        prices = self.downloader.prices(tickers, start_date, self.as_of_date)
        matrix = price_matrix(prices, require_volume=True)
        model_dates = self.reference_dates(matrix)
        sec_dates = self.sec_dates(model_dates)
        reference = self.downloader.reference(matrix.tickers, self.as_of_date)
        fundamentals = FundamentalHistory(
            self.downloader,
            self.previous_fundamentals,
        ).rows(matrix.tickers, sec_dates)
        fundamentals = self.with_daily_market_caps(fundamentals, matrix)
        fundamentals = add_factor_inputs(
            fundamentals,
            matrix,
            self.downloader.dividends(matrix.tickers, start_date, self.as_of_date),
            self.downloader.short_interest(matrix.tickers, start_date, self.as_of_date),
            self.downloader.finnhub_reported(matrix.tickers),
            self.downloader.analyst_ratings(matrix.tickers),
            self.downloader.analyst_estimates(matrix.tickers),
            sec_dates,
        )
        current_reference = self.merge_reference(
            reference,
            self.reference_as_of(fundamentals, matrix.dates[-1]),
        )
        return self.result_from_ready_inputs(matrix, prices, current_reference, fundamentals)

    def result_from_ready_inputs(self, matrix, prices, current_reference, fundamentals, metadata=None):
        """Return a complete BuildResult from model-ready inputs.

        Example:
            result_from_ready_inputs(matrix, prices, reference, fundamentals)
            computes exposures, factor returns, covariance, and specific risk.
        """
        exposures = normalize_exposures(
            self.compute_exposures(matrix, current_reference),
            self.market_cap_weights(current_reference),
        )
        factor_returns, residuals = factor_model_history(
            matrix,
            exposures,
            window=RISK_WINDOW,
            reference_history=fundamentals,
        )
        snapshot = Snapshot(
            as_of_date=str(matrix.dates[-1]),
            universe_name=self.universe_name,
            exposures=exposures,
            factor_returns=factor_returns,
            factor_covariance=factor_covariance(factor_returns),
            specific_risk=specific_risk_from_residuals(residuals),
            universe=self.universe_frame(matrix.tickers, matrix.dates[-1]),
            metadata=self.metadata(matrix, prices, factor_returns, metadata),
        )
        return BuildResult(snapshot, prices, self.reference_file(current_reference), fundamentals)

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

    def with_daily_market_caps(self, fundamentals, matrix):
        """Attach daily PIT market caps to SEC rows.

        Example:
            AAPL market_cap equals shares_outstanding times same-day close.
        """
        if fundamentals.empty:
            return fundamentals

        frame = fundamentals.copy()
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date.astype(str)
        frame = frame.merge(self.close_frame(matrix), on=["as_of_date", "ticker"], how="left")
        frame["market_cap"] = frame["shares_outstanding"] * frame["close"]
        return frame.drop(columns=["close"])

    def close_frame(self, matrix):
        """Return close prices as rows keyed by date and ticker.

        Example:
            close_frame(matrix) returns as_of_date, ticker, and close columns.
        """
        return pd.DataFrame(
            {
                "as_of_date": np.repeat(matrix.dates, len(matrix.tickers)),
                "ticker": np.tile(matrix.tickers, len(matrix.dates)),
                "close": matrix.close.reshape(-1),
            }
        )

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
        reference = reference.drop(columns=["market_cap"], errors="ignore")
        return reference.merge(sec_reference, on="ticker", how="left")

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

    def metadata(self, matrix, prices, factor_returns, extra=None):
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
            "market_cap_source": "sec_shares_outstanding_x_daily_adjusted_close",
        }
        if extra:
            data.update(extra)
        return data


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
    r2 = r2 or R2Client.from_env()
    snapshot = result.snapshot
    dated = f"factors/{snapshot.universe_name}/date={snapshot.as_of_date}"
    latest = f"factors/{snapshot.universe_name}/latest"
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
    for folder, frame, filename in private_tables(result):
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


def upload_snapshot_files(r2, bucket, prefix, snapshot):
    """Upload public snapshot files under one R2 prefix.

    Example:
        upload_snapshot_files(r2, "openfactor-public", "factors/x/latest", snapshot)
        uploads CSVs and metadata.json under that prefix.
    """
    for filename, text in snapshot_csvs(snapshot):
        r2.upload_text(text, bucket, f"{prefix}/{filename}", "text/csv; charset=utf-8")
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
        ("reference", sort_rows(result.reference, ["ticker"]), "reference.csv"),
        ("fundamentals_pit", fundamentals_file(result.fundamentals), "fundamentals.csv"),
        ("fundamentals_pit", fundamentals_audit_file(result.fundamentals), "audit.csv"),
    ]


def sort_rows(frame, columns):
    """Return rows sorted by available columns.

    Example:
        sort_rows(prices, ["ticker", "date"]) groups each ticker's prices together.
    """
    columns = [column for column in columns if column in frame]
    if not columns:
        return frame
    return frame.sort_values(columns).reset_index(drop=True)
