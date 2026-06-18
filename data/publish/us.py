import argparse
from dataclasses import dataclass
from io import BytesIO
import logging
from pathlib import Path
import sys
import urllib.error

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.build.calendar import default_as_of_date
from data.build.downloads import ProviderDownloader
from data.build.fundamentals import TTM_TO_INTERNAL
from data.build.snapshot import DatasetBuilder, publish_dataset
from data.build.quality import validate_snapshot
from data.publish.r2 import R2Client


DEFAULT_LIMIT = 1000
PUBLIC_FILES = [
    "exposures.csv",
    "details/exposures_long.csv",
    "factor_returns.csv",
    "factor_covariance.csv",
    "specific_risk.csv",
    "universe.csv",
    "metadata.json",
]
PRIVATE_FILES = [
    ("prices", "prices.csv"),
    ("reference", "reference.csv"),
    ("fundamentals_pit", "fundamentals.csv"),
    ("fundamentals_pit", "audit.csv"),
]


@dataclass(frozen=True)
class UsConfig:
    """US dataset publish settings.

    Example:
        UsConfig(as_of_date="2026-06-16") publishes openfactor-us1000.
    """

    as_of_date: str
    limit: int = DEFAULT_LIMIT
    workers: int = 8
    sec_workers: int = 5
    public_bucket: str = "openfactor-public"
    private_bucket: str = "openfactor-private"
    universe_name: str = ""


class UsPublisher:
    """Build and publish one US factor dataset.

    Example:
        UsPublisher(UsConfig(default_as_of_date())).run()
        uploads public factors and private inputs to R2.
    """

    def __init__(self, config, downloader=None, r2=None):
        self.config = config
        self.downloader = downloader or ProviderDownloader(config.workers, config.sec_workers)
        self.r2 = r2

    def run(self):
        """Publish the dataset unless all R2 files already exist.

        Example:
            run() skips a complete openfactor-us1000 snapshot.
        """
        if self.already_published():
            print(f"skip as_of_date={self.config.as_of_date} reason=already_published")
            return None

        LOGGER.info(
            "publish run started as_of_date=%s universe=%s limit=%s",
            self.config.as_of_date,
            self.universe_name(),
            self.config.limit,
        )
        result = DatasetBuilder(
            as_of_date=self.config.as_of_date,
            universe_limit=self.config.limit,
            universe_name=self.universe_name(),
            workers=self.config.workers,
            sec_workers=self.config.sec_workers,
            downloader=self.downloader,
            previous_fundamentals=self.previous_fundamentals(),
        ).build()
        LOGGER.info("validating snapshot")
        validate_snapshot(result.snapshot)
        LOGGER.info("validation finished")
        publish_dataset(
            result,
            self.config.public_bucket,
            self.config.private_bucket,
            self.r2_client(),
        )
        print("published", result.snapshot.as_of_date, "tickers", len(result.snapshot.universe))
        return result

    def already_published(self):
        """Return True when dated and latest public/private files exist.

        Example:
            a complete openfactor-us1000 folder skips provider downloads.
        """
        universe = self.universe_name()
        public = set(self.r2_client().list_keys(self.config.public_bucket, public_prefix(universe)))
        private = set(self.r2_client().list_keys(self.config.private_bucket, private_prefix(universe)))
        return set(public_keys(self.config.as_of_date, universe)) <= public and set(
            private_keys(self.config.as_of_date, universe)
        ) <= private

    def universe_name(self):
        """Return the published universe name.

        Example:
            limit=1000 returns openfactor-us1000.
        """
        return self.config.universe_name or f"openfactor-us{self.config.limit}"

    def previous_fundamentals(self):
        """Return private latest PIT fundamentals when present.

        Example:
            yesterday's latest rows reduce SEC work for today's build.
        """
        try:
            frame = self.read_latest_fundamentals()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise

        frame = frame.rename(columns=TTM_TO_INTERNAL)
        print("fundamentals cache hit rows", len(frame))
        return frame

    def read_latest_fundamentals(self):
        """Read private latest fundamentals plus filing audit columns.

        Example:
            latest fundamentals.csv and audit.csv merge on ticker/as_of_date.
        """
        root = f"{private_prefix(self.universe_name())}/fundamentals_pit/latest"
        model = self.read_private_csv(f"{root}/fundamentals.csv")
        audit = self.read_private_csv(f"{root}/audit.csv")
        return model.merge(audit, on=["ticker", "as_of_date"], how="left")

    def read_private_csv(self, key):
        """Read one private R2 CSV object.

        Example:
            read_private_csv("inputs/openfactor-us1000/prices/latest/prices.csv")
            returns a DataFrame.
        """
        body = self.r2_client().request(
            "GET",
            self.config.private_bucket,
            key,
            return_body=True,
        )
        return pd.read_csv(BytesIO(body))

    def r2_client(self):
        """Return the configured R2 client.

        Example:
            r2_client().list_keys("openfactor-public", "factors") lists files.
        """
        if self.r2 is None:
            self.r2 = R2Client.from_env()
        return self.r2


def public_prefix(universe):
    """Return the public object prefix.

    Example:
        public_prefix("openfactor-us1000") returns factors/openfactor-us1000.
    """
    return f"factors/{universe}"


def private_prefix(universe):
    """Return the private object prefix.

    Example:
        private_prefix("openfactor-us1000") returns inputs/openfactor-us1000.
    """
    return f"inputs/{universe}"


def public_keys(as_of_date, universe=None):
    """Return required public R2 keys for one date.

    Example:
        public_keys("2026-06-16") includes latest/exposures.csv.
    """
    prefix = public_prefix(universe or f"openfactor-us{DEFAULT_LIMIT}")
    keys = [f"{prefix}/date={as_of_date}/{name}" for name in PUBLIC_FILES]
    keys += [f"{prefix}/latest/{name}" for name in PUBLIC_FILES]
    keys.append(f"{prefix}/latest.json")
    return keys


def private_keys(as_of_date, universe=None):
    """Return required private R2 keys for one date.

    Example:
        private_keys("2026-06-16") includes prices/latest/prices.csv.
    """
    prefix = private_prefix(universe or f"openfactor-us{DEFAULT_LIMIT}")
    keys = []
    for folder, filename in PRIVATE_FILES:
        keys.append(f"{prefix}/{folder}/date={as_of_date}/{filename}")
        keys.append(f"{prefix}/{folder}/latest/{filename}")
    return keys


def parse_args():
    """Parse US publish options.

    Example:
        no arguments publishes openfactor-us1000.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-date", default=default_as_of_date())
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sec-workers", type=int, default=5)
    parser.add_argument("--public-bucket", default="openfactor-public")
    parser.add_argument("--private-bucket", default="openfactor-private")
    parser.add_argument("--universe-name", default="")
    return parser.parse_args()


def main():
    """Build and publish the US dataset.

    Example:
        python3.10 -m data.publish.us
        publishes openfactor-us1000 unless it already exists.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    UsPublisher(
        UsConfig(
            as_of_date=args.as_of_date,
            limit=args.limit,
            workers=args.workers,
            sec_workers=args.sec_workers,
            public_bucket=args.public_bucket,
            private_bucket=args.private_bucket,
            universe_name=args.universe_name,
        )
    ).run()


if __name__ == "__main__":
    main()
