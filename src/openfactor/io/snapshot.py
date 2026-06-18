from dataclasses import dataclass
from io import BytesIO
import json
import time
import urllib.error
import urllib.request

import pandas as pd


PUBLIC_BASE_URL = "https://openfactor-data.rallies.ai"
SNAPSHOT_FILES = {
    "exposures": "exposures.csv",
    "exposures_detail": "details/exposures_long.csv",
    "factor_returns": "factor_returns.csv",
    "factor_covariance": "factor_covariance.csv",
    "specific_risk": "specific_risk.csv",
    "universe": "universe.csv",
}


@dataclass(frozen=True)
class Snapshot:
    """One OpenFactor model snapshot.

    Example:
        snapshot = load_snapshot("openfactor-us1000")
        snapshot.exposures contains ticker/factor/value rows.
    """

    as_of_date: str
    universe_name: str
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_risk: pd.DataFrame
    universe: pd.DataFrame
    metadata: dict


def load_snapshot(universe, as_of_date="latest"):
    """Load a public OpenFactor snapshot.

    Example:
        load_snapshot("openfactor-us1000")
        reads factors/openfactor-us1000 from the public OpenFactor bucket.
    """
    universe = require_value(universe, "universe")
    cache_bust = None
    if as_of_date == "latest":
        cache_bust = f"v={time.time_ns()}"
        meta = read_json(with_query(f"{PUBLIC_BASE_URL}/factors/{universe}/latest.json", cache_bust))
        as_of_date = meta["latest"]
        universe = meta.get("universe", universe)
        prefix = f"{PUBLIC_BASE_URL}/factors/{universe}/latest"
    else:
        prefix = f"{PUBLIC_BASE_URL}/factors/{universe}/date={as_of_date}"

    return load_snapshot_url(prefix, as_of_date, universe, cache_bust)


def load_snapshot_url(prefix, as_of_date, universe, cache_bust=None):
    """Load a snapshot from public bucket URLs.

    Example:
        load_snapshot_url("https://bucket/factors/x/latest", "2026-06-16", "x")
        returns a Snapshot.
    """
    metadata = read_json(with_query(f"{prefix}/metadata.json", cache_bust))
    return Snapshot(
        as_of_date=str(as_of_date),
        universe_name=str(universe),
        exposures=read_csv(with_query(f"{prefix}/{SNAPSHOT_FILES['exposures_detail']}", cache_bust)),
        factor_returns=read_csv(
            with_query(f"{prefix}/{SNAPSHOT_FILES['factor_returns']}", cache_bust),
            index_col=0,
        ),
        factor_covariance=read_csv(
            with_query(f"{prefix}/{SNAPSHOT_FILES['factor_covariance']}", cache_bust),
            index_col=0,
        ),
        specific_risk=read_csv(with_query(f"{prefix}/{SNAPSHOT_FILES['specific_risk']}", cache_bust)),
        universe=read_csv(with_query(f"{prefix}/{SNAPSHOT_FILES['universe']}", cache_bust)),
        metadata=metadata,
    )


def with_query(path, query):
    """Add a query string to an HTTP path when needed.

    Example:
        with_query("https://x/a.csv", "v=1") returns "https://x/a.csv?v=1".
    """
    if not query:
        return path
    separator = "&" if "?" in str(path) else "?"
    return f"{path}{separator}{query}"


def read_csv(path, **kwargs):
    """Read one snapshot CSV with a helpful missing-data error.

    Example:
        read_csv("https://example.com/missing.csv")
        raises FileNotFoundError with that URL.
    """
    try:
        return pd.read_csv(BytesIO(read_url(path)), **kwargs)
    except urllib.error.HTTPError as error:
        raise FileNotFoundError(f"OpenFactor snapshot file is unavailable: {path}") from error


def read_json(path):
    """Read one URL JSON file.

    Example:
        read_json("https://.../metadata.json") returns a Python dict.
    """
    try:
        return json.loads(read_url(path).decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise FileNotFoundError(f"OpenFactor metadata is unavailable: {path}") from error


def read_url(path):
    """Read a public OpenFactor URL with a normal client header.

    Example:
        read_url("https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest.json")
        returns bytes.
    """
    request = urllib.request.Request(str(path), headers={"User-Agent": "OpenFactor/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def require_value(value, name):
    """Return a required string value.

    Example:
        require_value("openfactor-us1000", "universe") returns that value.
    """
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} is required")
    return str(value)
