from pathlib import Path
from io import StringIO
import os
import re
import time
import urllib.error

import pandas as pd

from openfactor.io.r2 import R2Client
from openfactor.io.snapshot import PUBLIC_BASE_URL, read_url


DEFAULT_SEMANTIC_CACHE = "r2://openfactor-public/semantic_factors.csv"


def read_semantic_cache(path=DEFAULT_SEMANTIC_CACHE):
    """Read the wide semantic membership cache.

    Example:
        r2://openfactor-public/semantic_factors.csv reads the public cache.
    """
    if not path:
        return empty_cache()

    text = read_cache_text(path)
    if text is None:
        return empty_cache()

    frame = pd.read_csv(StringIO(text))
    if "ticker" not in frame:
        raise ValueError("semantic cache must have a ticker column")
    frame = frame.drop_duplicates("ticker", keep="last").copy()
    frame["ticker"] = frame["ticker"].astype(str)
    for column in semantic_columns(frame):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("ticker").reset_index(drop=True)


def write_semantic_cache(cache, path=DEFAULT_SEMANTIC_CACHE):
    """Write the semantic membership cache.

    Example:
        write_semantic_cache(cache) uploads semantic_factors.csv to public R2.
    """
    if not path:
        return

    text = cache.to_csv(index=False)
    if is_r2_path(path):
        bucket, key = r2_parts(path)
        R2Client.from_env().upload_text(text, bucket, key, "text/csv; charset=utf-8")
        return

    if str(path).startswith(("http://", "https://")):
        raise ValueError("semantic cache HTTP URLs are read-only; use r2://bucket/key to write")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def try_write_semantic_cache(cache, path=DEFAULT_SEMANTIC_CACHE):
    """Write the semantic cache when the target is writable in this environment.

    Example:
        default public R2 cache writes on maintainer machines, but local
        discovery still succeeds without R2 credentials.
    """
    if not path:
        return False

    if is_r2_path(path) and not r2_write_env_present():
        return False

    if str(path).startswith(("http://", "https://")):
        return False

    write_semantic_cache(cache, path)
    return True


def read_cache_text(path):
    """Return semantic cache CSV text.

    Example:
        public R2 cache missing returns None instead of failing discovery.
    """
    if is_r2_path(path):
        bucket, key = r2_parts(path)
        if bucket == "openfactor-public":
            return read_public_cache(key)
        return R2Client.from_env().read_text(bucket, key)

    if str(path).startswith(("http://", "https://")):
        return read_url(path).decode("utf-8")

    path = Path(path)
    return path.read_text() if path.exists() else None


def read_public_cache(key):
    """Read the public cache through the public Cloudflare URL.

    Example:
        semantic_factors.csv is read without R2 credentials.
    """
    try:
        return read_url(f"{PUBLIC_BASE_URL}/{key}?v={time.time_ns()}").decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def is_r2_path(path):
    """Return True for r2://bucket/key paths.

    Example:
        r2://openfactor-public/semantic_factors.csv returns True.
    """
    return str(path).startswith("r2://")


def r2_parts(path):
    """Return bucket and key from an r2 path.

    Example:
        r2://openfactor-public/a.csv returns ("openfactor-public", "a.csv").
    """
    value = str(path).removeprefix("r2://")
    bucket, _, key = value.partition("/")
    if not bucket or not key:
        raise ValueError("semantic cache R2 path must look like r2://bucket/key")
    return bucket, key


def r2_write_env_present():
    """Return True when all R2 write credentials are available."""
    return all(
        os.getenv(name)
        for name in [
            "OPENFACTOR_R2_ACCOUNT_ID",
            "OPENFACTOR_R2_ACCESS_KEY_ID",
            "OPENFACTOR_R2_SECRET_ACCESS_KEY",
        ]
    )


def semantic_factor_context(cache, limit=100):
    """Return existing semantic factor ids for the discovery prompt.

    Example:
        ai_infra with 40 known members becomes one prompt row.
    """
    rows = []
    for column in semantic_columns(cache)[:limit]:
        values = pd.to_numeric(cache[column], errors="coerce")
        rows.append(
            {
                "id": column,
                "known_labels": int(values.notna().sum()),
                "members": int(values.fillna(0).sum()),
            }
        )
    return rows


def cached_memberships(cache, factor_id, tickers):
    """Return cached rows for one factor and ticker list.

    Example:
        cached 0 and 1 values both count as known labels; blanks are missing.
    """
    columns = ["factor_id", "ticker", "member", "reason"]
    if cache.empty or factor_id not in cache:
        return pd.DataFrame(columns=columns)

    values = cache.set_index("ticker")[factor_id].reindex([str(ticker) for ticker in tickers])
    values = values[values.notna()]
    if values.empty:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        {
            "factor_id": factor_id,
            "ticker": values.index.astype(str),
            "member": values.astype(int).clip(0, 1).to_numpy(),
            "reason": "semantic_factors.csv",
        }
    )


def semantic_factor_members(factor, cache=DEFAULT_SEMANTIC_CACHE):
    """Return tickers that belong to one semantic factor.

    Example:
        semantic_factor_members("Retail Speculation") returns ["GME", "HOOD", "RDDT"].
    """
    frame = read_semantic_cache(cache)
    column = semantic_factor_column(frame, factor)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return sorted(frame.loc[values > 0, "ticker"].astype(str).tolist())


def semantic_factor_column(cache, factor):
    """Return the cache column for a display name or factor id.

    Example:
        "Retail Speculation" and "retail_speculation" both map to retail_speculation.
    """
    wanted = semantic_factor_id(factor)
    columns = semantic_columns(cache)
    for column in columns:
        if semantic_factor_id(column) == wanted:
            return column
    raise ValueError(f"unknown semantic factor: {factor}. available: {columns}")


def semantic_factor_id(value):
    """Return a semantic factor id from a readable name.

    Example:
        "Retail Speculation" becomes retail_speculation.
    """
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")


def update_semantic_cache(cache, memberships):
    """Merge binary memberships into the wide semantic cache.

    Example:
        a later 1000-name run keeps the first 25 cached rows and fills the rest.
    """
    if memberships.empty:
        return cache

    matrix = cache_frame(cache).set_index("ticker")
    for row in memberships[["factor_id", "ticker", "member"]].itertuples(index=False):
        matrix.loc[str(row.ticker), str(row.factor_id)] = int(row.member)

    matrix = matrix.sort_index().reset_index().rename(columns={"index": "ticker"})
    columns = ["ticker"] + sorted(semantic_columns(matrix))
    return matrix[columns]


def cache_frame(cache):
    """Return a valid cache frame.

    Example:
        an empty cache becomes a DataFrame with only ticker.
    """
    if cache is None or cache.empty:
        return empty_cache()
    frame = cache.copy()
    if "ticker" not in frame:
        frame.insert(0, "ticker", [])
    frame["ticker"] = frame["ticker"].astype(str)
    return frame


def semantic_columns(cache):
    """Return semantic factor columns.

    Example:
        ticker,ai_infra returns ["ai_infra"].
    """
    if cache is None or cache.empty:
        return []
    return [column for column in cache.columns if column != "ticker"]


def empty_cache():
    """Return an empty semantic cache table.

    Example:
        empty_cache().columns is ["ticker"].
    """
    return pd.DataFrame(columns=["ticker"])
