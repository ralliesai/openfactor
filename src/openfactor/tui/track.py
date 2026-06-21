from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252
FIELDS = ["date", "holdings", "portfolio_return", "benchmark_return", "active_return",
          "tracking_error", "beta", "total_risk", "idio_share_te"]


def record_for(report):
    """Return one stored row: the day's risk plus its realized active return.

    Example:
        record_for(report)["active_return"] is the snapshot date's
        portfolio-minus-benchmark return.
    """
    summary, meta = report["summary"], report["meta"]
    return {
        "date": meta["as_of_date"],
        "holdings": ";".join(f"{h['ticker']}:{h['weight']:.4f}" for h in meta["holdings"]),
        "portfolio_return": report["portfolio_ret"][0],
        "benchmark_return": report["benchmark_ret"][0],
        "active_return": report["active_ret"][0],
        "tracking_error": summary["tracking_error"],
        "beta": summary["beta"],
        "total_risk": summary["total_risk"],
        "idio_share_te": summary["specific_share_te"],
    }


def update_track(path, record):
    """Upsert one day's record into the track file, keyed by date.

    Example:
        re-running a date overwrites its row; a new date appends and re-sorts.
    """
    path = Path(path)
    new = pd.DataFrame([record])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["date"].astype(str) != str(record["date"])]
        frame = new if existing.empty else pd.concat([existing, new], ignore_index=True)
    else:
        frame = new
    frame = frame.reindex(columns=FIELDS).sort_values("date").reset_index(drop=True)
    frame.to_csv(path, index=False)
    return frame


def realized_stats(frame):
    """Return realized track-record statistics from the accumulated daily rows.

    Example:
        the stored daily active returns give a realized information ratio,
        hit rate, and cumulative active return.
    """
    active = pd.to_numeric(frame.get("active_return"), errors="coerce").dropna()
    days = int(len(active))
    stats = {"days": days, "ir": None, "mean": None, "hit_rate": None, "cumulative": None}
    if days:
        stats["cumulative"] = float(active.sum())
        stats["mean"] = float(active.mean())
        stats["hit_rate"] = float((active > 0).mean())
    if days >= 2 and active.std(ddof=1) > 0:
        stats["ir"] = float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return stats
