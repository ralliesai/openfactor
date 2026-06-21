from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252
FIELDS = ["date", "holdings", "portfolio_return", "benchmark_return", "active_return",
          "tracking_error", "beta", "predicted_beta", "total_risk",
          "idiosyncratic_share_of_tracking_error", "factor_contrib",
          "idiosyncratic_contribution"]


def record_for(report):
    """Return one stored row: the day's risk plus its realized active return.

    Example:
        record_for(report)["active_return"] is the snapshot date's
        portfolio-minus-benchmark return; factor_contrib is that day's
        per-factor return breakdown, so a real holding path can be summed later.
    """
    summary, meta = report["summary"], report["meta"]
    today = report.get("today") or {"factor": {}, "idiosyncratic": None}
    return {
        "date": meta["as_of_date"],
        "holdings": ";".join(f"{h['ticker']}:{h['weight']:.4f}" for h in meta["holdings"]),
        "portfolio_return": report["portfolio_ret"][0],
        "benchmark_return": report["benchmark_ret"][0],
        "active_return": report["active_ret"][0],
        "tracking_error": summary["tracking_error"],
        "beta": summary["beta"],
        "predicted_beta": summary["predicted_beta"],
        "total_risk": summary["total_risk"],
        "idiosyncratic_share_of_tracking_error": summary["idiosyncratic_share_of_tracking_error"],
        "factor_contrib": ";".join(f"{f}:{v:.6f}" for f, v in today["factor"].items()),
        "idiosyncratic_contribution": today["idiosyncratic"],
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
        if "predicted_beta" not in existing and "beta" in existing:
            existing["predicted_beta"] = existing["beta"]
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
    stats = {
        "days": days,
        "ir": None,
        "mean": None,
        "hit_rate": None,
        "cumulative": None,
        "realized_beta": realized_beta(frame),
    }
    if days:
        stats["cumulative"] = float(active.sum())
        stats["mean"] = float(active.mean())
        stats["hit_rate"] = float((active > 0).mean())
    if days >= 2 and active.std(ddof=1) > 0:
        stats["ir"] = float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return stats


def realized_beta(frame):
    """Return realized beta from stored portfolio and benchmark returns."""
    if "portfolio_return" not in frame or "benchmark_return" not in frame:
        return None
    returns = frame[["portfolio_return", "benchmark_return"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(returns) < 2:
        return None
    variance = returns["benchmark_return"].var(ddof=1)
    if not np.isfinite(variance) or variance <= 0:
        return None
    covariance = returns["portfolio_return"].cov(returns["benchmark_return"])
    return float(covariance / variance)


def parse_contrib(text):
    """Return a {factor: value} map from a serialized contribution string."""
    out = {}
    if not isinstance(text, str):
        return out
    for part in text.split(";"):
        factor, sep, value = part.rpartition(":")
        if sep:
            try:
                out[factor] = float(value)
            except ValueError:
                continue
    return out


def realized_attribution(frame):
    """Return multi-period attribution summed over the real daily holdings.

    Example:
        each stored day contributes its own factor breakdown, so the total is
        the honest "what drove the book" — not today's weights run backward.
    """
    if "factor_contrib" not in frame.columns:
        return None
    rows = frame[frame["factor_contrib"].apply(lambda t: bool(parse_contrib(t)))]
    if rows.empty:
        return None
    factor = {}
    for text in rows["factor_contrib"]:
        for name, value in parse_contrib(text).items():
            factor[name] = factor.get(name, 0.0) + value
    idiosyncratic = float(pd.to_numeric(rows["idiosyncratic_contribution"], errors="coerce").fillna(0.0).sum())
    factor.pop("market", None)
    benchmark = realized_sum(rows, "benchmark_return")
    portfolio = realized_sum(rows, "portfolio_return")
    active = realized_sum(rows, "active_return")
    if active is None:
        active = sum(factor.values()) + idiosyncratic
    if benchmark is None:
        benchmark = portfolio - active if portfolio is not None and active is not None else None
    if portfolio is None and benchmark is not None and active is not None:
        portfolio = benchmark + active
    dates = sorted(rows["date"].astype(str))
    return {
        "days": int(len(rows)),
        "date_range": dates[-1] if len(dates) == 1 else f"{dates[0]} → {dates[-1]}",
        "factor": factor,
        "idiosyncratic": idiosyncratic,
        "benchmark": benchmark,
        "active": active,
        "portfolio": portfolio,
    }


def realized_sum(rows, column):
    """Return a numeric column sum from realized track rows, or None."""
    if column not in rows:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(values.sum()) if len(values) else None
