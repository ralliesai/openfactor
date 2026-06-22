from pathlib import Path
import json

import numpy as np
import pandas as pd


TRADING_DAYS = 252
TRACK_VERSION = 2
SUMMARY_FIELDS = [
    "date", "schema_version", "universe", "benchmark", "benchmark_kind", "benchmark_ticker",
    "holdings", "portfolio_return", "benchmark_return", "active_return",
    "tracking_error", "beta", "predicted_beta", "total_risk",
    "idiosyncratic_share_of_tracking_error", "idiosyncratic_contribution",
]
TABLES = ["holdings", "factor_contrib", "idiosyncratic_returns", "idiosyncratic_risk", "active_risk", "risk_rows"]
REALIZED_WINDOWS = [
    {"key": "1w", "label": "1W", "days": 7},
    {"key": "1m", "label": "1M", "days": 22},
    {"key": "1q", "label": "1Q", "days": 63},
    {"key": "1y", "label": "1Y", "days": TRADING_DAYS},
]


def update_track(path, report):
    """Store one detailed local report day and return all accumulated track tables."""
    root = track_root(path)
    date = str(report["meta"]["as_of_date"])
    day = root / "days" / date
    day.mkdir(parents=True, exist_ok=True)
    write_json(day / "report.json", json_safe(report))
    write_json(day / "summary.json", record_for(report))
    for name, frame in daily_tables(report).items():
        write_csv(day / f"{name}.csv", frame)
    return rebuild_track(root)


def track_root(path):
    """Return a local track directory, failing loudly for old thin CSV paths."""
    root = Path(path)
    if root.exists() and root.is_file():
        raise ValueError("--track now expects a local directory, not a CSV file")
    root.mkdir(parents=True, exist_ok=True)
    (root / "days").mkdir(exist_ok=True)
    return root


def rebuild_track(root):
    """Rebuild aggregate CSVs from all stored day folders."""
    root = Path(root)
    summaries = []
    tables = {name: [] for name in TABLES}
    for day in sorted((root / "days").glob("*")):
        if not day.is_dir():
            continue
        summary = read_day_summary(day)
        if summary:
            summaries.append(summary)
        for name in TABLES:
            path = day / f"{name}.csv"
            if path.exists():
                tables[name].append(pd.read_csv(path))
    summary_frame = pd.DataFrame(summaries).reindex(columns=SUMMARY_FIELDS)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values("date").reset_index(drop=True)
    write_csv(root / "track.csv", summary_frame)
    result = {"summary": summary_frame}
    for name, frames in tables.items():
        frame = concat_frames(frames)
        write_csv(root / f"{name}.csv", frame)
        result[name] = frame
    return result


def read_day_summary(day):
    """Read one day summary JSON."""
    path = day / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def concat_frames(frames):
    """Return one dataframe from a list, preserving empty output."""
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    sort_columns = [col for col in ["date", "ticker", "factor"] if col in frame]
    return frame.sort_values(sort_columns) if sort_columns else frame


def record_for(report):
    """Return one summary row for daily track statistics."""
    summary, meta = report["summary"], report["meta"]
    benchmark = meta["benchmark"]
    today = report.get("today") or {"idiosyncratic": None}
    return {
        "date": meta["as_of_date"],
        "schema_version": TRACK_VERSION,
        "universe": meta["universe"],
        "benchmark": benchmark.get("name"),
        "benchmark_kind": benchmark.get("kind"),
        "benchmark_ticker": benchmark.get("ticker"),
        "holdings": ";".join(f"{h['ticker']}:{h['weight']:.6f}" for h in meta["holdings"]),
        "portfolio_return": first(report.get("portfolio_ret")),
        "benchmark_return": first(report.get("benchmark_ret")),
        "active_return": first(report.get("active_ret")),
        "tracking_error": summary["tracking_error"],
        "beta": summary["beta"],
        "predicted_beta": summary["predicted_beta"],
        "total_risk": summary["total_risk"],
        "idiosyncratic_share_of_tracking_error": summary["idiosyncratic_share_of_tracking_error"],
        "idiosyncratic_contribution": today["idiosyncratic"],
    }


def daily_tables(report):
    """Return detailed per-day tables for future local analysis."""
    date = str(report["meta"]["as_of_date"])
    return {
        "holdings": holdings_frame(date, report),
        "factor_contrib": factor_contrib_frame(date, report),
        "idiosyncratic_returns": dated_frame(date, report.get("idiosyncratic_return_by_name") or []),
        "idiosyncratic_risk": dated_frame(date, report.get("idiosyncratic_risk_by_name", {}).get("rows") or []),
        "active_risk": active_risk_frame(date, report),
        "risk_rows": dated_frame(date, report.get("risk_rows") or []),
    }


def holdings_frame(date, report):
    """Return holdings for one stored day."""
    rows = [{"date": date, "ticker": h["ticker"], "weight": h["weight"]} for h in report["meta"]["holdings"]]
    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])


def factor_contrib_frame(date, report):
    """Return one-day factor return contributions with labels and families."""
    rows = []
    for row in report.get("active_rows", []):
        ret = first(row.get("ret"))
        if ret is None or str(row.get("factor")) == "market":
            continue
        rows.append({
            "date": date,
            "factor": row.get("factor"),
            "label": row.get("label"),
            "family": row.get("family"),
            "contribution": ret,
            "te_share": row.get("te_share"),
        })
    return pd.DataFrame(rows, columns=["date", "factor", "label", "family", "contribution", "te_share"])


def active_risk_frame(date, report):
    """Return active-risk rows for one stored day."""
    rows = []
    for row in report.get("active_rows", []):
        out = {"date": date, **{k: v for k, v in row.items() if k != "ret"}}
        out["return_contribution"] = first(row.get("ret"))
        rows.append(out)
    return pd.DataFrame(rows)


def dated_frame(date, rows):
    """Return report dictionaries as a dated dataframe."""
    frame = pd.DataFrame([json_safe(row) for row in rows])
    if frame.empty:
        return pd.DataFrame({"date": []})
    frame.insert(0, "date", date)
    return frame


def realized_stats(track):
    """Return realized track-record statistics from accumulated daily rows."""
    frame = summary_frame(track)
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


def realized_attribution(track, days=None):
    """Return return attribution summed over stored daily holdings."""
    summary = window_summary(summary_frame(track), days)
    if summary.empty:
        return None
    dates = set(summary["date"].astype(str))
    factors = filter_dates(track.get("factor_contrib"), dates)
    idio_names = filter_dates(track.get("idiosyncratic_returns"), dates)
    holdings = filter_dates(track.get("holdings"), dates)
    factor = grouped_sum(factors, "factor", "contribution")
    idiosyncratic = float(pd.to_numeric(summary["idiosyncratic_contribution"], errors="coerce").fillna(0.0).sum())
    active = realized_sum(summary, "active_return")
    if active is None:
        active = sum(factor.values()) + idiosyncratic
    benchmark = realized_sum(summary, "benchmark_return")
    portfolio = realized_sum(summary, "portfolio_return")
    if benchmark is None and portfolio is not None and active is not None:
        benchmark = portfolio - active
    if portfolio is None and benchmark is not None and active is not None:
        portfolio = benchmark + active
    date_list = sorted(summary["date"].astype(str))
    return {
        "days": int(len(summary)),
        "date_range": date_list[-1] if len(date_list) == 1 else f"{date_list[0]} -> {date_list[-1]}",
        "factor": factor,
        "idiosyncratic": idiosyncratic,
        "idiosyncratic_by_name": idiosyncratic_by_name(idio_names, holdings, date_list),
        "benchmark": benchmark,
        "active": active,
        "portfolio": portfolio,
    }


def realized_windows(track):
    """Return available realized attribution windows from stored daily records."""
    frame = summary_frame(track)
    stored_days = len(frame)
    windows = {}
    for spec in REALIZED_WINDOWS:
        if stored_days >= spec["days"]:
            window = labeled_attribution(track, spec["key"], spec["label"], spec["days"])
            if window:
                windows[spec["key"]] = window
    if stored_days > TRADING_DAYS:
        window = labeled_attribution(track, "all", "All", None)
        if window:
            windows["all"] = window
    return windows


def labeled_attribution(track, key, label, days):
    """Return one labeled realized attribution window."""
    result = realized_attribution(track, days=days)
    if not result:
        return None
    result["key"] = key
    result["label"] = label
    result["window_days"] = days
    return result


def summary_frame(track):
    """Return the summary dataframe from a track result or dataframe."""
    return track.get("summary", pd.DataFrame()) if isinstance(track, dict) else track


def window_summary(frame, days):
    """Return sorted summary rows, optionally limited to the trailing N days."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result["date"] = result["date"].astype(str)
    result = result.sort_values("date").reset_index(drop=True)
    return result.tail(int(days)) if days else result


def filter_dates(frame, dates):
    """Return rows whose date is in the selected window."""
    if frame is None or frame.empty or "date" not in frame:
        return pd.DataFrame()
    return frame[frame["date"].astype(str).isin(dates)].copy()


def grouped_sum(frame, key, value):
    """Return numeric sums keyed by one column."""
    if frame.empty or key not in frame or value not in frame:
        return {}
    values = frame.copy()
    values[value] = pd.to_numeric(values[value], errors="coerce").fillna(0.0)
    return {str(k): float(v) for k, v in values.groupby(key)[value].sum().items()}


def idiosyncratic_by_name(frame, holdings=None, dates=None):
    """Return trailing idiosyncratic return contribution by ticker."""
    if frame.empty or "ticker" not in frame or "contribution" not in frame:
        return []
    rows = []
    values = frame.copy()
    values["contribution"] = pd.to_numeric(values["contribution"], errors="coerce").fillna(0.0)
    if "raw_contribution" in values:
        values["raw_contribution"] = pd.to_numeric(values["raw_contribution"], errors="coerce").fillna(0.0)
    if "weight" in values:
        values["weight"] = pd.to_numeric(values["weight"], errors="coerce")
    total = float(values["contribution"].sum())
    grouped = values.groupby("ticker", as_index=False).agg(
        contribution=("contribution", "sum"),
        raw_contribution=("raw_contribution", "sum") if "raw_contribution" in values else ("contribution", "sum"),
        weight=("weight", "mean") if "weight" in values else ("contribution", "size"),
    )
    weight_lookup = average_weights(holdings, dates)
    grouped = grouped.reindex(grouped["contribution"].abs().sort_values(ascending=False).index)
    for row in grouped.to_dict("records"):
        contribution = float(row["contribution"])
        weight = clean_float(weight_lookup.get(row["ticker"], row["weight"]))
        rows.append({
            "ticker": str(row["ticker"]),
            "weight": weight,
            "raw_contribution": float(row["raw_contribution"]),
            "contribution": contribution,
            "share": None if abs(total) < 1e-12 else contribution / total,
        })
    return rows


def average_weights(holdings, dates):
    """Return average portfolio weight by ticker across the selected stored days."""
    if holdings is None or holdings.empty or not dates or "ticker" not in holdings or "weight" not in holdings:
        return {}
    values = holdings.copy()
    values["weight"] = pd.to_numeric(values["weight"], errors="coerce").fillna(0.0)
    totals = values.groupby("ticker")["weight"].sum()
    return {str(ticker): float(weight) / len(dates) for ticker, weight in totals.items()}


def clean_float(value):
    """Return a finite float, or None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def realized_sum(rows, column):
    """Return a numeric column sum from realized track rows, or None."""
    if column not in rows:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(values.sum()) if len(values) else None


def first(values):
    """Return the first value from a list-like object."""
    if values is None or len(values) == 0:
        return None
    value = values[0]
    return None if value != value else float(value)


def json_safe(value):
    """Return JSON-safe report values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def write_csv(path, frame):
    """Write one local CSV."""
    frame.to_csv(path, index=False)


def write_json(path, data):
    """Write one local JSON file."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
