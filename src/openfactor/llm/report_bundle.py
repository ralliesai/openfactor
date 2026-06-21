import json
from pathlib import Path

import pandas as pd


EXPOSURE_PANEL_DAYS = 14

FILE_DESCRIPTIONS = {
    "snapshot_pointer.json": "Universe and as-of date for the report data bundle.",
    "portfolio.csv": "Current submitted portfolio weights.",
    "report.json": "Complete rendered portfolio report JSON.",
    "report_context.md": "Compact report context shown to the PM analyst.",
    "metadata.json": "OpenFactor model metadata for this report date.",
    "exposures.csv": "Wide factor exposure table for the report date.",
    "exposures_long.csv": "Long factor exposure table for the report date.",
    "exposures_panel_last14.csv": "Last 14 exposure dates from the historical exposure panel.",
    "factor_returns.csv": "Realized factor returns.",
    "residual_returns.csv": "Realized idiosyncratic returns by ticker.",
    "factor_covariance.csv": "Current factor covariance matrix.",
    "idiosyncratic_risk.csv": "Current per-ticker idiosyncratic risk.",
    "universe.csv": "Current model universe.",
    "indexes.csv": "Public benchmark/index instrument metadata.",
    "index_prices.csv": "Public benchmark/index prices.",
    "index_returns.csv": "Public benchmark/index returns.",
}


def write_report_bundle(directory, snapshot, report, report_context, report_json):
    """Write the files Code Interpreter should see for one report session."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = [
        write_json(
            directory / "snapshot_pointer.json",
            {"universe": snapshot.universe_name, "as_of_date": snapshot.as_of_date},
        ),
        write_csv(directory / "portfolio.csv", portfolio_frame(report)),
        write_text(directory / "report_context.md", report_context),
        write_json(directory / "report.json", json.loads(report_json)),
        write_json(directory / "metadata.json", snapshot.metadata),
        write_csv(directory / "exposures.csv", wide_exposures(snapshot)),
        write_csv(directory / "exposures_long.csv", snapshot.exposures),
        write_csv(directory / "exposures_panel_last14.csv", last_exposure_panel(snapshot.exposures_panel, EXPOSURE_PANEL_DAYS)),
        write_csv(directory / "factor_returns.csv", index_frame(snapshot.factor_returns, "date")),
        write_csv(directory / "residual_returns.csv", snapshot.residual_returns),
        write_csv(directory / "factor_covariance.csv", index_frame(snapshot.factor_covariance, "factor")),
        write_csv(directory / "idiosyncratic_risk.csv", snapshot.idiosyncratic_risk),
        write_csv(directory / "universe.csv", snapshot.universe),
    ]
    files += optional_csvs(
        directory,
        [
            ("indexes.csv", snapshot.indexes),
            ("index_prices.csv", snapshot.index_prices),
            ("index_returns.csv", snapshot.index_returns),
        ],
    )
    return files


def portfolio_frame(report):
    """Return current holdings as a ticker/allocation table."""
    return pd.DataFrame(report["meta"]["holdings"]).rename(columns={"weight": "allocation"})


def wide_exposures(snapshot):
    """Return wide exposures from the same long rows that built the report."""
    return snapshot.exposures.pivot_table(index="ticker", columns="factor", values="value").reset_index()


def last_exposure_panel(panel, days):
    """Return the last N exposure dates from the historical exposure panel."""
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["as_of_date", "ticker", "factor", "value"])
    frame = panel.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date.astype(str)
    dates = sorted(frame["as_of_date"].dropna().unique())[-int(days):]
    return frame[frame["as_of_date"].isin(dates)].sort_values(["as_of_date", "ticker", "factor"]).reset_index(drop=True)


def optional_csvs(directory, items):
    """Write optional dataframes that exist in the snapshot."""
    files = []
    for name, frame in items:
        if frame is not None:
            files.append(write_csv(Path(directory) / name, frame))
    return files


def metric_glossary(files):
    """Return the file and report metric glossary for the analyst prompt."""
    lines = [
        "portfolio weights are current signed allocations for the submitted book.",
        "summary.total_risk is annualized ex-ante total portfolio volatility.",
        "summary.tracking_error is annualized ex-ante active risk versus the model risk proxy.",
        "summary.beta / predicted_beta is ex-ante model beta to the model risk proxy.",
        "summary.factor_share and summary.idiosyncratic_share_of_total_variance split total variance.",
        "idiosyncratic_te_share and summary.idiosyncratic_share_of_tracking_error are idiosyncratic variance share of tracking error.",
        "active_rows are active-risk / tracking-error driver rows.",
        "family_ret groups realized factor return attribution into Style, Sector, and Industry buckets.",
        "idiosyncratic_return is benchmark-relative idiosyncratic return after reconciling active return.",
        "idiosyncratic_return_by_name ranks latest-day idiosyncratic return by holding.",
        "idiosyncratic_risk_by_name.rows ranks idiosyncratic risk by holding.",
        "portfolio_ret, benchmark_ret, and active_ret are realized portfolio, benchmark, and excess returns.",
    ]
    lines.extend(f"{path.name}: {FILE_DESCRIPTIONS.get(path.name, 'OpenFactor report data file.')}" for path in files)
    return lines


def file_previews(files):
    """Return file headers and first rows for the analyst prompt."""
    return "\n\n".join(file_preview(path) for path in files)


def file_preview(path):
    """Return a short markdown preview for one attached file."""
    description = FILE_DESCRIPTIONS.get(path.name, "")
    if path.suffix == ".csv":
        frame = pd.read_csv(path, nrows=5)
        body = preview_frame(frame).to_csv(index=False).strip()
        return f"### {path.name}\n{description}\ncolumns: {', '.join(map(str, frame.columns))}\n```csv\n{body}\n```"
    data = json.loads(path.read_text()) if path.suffix == ".json" else path.read_text(errors="ignore")
    text = json.dumps(data, sort_keys=True) if isinstance(data, dict) else str(data)
    return f"### {path.name}\n{description}\n```text\n{text[:1200]}\n```"


def preview_frame(frame):
    """Return a display-bounded copy of a small dataframe preview."""
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].map(preview_value)
    return result


def preview_value(value):
    """Return one compact preview cell."""
    if pd.isna(value):
        return ""
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def index_frame(frame, name):
    """Return a dataframe with its index inserted as a named first column."""
    result = frame.copy()
    result.insert(0, name, result.index)
    return result.reset_index(drop=True)


def write_csv(path, frame):
    """Write a dataframe and return its path."""
    frame.to_csv(path, index=False)
    return path


def write_json(path, data):
    """Write JSON and return its path."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def write_text(path, text):
    """Write text and return its path."""
    path.write_text(str(text).strip() + "\n")
    return path
