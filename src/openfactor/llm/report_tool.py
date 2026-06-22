import json

import pandas as pd

try:  # pydantic needs typing_extensions.TypedDict for tool schemas on Python < 3.12
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict

from openfactor.tui.report import tui_report


class Holding(TypedDict):
    """One target position for a what-if report.

    Example:
        {"ticker": "NVDA", "allocation": 0.35}
    """

    ticker: str
    allocation: float


def report_json(snapshot, holdings, sections=None):
    """Build the OpenFactor report for a set of holdings as a JSON-safe dict.

    Shares the same tui_report engine the terminal and report chat use, so a
    hypothetical book's numbers tie out to the displayed report. sections is an
    optional list of top-level keys to keep; None returns the full report.

    Example:
        report_json(snapshot, [{"ticker": "AAPL", "allocation": 1.0}])["summary"]
        returns the same total_risk/tracking_error/beta the report shows.
    """
    report = json_safe(tui_report(portfolio_frame(holdings), snapshot))
    return select_sections(report, sections)


def select_sections(report, sections):
    """Return only the requested top-level report sections, or all of them.

    Example:
        select_sections(report, ["summary", "active_rows"]) drops name-level rows.
        Unknown keys are ignored; an empty match falls back to the full report.
    """
    if not sections:
        return report
    chosen = {key: report[key] for key in sections if key in report}
    return chosen or report


def portfolio_frame(holdings):
    """Return a ticker/allocation DataFrame from tool holdings.

    Example:
        [{"ticker": "AAPL", "allocation": 0.5}] becomes one AAPL row at 0.5.
    """
    frame = pd.DataFrame(list(holdings))
    if frame.empty or "ticker" not in frame or "allocation" not in frame:
        raise ValueError("holdings must be rows of {ticker, allocation}")
    frame = frame[["ticker", "allocation"]].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["allocation"] = pd.to_numeric(frame["allocation"], errors="coerce")
    return frame


def portfolio_report_tool(snapshot):
    """Return an Agents SDK function tool that reruns the report for new weights.

    Example:
        agent has tools=[portfolio_report_tool(snapshot)] and can answer
        "what if I cut META to 10%" against the same model.
    """
    from agents import function_tool

    @function_tool
    def portfolio_report(holdings: list[Holding], sections: list[str] | None = None) -> str:
        """Re-run the OpenFactor risk report for a hypothetical portfolio as JSON.

        Use this for any what-if, hedge, or rebalance question so the numbers
        come from the same engine as the displayed report and tie out. Pass the
        FULL target book (every ticker and its allocation), not just the change.
        allocation is a signed decimal weight (0.30 = 30% long, -0.10 = 10%
        short); a long-only book sums to 1.0.

        The full JSON includes: summary (total_risk, tracking_error, beta,
        factor_share, idiosyncratic_share_of_total_variance,
        idiosyncratic_share_of_tracking_error); active_rows (per-factor active
        exposure, te_share, ret); risk_rows (absolute risk decomposition);
        family_ret (Market/Style/Sector/Industry return); idiosyncratic_risk_by_name
        (name-level idiosyncratic vol and variance share); idiosyncratic_return_by_name
        (latest-day name-level return); and meta.missing (tickers absent from the
        model universe).

        Args:
            holdings: target portfolio rows, each a ticker and signed allocation.
            sections: optional top-level keys to return (e.g. "summary",
                "active_rows", "risk_rows", "idiosyncratic_risk_by_name",
                "idiosyncratic_return_by_name", "family_ret", "meta"). Omit for the
                full report; pass a subset to stay concise on iterative what-ifs.
        """
        return json.dumps(report_json(snapshot, holdings, sections), default=str)

    return portfolio_report


def json_safe(value):
    """Return JSON-safe Python values for report payloads.

    Example:
        numpy floats, pandas Timestamps, and NaN become plain JSON values.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value
