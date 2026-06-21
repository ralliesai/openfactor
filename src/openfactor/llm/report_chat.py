import asyncio
import json
import os
from threading import Thread

import pandas as pd


DEFAULT_REPORT_CHAT_MODEL = "gpt-5.5"
DEFAULT_REPORT_CHAT_TURNS = 8


class ReportChat:
    """PM-facing chat over one rendered report."""

    def __init__(self, report, model=None, api_key=None, timeout=None):
        self.report = report
        self.model = model or os.getenv("OPENFACTOR_REPORT_CHAT_MODEL", DEFAULT_REPORT_CHAT_MODEL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = float(timeout or os.getenv("OPENFACTOR_REPORT_CHAT_TIMEOUT", "90"))

    @classmethod
    def from_env(cls, report):
        """Return a chat client only when OPENAI_API_KEY is exported."""
        return cls(report) if report_chat_enabled() else None

    def answer(self, question, history=None):
        """Answer one PM question with an Agent SDK code-interpreter agent."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not exported")

        return run_sync(self.answer_async(question, history))

    async def answer_async(self, question, history=None):
        """Answer one PM question from async Textual or script code."""
        from agents import Agent, CodeInterpreterTool, ModelSettings, Runner, set_default_openai_key
        from openai.types.shared.reasoning import Reasoning

        set_default_openai_key(self.api_key)
        agent = Agent(
            name="OpenFactor PM report analyst",
            model=self.model,
            tools=[CodeInterpreterTool(code_interpreter_config())],
            model_settings=ModelSettings(
                tool_choice="auto",
                max_tokens=int(os.getenv("OPENFACTOR_REPORT_CHAT_MAX_TOKENS", "1800")),
                reasoning=Reasoning(effort="medium"),
                verbosity="medium",
            ),
            instructions=report_chat_instructions(),
        )
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                report_chat_input(self.report, question, history),
                max_turns=int(os.getenv("OPENFACTOR_REPORT_CHAT_MAX_TURNS", DEFAULT_REPORT_CHAT_TURNS)),
            ),
            timeout=self.timeout,
        )
        return str(result.final_output).strip()


def report_chat_enabled():
    """Return True when the report should show the chat sidebar."""
    return bool(os.getenv("OPENAI_API_KEY"))


def code_interpreter_config():
    """Return an automatic Code Interpreter container config."""
    return {"type": "code_interpreter", "container": {"type": "auto"}}


def report_chat_input(report, question, history=None):
    """Return one report-grounded PM question for the Agent SDK runner."""
    parts = [report_payload(report)]
    history_text = recent_history(history)
    if history_text:
        parts.append(f"Recent chat:\n{history_text}")
    parts.append(f"Current PM question:\n{str(question)}")
    return "\n\n".join(parts)


def recent_history(history):
    """Return the latest turns as bounded plain text."""
    lines = []
    for item in (history or [])[-6:]:
        role = item.get("role") if isinstance(item, dict) else None
        content = item.get("content") if isinstance(item, dict) else None
        if role in {"user", "assistant"} and content:
            lines.append(f"{role}: {str(content)[:5000]}")
    return "\n".join(lines)


def report_chat_instructions():
    """Return the product contract for the PM report chat."""
    return (
        "You are an institutional PM-facing OpenFactor analyst. Use only the "
        "provided report context and JSON. Be direct, numerical, and clear "
        "about assumptions. The report is the current portfolio report, not a backtest. "
        "Format answers as concise Markdown for a narrow terminal sidebar: short "
        "paragraphs and bullets, no wide tables unless the PM explicitly asks. "
        "Use Code Interpreter for arithmetic, hedge sizing, and reconciliation "
        "checks. The report is the source of truth; do not assume hidden "
        "bucket files or unpublished data. "
        "If the PM asks for a beta-neutral hedge and gives no instrument, assume "
        "a SPY overlay and state the notional as a signed weight on a $1 gross "
        "book when portfolio notional is not supplied. If the PM asks for a new "
        "stock-only portfolio, only construct it when the report contains enough "
        "per-name exposure data; otherwise say the report is insufficient "
        "and explain the missing fields. Explain what remains after hedges "
        "instead of implying risk is gone. Use idiosyncratic for residual/name-level risk and return. "
        "Avoid the legacy S-term formed by 'spec' + 'ific'; use idiosyncratic, "
        "name-level, or trade ticket instead. Do not invent data that is not in "
        "the report."
    )


def report_payload(report):
    """Return the compact and complete report context sent to the model."""
    return (
        "Rendered report context:\n"
        f"{report_context(report)}\n\n"
        "Complete report JSON:\n"
        "```json\n"
        f"{full_report_json(report)}\n"
        "```"
    )


def report_context(report):
    """Return compact CSV blocks that explain the report."""
    summary = report["summary"]
    meta = report["meta"]
    frames = {
        "report_meta": pd.DataFrame(
            [
                {"field": "universe", "value": meta["universe"]},
                {"field": "as_of_date", "value": meta["as_of_date"]},
                {"field": "benchmark_name", "value": meta["benchmark"]["name"]},
                {"field": "benchmark_kind", "value": meta["benchmark"]["kind"]},
                {"field": "benchmark_tagline", "value": meta["benchmark"]["tagline"]},
            ]
        ),
        "headline_summary": pd.DataFrame(
            [
                {
                    "metric": "total_risk",
                    "value": summary["total_risk"],
                    "meaning": "annualized total portfolio volatility",
                },
                {
                    "metric": "tracking_error",
                    "value": summary["tracking_error"],
                    "meaning": "annualized active risk versus model risk proxy",
                },
                {
                    "metric": "ex_ante_model_beta",
                    "value": summary["beta"],
                    "meaning": "model beta versus model risk proxy",
                },
                {
                    "metric": "common_factor_share_total_variance",
                    "value": summary["factor_share"],
                    "meaning": "common-factor share of total portfolio variance, not tracking error",
                },
                {
                    "metric": "idiosyncratic_share_total_variance",
                    "value": summary["idiosyncratic_share_of_total_variance"],
                    "meaning": "idiosyncratic share of total portfolio variance, not tracking error",
                },
                {
                    "metric": "idiosyncratic_share_tracking_error",
                    "value": summary["idiosyncratic_share_of_tracking_error"],
                    "meaning": "idiosyncratic variance share of tracking error",
                },
                {
                    "metric": "common_factor_share_tracking_error",
                    "value": 1.0 - summary["idiosyncratic_share_of_tracking_error"],
                    "meaning": "common-factor variance share of tracking error",
                },
            ]
        ),
        "active_return_reconciliation": active_return_reconciliation(report),
        "top_active_return_contributors": top_active_return_contributors(report, 12),
        "idiosyncratic_return_by_name": rows_frame(report["idiosyncratic_return_by_name"]),
        "idiosyncratic_risk_by_name": rows_frame(report["idiosyncratic_risk_by_name"]["rows"]),
        "top_active_risk_rows": rows_frame(
            sorted(report["active_rows"], key=lambda row: -abs(row["te_share"] or 0))[:12]
        ),
    }
    parts = []
    for name, frame in frames.items():
        parts.append(f"{name}:\n```csv\n{frame.to_csv(index=False).strip()}\n```")
    return "\n\n".join(parts)


def active_return_reconciliation(report):
    """Return the active-return bridge as a small table."""
    active = first(report.get("active_ret"))
    rows = []
    for label, value in [
        ("Style factors", family_value(report, "Style")),
        ("Sector factors", family_value(report, "Sector")),
        ("Industry factors", family_value(report, "Industry")),
        ("Idiosyncratic return", first(report.get("idiosyncratic_return"))),
        ("Active return", active),
        ("Portfolio return", first(report.get("portfolio_ret"))),
        ("Benchmark return", first(report.get("benchmark_ret"))),
    ]:
        rows.append({"source": label, "contribution": value, "pct_active": ratio(value, active)})
    return pd.DataFrame(rows)


def top_active_return_contributors(report, limit):
    """Return ranked realized factor contributors for the latest day."""
    active = first(report.get("active_ret"))
    rows = []
    for row in report.get("active_rows", []):
        values = row.get("ret")
        ret = first(values)
        if ret is None or str(row.get("factor")) == "market":
            continue
        rows.append(
            {
                "group": row.get("family"),
                "factor": row.get("label"),
                "contribution": ret,
                "pct_active": ratio(ret, active),
                "tracking_error_share": row.get("te_share"),
            }
        )
    rows.sort(key=lambda item: abs(item["contribution"] or 0.0), reverse=True)
    return pd.DataFrame(rows[:limit])


def family_value(report, name):
    """Return the latest-day return contribution for one factor family."""
    values = report.get("family_ret", {}).get(name)
    return first(values)


def first(values):
    """Return the first numeric value from a list-like object."""
    if values is None or len(values) == 0:
        return None
    value = values[0]
    return None if pd.isna(value) else float(value)


def ratio(value, denominator):
    """Return value divided by denominator, preserving empty denominators."""
    if value is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return float(value / denominator)


def full_report_json(report):
    """Return JSON for the complete report."""
    return json.dumps(json_safe(report), indent=2, sort_keys=True)


def rows_frame(rows):
    """Return a DataFrame from report row dictionaries."""
    return pd.DataFrame([json_safe(row) for row in rows])


def json_safe(value):
    """Return JSON-safe Python values for report payloads."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def run_sync(coro):
    """Run an async Agent SDK call from sync report code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}

    def target():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as error:
            result["error"] = error

    thread = Thread(target=target)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]
