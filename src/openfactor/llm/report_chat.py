import asyncio
import json
import os
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

import pandas as pd

from openfactor.llm.report_bundle import file_previews, metric_glossary, write_report_bundle
from openfactor.llm.report_tool import json_safe, portfolio_report_tool


DEFAULT_REPORT_CHAT_MODEL = "gpt-5.5"
DEFAULT_REPORT_CHAT_TURNS = 50


class ReportChat:
    """PM-facing chat over one rendered report."""

    def __init__(self, report, snapshot=None, model=None, api_key=None, timeout=None):
        self.report = report
        self.snapshot = snapshot
        self.model = model or os.getenv("OPENFACTOR_REPORT_CHAT_MODEL", DEFAULT_REPORT_CHAT_MODEL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = float(timeout or os.getenv("OPENFACTOR_REPORT_CHAT_TIMEOUT", "600"))
        self.client = None
        self.local_bundle = None
        self.files = []
        self.file_ids = []

    @classmethod
    def from_env(cls, report, snapshot=None):
        """Return a chat client only when OPENAI_API_KEY is exported."""
        return cls(report, snapshot=snapshot) if report_chat_enabled() else None

    def answer(self, question, history=None):
        """Answer one PM question with an Agent SDK code-interpreter agent."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not exported")

        return run_sync(self.answer_async(question, history))

    async def answer_async(self, question, history=None):
        """Answer one PM question from async Textual or script code."""
        from agents import Agent, CodeInterpreterTool, ModelSettings, Runner, WebSearchTool, set_default_openai_key
        from openai.types.shared.reasoning import Reasoning

        set_default_openai_key(self.api_key)
        file_ids = await asyncio.to_thread(self.ensure_files_uploaded)
        tools = [
            CodeInterpreterTool(code_interpreter_config(file_ids)),
            WebSearchTool(user_location={"type": "approximate", "country": "US"}),
        ]
        if self.snapshot is not None:
            tools.append(portfolio_report_tool(self.snapshot))
        agent = Agent(
            name="OpenFactor PM report analyst",
            model=self.model,
            tools=tools,
            model_settings=ModelSettings(
                tool_choice="auto",
                reasoning=Reasoning(effort="medium"),
                verbosity="medium",
            ),
            instructions=report_chat_instructions(self.files),
        )
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                report_chat_input(self.report, question, history, self.files),
                max_turns=int(os.getenv("OPENFACTOR_REPORT_CHAT_MAX_TURNS", DEFAULT_REPORT_CHAT_TURNS)),
            ),
            timeout=self.timeout,
        )
        return str(result.final_output).strip()

    def ensure_files_uploaded(self):
        """Write, upload, and retain Code Interpreter files for this report."""
        if self.file_ids:
            return self.file_ids
        if self.snapshot is None:
            return []

        try:
            self.local_bundle = TemporaryDirectory(prefix="openfactor-report-chat-")
            directory = Path(self.local_bundle.name)
            context = report_context(self.report)
            report_json = full_report_json(self.report)
            self.files = write_report_bundle(directory, self.snapshot, self.report, context, report_json)
            self.file_ids = upload_files(self.openai_client(), self.files)
            return self.file_ids
        except Exception:
            self.files = []
            self.file_ids = []
            if self.local_bundle is not None:
                self.local_bundle.cleanup()
                self.local_bundle = None
            raise

    def openai_client(self):
        """Return the OpenAI client used for file upload and cleanup."""
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        return self.client

    def close(self):
        """Delete uploaded OpenAI files and remove the local temporary bundle."""
        try:
            if self.client is not None and self.file_ids:
                delete_files(self.client, self.file_ids)
        finally:
            self.file_ids = []
            self.files = []
            if self.local_bundle is not None:
                self.local_bundle.cleanup()
                self.local_bundle = None


def report_chat_enabled():
    """Return True when the report should show the chat sidebar."""
    return bool(os.getenv("OPENAI_API_KEY"))


def code_interpreter_config(file_ids):
    """Return an automatic Code Interpreter container config."""
    container = {"type": "auto"}
    if file_ids:
        container["file_ids"] = file_ids
    return {"type": "code_interpreter", "container": container}


def report_chat_input(report, question, history=None, files=None):
    """Return one report-grounded PM question for the Agent SDK runner."""
    parts = [report_payload(report, files)]
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


def report_chat_instructions(files=None):
    """Return the product contract for the PM report chat."""
    file_clause = (
        "Use Code Interpreter and the attached report/data files for calculations."
        if files else
        "Use Code Interpreter for arithmetic, hedge sizing, and reconciliation checks."
    )
    source_clause = (
        "The report and attached files are the source of truth; do not assume hidden "
        if files else
        "The report is the source of truth; do not assume hidden "
    )
    return prompt_text("report_chat.txt").format(
        file_clause=file_clause,
        source_clause=source_clause,
    )


def prompt_text(name):
    """Return one packaged LLM prompt."""
    return resources.files("openfactor.llm.prompts").joinpath(name).read_text()


def report_payload(report, files=None):
    """Return the compact and complete report context sent to the model."""
    payload = (
        "Rendered report context:\n"
        f"{report_context(report)}\n\n"
        "Complete report JSON:\n"
        "```json\n"
        f"{full_report_json(report)}\n"
        "```"
    )
    if files:
        payload += (
            "\n\nAttached files:\n"
            f"{attached_files(files)}\n\n"
            "Data glossary:\n"
            f"{glossary_text(files)}\n\n"
            "File headers and first rows:\n"
            f"{file_previews(files)}"
        )
    return payload


def upload_files(client, files):
    """Upload report files for Code Interpreter and return OpenAI file IDs."""
    ids = []
    try:
        for path in files:
            with path.open("rb") as handle:
                ids.append(client.files.create(file=handle, purpose="assistants").id)
        return ids
    except Exception:
        if ids:
            delete_files(client, ids)
        raise


def delete_files(client, file_ids):
    """Delete OpenAI files created for this report chat."""
    errors = []
    for file_id in file_ids:
        try:
            client.files.delete(file_id)
        except Exception as error:
            errors.append(f"{file_id}: {error}")
    if errors:
        raise RuntimeError("failed to delete OpenAI report files: " + "; ".join(errors))


def attached_files(files):
    """Return a markdown list of attached file names."""
    return "\n".join(f"- {path.name}" for path in files)


def glossary_text(files):
    """Return the metric and file glossary for the analyst prompt."""
    return "\n".join(f"- {line}" for line in metric_glossary(files))


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
        "realized_return_windows": realized_return_windows(report),
        "realized_window_contributors": realized_window_contributors(report, 8),
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


def realized_return_windows(report):
    """Return available multi-day realized attribution windows."""
    rows = []
    for key, window in (report.get("realized_windows") or {}).items():
        rows.append(
            {
                "key": key,
                "label": window.get("label"),
                "days": window.get("days"),
                "date_range": window.get("date_range"),
                "benchmark_return": window.get("benchmark"),
                "active_return": window.get("active"),
                "portfolio_return": window.get("portfolio"),
                "idiosyncratic_return": window.get("idiosyncratic"),
            }
        )
    return pd.DataFrame(rows)


def realized_window_contributors(report, limit):
    """Return top factor contributors for each multi-day realized window."""
    rows = []
    lookup = {row["factor"]: (row["label"], row["family"]) for row in report.get("active_rows", [])}
    for key, window in (report.get("realized_windows") or {}).items():
        active = window.get("active")
        factors = sorted(
            (window.get("factor") or {}).items(),
            key=lambda item: abs(item[1] or 0.0),
            reverse=True,
        )
        for factor, contribution in factors[:limit]:
            label, family = lookup.get(factor, (factor, None))
            rows.append(
                {
                    "window": key,
                    "label": window.get("label"),
                    "group": family,
                    "factor": label,
                    "contribution": contribution,
                    "pct_active": ratio(contribution, active),
                }
            )
    return pd.DataFrame(rows)


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
