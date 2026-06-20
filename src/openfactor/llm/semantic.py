from dataclasses import asdict, dataclass
from importlib.resources import files
import json
import re
from textwrap import fill

import numpy as np
import pandas as pd
from tqdm import tqdm

from openfactor.llm.cache import (
    DEFAULT_SEMANTIC_CACHE,
    cached_memberships,
    read_semantic_cache,
    semantic_factor_context,
    update_semantic_cache,
    write_semantic_cache,
)
from openfactor.model.exposures import model_exposure_matrix
from openfactor.portfolio.report import portfolio_report


DEFAULT_RESIDUAL_THRESHOLD = 0.10


@dataclass(frozen=True)
class SemanticCandidate:
    """One LLM-proposed residual factor.

    Example:
        SemanticCandidate("ai_infra", "AI Infrastructure", "GPU supply-chain risk.")
        represents one possible factor to classify across the universe.
    """

    id: str
    name: str
    description: str
    kind: str = "other"
    tickers: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticDiscoveryResult:
    """Result from on-demand semantic residual discovery.

    Example:
        result.accepted contains factors that reduced residual risk enough.
        result.skipped contains rejected candidates and skip reasons.
    """

    residual_share: float
    threshold: float
    skipped_holdings: list[str]
    pca: pd.DataFrame
    candidates: pd.DataFrame
    memberships: pd.DataFrame
    accepted: pd.DataFrame
    skipped: pd.DataFrame


def discover_semantic_factors(
    portfolio,
    snapshot,
    llm=None,
    threshold=DEFAULT_RESIDUAL_THRESHOLD,
    window=63,
    batch_size=50,
    semantic_cache=DEFAULT_SEMANTIC_CACHE,
    progress=True,
    logger=print,
):
    """Run on-demand semantic discovery when residual risk is high.

    Example:
        discover_semantic_factors(portfolio, snapshot, threshold=0.20)
        only asks the LLM when stock-specific variance share is at least 20%.
    """
    weights, skipped_holdings = usable_weights(portfolio, snapshot, window)
    if len(weights) < 2:
        return skipped_result(0.0, threshold, skipped_holdings, "not_enough_residual_history", logger=logger)

    share = residual_share(snapshot, weights)
    residuals = residual_matrix(snapshot.residual_returns, list(weights), window)
    _, pca, _ = residual_pca(residuals, weights)
    log_start(logger, share, threshold, skipped_holdings, pca)

    if share < threshold:
        return skipped_result(share, threshold, skipped_holdings, "residual_below_threshold", pca, logger)

    llm = llm or default_llm()
    cache = read_semantic_cache(semantic_cache)
    existing_semantics = semantic_factor_context(cache)
    log_semantic_cache(logger, semantic_cache, existing_semantics)
    context = deterministic_context(snapshot, weights)
    candidates = discover_candidates(llm, snapshot, weights, share, pca, context, existing_semantics)
    candidates_frame = candidates_table(candidates)
    log_candidates(logger, candidates)
    if not candidates:
        return skipped_result(share, threshold, skipped_holdings, "llm_returned_no_candidates", pca, logger)

    universe = snapshot.universe["ticker"].dropna().astype(str).tolist()
    stocks = stock_contexts(snapshot, universe)
    memberships = classify_members(llm, candidates, stocks, context, cache, batch_size, progress)
    write_semantic_cache(update_semantic_cache(cache, memberships), semantic_cache)
    log_portfolio_memberships(logger, memberships, weights)

    universe_residuals = residual_matrix(snapshot.residual_returns, universe, window)
    accepted, skipped = refit_candidates(candidates, memberships, universe_residuals, weights)
    log_factor_summary(logger, "semantic accepted factors", accepted)
    log_factor_summary(logger, "semantic rejected factors", skipped)
    log_accepted_details(logger, accepted)
    return SemanticDiscoveryResult(share, threshold, skipped_holdings, pca, candidates_frame, memberships, accepted, skipped)


def usable_weights(portfolio, snapshot, window, min_rows=20):
    """Return holdings with enough residual history.

    Example:
        if AMC has no residual rows, it is skipped and the rest keep the same gross exposure.
    """
    weights = portfolio[["ticker", "allocation"]].copy()
    weights["ticker"] = weights["ticker"].astype(str)
    weights = weights.set_index("ticker")["allocation"].astype(float).to_dict()
    residuals = residual_matrix(snapshot.residual_returns, list(weights), window)
    counts = residuals.notna().sum()
    keep = {ticker: weight for ticker, weight in weights.items() if counts.get(ticker, 0) >= min_rows}
    skipped = [ticker for ticker in weights if ticker not in keep]
    gross = sum(abs(value) for value in keep.values())
    original = sum(abs(value) for value in weights.values())
    if gross > 0:
        keep = {ticker: value * original / gross for ticker, value in keep.items()}
    return keep, skipped


def residual_share(snapshot, weights):
    """Return stock-specific variance share as a decimal.

    Example:
        25 in report["risk_share"] becomes 0.25.
    """
    report = portfolio_report(portfolio_frame(weights), snapshot)
    value = report["risk_share"].loc["residual_unexplained_percent", "value"]
    return float(value) / 100


def residual_matrix(residuals, tickers, window):
    """Return date x ticker residual returns.

    Example:
        rows for AAPL and MSFT become columns AAPL and MSFT over the last window dates.
    """
    tickers = [str(ticker) for ticker in tickers]
    frame = residuals[residuals["ticker"].astype(str).isin(tickers)].copy()
    if frame.empty:
        return pd.DataFrame(columns=tickers)

    frame["date"] = frame["date"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    dates = sorted(frame["date"].unique())[-window:]
    date_index = {date: row for row, date in enumerate(dates)}
    ticker_index = {ticker: col for col, ticker in enumerate(tickers)}
    values = np.full((len(dates), len(tickers)), np.nan)
    for row in frame[["date", "ticker", "residual_return"]].itertuples(index=False):
        if row.date in date_index and row.ticker in ticker_index:
            values[date_index[row.date], ticker_index[row.ticker]] = float(row.residual_return)
    return pd.DataFrame(values, index=dates, columns=tickers)


def residual_pca(residuals, weights, components=3):
    """Return PCA loadings from portfolio residual returns.

    Example:
        if NVDA and AMD residuals move together, pc1 shows same-sign loadings.
    """
    clean = residuals.dropna()
    tickers = list(weights)
    if len(clean) < 20 or len(tickers) < 2:
        return clean, pd.DataFrame(), []

    x = clean[tickers].to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    x = x * np.sqrt(np.abs(np.array([weights[ticker] for ticker in tickers], dtype=float)))
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    explained = (s**2) / np.sum(s**2)
    rows = []
    for i in range(min(components, len(explained))):
        row = {"component": f"pc{i + 1}", "explained": explained[i]}
        row.update({ticker: vt[i, j] for j, ticker in enumerate(tickers)})
        rows.append(row)
    return clean, pd.DataFrame(rows), explained[:components]


def discover_candidates(llm, snapshot, weights, share, pca, context, existing_semantics=None):
    """Ask the LLM for candidate residual factors.

    Example:
        a high meme-stock residual cluster can return a community-speculation factor.
    """
    payload = {
        "portfolio": weights,
        "residual_share": share,
        "pca_loadings": pca.round(4).to_dict("records"),
        "deterministic_model_context": context,
        "existing_semantic_factors": existing_semantics or [],
        "portfolio_stocks": stock_contexts(snapshot, list(weights)),
    }
    data = llm_json(llm, prompt("discovery.txt"), payload)
    return [candidate(row) for row in json_rows(data, "factors")]


def classify_members(llm, candidates, stocks, context, cache, batch_size, progress):
    """Classify semantic factor membership across the snapshot universe.

    Example:
        1000 stocks with batch_size=50 makes 20 LLM membership calls per factor.
    """
    rows = []
    instructions = prompt("scoring.txt")
    for factor in candidates:
        bar = tqdm(total=len(stocks), desc=f"semantic member {factor.id}", unit="ticker", dynamic_ncols=True, disable=not progress)
        cached = cached_memberships(cache, factor.id, [stock["ticker"] for stock in stocks])
        rows += cached.to_records(index=False).tolist()
        if not cached.empty:
            bar.update(len(cached))

        cached_tickers = set(cached["ticker"]) if not cached.empty else set()
        missing = [stock for stock in stocks if stock["ticker"] not in cached_tickers]
        for batch in batches(missing, batch_size):
            payload = {
                "factor": asdict(factor),
                "deterministic_model_context": context,
                "stocks": batch,
            }
            data = llm_json(llm, instructions, payload)
            rows += membership_rows(factor.id, json_rows(data, "memberships"))
            bar.update(len(batch))
        bar.close()
    return clean_memberships(pd.DataFrame(rows, columns=["factor_id", "ticker", "member", "reason"]))


def refit_candidates(candidates, memberships, residuals, weights):
    """Keep candidates that explain portfolio idiosyncratic returns.

    Example:
        any lower idiosyncratic variance after refit means the factor is accepted.
    """
    rows = []
    for factor in candidates:
        members = memberships[memberships["factor_id"] == factor.id]
        refit = semantic_refit(residuals, weights, members)
        accepted = refit["after_var"] < refit["before_var"]
        rows.append(
            {
                "factor_id": factor.id,
                "name": factor.name,
                "description": factor.description,
                "kind": factor.kind,
                "members": int(members["member"].sum()) if not members.empty else 0,
                "idio_explained_percent": 100 * refit["idio_explained"],
                "before_var": refit["before_var"],
                "after_var": refit["after_var"],
                "before_risk": refit["before_risk"],
                "after_risk": refit["after_risk"],
                "risk_reduction_percent": 100 * refit["risk_reduction"],
                "decision": "accepted" if accepted else "rejected",
                "reason": "" if accepted else refit["reason"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, frame
    return frame[frame["decision"] == "accepted"], frame[frame["decision"] != "accepted"]


def semantic_refit(residuals, weights, memberships):
    """Return idiosyncratic variance explained by one semantic membership.

    Example:
        member=1 rows define a cross-sectional residual-return factor.
    """
    if memberships.empty:
        return refit_result(reason="no_memberships")

    memberships = memberships.set_index("ticker")["member"].astype(float)
    selected = sorted(set(memberships[memberships > 0].index) | set(weights))
    selected = [ticker for ticker in selected if ticker in residuals.columns and residuals[ticker].notna().sum() >= 20]
    clean = residuals[selected].dropna() if selected else pd.DataFrame()
    portfolio_tickers = [ticker for ticker in weights if ticker in clean.columns]
    if len(clean) < 20 or len(selected) < 2 or not portfolio_tickers:
        return refit_result(reason="not_enough_residual_history")

    x = memberships.reindex(selected).fillna(0.0).to_numpy(dtype=float)
    x = (x - x.mean()).reshape(-1, 1)
    if np.allclose(x, 0):
        return refit_result(reason="flat_membership")

    y = clean[selected].to_numpy(dtype=float)
    fitted = []
    for row in y:
        beta = np.linalg.lstsq(x, row, rcond=None)[0]
        fitted.append(row - x @ beta)
    after = pd.DataFrame(fitted, index=clean.index, columns=selected)

    w = np.array([weights[ticker] for ticker in portfolio_tickers], dtype=float)
    before_returns = clean[portfolio_tickers].to_numpy(dtype=float) @ w
    after_returns = after[portfolio_tickers].to_numpy(dtype=float) @ w
    before_risk = annualized_vol(before_returns)
    after_risk = annualized_vol(after_returns)
    before_var = variance(before_returns)
    after_var = variance(after_returns)
    idio_explained = (before_var - after_var) / before_var if before_var > 0 else 0.0
    risk_reduction = (before_risk - after_risk) / before_risk if before_risk > 0 else 0.0
    return {
        "idio_explained": idio_explained,
        "before_var": before_var,
        "after_var": after_var,
        "before_risk": before_risk,
        "after_risk": after_risk,
        "risk_reduction": risk_reduction,
        "reason": "",
    }


def deterministic_context(snapshot, weights):
    """Return existing factors the LLM must not rename.

    Example:
        sector:Technology appears here, so the LLM should not rediscover technology.
    """
    matrix = model_exposure_matrix(snapshot.exposures).reindex(list(weights)).fillna(0.0)
    exposure = matrix.multiply(pd.Series(weights), axis=0).sum()
    groups = factor_groups(snapshot.exposures)
    rows = []
    for factor, value in exposure.dropna().sort_index().items():
        group = groups.get(str(factor), "model")
        if group in ["sector", "industry"] and abs(value) < 0.01:
            continue
        if group not in ["sector", "industry"] and abs(value) < 0.25:
            continue
        rows.append({"factor": str(factor), "group": group, "portfolio_exposure": round(float(value), 4)})
    return rows


def stock_contexts(snapshot, tickers):
    """Return compact stock rows for LLM prompts.

    Example:
        NVDA becomes ticker plus sector, industry, and largest style exposures.
    """
    groups = factor_groups(snapshot.exposures)
    matrix = model_exposure_matrix(snapshot.exposures).reindex(tickers).fillna(0.0)
    rows = []
    for ticker, values in matrix.iterrows():
        row = {"ticker": str(ticker)}
        categories = [factor for factor, value in values.items() if groups.get(str(factor)) in ["sector", "industry"] and value > 0.5]
        if categories:
            row["categories"] = categories
        styles = values[[factor for factor in values.index if groups.get(str(factor)) not in ["sector", "industry"]]]
        styles = styles[styles.abs() >= 1.5].sort_values(key=np.abs, ascending=False).head(5)
        if not styles.empty:
            row["large_exposures"] = {str(key): round(float(value), 3) for key, value in styles.items()}
        rows.append(row)
    return rows


def llm_json(llm, instructions, payload):
    """Return parsed JSON from a callable or complete_json client.

    Example:
        llm_json(client, "Return JSON.", {}) returns a dict.
    """
    data = llm.complete_json(instructions, payload) if hasattr(llm, "complete_json") else llm(instructions, payload)
    return json.loads(data) if isinstance(data, str) else data


def default_llm():
    """Return the default optional semantic LLM client.

    Example:
        default_llm() uses OPENFACTOR_SEMANTIC_MODEL when set.
    """
    from openfactor.llm.client import SemanticLLMClient

    return SemanticLLMClient()


def candidate(row):
    """Return a candidate from one LLM JSON row.

    Example:
        {"name": "AI Infra"} becomes a SemanticCandidate with id ai_infra.
    """
    name = str(row.get("name") or row.get("id") or "semantic_factor")
    return SemanticCandidate(
        id=safe_id(row.get("id") or name),
        name=name,
        description=str(row.get("description") or ""),
        kind=str(row.get("kind") or "other"),
        tickers=tuple(str(ticker) for ticker in row.get("tickers", [])),
        evidence=tuple(str(item) for item in row.get("evidence", [])),
    )


def membership_rows(factor_id, rows):
    """Return normalized binary membership rows.

    Example:
        member=true becomes 1.
    """
    output = []
    for row in rows:
        output.append((factor_id, str(row.get("ticker", "")), clean_member(row.get("member")), str(row.get("reason", ""))))
    return output


def clean_memberships(rows):
    """Return one binary membership row per factor and ticker.

    Example:
        duplicate HOOD rows with member 1 and 0 become one member 1 row.
    """
    if rows.empty:
        return rows
    rows = rows[rows["ticker"].astype(str).str.strip() != ""].copy()
    rows["ticker"] = rows["ticker"].astype(str)
    rows["member"] = rows["member"].astype(int).clip(0, 1)
    return (
        rows.groupby(["factor_id", "ticker"], as_index=False)
        .agg(
            member=("member", "max"),
            reason=("reason", join_reasons),
        )
        .sort_values(["factor_id", "ticker"])
    )


def join_reasons(values):
    """Return a compact reason string from duplicate LLM rows.

    Example:
        repeated reasons are shown once.
    """
    reasons = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in reasons:
            reasons.append(text)
    return " / ".join(reasons[:2])


def json_rows(data, key):
    """Return rows from either {"key": [...]} or a raw list.

    Example:
        json_rows({"memberships": []}, "memberships") returns [].
    """
    if isinstance(data, list):
        return data
    return data.get(key, [])


def clean_member(value):
    """Return a binary membership value.

    Example:
        clean_member("yes") returns 1.
    """
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y"} else 0
    return 1 if value else 0


def candidates_table(candidates):
    """Return candidates as a DataFrame.

    Example:
        two candidates become two rows with name and description.
    """
    return pd.DataFrame([asdict(factor) for factor in candidates])


def portfolio_memberships(memberships, weights):
    """Return LLM memberships for portfolio tickers.

    Example:
        only AAPL/MSFT rows remain for an AAPL/MSFT portfolio.
    """
    if memberships.empty:
        return memberships
    return memberships[memberships["ticker"].isin(list(weights))].sort_values(["factor_id", "ticker"])


def empty_result(share, threshold, skipped_holdings, reason, pca=None):
    """Return an empty discovery result with a skip reason.

    Example:
        residual_below_threshold appears in result.skipped.
    """
    skipped = pd.DataFrame([{"reason": reason}])
    empty = pd.DataFrame()
    return SemanticDiscoveryResult(share, threshold, skipped_holdings, pca if pca is not None else empty, empty, empty, empty, skipped)


def skipped_result(share, threshold, skipped_holdings, reason, pca=None, logger=print):
    """Return and print one skipped semantic result.

    Example:
        residual_below_threshold is shown in semantic skipped factors.
    """
    result = empty_result(share, threshold, skipped_holdings, reason, pca)
    log_frame(logger, "semantic skipped factors", result.skipped)
    return result


def log_start(logger, share, threshold, skipped_holdings, pca):
    """Print the semantic run header.

    Example:
        residual_share=0.18 threshold=0.10 is printed before LLM calls.
    """
    if not logger:
        return
    logger(f"semantic residual_share={share:.2%} threshold={threshold:.2%}")
    if skipped_holdings:
        logger(f"semantic skipped_holdings={', '.join(skipped_holdings)}")
    log_frame(logger, "semantic residual pca", pca)


def log_frame(logger, title, frame):
    """Print one semantic table.

    Example:
        empty frames print 'none'.
    """
    if not logger:
        return
    logger(title)
    logger("none" if frame is None or frame.empty else frame.to_string(index=False))


def log_candidates(logger, candidates):
    """Print compact semantic candidate rows.

    Example:
        one candidate prints name, id, kind, and tickers.
    """
    if not logger:
        return
    logger("semantic candidate factors")
    if not candidates:
        logger("none")
        return

    for index, factor in enumerate(candidates, 1):
        logger(f"[{index}] {factor.name}")
        logger(f"    id: {factor.id}")
        logger(f"    kind: {factor.kind}")
        if factor.tickers:
            logger(wrapped("tickers", ", ".join(factor.tickers)))
        logger("")


def log_accepted_details(logger, accepted):
    """Print accepted semantic factor details at the end.

    Example:
        accepted factors show description only after refit approval.
    """
    if not logger or accepted is None or accepted.empty:
        return

    logger("semantic accepted factor details")
    for row in accepted.itertuples(index=False):
        logger(f"[{row.factor_id}] {row.name}")
        if getattr(row, "description", ""):
            logger("    description:")
            logger(wrapped_text(row.description, "      "))
        logger("")


def log_portfolio_memberships(logger, memberships, weights):
    """Print portfolio semantic memberships compactly.

    Example:
        a factor prints included and excluded portfolio tickers.
    """
    if not logger:
        return
    logger("semantic portfolio memberships")
    rows = portfolio_memberships(memberships, weights)
    if rows.empty:
        logger("none")
        return

    for factor_id, group in rows.groupby("factor_id", sort=True):
        in_names = sorted(group[group["member"] == 1]["ticker"])
        out_names = sorted(group[group["member"] == 0]["ticker"])
        parts = []
        if in_names:
            parts.append(f"in: {', '.join(in_names)}")
        if out_names:
            parts.append(f"out: {', '.join(out_names)}")
        logger(f"[{factor_id}] " + (" | ".join(parts) if parts else "none"))
    logger("")


def log_factor_summary(logger, title, frame):
    """Print accepted or rejected semantic factor decisions.

    Example:
        rejected factors print one compact idio-explained line each.
    """
    if not logger:
        return
    logger(title)
    if frame is None or frame.empty:
        logger("none")
        return

    for row in frame.itertuples(index=False):
        line = (
            f"[{row.decision}] {row.name} "
            f"members={row.members} "
            f"idio_explained={row.idio_explained_percent:.2f}% "
            f"risk={percent(row.before_risk)}->{percent(row.after_risk)} "
            f"risk_reduction={row.risk_reduction_percent:.2f}%"
        )
        if getattr(row, "reason", ""):
            line = f"{line} reason={row.reason}"
        logger(line)
    logger("")


def percent(value):
    """Return a compact percent string.

    Example:
        0.1234 becomes 12.34%.
    """
    return "nan" if not np.isfinite(value) else f"{100 * value:.2f}%"


def variance(values):
    """Return sample variance for a return series.

    Example:
        variance([0.01, -0.01]) returns a positive number.
    """
    return float(np.var(values, ddof=1)) if len(values) > 1 else 0.0


def wrapped(label, value):
    """Return one wrapped labeled line.

    Example:
        wrapped("tickers", "AAPL, MSFT") prints under the label.
    """
    return fill(
        clean_text(value),
        width=110,
        initial_indent=f"    {label}: ",
        subsequent_indent=" " * (len(label) + 6),
    )


def wrapped_text(value, indent, extra_indent=None):
    """Return wrapped free text.

    Example:
        wrapped_text("long text", "  ") keeps text readable in terminals.
    """
    return fill(
        clean_text(value),
        width=110,
        initial_indent=indent,
        subsequent_indent=extra_indent or indent,
    )


def clean_text(value):
    """Return single-line text before wrapping.

    Example:
        embedded newlines become spaces.
    """
    return re.sub(r"\s+", " ", str(value)).strip()


def log_semantic_cache(logger, path, existing_semantics):
    """Print the semantic cache summary.

    Example:
        semantic cache factors=3 tells the LLM has reusable labels.
    """
    if not logger:
        return
    logger(f"semantic cache path={path} factors={len(existing_semantics)}")


def prompt(name):
    """Read one bundled prompt file.

    Example:
        prompt("discovery.txt") returns discovery instructions.
    """
    return files("openfactor.llm.prompts").joinpath(name).read_text()


def batches(items, size):
    """Yield fixed-size item batches.

    Example:
        batches([1, 2, 3], 2) yields [1, 2] then [3].
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def factor_groups(exposures):
    """Return factor id to group name.

    Example:
        sector:Technology maps to sector.
    """
    return exposures.drop_duplicates("factor").set_index("factor")["group"].astype(str).to_dict()


def portfolio_frame(weights):
    """Return a portfolio DataFrame from a weight dict.

    Example:
        {"AAPL": 0.5} becomes ticker/allocation columns.
    """
    return pd.DataFrame({"ticker": list(weights), "allocation": list(weights.values())})


def annualized_vol(values):
    """Return annualized volatility from daily returns.

    Example:
        daily standard deviation 1% becomes about 15.9%.
    """
    return float(np.std(values, ddof=1) * np.sqrt(252))


def refit_result(
    idio_explained=0.0,
    before_var=np.nan,
    after_var=np.nan,
    before_risk=np.nan,
    after_risk=np.nan,
    risk_reduction=0.0,
    reason="",
):
    """Return a standard refit result dict.

    Example:
        refit_result(reason="flat_membership") marks a rejected factor.
    """
    return {
        "idio_explained": idio_explained,
        "before_var": before_var,
        "after_var": after_var,
        "before_risk": before_risk,
        "after_risk": after_risk,
        "risk_reduction": risk_reduction,
        "reason": reason,
    }


def safe_id(value):
    """Return a machine-readable factor id.

    Example:
        "AI Infrastructure" becomes "ai_infrastructure".
    """
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")
    return text or "semantic_factor"
