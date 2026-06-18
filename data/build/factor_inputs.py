from datetime import timedelta

import numpy as np
import pandas as pd

from data.sec.ttm import last_four_quarters, quarter_values
from data.build.forward_inputs import FORWARD_COLUMNS, forward_estimate_inputs
from openfactor.factors.price.momentum import momentum_for_stock


def add_factor_inputs(
    fundamentals,
    matrix,
    dividends,
    short_interest,
    finnhub,
    analyst_ratings,
    analyst_estimates,
    keep_dates,
):
    """Attach daily point-in-time factor inputs.

    Example:
        AAPL rows get dividend_yield, sentiment, growth, and forward estimates.
    """
    keep_dates = clean_dates(keep_dates)
    forward = existing_forward_inputs(fundamentals)
    frame = clean_fundamentals(fundamentals).reset_index(drop=True)
    frame = frame.merge(close_frame(matrix), on=["ticker", "as_of_date"], how="left")
    frame["dividend_yield"] = dividend_yields(frame, dividends)
    frame["short_interest"] = short_interest_ratios(frame, short_interest)
    frame["growth"] = growth_values(frame)
    frame = frame.join(industry_momentum_values(frame, matrix))
    frame = frame.join(analyst_sentiment_values(frame, analyst_ratings))
    frame = frame.merge(
        finnhub_inputs(finnhub, frame[["ticker", "as_of_date"]]),
        on=["ticker", "as_of_date"],
        how="left",
    )
    frame["investment_quality"] = investment_quality(frame)
    frame = merge_forward_inputs(frame, forward)
    frame = merge_forward_inputs(frame, forward_estimate_inputs(frame, analyst_estimates))
    frame = frame[frame["as_of_date"].isin(keep_dates)].copy()
    return frame.drop(columns=["close"], errors="ignore")


def clean_fundamentals(fundamentals):
    """Return fundamentals with standard ticker/date columns.

    Example:
        datetime as_of_date values become ISO date strings.
    """
    frame = fundamentals.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date.astype(str)
    frame = frame.drop(columns=DERIVED_COLUMNS, errors="ignore")
    return frame


def existing_forward_inputs(fundamentals):
    """Return previously stored forward estimate snapshots.

    Example:
        yesterday's fundamentals.csv keeps yesterday's forward_growth value.
    """
    columns = [column for column in FORWARD_COLUMNS if column in fundamentals]
    if len(columns) != len(FORWARD_COLUMNS):
        return pd.DataFrame(columns=FORWARD_COLUMNS)
    frame = fundamentals[FORWARD_COLUMNS].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date.astype(str)
    return frame.dropna(how="all", subset=FORWARD_COLUMNS[2:])


def merge_forward_inputs(frame, inputs):
    """Attach forward estimate snapshots to fundamentals.

    Example:
        new rows overwrite blank forward estimate cells for the same ticker/date.
    """
    if inputs.empty:
        return frame
    rows = inputs.copy()
    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    rows["as_of_date"] = pd.to_datetime(rows["as_of_date"]).dt.date.astype(str)
    rows = rows.drop_duplicates(["ticker", "as_of_date"], keep="last")
    frame = frame.merge(rows, on=["ticker", "as_of_date"], how="left", suffixes=("", "_new"))
    for column in FORWARD_COLUMNS[2:]:
        new = f"{column}_new"
        if new in frame:
            frame[column] = frame[new].combine_first(frame.get(column))
            frame = frame.drop(columns=new)
    return frame


DERIVED_COLUMNS = [
    "dividend_yield",
    "short_interest",
    "growth",
    "industry_momentum",
    "industry_momentum_observations",
    "analyst_sentiment",
    "analyst_sentiment_observations",
    "earnings_quality",
    "earnings_variability",
    "investment_quality",
    "forward_earnings_yield",
    "forward_earnings_yield_observations",
    "forward_growth",
    "forward_growth_observations",
]


def close_frame(matrix):
    """Return close prices as ticker/date rows.

    Example:
        matrix.close[date, ticker] becomes one close row.
    """
    return pd.DataFrame(
        {
            "as_of_date": np.repeat(matrix.dates, len(matrix.tickers)),
            "ticker": np.tile(matrix.tickers, len(matrix.dates)),
            "close": matrix.close.reshape(-1),
        }
    )


def dividend_yields(frame, dividends):
    """Return trailing one-year dividends divided by same-day close.

    Example:
        $1 of trailing dividends and $100 close returns 0.01.
    """
    dividends = clean_dates_frame(dividends, "ex_dividend_date")
    groups = {ticker: rows for ticker, rows in dividends.groupby("ticker")}
    values = []
    for row in frame[["ticker", "as_of_date", "close"]].itertuples(index=False):
        rows = groups.get(row.ticker)
        close = number(row.close)
        if not positive(close):
            values.append(np.nan)
            continue
        if rows is None:
            values.append(0.0)
            continue
        day = pd.to_datetime(row.as_of_date).date()
        cash = rows[
            (rows["ex_dividend_date"] > day - timedelta(days=365))
            & (rows["ex_dividend_date"] <= day)
        ]["cash_amount"].sum()
        values.append(cash / close)
    return values


def short_interest_ratios(frame, short_interest):
    """Return latest reported short interest divided by shares.

    Example:
        10 shorted shares over 100 shares outstanding returns 0.10.
    """
    rows = clean_dates_frame(short_interest, "settlement_date")
    if not rows.empty:
        rows["available_date"] = rows["settlement_date"] + timedelta(days=8)
    groups = {ticker: item.sort_values("available_date") for ticker, item in rows.groupby("ticker")}

    values = []
    for row in frame[["ticker", "as_of_date", "shares_outstanding"]].itertuples(index=False):
        shares = number(row.shares_outstanding)
        ticker_rows = groups.get(row.ticker)
        if ticker_rows is None or not positive(shares):
            values.append(np.nan)
            continue
        day = pd.to_datetime(row.as_of_date).date()
        available = ticker_rows[ticker_rows["available_date"] <= day]
        if available.empty:
            values.append(np.nan)
        else:
            values.append(number(available.iloc[-1]["short_interest"]) / shares)
    return values


def growth_values(frame):
    """Return average revenue and net-income year-over-year growth.

    Example:
        revenue +10% and income +20% returns 15%.
    """
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("ticker"):
        group = group.sort_values("as_of_date")
        for index, row in group.iterrows():
            prior = group[group["as_of_date"] <= year_ago(row["as_of_date"])].tail(1)
            if not prior.empty:
                values.loc[index] = average_growth(row, prior.iloc[0])
    return values


def industry_momentum_values(frame, matrix):
    """Return peer momentum for each ticker's industry.

    Example:
        if NVDA's chip peers are up 20%, NVDA gets industry_momentum=0.20.
    """
    stock = stock_momentum_values(matrix, frame[["ticker", "as_of_date"]])
    rows = frame[["ticker", "as_of_date", "industry"]].merge(stock, on=["ticker", "as_of_date"], how="left")
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    observations = pd.Series(0, index=frame.index, dtype=int)

    for _, group in rows.groupby(["as_of_date", "industry"], dropna=False):
        peer_values = pd.to_numeric(group["stock_momentum"], errors="coerce")
        count = int(peer_values.notna().sum())
        total = float(peer_values.sum())
        for index, value in zip(group.index, peer_values):
            observations.loc[index] = count
            if count == 0:
                continue
            if count > 1 and np.isfinite(value):
                values.loc[index] = (total - value) / (count - 1)
            else:
                values.loc[index] = total / count

    return pd.DataFrame(
        {
            "industry_momentum": values,
            "industry_momentum_observations": observations,
        }
    )


def stock_momentum_values(matrix, requested):
    """Return stock momentum by requested ticker/date pairs.

    Example:
        requested AAPL on 2026-06-16 gets momentum using prices known that day.
    """
    requested = requested.drop_duplicates().copy()
    requested["as_of_date"] = pd.to_datetime(requested["as_of_date"]).dt.date.astype(str)
    date_rows = {date: price_row_on_or_before(matrix.dates, date) for date in requested["as_of_date"].unique()}
    ticker_index = {ticker: column for column, ticker in enumerate(matrix.tickers)}
    rows = []
    for item in requested.itertuples(index=False):
        row = date_rows[item.as_of_date]
        column = ticker_index.get(item.ticker)
        value = np.nan
        if row is not None and column is not None:
            value = momentum_for_stock(matrix.close[: row + 1, column]).value
        rows.append({"ticker": item.ticker, "as_of_date": item.as_of_date, "stock_momentum": value})
    return pd.DataFrame(rows)


def price_row_on_or_before(dates, value):
    """Return the last price row at or before a date.

    Example:
        a Sunday target uses the prior trading day row.
    """
    dates = pd.to_datetime(dates).to_numpy(dtype="datetime64[D]")
    target = np.datetime64(pd.to_datetime(value).date())
    index = np.searchsorted(dates, target, side="right") - 1
    return None if index < 0 else int(index)


def analyst_sentiment_values(frame, analyst_ratings):
    """Return time-decayed analyst sentiment values.

    Example:
        recent Buy ratings produce a positive analyst_sentiment.
    """
    rows = clean_analyst_ratings(analyst_ratings)
    groups = {ticker: item.sort_values("event_date") for ticker, item in rows.groupby("ticker")}
    values = []
    observations = []
    for row in frame[["ticker", "as_of_date"]].itertuples(index=False):
        value, count = analyst_sentiment_for_day(groups.get(row.ticker), row.as_of_date)
        values.append(value)
        observations.append(count)
    return pd.DataFrame(
        {
            "analyst_sentiment": values,
            "analyst_sentiment_observations": observations,
        },
        index=frame.index,
    )


def clean_analyst_ratings(frame):
    """Return analyst ratings with comparable dates.

    Example:
        event_date strings become date objects and score becomes float.
    """
    columns = ["ticker", "event_date", "score"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = frame.copy()
    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    rows["event_date"] = pd.to_datetime(rows["event_date"], errors="coerce").dt.date
    rows["score"] = pd.to_numeric(rows["score"], errors="coerce")
    return rows.dropna(subset=["ticker", "event_date", "score"])


def analyst_sentiment_for_day(rows, as_of_date, days=365, half_life=90):
    """Return one date's decayed analyst score and event count.

    Example:
        one current Buy and one old Hold returns a score between 0 and 1.
    """
    if rows is None or rows.empty:
        return np.nan, 0
    day = pd.to_datetime(as_of_date).date()
    available = rows[(rows["event_date"] <= day) & (rows["event_date"] > day - timedelta(days=days))]
    if available.empty:
        return np.nan, 0
    scores = available["score"].to_numpy(dtype=float)
    ages = np.array([(day - value).days for value in available["event_date"]], dtype=float)
    weights = 0.5 ** (ages / half_life)
    return float(np.average(scores, weights=weights)), len(scores)


def average_growth(current, prior):
    """Return mean growth across available fundamental metrics.

    Example:
        revenue from 100 to 120 returns one 20% growth input.
    """
    values = [ratio_growth(current, prior, name) for name in ["revenue", "net_income"]]
    values = [value for value in values if np.isfinite(value)]
    return np.nan if not values else float(np.mean(values))


def ratio_growth(current, prior, column):
    """Return current / prior - 1 for one metric.

    Example:
        120 over 100 returns 0.20.
    """
    top = number(current.get(column))
    bottom = number(prior.get(column))
    if not np.isfinite(top) or not np.isfinite(bottom) or bottom == 0:
        return np.nan
    return top / abs(bottom) - 1


def finnhub_inputs(finnhub, requested):
    """Return daily Finnhub earnings-quality inputs.

    Example:
        requested AAPL dates get earnings_quality and earnings_variability.
    """
    columns = ["ticker", "as_of_date", "earnings_quality", "earnings_variability"]
    if finnhub.empty:
        return pd.DataFrame(columns=columns)

    requested = requested.drop_duplicates().copy()
    requested["as_of_date"] = pd.to_datetime(requested["as_of_date"]).dt.date
    rows = []
    for ticker, dates in requested.groupby("ticker"):
        filings = clean_finnhub(finnhub[finnhub["ticker"] == ticker])
        current_key = None
        current_values = empty_earnings()
        for day in sorted(dates["as_of_date"]):
            available = filings[filings["accepted_date"] <= day]
            if not available.empty and available.iloc[-1]["accession_no"] != current_key:
                current_key = available.iloc[-1]["accession_no"]
                current_values = earnings_values(available)
            rows.append({"ticker": ticker, "as_of_date": str(day), **current_values})
    return pd.DataFrame(rows, columns=columns)


def clean_finnhub(filings):
    """Return Finnhub rows with date objects.

    Example:
        accepted_date strings become comparable dates.
    """
    frame = filings.copy()
    for column in ["accepted_date", "start_date", "end_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    return frame.dropna(subset=["accepted_date", "start_date", "end_date"]).sort_values("accepted_date")


def earnings_values(filings):
    """Return earnings quality and variability for available filings.

    Example:
        quality = (cash flow from operations TTM - net income TTM) / assets.
    """
    latest = filings.iloc[-1]
    report_date = latest["end_date"]
    assets = latest_asset(filings, report_date)
    net_income = ttm_value(filings, "net_income", report_date)
    cash_flow = ttm_value(filings, "operating_cash_flow", report_date)
    quarters = quarter_series(filings, "net_income", report_date).tail(8)
    return {
        "earnings_quality": safe_divide(cash_flow - net_income, assets),
        "earnings_variability": earnings_variability(quarters, assets),
    }


def ttm_value(filings, metric, report_date):
    """Return trailing-twelve-month value from Finnhub rows.

    Example:
        four net-income quarters sum to TTM net income.
    """
    quarters = quarter_series(filings, metric, report_date)
    last4 = last_four_quarters(quarters, report_date)
    if len(last4) != 4:
        return np.nan
    return float(last4["value"].sum())


def quarter_series(filings, metric, report_date):
    """Return direct and derived quarterly values.

    Example:
        six-month YTD 230 and Q1 100 derives Q2 as 130.
    """
    rows = []
    for row in filings.itertuples(index=False):
        value = number(getattr(row, metric))
        if np.isfinite(value) and row.start_date and row.end_date and row.end_date <= report_date:
            rows.append(
                {
                    "start_date": row.start_date,
                    "end_date": row.end_date,
                    "value": value,
                    "days": (row.end_date - row.start_date).days,
                }
            )
    periods = pd.DataFrame(rows).drop_duplicates(["start_date", "end_date"], keep="last")
    if periods.empty:
        return pd.DataFrame(columns=["start_date", "end_date", "value", "source"])
    return quarter_values(periods)


def latest_asset(filings, report_date):
    """Return latest assets by fiscal period.

    Example:
        assets from the latest filing period are used as denominator.
    """
    rows = filings[(filings["end_date"] <= report_date) & filings["total_assets"].notna()]
    if rows.empty:
        return np.nan
    return number(rows.iloc[-1]["total_assets"])


def earnings_variability(quarters, assets):
    """Return variability of quarterly earnings scaled by assets.

    Example:
        stable quarterly income returns a low variability value.
    """
    if len(quarters) < 6 or not positive(assets):
        return np.nan
    return float((quarters["value"] / assets).std(ddof=1))


def investment_quality(frame):
    """Return SEC investment-quality exposure input.

    Example:
        low asset growth, low capex, and high buybacks produce a higher value.
    """
    assets = numeric(frame, "total_assets")
    market_cap = numeric(frame, "market_cap")
    values = -numeric(frame, "asset_growth")
    values -= safe_ratio(numeric(frame, "capex"), assets)
    values += safe_ratio(numeric(frame, "buybacks"), market_cap)
    values -= safe_ratio(numeric(frame, "share_issuance"), market_cap)
    values[~np.isfinite(numeric(frame, "asset_growth"))] = np.nan
    return values


def clean_dates_frame(frame, date_column):
    """Return provider rows with uppercase tickers and date objects.

    Example:
        ex_dividend_date strings become date objects.
    """
    if frame.empty or "ticker" not in frame or date_column not in frame:
        return pd.DataFrame(columns=["ticker", date_column])
    rows = frame.copy()
    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    rows[date_column] = pd.to_datetime(rows[date_column], errors="coerce").dt.date
    return rows.dropna(subset=[date_column])


def clean_dates(values):
    """Return ISO date strings.

    Example:
        Timestamp("2026-06-16") becomes "2026-06-16".
    """
    return [pd.to_datetime(value).date().isoformat() for value in values]


def year_ago(value):
    """Return the same date one year earlier.

    Example:
        2026-06-16 becomes 2025-06-16.
    """
    day = pd.to_datetime(value).date()
    try:
        return day.replace(year=day.year - 1).isoformat()
    except ValueError:
        return day.replace(year=day.year - 1, month=2, day=28).isoformat()


def safe_ratio(top, bottom):
    """Return top / bottom with bad rows as zero components.

    Example:
        missing capex contributes 0.0 to investment_quality.
    """
    return np.divide(
        top,
        bottom,
        out=np.zeros(len(top)),
        where=np.isfinite(top) & np.isfinite(bottom) & (bottom != 0),
    )


def safe_divide(top, bottom):
    """Return top / bottom or NaN.

    Example:
        10 / 100 returns 0.10.
    """
    if not np.isfinite(top) or not positive(bottom):
        return np.nan
    return top / bottom


def numeric(frame, column):
    """Return a numeric column or NaNs.

    Example:
        missing capex returns all NaNs.
    """
    if column not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def empty_earnings():
    """Return empty Finnhub factor values.

    Example:
        missing Finnhub filings produce NaN factor inputs.
    """
    return {"earnings_quality": np.nan, "earnings_variability": np.nan}


def number(value):
    """Return a float or NaN.

    Example:
        number("10") returns 10.0.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def positive(value):
    """Return True for finite positive numbers.

    Example:
        positive(1.0) is True; positive(0.0) is False.
    """
    return np.isfinite(value) and value > 0
