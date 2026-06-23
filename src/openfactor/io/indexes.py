import pandas as pd

from openfactor.core.checks import require_columns


DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_INDEX_TICKERS = (
    "SPY", "QQQ", "IWM", "IJH", "IJR",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    "MTUM", "VLUE", "QUAL", "USMV", "SPHB", "SPLV", "IWD", "IWF", "VTV", "VUG", "VYM", "SCHD",
)
INDEX_COLUMNS = ["ticker", "name", "benchmark", "provider", "default_benchmark"]
INDEX_PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "unadjusted_close"]
INDEX_RETURN_COLUMNS = ["date", "ticker", "return"]


def _index(name, benchmark, default_benchmark=False):
    """Return one index/ETF metadata entry.

    Example:
        _index("SPDR S&P 500 ETF Trust", "S&P 500", True) describes the benchmark.
    """
    return {"name": name, "benchmark": benchmark, "provider": "Massive/Polygon", "default_benchmark": default_benchmark}


INDEXES = {
    # Broad market and size
    "SPY": _index("SPDR S&P 500 ETF Trust", "S&P 500", True),
    "QQQ": _index("Invesco QQQ Trust", "Nasdaq 100"),
    "IWM": _index("iShares Russell 2000 ETF", "Russell 2000"),
    "IJH": _index("iShares Core S&P Mid-Cap ETF", "S&P MidCap 400"),
    "IJR": _index("iShares Core S&P Small-Cap ETF", "S&P SmallCap 600"),
    # GICS sectors (SPDR Select Sector) — map ~1:1 to the model sector factors
    "XLK": _index("Technology Select Sector SPDR Fund", "S&P 500 Technology"),
    "XLF": _index("Financial Select Sector SPDR Fund", "S&P 500 Financials"),
    "XLE": _index("Energy Select Sector SPDR Fund", "S&P 500 Energy"),
    "XLV": _index("Health Care Select Sector SPDR Fund", "S&P 500 Health Care"),
    "XLY": _index("Consumer Discretionary Select Sector SPDR Fund", "S&P 500 Consumer Discretionary"),
    "XLP": _index("Consumer Staples Select Sector SPDR Fund", "S&P 500 Consumer Staples"),
    "XLI": _index("Industrial Select Sector SPDR Fund", "S&P 500 Industrials"),
    "XLU": _index("Utilities Select Sector SPDR Fund", "S&P 500 Utilities"),
    "XLB": _index("Materials Select Sector SPDR Fund", "S&P 500 Materials"),
    "XLRE": _index("Real Estate Select Sector SPDR Fund", "S&P 500 Real Estate"),
    "XLC": _index("Communication Services Select Sector SPDR Fund", "S&P 500 Communication Services"),
    # Style / factor proxies
    "MTUM": _index("iShares MSCI USA Momentum Factor ETF", "USA Momentum"),
    "VLUE": _index("iShares MSCI USA Value Factor ETF", "USA Value"),
    "QUAL": _index("iShares MSCI USA Quality Factor ETF", "USA Quality"),
    "USMV": _index("iShares MSCI USA Min Vol Factor ETF", "USA Minimum Volatility"),
    "SPHB": _index("Invesco S&P 500 High Beta ETF", "S&P 500 High Beta"),
    "SPLV": _index("Invesco S&P 500 Low Volatility ETF", "S&P 500 Low Volatility"),
    "IWD": _index("iShares Russell 1000 Value ETF", "Russell 1000 Value"),
    "IWF": _index("iShares Russell 1000 Growth ETF", "Russell 1000 Growth"),
    "VTV": _index("Vanguard Value ETF", "CRSP US Large Cap Value"),
    "VUG": _index("Vanguard Growth ETF", "CRSP US Large Cap Growth"),
    "VYM": _index("Vanguard High Dividend Yield ETF", "FTSE High Dividend Yield"),
    "SCHD": _index("Schwab U.S. Dividend Equity ETF", "Dow Jones U.S. Dividend 100"),
}


def index_metadata(tickers=DEFAULT_INDEX_TICKERS):
    """Return public benchmark/index rows.

    Example:
        index_metadata(["SPY"]) returns one row describing the S&P 500 proxy.
    """
    rows = []
    for ticker in tickers:
        ticker = str(ticker).upper()
        item = INDEXES.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "name": item.get("name", ticker),
                "benchmark": item.get("benchmark", ticker),
                "provider": item.get("provider", "Massive/Polygon"),
                "default_benchmark": bool(item.get("default_benchmark", False)),
            }
        )
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def index_returns_from_prices(prices):
    """Return daily simple returns from adjusted index closes.

    Example:
        SPY closes 100 then 101 returns 0.01 on the second date.
    """
    require_columns(prices, ["date", "ticker", "close"])
    frame = prices[["date", "ticker", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values(["ticker", "date"])
    frame["return"] = frame.groupby("ticker")["close"].pct_change()
    return (
        frame[INDEX_RETURN_COLUMNS]
        .dropna(subset=["return"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def index_return_series(index_returns, ticker=DEFAULT_BENCHMARK_TICKER):
    """Return one ticker's index returns keyed by date."""
    if index_returns is None or index_returns.empty:
        return pd.Series(dtype=float, name=str(ticker).upper())
    require_columns(index_returns, INDEX_RETURN_COLUMNS)
    ticker = str(ticker).upper()
    frame = index_returns[index_returns["ticker"].astype(str).str.upper() == ticker].copy()
    if frame.empty:
        return pd.Series(dtype=float, name=ticker)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce")
    frame = frame.dropna(subset=["return"]).drop_duplicates("date", keep="last").sort_values("date")
    return frame.set_index("date")["return"].rename(ticker)


def trailing_index_returns(index_returns, dates, windows, ticker=DEFAULT_BENCHMARK_TICKER):
    """Return additive trailing index returns for report horizons."""
    series = index_return_series(index_returns, ticker)
    dates = [str(pd.to_datetime(date).date()) for date in dates]
    values = []
    for window in windows:
        window_dates = dates[-int(window):]
        if not window_dates:
            values.append(None)
            continue
        observed = series.reindex(window_dates)
        values.append(float(observed.sum()) if observed.notna().sum() == len(window_dates) else None)
    return values


def index_label(indexes, ticker=DEFAULT_BENCHMARK_TICKER):
    """Return a report label such as S&P 500 (SPY)."""
    ticker = str(ticker).upper()
    row = index_row(indexes, ticker)
    benchmark = row.get("benchmark") if row else None
    return f"{benchmark} ({ticker})" if benchmark else ticker


def index_row(indexes, ticker=DEFAULT_BENCHMARK_TICKER):
    """Return one metadata row for an index ticker."""
    if indexes is None or indexes.empty:
        return None
    require_columns(indexes, ["ticker"])
    ticker = str(ticker).upper()
    frame = indexes[indexes["ticker"].astype(str).str.upper() == ticker]
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()
