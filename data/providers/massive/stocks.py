import pandas as pd


def stock_bars(client, ticker, start_date, end_date, adjusted=True):
    """Download daily bars for one ticker.

    Example:
        stock_bars(client, "AAPL", "2024-01-01", "2024-01-31")
        returns rows with date, ticker, close, and volume.
    """
    path = f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start_date}/{end_date}"
    data = client.get(path, {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000})
    return bars_to_frame(data.get("results", []), ticker.upper())


def daily_market(client, date, adjusted=True, include_otc=False):
    """Download all US stock daily bars for one date.

    Example:
        daily_market(client, "2024-01-02")
        returns one price row per ticker traded that day.
    """
    path = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"
    data = client.get(
        path,
        {"adjusted": str(adjusted).lower(), "include_otc": str(include_otc).lower()},
    )
    return bars_to_frame(data.get("results", []), ticker_key="T")


def all_tickers(client, date=None, active=True):
    """Download reference rows for listed stock tickers.

    Example:
        all_tickers(client, date="2024-01-02")
        returns a DataFrame of ticker metadata.
    """
    params = {"market": "stocks", "active": str(active).lower(), "limit": 1000}
    if date:
        params["date"] = date

    rows = []
    for page in client.pages("/v3/reference/tickers", params):
        rows.extend(page.get("results", []))
    return pd.DataFrame(rows)


def ticker_overview(client, ticker, date=None):
    """Download reference metadata for one ticker.

    Example:
        ticker_overview(client, "AAPL")
        returns one DataFrame row for AAPL.
    """
    params = {"date": date} if date else None
    data = client.get(f"/v3/reference/tickers/{ticker.upper()}", params)
    return pd.DataFrame([data.get("results", {})])


def stock_dividends(client, ticker, start_date, end_date):
    """Download cash dividends for one ticker.

    Example:
        stock_dividends(client, "AAPL", "2025-01-01", "2026-01-01")
        returns ex-date and split-adjusted cash rows.
    """
    params = {
        "ticker": ticker.upper(),
        "ex_dividend_date.gte": start_date,
        "ex_dividend_date.lte": end_date,
        "sort": "ex_dividend_date",
        "limit": 1000,
    }
    rows = []
    for page in client.pages("/stocks/v1/dividends", params):
        rows.extend(page.get("results", []))
    return dividends_to_frame(rows, ticker.upper())


def stock_short_interest(client, ticker, start_date, end_date):
    """Download reported short interest for one ticker.

    Example:
        stock_short_interest(client, "AAPL", "2025-01-01", "2026-01-01")
        returns settlement-date short interest rows.
    """
    params = {
        "ticker": ticker.upper(),
        "settlement_date.gte": start_date,
        "settlement_date.lte": end_date,
        "sort": "settlement_date",
        "limit": 1000,
    }
    rows = []
    for page in client.pages("/stocks/v1/short-interest", params):
        rows.extend(page.get("results", []))
    return short_interest_to_frame(rows, ticker.upper())


def bars_to_frame(rows, ticker_key):
    """Turn Massive bar JSON into OpenFactor price rows.

    Example input:
        {"T": "AAPL", "t": 1704153600000, "o": 184, "h": 186, "l": 183, "c": 185, "v": 10}

    Example output row:
        date        ticker  open  high  low  close  volume
        2024-01-02  AAPL    184   186   183  185    10

    Empty input returns the same columns with zero rows.
    """
    frame = pd.DataFrame(rows)
    columns = ["date", "ticker", "open", "high", "low", "close", "volume", "vwap"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    ticker = frame[ticker_key] if ticker_key in frame else ticker_key
    return pd.DataFrame(
        {
            "date": pd.to_datetime(frame["t"], unit="ms").dt.date.astype(str),
            "ticker": ticker,
            "open": frame["o"],
            "high": frame["h"],
            "low": frame["l"],
            "close": frame["c"],
            "volume": frame["v"],
            "vwap": frame.get("vw"),
        }
    )


def dividends_to_frame(rows, ticker):
    """Turn Massive dividend JSON into rows.

    Example:
        cash_amount 1.0 and split_adjusted_cash_amount 0.5 returns cash_amount=0.5.
    """
    frame = pd.DataFrame(rows)
    columns = ["ticker", "ex_dividend_date", "cash_amount"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    cash = frame.get("split_adjusted_cash_amount", frame.get("cash_amount"))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "ex_dividend_date": frame["ex_dividend_date"],
            "cash_amount": pd.to_numeric(cash, errors="coerce"),
        }
    )


def short_interest_to_frame(rows, ticker):
    """Turn Massive short-interest JSON into rows.

    Example:
        short_interest 100 and settlement_date 2026-01-15 becomes one row.
    """
    frame = pd.DataFrame(rows)
    columns = ["ticker", "settlement_date", "short_interest"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        {
            "ticker": ticker,
            "settlement_date": frame["settlement_date"],
            "short_interest": pd.to_numeric(frame["short_interest"], errors="coerce"),
        }
    )
