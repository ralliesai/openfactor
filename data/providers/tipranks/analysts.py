import numpy as np
import pandas as pd


RATING_SCORE = {
    "strong buy": 2.0,
    "buy": 1.0,
    "hold": 0.0,
    "sell": -1.0,
    "strong sell": -2.0,
}
TIPRANKS_TICKER_ALIAS = {"BRK.B": "BRK-B", "GOOG": "GOOGL"}


def analyst_ratings(client, ticker, count=50):
    """Download normalized analyst rating events.

    Example:
        analyst_ratings(client, "AAPL") returns event_date, recommendation, and score rows.
    """
    provider_ticker = TIPRANKS_TICKER_ALIAS.get(ticker.upper(), ticker.upper())
    data = client.get(f"analysts/{provider_ticker.lower()}", {"num": count})
    return clean_rows(ticker.upper(), data if isinstance(data, list) else [])


def clean_rows(ticker, items):
    """Return slim analyst-rating rows.

    Example:
        one Buy item becomes score 1.0 for that event date.
    """
    rows = []
    for item in items:
        event_date = rating_date(item)
        score = rating_score(item.get("recommendation"))
        if event_date and np.isfinite(score):
            rows.append(
                {
                    "ticker": ticker,
                    "event_date": event_date,
                    "recommendation": item.get("recommendation"),
                    "analyst_action": item.get("analystAction"),
                    "price_target": number(item.get("priceTarget")),
                    "score": score,
                }
            )
    columns = ["ticker", "event_date", "recommendation", "analyst_action", "price_target", "score"]
    return pd.DataFrame(rows, columns=columns).drop_duplicates(keep="last")


def rating_date(item):
    """Return the publication date for one analyst event.

    Example:
        timestamp 2026-06-16T09:00:00 returns 2026-06-16.
    """
    for key in ["timestamp", "publishedDate", "recommendationDate"]:
        value = pd.to_datetime(item.get(key), errors="coerce")
        if not pd.isna(value):
            return value.date().isoformat()
    return None


def rating_score(value):
    """Return the numeric score for a recommendation.

    Example:
        Buy returns 1.0 and Sell returns -1.0.
    """
    return RATING_SCORE.get(str(value or "").strip().lower(), np.nan)


def number(value):
    """Return a float or NaN.

    Example:
        number("315") returns 315.0.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
