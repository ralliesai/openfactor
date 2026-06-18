import os
import threading
import time

import requests


BASE_URL = "https://finnhub.io/api/v1"
SECONDS_BETWEEN_REQUESTS = float(os.getenv("OPENFACTOR_FINNHUB_SLEEP", "1"))
RATE_LOCK = threading.Lock()
NEXT_REQUEST_AT = 0.0


class FinnhubClient:
    """Tiny Finnhub JSON client.

    Example:
        FinnhubClient(api_key="...").get("stock/financials-reported", {"symbol": "AAPL"})
        returns Finnhub JSON.
    """

    def __init__(self, api_key=None, timeout=30):
        """Create a client from a key or OPENFACTOR_FINNHUB_API_KEY.

        Example:
            OPENFACTOR_FINNHUB_API_KEY=... lets FinnhubClient() work.
        """
        self.api_key = api_key or os.getenv("OPENFACTOR_FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("OPENFACTOR_FINNHUB_API_KEY is required")
        self.timeout = timeout

    def get(self, path, params=None):
        """GET one Finnhub path.

        Example:
            get("stock/financials-reported", {"symbol": "AAPL", "freq": "quarterly"})
            returns reported financial filings.
        """
        params = dict(params or {})
        params["token"] = self.api_key
        return request_json(f"{BASE_URL}/{path.lstrip('/')}", params, self.timeout)


def request_json(url, params, timeout):
    """Return one Finnhub JSON response.

    Example:
        request_json(url, {"symbol": "AAPL"}, 30) returns one response dict.
    """
    for attempt in range(4):
        wait_for_rate_slot()
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as error:
            raise RuntimeError(
                f"Finnhub request failed: {type(error).__name__} {clean_url(url)}"
            ) from None

        if response.status_code == 429 and attempt < 3:
            time.sleep(reset_seconds(response))
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(
                f"Finnhub request failed: HTTP {response.status_code} {clean_url(url)}"
            ) from None
        return response.json()

    raise RuntimeError(f"Finnhub request failed: HTTP 429 {clean_url(url)}")


def wait_for_rate_slot():
    """Wait so Finnhub gets one request per second.

    Example:
        three calls run at least one second apart across all threads.
    """
    global NEXT_REQUEST_AT
    with RATE_LOCK:
        now = time.monotonic()
        wait = max(0.0, NEXT_REQUEST_AT - now)
        NEXT_REQUEST_AT = max(now, NEXT_REQUEST_AT) + SECONDS_BETWEEN_REQUESTS
    if wait:
        time.sleep(wait)


def reset_seconds(response):
    """Return seconds until Finnhub's rate-limit reset.

    Example:
        an X-Ratelimit-Reset epoch two seconds away returns about 3 seconds.
    """
    reset = response.headers.get("X-Ratelimit-Reset")
    if not reset:
        return 5.0
    try:
        return max(1.0, float(reset) - time.time() + 1.0)
    except ValueError:
        return 5.0


def clean_url(url):
    """Return a provider URL without query secrets.

    Example:
        clean_url("https://x/a?token=secret") returns "https://x/a".
    """
    return str(url).split("?", 1)[0]
