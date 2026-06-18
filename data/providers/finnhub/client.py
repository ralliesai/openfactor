import os

import requests


BASE_URL = "https://finnhub.io/api/v1"


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
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError:
        raise RuntimeError(
            f"Finnhub request failed: HTTP {response.status_code} {clean_url(url)}"
        ) from None
    except requests.RequestException as error:
        raise RuntimeError(
            f"Finnhub request failed: {type(error).__name__} {clean_url(url)}"
        ) from None
    return response.json()


def clean_url(url):
    """Return a provider URL without query secrets.

    Example:
        clean_url("https://x/a?token=secret") returns "https://x/a".
    """
    return str(url).split("?", 1)[0]
