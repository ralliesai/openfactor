import os
import time

import requests


class MassiveClient:
    """Small Massive JSON client.

    Example:
        client = MassiveClient(api_key="...")
        data = client.get("/v3/reference/tickers", {"limit": 1})
    """

    def __init__(self, api_key=None, base_url="https://api.massive.com"):
        """Create a client from an explicit key or OPENFACTOR_MASSIVE_API_KEY.

        Example:
            client = MassiveClient(api_key="...")
        """
        self.api_key = api_key or os.getenv("OPENFACTOR_MASSIVE_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENFACTOR_MASSIVE_API_KEY or pass api_key.")

        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get(self, path, params=None):
        """GET one Massive API path.

        Example:
            client.get("/v3/reference/tickers", {"limit": 1})
            returns one JSON response dict.
        """
        params = dict(params or {})
        params["apiKey"] = self.api_key
        return self.request_json(f"{self.base_url}{path}", params)

    def get_url(self, url):
        """GET a full URL, usually from next_url pagination.

        Example:
            page = client.get_url(next_url)
        """
        params = None if "apiKey=" in url else {"apiKey": self.api_key}
        return self.request_json(url, params)

    def request_json(self, url, params=None):
        """GET JSON and honor server rate-limit waits.

        Example:
            if Massive returns 429 with Retry-After=2, wait 2 seconds before asking again.
        """
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 429 and response.headers.get("Retry-After"):
                time.sleep(float(response.headers["Retry-After"]))
                response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                f"Massive request failed: {type(error).__name__} {clean_url(url)}"
            ) from None
        return response.json()

    def close(self):
        """Close the underlying HTTP session.

        Example:
            client.close()
        """
        self.session.close()

    def pages(self, path, params=None):
        """Yield every page for an endpoint with next_url pagination.

        Example:
            for page in client.pages("/v3/reference/tickers", {"limit": 1000}):
                rows = page["results"]
        """
        data = self.get(path, params)
        while True:
            yield data
            next_url = data.get("next_url")
            if not next_url:
                return
            data = self.get_url(next_url)


def clean_url(url):
    """Return a provider URL without query secrets.

    Example:
        clean_url("https://x/a?apiKey=secret") returns "https://x/a".
    """
    return str(url).split("?", 1)[0]
