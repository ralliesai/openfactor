import os

import requests


BASE_URL = "https://api.tipranks.com/api"


class TipRanksClient:
    """Tiny TipRanks JSON client.

    Example:
        TipRanksClient(api_key="key", api_token="token").get("analysts/aapl", {"num": 50})
        returns analyst rating JSON.
    """

    def __init__(self, api_key=None, api_token=None, timeout=30):
        """Create a client from explicit keys or TipRanks env vars.

        Example:
            OPENFACTOR_TIPRANKS_API_KEY=... lets TipRanksClient() work.
        """
        self.api_key = api_key or os.getenv("OPENFACTOR_TIPRANKS_API_KEY") or os.getenv("TIPRANKS_API_KEY")
        self.api_token = api_token or os.getenv("OPENFACTOR_TIPRANKS_API_TOKEN") or os.getenv("TIPRANKS_API_TOKEN")
        if not self.api_key or not self.api_token:
            raise ValueError("OPENFACTOR_TIPRANKS_API_KEY and OPENFACTOR_TIPRANKS_API_TOKEN are required")
        self.timeout = timeout

    def get(self, path, params=None):
        """GET one TipRanks path.

        Example:
            get("analysts/aapl", {"num": 50}) returns recent analyst ratings.
        """
        headers = {"X-APIKey": self.api_key, "X-APIToken": self.api_token}
        try:
            response = requests.get(
                f"{BASE_URL}/{path.lstrip('/')}",
                params=params or {},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"TipRanks request failed: {type(error).__name__} {path}") from None
        return response.json()
