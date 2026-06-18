import os

import requests


BASE_URL = "https://financialmodelingprep.com/stable"


class FmpClient:
    """Tiny Financial Modeling Prep JSON client.

    Example:
        FmpClient(api_key="key").get("analyst-estimates", {"symbol": "AAPL"})
        returns analyst estimate JSON.
    """

    def __init__(self, api_key=None, timeout=30):
        """Create a client from a key or OPENFACTOR_FMP_API_KEY.

        Example:
            OPENFACTOR_FMP_API_KEY=... lets FmpClient() work.
        """
        self.api_key = api_key or os.getenv("OPENFACTOR_FMP_API_KEY") or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("OPENFACTOR_FMP_API_KEY is required")
        self.timeout = timeout

    def get(self, path, params=None):
        """GET one FMP path.

        Example:
            get("analyst-estimates", {"symbol": "AAPL"}) returns estimate rows.
        """
        params = dict(params or {})
        params["apikey"] = self.api_key
        try:
            response = requests.get(
                f"{BASE_URL}/{path.lstrip('/')}",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"FMP request failed: {type(error).__name__} {path}") from None
        return response.json()
