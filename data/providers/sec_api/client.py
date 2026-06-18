import os
import time

import pandas as pd
import requests

from data.sec.schema import FILING_COLUMNS


BASE_URL = "https://api.sec-api.io"


class SecApiClient:
    """Small SEC-API.io client.

    Example:
        SecApiClient(api_key="key").filings("AAPL", "2026-06-16")
        returns 10-K and 10-Q metadata available by that date.
    """

    def __init__(self, api_key=None, timeout=30):
        """Create a client from an explicit key or environment variable.

        Example:
            OPENFACTOR_SEC_API_KEY=... lets SecApiClient() work.
        """
        self.api_key = (
            api_key
            or os.getenv("OPENFACTOR_SEC_API_KEY")
            or os.getenv("SEC_API_KEY")
        )
        if not self.api_key:
            raise ValueError("OPENFACTOR_SEC_API_KEY or SEC_API_KEY is required")
        self.timeout = timeout

    def mapping(self, ticker):
        """Return SEC-API ticker mapping rows.

        Example:
            mapping("AAPL") returns CIK, SIC, sector, and industry metadata.
        """
        return request_json(
            "GET",
            f"{BASE_URL}/mapping/ticker/{str(ticker).upper()}",
            headers={"Authorization": self.api_key},
            timeout=self.timeout,
        )

    def filings(
        self,
        ticker,
        as_of_date,
        forms=("10-K", "10-Q"),
        size=50,
        start_date="1900-01-01",
    ):
        """Return filing rows available by one as-of date.

        Example:
            filings("AAPL", "2026-06-16")
            returns recent 10-K and 10-Q rows filed on or before that date.
        """
        form_query = " OR ".join([f'formType:"{form}"' for form in forms])
        payload = {
            "query": (
                f"ticker:{str(ticker).upper()} "
                f"AND filedAt:[{str(start_date)[:10]} TO {str(as_of_date)[:10]}] "
                f"AND ({form_query})"
            ),
            "from": "0",
            "size": str(size),
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        response = request_json(
            "POST",
            f"{BASE_URL}/",
            headers={"Authorization": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        return exact_forms(filing_frame(response.get("filings", [])), forms)

    def filings_between(self, start_date, end_date, forms=("10-K", "10-Q"), size=50):
        """Return filings in one date range.

        Example:
            filings_between("2026-06-16", "2026-06-16")
            returns 10-K/10-Q rows filed that day.
        """
        form_query = " OR ".join([f'formType:"{form}"' for form in forms])
        rows = []
        start = 0
        while True:
            payload = {
                "query": (
                    f"filedAt:[{str(start_date)[:10]} TO {str(end_date)[:10]}] "
                    f"AND ({form_query})"
                ),
                "from": str(start),
                "size": str(size),
                "sort": [{"filedAt": {"order": "desc"}}],
            }
            response = request_json(
                "POST",
                f"{BASE_URL}/",
                headers={"Authorization": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            filings = response.get("filings", [])
            rows.extend(filings)
            if len(filings) < size:
                return exact_forms(filing_frame(rows), forms)
            start += size

    def xbrl(self, accession_no):
        """Return normalized XBRL JSON for one accession number.

        Example:
            xbrl("0000320193-26-000013") returns statement dictionaries.
        """
        return request_json(
            "GET",
            f"{BASE_URL}/xbrl-to-json",
            params={"accession-no": accession_no, "token": self.api_key},
            timeout=max(self.timeout, 60),
        )


def request_json(method, url, **kwargs):
    """Return one SEC-API JSON response.

    Example:
        request_json("GET", url, timeout=30) returns one response dict.
    """
    while True:
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError:
            if response.status_code == 429:
                time.sleep(retry_after_seconds(response))
                continue
            raise RuntimeError(
                f"SEC-API request failed: HTTP {response.status_code} {method} {clean_url(url)}"
            ) from None
        except requests.RequestException as error:
            raise RuntimeError(
                f"SEC-API request failed: {type(error).__name__} {method} {clean_url(url)}"
            ) from None


def retry_after_seconds(response):
    """Return provider rate-limit wait seconds.

    Example:
        Retry-After: 2 waits two seconds before the next request.
    """
    try:
        return max(1.0, float(response.headers.get("Retry-After", 5)))
    except ValueError:
        return 5.0


def clean_url(url):
    """Return a provider URL without query secrets.

    Example:
        clean_url("https://x/a?token=secret") returns "https://x/a".
    """
    return str(url).split("?", 1)[0]


def filing_frame(filings):
    """Return OpenFactor filing rows from SEC-API metadata.

    Example:
        accessionNo becomes accession_no and formType becomes form_type.
    """
    rows = []
    for filing in filings:
        rows.append(
            {
                "ticker": str(filing.get("ticker", "")).upper(),
                "accession_no": filing.get("accessionNo"),
                "form_type": filing.get("formType"),
                "filed_at": str(filing.get("filedAt", ""))[:10],
                "period_of_report": filing.get("periodOfReport"),
                "link_to_html": filing.get("linkToHtml"),
                "link_to_details": filing.get("linkToFilingDetails"),
            }
        )
    return pd.DataFrame(rows, columns=FILING_COLUMNS)


def exact_forms(frame, forms):
    """Keep only exact SEC form types.

    Example:
        forms ("10-K", "10-Q") drops 10-K/A and NT 10-Q rows.
    """
    if frame.empty:
        return frame
    wanted = {str(form) for form in forms}
    return frame[frame["form_type"].isin(wanted)].reset_index(drop=True)
