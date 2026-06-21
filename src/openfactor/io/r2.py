import datetime as dt
import hashlib
import hmac
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


class R2Client:
    """Tiny Cloudflare R2 client.

    Example:
        R2Client.from_env().upload_text("a,b\n1,2\n", "openfactor-public", "x.csv")
        stores one public-bucket object.
    """

    def __init__(self, account_id=None, access_key_id=None, secret_access_key=None):
        self.account_id = account_id or require_env("OPENFACTOR_R2_ACCOUNT_ID")
        self.access_key_id = access_key_id or require_env("OPENFACTOR_R2_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or require_env("OPENFACTOR_R2_SECRET_ACCESS_KEY")
        self.host = f"{self.account_id}.r2.cloudflarestorage.com"

    @classmethod
    def from_env(cls):
        """Build a client from OPENFACTOR_R2_* environment variables.

        Example:
            R2Client.from_env().list_keys("openfactor-public", "factors")
            lists object keys.
        """
        return cls()

    def upload_text(self, text, bucket, key, content_type="text/plain; charset=utf-8"):
        """Upload one text object.

        Example:
            upload_text("ticker\nAAPL\n", "openfactor-public", "semantic_factors.csv")
            writes a CSV object.
        """
        return self.request("PUT", bucket, key, text.encode("utf-8"), content_type)

    def upload_bytes(self, data, bucket, key, content_type="application/octet-stream"):
        """Upload one binary object.

        Example:
            upload_bytes(gzipped, "openfactor-public", "panel.csv.gz", "application/gzip")
            writes a compressed object.
        """
        return self.request("PUT", bucket, key, data, content_type)

    def read_text(self, bucket, key):
        """Read one private R2 text object with signed credentials.

        Example:
            read_text("openfactor-public", "semantic_factors.csv") returns CSV text.
        """
        return self.request("GET", bucket, key, return_body=True).decode("utf-8")

    def list_keys(self, bucket, prefix=""):
        """Return object keys under one prefix.

        Example:
            list_keys("openfactor-public", "factors/openfactor-us1000")
            returns uploaded factor files.
        """
        keys = []
        token = None
        while True:
            body = self.request("GET", bucket, query=list_query(prefix, token), return_body=True)
            root = ET.fromstring(body)
            keys.extend(item.text for item in root.findall(".//{*}Key"))
            token = text_or_none(root.find(".//{*}NextContinuationToken"))
            if text_or_none(root.find(".//{*}IsTruncated")) != "true":
                return sorted(keys)

    def delete_key(self, bucket, key):
        """Delete one object.

        Example:
            delete_key("openfactor-public", "factors/x/latest.json") removes it.
        """
        return self.request("DELETE", bucket, key)

    def delete_prefix(self, bucket, prefix=""):
        """Delete every object under one prefix.

        Example:
            delete_prefix("openfactor-public", "factors/openfactor-us1000")
            removes all matching objects.
        """
        keys = self.list_keys(bucket, prefix)
        for key in keys:
            self.delete_key(bucket, key)
        return keys

    def request(self, method, bucket, key="", body=b"", content_type=None, query="", return_body=False):
        """Send one signed S3-compatible R2 request.

        Example:
            request("PUT", "openfactor-public", "hello.txt", b"hello")
            stores hello.txt.
        """
        path = f"/{quote(bucket)}"
        if key:
            path = f"{path}/{quote(key)}"
        now = dt.datetime.utcnow()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        datestamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {
            "host": self.host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        if content_type:
            headers["content-type"] = content_type

        signed_headers = ";".join(sorted(headers))
        canonical_headers = "".join(f"{name}:{headers[name]}\n" for name in sorted(headers))
        canonical = "\n".join([method, path, query, canonical_headers, signed_headers, payload_hash])
        scope = f"{datestamp}/auto/s3/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                scope,
                hashlib.sha256(canonical.encode()).hexdigest(),
            ]
        )
        signature = hmac.new(self.signing_key(datestamp), string_to_sign.encode(), hashlib.sha256).hexdigest()
        headers["authorization"] = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self.access_key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        url = f"https://{self.host}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(url, data=body or None, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read() if return_body else response.status

    def signing_key(self, datestamp):
        """Return the SigV4 signing key for one date.

        Example:
            signing_key("20260616") returns bytes used to sign R2 requests.
        """
        key = hmac.new(("AWS4" + self.secret_access_key).encode(), datestamp.encode(), hashlib.sha256).digest()
        for value in ["auto", "s3", "aws4_request"]:
            key = hmac.new(key, value.encode(), hashlib.sha256).digest()
        return key


def list_query(prefix, token=None):
    """Return a list-objects query string.

    Example:
        list_query("factors") returns "list-type=2&prefix=factors".
    """
    params = {"list-type": "2", "prefix": prefix}
    if token:
        params["continuation-token"] = token
    return urllib.parse.urlencode(sorted(params.items()))


def text_or_none(node):
    """Return XML node text or None.

    Example:
        text_or_none(None) returns None.
    """
    return None if node is None else node.text


def quote(value):
    """URL-quote a bucket or object path.

    Example:
        quote("a/b c") returns "a/b%20c".
    """
    return urllib.parse.quote(str(value), safe="/-_.~")


def require_env(name):
    """Return an environment variable or raise.

    Example:
        require_env("OPENFACTOR_R2_ACCOUNT_ID") returns the account id.
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value
