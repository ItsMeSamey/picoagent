"""SearXNG JSON search over an operator-configured HTTP(S) endpoint."""
from __future__ import annotations

import json
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("search endpoint redirects are disabled")


class SearXNGSearch:
    def __init__(self, endpoint: str, *, timeout: float = 10, max_response_bytes: int = 1_048_576, opener=None):
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("search endpoint must be an HTTP(S) URL without credentials or fragment")
        if timeout <= 0 or max_response_bytes <= 0:
            raise ValueError("search limits must be positive")
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self._opener = opener or build_opener(ProxyHandler({}), _NoRedirects())

    def search(self, query: str, limit: int = 5) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must be 1-2000 characters")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10:
            raise ValueError("limit must be an integer from 1 to 10")
        parts = urlsplit(self.endpoint)
        parameters = [(key, value) for key, value in parse_qsl(parts.query) if key not in {"q", "format"}]
        parameters.extend([("q", query), ("format", "json")])
        url = urlunsplit((parts.scheme, parts.netloc, parts.path or "/search", urlencode(parameters), ""))
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "picoagent/0.1"})
        with self._opener.open(request, timeout=self.timeout) as response:
            raw = response.read(self.max_response_bytes + 1)
        if len(raw) > self.max_response_bytes:
            raise ValueError("search response exceeds byte limit")
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
            raise ValueError("invalid SearXNG response")
        results = []
        for row in data.get("results", [])[:limit]:
            if not isinstance(row, dict):
                continue
            results.append({key: str(row.get(key, ""))[:4000] for key in ("title", "url", "content")})
        return {"query": query, "results": results, "source": "searxng", "untrusted": True}
