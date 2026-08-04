"""Built-in HTTP fetch tool (``http_fetch``).

A single :class:`core.tools.registry.Tool` wrapping :mod:`httpx`. The model
can request a URL with an allow-listed method, optional headers/body, and an
optional timeout capped by configuration
"""

from __future__ import annotations

import httpx

from core.tools.registry import Tool, ToolResult

__all__ = ["HttpFetch"]


_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"})


_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Absolute HTTP(S) URL to fetch."},
        "method": {
            "type": "string",
            "description": "HTTP method. One of GET/POST/PUT/PATCH/DELETE/HEAD.",
            "default": "GET",
        },
        "headers": {
            "type": "object",
            "description": "Extra request headers.",
            "additionalProperties": {"type": "string"},
        },
        "body": {"type": "string", "description": "Request body (for methods that take one)."},
        "timeout": {
            "type": "number",
            "description": "Per-request timeout in seconds, capped by the server.",
        },
    },
    "required": ["url"],
}


class HttpFetch(Tool):
    """``http_fetch`` -- single-shot HTTP request returning the response body.

    Construct with the server-side caps from :class:`ToolsSettings`
    (``max_bytes``, ``default_timeout``). One instance per core app is fine:
    httpx reuses async transport lazily on each request.
    """

    name = "http_fetch"
    
    parameters = _SCHEMA

    def __init__(self, *, max_bytes: int, default_timeout: float) -> None:
        self._max_bytes = int(max_bytes)
        self._default_timeout = float(default_timeout)

    async def run(self, arguments: dict) -> ToolResult:
        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            return ToolResult(content="missing 'url'", is_error=True)
        method = str(arguments.get("method") or "GET").upper()
        if method not in _ALLOWED_METHODS:
            return ToolResult(
                content=f"method not allowed: {method!r}", is_error=True
            )
        headers = arguments.get("headers") or {}
        if not isinstance(headers, dict):
            return ToolResult(content="'headers' must be an object", is_error=True)
        body = arguments.get("body")
        if body is not None and not isinstance(body, str):
            return ToolResult(content="'body' must be a string", is_error=True)

        # Caller timeout is capped at the server default.
        t = arguments.get("timeout")
        if t is None:
            timeout = self._default_timeout
        else:
            try:
                timeout = min(float(t), self._default_timeout)
            except (TypeError, ValueError):
                return ToolResult(content="'timeout' must be a number", is_error=True)

        # Read up to max_bytes+1 so we can detect oversize and marker-truncate.
        cap = self._max_bytes
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.request(
                    method,
                    url,
                    headers={str(k): str(v) for k, v in headers.items()},
                    content=body if method in {"POST", "PUT", "PATCH"} else None,
                )
                raw = await resp.aread()
        except httpx.HTTPError as e:
            return ToolResult(
                content=f"http request failed: {type(e).__name__}: {e}",
                is_error=True,
            )
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(
                content=f"http request failed: {type(e).__name__}: {e}",
                is_error=True,
            )

        truncated = len(raw) > cap
        if truncated:
            body_bytes = raw[:cap]
        else:
            body_bytes = raw
        # Decode as UTF-8 with replacement so binary bodies don't blow up;
        # the model gets best-effort text plus a truncation marker when oversize.
        text = body_bytes.decode("utf-8", errors="replace")

        summary = f"{resp.status_code} {resp.reason_phrase}\n"
        # A few useful headers (case-insensitive), not the whole set.
        for h in ("content-type", "content-length"):
            v = resp.headers.get(h)
            if v is not None:
                summary += f"{h}: {v}\n"
        summary += "---\n"
        out = summary + text
        if truncated:
            out += f"\n…[truncated at {cap} bytes]"
        return ToolResult(content=out, is_error=False)
