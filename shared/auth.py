"""Bearer-token auth helper shared by core and endpoint.

The core<->endpoint WebSocket link is protected by a single shared token kept
in each service's config. Core validates incoming tokens; endpoint sends its
token. Keeping the helper here avoids either side reaching into the other.
"""

from __future__ import annotations

from collections.abc import Mapping

#: HTTP header name carrying the bearer token on the WS handshake.
AUTH_HEADER = "Authorization"


def bearer_header(token: str) -> dict[str, str]:
    """Build the headers an endpoint sends when opening the link."""
    return {AUTH_HEADER: f"Bearer {token}"}


def read_bearer(headers: Mapping[str, str]) -> str | None:
    """Extract a bearer token from request headers, case-insensitively.

    Returns ``None`` if the header is absent or malformed.
    """
    for key, value in headers.items():
        if key.lower() == AUTH_HEADER.lower():
            value = value.strip()
            if value.lower().startswith("bearer "):
                return value[7:].strip() or None
            return None
    return None


def check_token(headers: Mapping[str, str], expected: str) -> bool:
    """True iff the request carries the expected bearer token.

    Uses :func:`hmac.compare_digest`-style constant-time comparison when the
    lengths match; mismatched lengths return False without leaking timing.
    """
    from secrets import compare_digest

    presented = read_bearer(headers)
    if presented is None:
        return False
    return compare_digest(presented.encode(), expected.encode())


__all__ = ["AUTH_HEADER", "bearer_header", "check_token", "read_bearer"]
