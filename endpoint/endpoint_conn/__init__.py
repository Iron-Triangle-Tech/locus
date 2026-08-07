"""Endpoint-side connection glue to core over the WS link."""

from __future__ import annotations

from .client import AdhocTool, AuthError, EndpointClient

__all__ = ["AdhocTool", "AuthError", "EndpointClient"]