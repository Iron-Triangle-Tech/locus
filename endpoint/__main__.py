"""``locus-endpoint`` console entry point.

Connect to a core server and drop into the stdin/stdout REPL. A handful of
overrides come from the command line; everything else comes from
``endpoint/config.toml`` + ``LOCUS_ENDPOINT_*`` env (see :mod:`endpoint.settings`).

Usage::

    locus-endpoint --core ws://localhost:7100/link --token my-secret

If ``--token`` is omitted, the shared bearer token must come from the
``LOCUS_ENDPOINT_LINK_TOKEN`` env var (secrets are intentionally not committed
to ``config.toml``). In real deployments core and endpoint must share the same
token.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Sequence

from endpoint.settings import EndpointSettings, get_settings


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="locus-endpoint",
        description="Connect to a Locus core server and enter the REPL.",
    )
    p.add_argument(
        "--core", dest="core_url",
        help="WS URL of core's /link endpoint (default: endpoint/config.toml [core].url)",
    )
    p.add_argument(
        "--token", dest="link_token",
        help="Shared bearer token for the core<->endpoint link "
             "(default: LOCUS_ENDPOINT_LINK_TOKEN env; not stored in config.toml)",
    )
    p.add_argument(
        "--endpoint-id", dest="endpoint_id",
        help="Identifier announced in the endpoint's Connect frame (default: endpoint-1)",
    )
    p.add_argument(
        "--no-stream", dest="stream_tokens", action="store_false",
        help="Buffer to final instead of streaming tokens inline",
    )
    return p


def _apply_overrides(
    settings: EndpointSettings, args: argparse.Namespace
) -> EndpointSettings:
    """Return a copy of ``settings`` with any CLI overrides applied.

    ``model_copy(update=...)`` keeps nested pydantic models immutable: we
    reconstruct only the sub-models whose fields actually changed so a one-off
    CLI flag beats both TOML and env without surprise.
    """
    update: dict[str, Any] = {}

    core_update: dict[str, Any] = {}
    if args.core_url:
        core_update["url"] = args.core_url
    if core_update:
        update["core"] = settings.core.model_copy(update=core_update)

    ui_update: dict[str, Any] = {}
    if args.stream_tokens is False:  # argparse stores False on --no-stream
        ui_update["stream_tokens"] = False
    if ui_update:
        update["ui"] = settings.ui.model_copy(update=ui_update)

    if args.link_token is not None:
        update["link_token"] = args.link_token

    if not update:
        return settings
    return settings.model_copy(update=update)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse args, build settings, run the REPL."""
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    import asyncio

    from endpoint.ui.repl import run_repl

    settings = _apply_overrides(get_settings(), args)
    return asyncio.run(run_repl(settings, endpoint_id=args.endpoint_id))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
