"""Endpoint service settings.

Loads from ``endpoint/config.toml`` with ``LOCUS_ENDPOINT_*`` env overrides
(``__`` delimiter). Secrets (the shared bearer token) via env only, not in
config.toml -- same convention as core.

The endpoint is deliberately a thin client: it only needs to know where core is
(``core_url``), the shared bearer token, and a couple of UI knobs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.sources import TomlConfigSettingsSource

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "config.toml"


class ServerSettings(BaseModel):
    """Local listen options. The endpoint is CLI-driven and does not serve
    anything today, but we keep a stub so ``-o`` templates line up and a
    future webui can reuse the same settings object."""

    host: str = "127.0.0.1"
    port: int = 7101


class CoreSettings(BaseModel):
    """How to reach core + the WS link's timeouts.

    ``core_url`` is the full WS URL (e.g. ``ws://localhost:7100/link`` or
    ``wss://core.example.com/link``). The endpoint authenticates the upgrade
    by carrying the shared bearer token in the ``Authorization`` header.

    ``proxy`` controls outbound proxy use for the core link. ``None`` (default)
    means never proxy -- a service-to-service link to a known host should not be
    silently routed through a dev machine's ``ALL_PROXY``/``HTTPS_PROXY`` env
    vars. Set it to a URL string to pin a proxy, or ``True`` to honor env.
    """

    url: str = "ws://localhost:7100/link"
    # Seconds to wait for the first Connect ack / a frame before giving up.
    connect_timeout: float = 10.0
    idle_timeout: float = 300.0
    proxy: str | bool | None = None


class UISettings(BaseModel):
    """REPL rendering toggles. Kept in config so a user can flip them without
    editing code."""

    show_tool_args: bool = True
    # Stream assistant tokens inline as they arrive (vs. buffering to final).
    stream_tokens: bool = True


class EndpointSettings(BaseSettings):
    """Aggregated endpoint settings.

    Precedence (highest first): env (``LOCUS_ENDPOINT_*``) -> explicit
    constructor args -> ``endpoint/config.toml`` -> field defaults. Same
    ordering rationale as :class:`core.settings.CoreSettings`: tests pass
    constructor args that must beat the committed TOML.
    """

    model_config = SettingsConfigDict(
        env_prefix="LOCUS_ENDPOINT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    core: CoreSettings = Field(default_factory=CoreSettings)
    ui: UISettings = Field(default_factory=UISettings)

    # Secret: bearer token shared with core for the WS link. NOT in config.toml.
    link_token: str = ""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # env -> constructor args -> TOML -> field defaults.
        toml_path = Path(DEFAULT_CONFIG_PATH)
        toml_source = TomlConfigSettingsSource(settings_cls, toml_path)
        return (env_settings, init_settings, toml_source)


def get_settings() -> EndpointSettings:
    """Construct EndpointSettings, loading from config.toml + env.

    A fresh instance per call so tests can monkeypatch env between cases.
    """
    return EndpointSettings()


__all__ = [
    "CoreSettings",
    "EndpointSettings",
    "ServerSettings",
    "UISettings",
    "get_settings",
]
