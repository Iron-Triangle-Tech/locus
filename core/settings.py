"""Core service settings.

Loads from ``core/config.toml`` with ``LOCUS_CORE_*`` env overrides (``__`` delimiter).
Secrets (link token, API keys) via env only, not in config.toml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.sources import TomlConfigSettingsSource

from shared.paths import db_path, ensure_data_dir, workspace_path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "config.toml"

# Built-in provider names always available (no [providers.<name>] entry needed).
BUILTIN_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai", "gemini"})


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7100
    ws_idle_timeout: float = 300.0


class StorageSettings(BaseModel):
    # Default lives under the per-user ~/Locus data dir, so the DB survives
    # across CWDs and isn't dropped in whatever directory launched the server.
    # Tests override with ":memory:" or an explicit path (init arg beats TOML).
    sqlite_path: str = str(db_path())


class ProviderSettings(BaseModel):
    """Top-level provider routing + built-in per-provider model ids."""

    default: str = "anthropic"
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o",
            "gemini": "gemini-1.5-flash",
        }
    )


class NamedProviderSettings(BaseModel):
    """A single configured provider entry under [providers.<name>].

    Currently only OpenAI-compatible gateways are supported via the
    ``openai_compat`` adapter; built-ins (anthropic/openai/gemini) don't need
    an entry here.
    """

    type: Literal["openai_compat"] = "openai_compat"
    base_url: str
    api_key: str = ""
    model: str = ""


class ToolsSettings(BaseModel):
    # Sandbox root under ~/Locus/workspace; created by ensure_data_dir().
    agent_root: str = str(workspace_path())
    http_max_bytes: int = 1048576
    http_timeout: float = 20.0


class CoreSettings(BaseSettings):
    """Aggregated core settings.

    Precedence (highest first): env (LOCUS_CORE_*) -> config.toml -> field
    defaults on these models.
    """

    model_config = SettingsConfigDict(
        env_prefix="LOCUS_CORE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    providers: dict[str, NamedProviderSettings] = Field(default_factory=dict)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)

    # Secret: bearer token for the core<->endpoint link. NOT in config.toml.
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
        # Order = precedence (first wins): env -> explicit constructor args
        # -> TOML file -> field defaults. Putting ``init_settings`` ahead of the
        # TOML source means tests that pass e.g. ``storage=StorageSettings(
        # sqlite_path='...')`` actually win over the committed config.toml
        # (which would otherwise force every test onto the shared ``locus.db``).
        toml_path = Path(DEFAULT_CONFIG_PATH)
        toml_source = TomlConfigSettingsSource(settings_cls, toml_path)
        return (env_settings, init_settings, toml_source)

    def resolved_model(self, provider: str) -> str:
        """Resolve a concrete model id for a provider name.

        Falls back through: named-provider's own model -> built-in models map ->
        empty string (caller must error).
        """
        if provider == "auto":
            provider = self.provider.default
        named = self.providers.get(provider)
        if named and named.model:
            return named.model
        return self.provider.models.get(provider, "")

    def is_known_provider(self, provider: str) -> bool:
        """True if ``provider`` is a built-in or a configured named provider."""
        if provider == "auto":
            return True
        return provider in BUILTIN_PROVIDERS or provider in self.providers


def get_settings() -> CoreSettings:
    """Construct the CoreSettings, loading from config.toml + env.

    A fresh instance per call so tests can monkeypatch env between cases.
    First-run bootstrap: make sure the per-user ~/Locus data dir + workspace
    exist before building settings, so the file-tool sandbox and SQLite DB
    have a home even on a fresh install. Idempotent; near-free after first call.
    """
    ensure_data_dir()
    return CoreSettings()


__all__ = [
    "BUILTIN_PROVIDERS",
    "CoreSettings",
    "NamedProviderSettings",
    "ProviderSettings",
    "ServerSettings",
    "StorageSettings",
    "ToolsSettings",
    "get_settings",
]
