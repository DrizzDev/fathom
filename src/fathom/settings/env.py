import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fathom.constants.llm import (
    DEFAULT_PRIORITY_FAILURE_THRESHOLD,
    DEFAULT_PRIORITY_LATENCY_THRESHOLD,
    DEFAULT_PRIORITY_RECOVERY_SUCCESSES,
    DEFAULT_PRIORITY_SLOW_THRESHOLD,
    DEFAULT_PRIORITY_WINDOW,
)
from fathom.constants.storage import (
    INTERACTION_DEFAULT_BACKEND,
    INTERACTION_POSTGRES_DEFAULT_MIGRATION_MODE,
    INTERACTION_POSTGRES_DEFAULT_POOL_MAX_SIZE,
    INTERACTION_POSTGRES_DEFAULT_POOL_MIN_SIZE,
    INTERACTION_POSTGRES_DEFAULT_PORT,
    INTERACTION_POSTGRES_DEFAULT_SCHEMA,
    INTERACTION_POSTGRES_DEFAULT_SSL,
    INTERACTION_POSTGRES_DEFAULT_STATEMENT_TIMEOUT,
    InteractionBackend,
    PostgresMigrationMode,
    PostgresSslMode,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class FathomSettings(BaseSettings):
    """
    Global settings loaded from environment variables and the .env file.
    """

    # Gemini settings
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")

    capacity_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "FATHOM_LLM_PRIORITY_ENABLED",
            "DRIZZ_FATHOM_LLM_PRIORITY_ENABLED",
        ),
    )
    capacity_mode: Literal["always", "adaptive"] = Field(
        default="always",
        validation_alias=AliasChoices(
            "FATHOM_LLM_PRIORITY_MODE",
            "DRIZZ_FATHOM_LLM_PRIORITY_MODE",
        ),
    )
    capacity_window: int = Field(
        default=DEFAULT_PRIORITY_WINDOW,
        validation_alias=AliasChoices(
            "FATHOM_LLM_PRIORITY_WINDOW",
            "DRIZZ_FATHOM_LLM_PRIORITY_WINDOW",
        ),
    )
    capacity_failures: int = Field(
        default=DEFAULT_PRIORITY_FAILURE_THRESHOLD,
        validation_alias=AliasChoices(
            "FATHOM_LLM_PRIORITY_FAILURE_THRESHOLD",
            "DRIZZ_FATHOM_LLM_PRIORITY_FAILURE_THRESHOLD",
        ),
    )
    capacity_slows: int = Field(
        default=DEFAULT_PRIORITY_SLOW_THRESHOLD,
        validation_alias=AliasChoices(
            "FATHOM_LLM_PRIORITY_SLOW_THRESHOLD",
            "DRIZZ_FATHOM_LLM_PRIORITY_SLOW_THRESHOLD",
        ),
    )
    capacity_latency: float = Field(
        default=DEFAULT_PRIORITY_LATENCY_THRESHOLD,
        validation_alias=AliasChoices(
            "FATHOM_LLM_PRIORITY_LATENCY_THRESHOLD",
            "DRIZZ_FATHOM_LLM_PRIORITY_LATENCY_THRESHOLD",
        ),
    )
    capacity_recovery: int = Field(
        default=DEFAULT_PRIORITY_RECOVERY_SUCCESSES,
        validation_alias=AliasChoices(
            "FATHOM_LLM_PRIORITY_RECOVERY_SUCCESSES",
            "DRIZZ_FATHOM_LLM_PRIORITY_RECOVERY_SUCCESSES",
        ),
    )

    vertex_location: str = Field(default="global", alias="VERTEX_LOCATION")
    vertex_project_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"),
    )

    google_application_credentials: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS_PATH"
        ),
    )

    # Device settings
    adb_path: str = Field(default="adb", alias="ADB_PATH")
    android_serial: Optional[str] = Field(default=None, alias="ANDROID_SERIAL")

    # Logging settings
    log_json: bool = Field(default=False, alias="LOG_JSON")
    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")

    # Workflow default limits
    max_steps: int = Field(default=100, alias="MAX_STEPS")

    # Temporal settings (used by TemporalSignalAdapter in interactive mode)
    temporal_host: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("TEMPORAL_HOST", "TEMPORAL_TARGET_HOST"),
    )
    temporal_api_key: Optional[str] = Field(default=None, alias="TEMPORAL_API_KEY")

    # Redis URL for telemetry streaming
    redis_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("REDIS_URL", "GATEWAY_REDIS_URL"),
    )

    # Google service account credentials (dict or JSON string from env var)
    google_credentials_json: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="GOOGLE_APPLICATION_CREDENTIALS_JSON",
    )

    # Assets path
    assets_path: Path = Field(default=PROJECT_ROOT / "assets", alias="FATHOM_ASSETS_PATH")

    # Interaction storage tunables
    interaction_backend: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FATHOM_INTERACTION_BACKEND",
            "DRIZZ_FATHOM_INTERACTION_BACKEND",
        ),
    )
    interaction_postgres_dsn: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_DSN",
            "FATHOM_INTERACTION_POSTGRES_DSN",
        ),
    )
    interaction_postgres_host: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_HOST",
            "FATHOM_INTERACTION_POSTGRES_HOST",
        ),
    )
    interaction_postgres_port: int = Field(
        default=INTERACTION_POSTGRES_DEFAULT_PORT,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_PORT",
            "FATHOM_INTERACTION_POSTGRES_PORT",
        ),
    )
    interaction_postgres_user: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_USER",
            "FATHOM_INTERACTION_POSTGRES_USER",
        ),
    )
    interaction_postgres_password: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_PASSWORD",
            "FATHOM_INTERACTION_POSTGRES_PASSWORD",
        ),
    )
    interaction_postgres_database: str = Field(
        default="fathom",
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_DATABASE",
            "FATHOM_INTERACTION_POSTGRES_DATABASE",
        ),
    )
    interaction_postgres_schema: str = Field(
        default=INTERACTION_POSTGRES_DEFAULT_SCHEMA,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_SCHEMA",
            "FATHOM_INTERACTION_POSTGRES_SCHEMA",
        ),
    )
    interaction_postgres_pool_min_size: int = Field(
        default=INTERACTION_POSTGRES_DEFAULT_POOL_MIN_SIZE,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_POOL_MIN_SIZE",
            "FATHOM_INTERACTION_POSTGRES_POOL_MIN_SIZE",
        ),
    )
    interaction_postgres_pool_max_size: int = Field(
        default=INTERACTION_POSTGRES_DEFAULT_POOL_MAX_SIZE,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_POOL_MAX_SIZE",
            "FATHOM_INTERACTION_POSTGRES_POOL_MAX_SIZE",
        ),
    )
    interaction_postgres_statement_timeout: int = Field(
        default=INTERACTION_POSTGRES_DEFAULT_STATEMENT_TIMEOUT,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_STATEMENT_TIMEOUT",
            "FATHOM_INTERACTION_POSTGRES_STATEMENT_TIMEOUT",
        ),
    )
    interaction_postgres_ssl: PostgresSslMode = Field(
        default=INTERACTION_POSTGRES_DEFAULT_SSL,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_SSL",
            "FATHOM_INTERACTION_POSTGRES_SSL",
        ),
    )
    interaction_postgres_migration_mode: PostgresMigrationMode = Field(
        default=INTERACTION_POSTGRES_DEFAULT_MIGRATION_MODE,
        validation_alias=AliasChoices(
            "DRIZZ_FATHOM_POSTGRES_MIGRATION_MODE",
            "FATHOM_INTERACTION_POSTGRES_MIGRATION_MODE",
        ),
    )

    # CLI principal fallbacks
    cli_tenant: Optional[str] = Field(default=None, alias="FATHOM_CLI_TENANT")
    cli_operator: Optional[str] = Field(default=None, alias="FATHOM_CLI_OPERATOR")
    cli_workspace: Optional[str] = Field(default=None, alias="FATHOM_CLI_WORKSPACE")

    # Per-workflow run-log directory (mirror of the structured log stream written when --log-file is passed on the CLI).
    # Lives under assets/ so artifact retention and cleanup share a single root.
    run_logs_path: Path = Field(
        alias="FATHOM_RUN_LOGS_PATH",
        default=PROJECT_ROOT / "assets" / "logs",
    )

    # Perception subsystem toggles. CV stays off by default because its anonymous visual-control boxes are still
    # too noisy for production grounding and can pollute both prompts and debug artifacts.
    oracle_enabled: bool = Field(default=True, alias="FATHOM_ORACLE_ENABLED")

    observation_ocr_enabled: bool = Field(default=True, alias="FATHOM_OBSERVATION_OCR")
    observation_cv_enabled: bool = Field(default=False, alias="FATHOM_OBSERVATION_CV")
    observation_icon_enabled: bool = Field(default=True, alias="FATHOM_OBSERVATION_ICON")
    observation_overlay_enabled: bool = Field(default=True, alias="FATHOM_OBSERVATION_OVERLAY")
    observation_keyboard_enabled: bool = Field(default=False, alias="FATHOM_OBSERVATION_KEYBOARD")

    # Ensemble vision-localizer (Gemini-vision + DocumentAI-layout) toggles. Enabled with both members so the
    # supervise cascade can fall back to name-based localization when the XML manifest snap fails or returns a
    # generic container.
    ensemble_localizer_enabled: bool = Field(default=True, alias="FATHOM_ENSEMBLE_LOCALIZER")
    ensemble_localizer_members: Optional[str] = Field(
        default="gemini_vision,document_ai_layout",
        alias="FATHOM_ENSEMBLE_LOCALIZER_MEMBERS",
    )

    # Runtime journal adapter (local JSONL) toggle.
    journal_local_enabled: bool = Field(default=False, alias="FATHOM_JOURNAL_LOCAL")

    # Document AI OCR provider credentials. Required only when observation_ocr_enabled is True and Document AI is
    # the active provider.
    document_ai_project: Optional[str] = Field(default=None, alias="FATHOM_DOCUMENT_AI_PROJECT")
    document_ai_location: Optional[str] = Field(default=None, alias="FATHOM_DOCUMENT_AI_LOCATION")
    document_ai_processor: Optional[str] = Field(default=None, alias="FATHOM_DOCUMENT_AI_PROCESSOR")

    @field_validator("google_credentials_json", mode="before")
    @classmethod
    def parse_google_credentials(cls, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, str):
            return cast("Dict[str, Any]", json.loads(value))

        return cast("Optional[Dict[str, Any]]", value)

    @property
    def google_credentials_dict(self) -> Optional[Dict[str, Any]]:
        return self.google_credentials_json

    @property
    def resolved_interaction_backend(self) -> str:
        """
        Resolve the durable interaction backend from explicit or detected config.
        """

        if self.interaction_backend:
            return self.interaction_backend

        if self.interaction_postgres_dsn:
            return InteractionBackend.POSTGRES.value

        if all(
            (
                self.interaction_postgres_host,
                self.interaction_postgres_user,
                self.interaction_postgres_password,
            )
        ):
            return InteractionBackend.POSTGRES.value

        return INTERACTION_DEFAULT_BACKEND.value

    # Environment file support
    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
        env_file_encoding="utf-8",
        env_file=[".env", str(PROJECT_ROOT / ".env")],
    )

    @field_validator("google_application_credentials")
    @classmethod
    def resolve_credentials_path(cls, value: Optional[str]) -> Optional[str]:
        """
        Resolve credentials path relative to project root.
        """

        if not value:
            return None

        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return str(path)
