import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Calculate project root from this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class FathomSettings(BaseSettings):
    """
    Global settings loaded from environment variables and .env file.

    This acts as the central point for accessing all environment configuration.
    """

    # Gemini settings
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3-flash-preview", alias="GEMINI_MODEL")

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

    # Per-workflow run-log directory (mirror of the structured log stream written when --log-file is passed on the CLI).
    # Lives under assets/ so artifact retention and cleanup share a single root.
    run_logs_path: Path = Field(
        alias="FATHOM_RUN_LOGS_PATH",
        default=PROJECT_ROOT / "assets" / "logs",
    )

    # Perception subsystem toggles. CV stays off by default because its
    # anonymous visual-control boxes are still too noisy for production
    # grounding and can pollute both prompts and debug artifacts.
    observation_ocr_enabled: bool = Field(default=True, alias="FATHOM_OBSERVATION_OCR")
    observation_cv_enabled: bool = Field(default=False, alias="FATHOM_OBSERVATION_CV")
    observation_icon_enabled: bool = Field(default=True, alias="FATHOM_OBSERVATION_ICON")
    observation_overlay_enabled: bool = Field(default=True, alias="FATHOM_OBSERVATION_OVERLAY")
    observation_keyboard_enabled: bool = Field(default=False, alias="FATHOM_OBSERVATION_KEYBOARD")

    # Ensemble vision-localizer (Gemini-vision + DocumentAI-layout)
    # toggles. Enabled with both members so the supervise cascade can
    # fall back to name-based localization when the XML manifest snap
    # fails or returns a generic container.
    ensemble_localizer_enabled: bool = Field(default=True, alias="FATHOM_ENSEMBLE_LOCALIZER")
    ensemble_localizer_members: Optional[str] = Field(
        default="gemini_vision,document_ai_layout",
        alias="FATHOM_ENSEMBLE_LOCALIZER_MEMBERS",
    )

    # Runtime journal adapter (local JSONL) toggle.
    journal_local_enabled: bool = Field(default=False, alias="FATHOM_JOURNAL_LOCAL")

    # Document AI OCR provider credentials. Required only when
    # observation_ocr_enabled is True and Document AI is the active provider.
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
