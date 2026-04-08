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
