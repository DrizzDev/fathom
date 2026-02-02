from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
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
    gemini_model: str = Field(default="gemini-2.0-flash-exp", alias="GEMINI_MODEL")

    vertex_location: str = Field(default="us-central1", alias="VERTEX_LOCATION")
    vertex_project_id: Optional[str] = Field(default=None, alias="VERTEX_PROJECT_ID")

    google_application_credentials: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS_PATH"
        ),
    )

    # Device settings
    android_serial: Optional[str] = Field(default=None, alias="ANDROID_SERIAL")
    adb_path: str = Field(default="adb", alias="ADB_PATH")

    # Logging settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")

    # Workflow default limits
    max_steps: int = Field(default=20, alias="MAX_STEPS")

    # Environment file support
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file_encoding="utf-8",
        env_file=[".env", str(PROJECT_ROOT / ".env")],
    )
