from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FathomSettings(BaseSettings):
    """
    Global settings loaded from environment variables.
    """

    # Gemini settings
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    vertex_location: str = Field(default="us-central1", alias="VERTEX_LOCATION")
    gemini_model: str = Field(default="gemini-2.0-flash-exp", alias="GEMINI_MODEL")
    vertex_project_id: Optional[str] = Field(default=None, alias="VERTEX_PROJECT_ID")
    google_application_credentials: Optional[str] = Field(
        default=None, alias="GOOGLE_APPLICATION_CREDENTIALS"
    )

    # Device settings
    adb_path: str = Field(default="adb", alias="ADB_PATH")
    android_serial: Optional[str] = Field(default=None, alias="ANDROID_SERIAL")

    # Logging settings
    log_json: bool = Field(default=False, alias="LOG_JSON")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Workflow default limits
    max_steps: int = Field(default=20, alias="MAX_STEPS")

    # Environment file support
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )
