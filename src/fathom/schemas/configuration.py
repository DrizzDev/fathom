from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class ADBConfig(BaseModel):
    """
    Configuration for ADB device tool.
    """

    model_config = ConfigDict(frozen=True)

    adb_path: str = Field(default="adb", description="Path to adb executable")
    device_serial: Optional[str] = Field(default=None, description="Target device serial")

    tap_duration: int = Field(default=50, description="Tap duration in ms")
    swipe_duration: int = Field(default=300, description="Swipe duration in ms")
    command_timeout: float = Field(default=30.0, description="Command timeout in seconds")
    long_press_duration: int = Field(default=1000, description="Long press duration in ms")


class ADBCaptureConfig(BaseModel):
    """
    Configuration for ADB capture tool.
    """

    model_config = ConfigDict(frozen=True)

    adb_path: str = Field(default="adb", description="Path to the ADB executable")
    timeout: float = Field(default=10.0, description="Timeout for capture operations in seconds")

    use_hybrid_hash: bool = Field(default=True, description="Whether to use hybrid hashing")
    device_serial: Optional[str] = Field(default=None, description="Specific device serial number")


class GeminiConfig(BaseModel):
    """
    Configuration for Gemini vision tool.
    """

    model_config = ConfigDict(frozen=True)

    api_key: Optional[str] = Field(
        default=None, description="Gemini API key (optional if using Vertex AI)"
    )
    credentials_path: Optional[str] = Field(
        default=None, description="Path to Google credentials JSON file"
    )
    project_id: Optional[str] = Field(default=None, description="GCP Project ID for Vertex AI")

    model: str = Field(default="gemini-2.5-flash-lite", description="High-speed VLM model")
    location: str = Field(default="global", description="Vertex AI location")

    timeout: float = Field(default=180.0, description="API request timeout")
    temperature: float = Field(default=0.0, description="Model temperature")
    max_output_tokens: int = Field(default=16384, description="Max output tokens")

    max_retries: int = Field(default=3, description="Max retries on API failure")
    retry_delay: float = Field(default=2.0, description="Base retry delay in seconds")
    gcs_bucket: str = Field(
        default="drizz-dev-crawler-artifacts", description="GCS bucket for screenshot uploads"
    )


class HasherConfig(BaseModel):
    """
    Configuration for hybrid hasher.
    """

    model_config = ConfigDict(frozen=True)

    use_perceptual: bool = Field(default=True, description="Enable perceptual hashing")
    use_structural: bool = Field(default=True, description="Enable structural hashing")
    thumbnail_size: Tuple[int, int] = Field(default=(8, 8), description="pHash thumbnail size")


class WorkflowConfig(BaseModel):
    """
    Configuration for workflow execution.
    """

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=100, ge=1, le=1000, description="Maximum steps")
    step_timeout: float = Field(
        default=15.0, ge=1.0, le=300.0, description="Per-step timeout in seconds"
    )
    total_timeout: float = Field(
        default=600.0, ge=10.0, le=86400.0, description="Total workflow timeout"
    )
    checkpoint_interval: int = Field(
        default=5, ge=1, le=50, description="Steps between checkpoints"
    )
    retry_limit: int = Field(default=3, ge=0, le=10, description="Max retries on failure")
    use_xml_bounding_boxes: bool = Field(
        default=False, description="Use XML hierarchy for bounding boxes"
    )
    package_name: str = Field(default="unknown_app", description="Target application package name")
