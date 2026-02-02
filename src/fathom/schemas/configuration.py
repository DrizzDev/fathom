from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, Field


class ADBConfig(BaseModel):
    """
    Configuration for ADB device tool.
    """

    model_config = {"frozen": True}

    adb_path: str = Field(default="adb", description="Path to adb executable")
    device_serial: Optional[str] = Field(default=None, description="Target device serial")

    tap_duration: int = Field(default=50, description="Tap duration in ms")
    swipe_duration: int = Field(default=300, description="Swipe duration in ms")
    command_timeout: float = Field(default=30.0, description="Command timeout in seconds")
    long_press_duration: int = Field(default=1000, description="Long press duration in ms")


class GeminiConfig(BaseModel):
    """
    Configuration for Gemini vision tool.
    """

    model_config = {"frozen": True}

    api_key: Optional[str] = Field(
        default=None, description="Gemini API key (optional if using Vertex AI)"
    )
    project_id: Optional[str] = Field(default=None, description="GCP Project ID for Vertex AI")

    model: str = Field(default="gemini-2.0-flash-exp", description="Model name")
    location: str = Field(default="us-central1", description="Vertex AI location")

    timeout: float = Field(default=30.0, description="API request timeout")
    temperature: float = Field(default=0.1, description="Model temperature")
    max_output_tokens: int = Field(default=2048, description="Max output tokens")


class HasherConfig(BaseModel):
    """
    Configuration for hybrid hasher.
    """

    model_config = {"frozen": True}

    use_perceptual: bool = Field(default=True, description="Enable perceptual hashing")
    use_structural: bool = Field(default=True, description="Enable structural hashing")
    thumbnail_size: Tuple[int, int] = Field(default=(8, 8), description="pHash thumbnail size")


class WorkflowConfig(BaseModel):
    """
    Configuration for workflow execution.
    """

    model_config = {"frozen": True}

    max_steps: int = Field(default=20, ge=1, le=1000, description="Maximum steps")
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
