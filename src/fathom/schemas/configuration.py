"""
Configuration schemas for Fathom execution.

These schemas define configurable parameters for execution engine,
strategies, and other components.
"""

from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, Field

from fathom.constants.execution import (
    BOUNDS_SWIPE_DISTANCE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_SCROLL_DISTANCE,
    DEFAULT_STABILITY_WAIT,
    DEFAULT_SWIPE_DISTANCE,
    DEFAULT_SWIPE_DURATION,
    VISUAL_HASH_LENGTH,
)


class ADBCaptureConfig(BaseModel):
    """Configuration for ADB capture tool."""

    adb_path: str = Field(
        default="adb",
        description="Path to the ADB executable",
    )

    timeout: float = Field(
        default=10.0,
        description="Timeout for capture operations in seconds",
        ge=1.0,
        le=60.0,
    )

    use_hybrid_hash: bool = Field(
        default=True,
        description="Whether to use hybrid hashing",
    )

    device_serial: Optional[str] = Field(
        default=None,
        description="Specific device serial number",
    )


class HasherConfig(BaseModel):
    """Configuration for hybrid hasher."""

    use_perceptual: bool = Field(
        default=True,
        description="Enable perceptual hashing",
    )

    use_structural: bool = Field(
        default=True,
        description="Enable structural hashing",
    )

    thumbnail_size: Tuple[int, int] = Field(
        default=(8, 8),
        description="pHash thumbnail size",
    )


class WorkflowConfig(BaseModel):
    """Configuration for workflow execution."""

    max_steps: int = Field(
        default=20,
        description="Maximum steps",
        ge=1,
        le=1000,
    )

    step_timeout: float = Field(
        default=15.0,
        description="Per-step timeout in seconds",
        ge=1.0,
        le=300.0,
    )

    total_timeout: float = Field(
        default=600.0,
        description="Total workflow timeout",
        ge=10.0,
        le=86400.0,
    )

    checkpoint_interval: int = Field(
        default=5,
        description="Steps between checkpoints",
        ge=1,
        le=50,
    )

    retry_limit: int = Field(
        default=3,
        description="Max retries on failure",
        ge=0,
        le=10,
    )

    use_xml_bounding_boxes: bool = Field(
        default=False,
        description="Use XML hierarchy for bounding boxes",
    )

    package_name: Optional[str] = Field(
        default=None,
        description="Target package name",
    )


class ADBConfig(BaseModel):
    """Configuration for ADB device adapter."""

    device_serial: Optional[str] = Field(
        default=None,
        description="Device serial number (None for default device)",
    )

    adb_path: str = Field(
        default="adb",
        description="Path to ADB executable",
    )

    command_timeout: float = Field(
        default=30.0,
        description="Timeout for ADB commands in seconds",
        ge=1.0,
        le=300.0,
    )

    swipe_duration: int = Field(
        default=DEFAULT_SWIPE_DURATION,
        description="Default swipe duration in milliseconds",
        ge=100,
        le=2000,
    )

    swipe_distance: float = Field(
        default=0.6,
        description="Default swipe distance as fraction of screen",
    )

    scroll_distance: float = Field(
        default=0.8,
        description="Default scroll distance as fraction of screen",
    )


class GeminiConfig(BaseModel):
    """Configuration for Gemini LLM adapter."""

    api_key: Optional[str] = Field(
        default=None,
        description="Gemini API key (if not using Vertex AI)",
    )

    model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Gemini model name",
    )

    project_id: Optional[str] = Field(
        default=None,
        description="Google Cloud project ID (for Vertex AI)",
    )

    location: Optional[str] = Field(
        default="global",
        description="Google Cloud location (for Vertex AI)",
    )

    credentials_path: Optional[str] = Field(
        default=None,
        description="Path to service account credentials JSON file",
    )

    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
        ge=10.0,
        le=300.0,
    )

    temperature: float = Field(
        default=0.0,
        description="Sampling temperature",
        ge=0.0,
        le=2.0,
    )

    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests",
        ge=0,
        le=10,
    )

    retry_delay: float = Field(
        default=1.0,
        description="Base delay for retries in seconds",
    )

    gcs_bucket: Optional[str] = Field(
        default=None,
        description="Google Cloud Storage bucket name",
    )


class ExecutionConfig(BaseModel):
    """Configuration for execution engine."""

    visual_hash_length: int = Field(
        default=VISUAL_HASH_LENGTH,
        description="Length of visual hash for screen identification",
        ge=8,
        le=64,
    )

    swipe_distance: int = Field(
        default=DEFAULT_SWIPE_DISTANCE,
        description="Default swipe distance in pixels",
        ge=50,
        le=1000,
    )

    scroll_distance: int = Field(
        default=DEFAULT_SCROLL_DISTANCE,
        description="Default scroll distance in pixels",
        ge=50,
        le=1000,
    )

    bounds_swipe_distance: int = Field(
        default=BOUNDS_SWIPE_DISTANCE,
        description="Swipe distance for bounded elements in pixels",
        ge=50,
        le=500,
    )

    swipe_duration: int = Field(
        default=DEFAULT_SWIPE_DURATION,
        description="Swipe duration in milliseconds",
        ge=100,
        le=2000,
    )

    stability_wait: int = Field(
        default=DEFAULT_STABILITY_WAIT,
        description="Wait time after action for screen to stabilize in milliseconds",
        ge=0,
        le=5000,
    )

    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        description="Maximum number of retries for failed actions",
        ge=0,
        le=10,
    )

    retry_delay: int = Field(
        default=DEFAULT_RETRY_DELAY,
        description="Base delay for exponential backoff in milliseconds",
        ge=100,
        le=5000,
    )


class IntentStrategyConfig(BaseModel):
    """Configuration for intent-based execution strategy."""

    max_steps: int = Field(
        default=20,
        description="Maximum number of execution steps",
        ge=1,
        le=100,
    )

    use_xml: bool = Field(
        default=False,
        description="Whether to use XML hierarchy for element detection",
    )

    enable_memory: bool = Field(
        default=True,
        description="Whether to store experiences in memory",
    )

    enable_audit: bool = Field(
        default=True,
        description="Whether to enable audit logging",
    )


class ExplorationStrategyConfig(BaseModel):
    """Configuration for exploration strategy."""

    max_steps: int = Field(
        default=100,
        description="Maximum number of exploration steps",
        ge=1,
        le=1000,
    )

    timeout: float = Field(
        default=3600.0,
        description="Maximum exploration time in seconds",
        ge=60.0,
        le=86400.0,
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible exploration",
    )

    exploration_limit: int = Field(
        default=5,
        description="Number of times to explore each screen before moving on",
        ge=1,
        le=20,
    )


class FathomConfig(BaseModel):
    """Complete Fathom configuration."""

    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Execution engine configuration",
    )

    intent_strategy: IntentStrategyConfig = Field(
        default_factory=IntentStrategyConfig,
        description="Intent strategy configuration",
    )

    exploration_strategy: ExplorationStrategyConfig = Field(
        default_factory=ExplorationStrategyConfig,
        description="Exploration strategy configuration",
    )
