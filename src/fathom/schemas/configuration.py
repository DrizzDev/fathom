from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class LLMConfiguration(BaseModel):
    """
    Generic configuration for LLM providers.
    Supports any backend (Gemini, OpenAI, Anthropic) via provider field.
    """

    provider: Literal["gemini", "openai", "anthropic", "vertex_ai"] = Field(
        default="gemini", description="LLM provider name"
    )
    model: str = Field(default="gemini-2.0-flash-exp", description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="API access key")

    # Provider-specific settings (GCP/Azure/OpenAI specific)
    location: Optional[str] = Field(default=None, description="Deployment location/region")
    project_id: Optional[str] = Field(
        default=None, description="Project identifier (if applicable)"
    )
    credentials_path: Optional[str] = Field(
        default=None, description="Path to authentication artifacts"
    )

    # Common hyper-parameters
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    retry_delay: float = Field(default=1.0, description="Base retry delay in seconds")

    # Backend storage (for artifacts like image caching)
    storage_bucket: Optional[str] = Field(default=None, description="Cloud storage bucket name")

    # Extension hook for arbitrary provider settings
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )


class ADBConfiguration(BaseModel):
    """
    Configuration for local ADB device interactions.
    """

    executable_path: str = Field(default="adb", description="Path to ADB executable")
    serial_number: Optional[str] = Field(default=None, description="Target device serial")
    command_timeout: float = Field(default=10.0, description="Shell command timeout in seconds")
    swipe_duration: int = Field(default=300, description="Default swipe gesture duration in ms")
    swipe_distance: float = Field(default=0.7, description="Percentage of screen to swipe")
    scroll_distance: float = Field(default=0.2, description="Percentage of screen height to scroll")


class DeviceConfiguration(BaseModel):
    """
    Unified configuration for device connection.
    Determines whether to use local ADB or a remote provider.
    """

    type: Literal["LOCAL", "REMOTE"] = Field(
        default="LOCAL", description="Device connection type: LOCAL or REMOTE"
    )

    # Connectivity Details
    session_id: Optional[str] = Field(default=None, description="Remote session identifier")
    provider_url: Optional[str] = Field(default=None, description="Remote provider endpoint")
    serial: Optional[str] = Field(default=None, description="Local serial or remote identifier")
    authentication_token: Optional[str] = Field(
        default=None, description="Access token for remote provider"
    )

    # Generic parameters for future adapters
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional device metadata")


class ExplorationConfiguration(BaseModel):
    """
    Configuration for application exploration strategy.
    """

    max_steps: int = Field(default=50, description="Maximum exploration depth")
    timeout: int = Field(default=300, description="Global timeout for the run")
    random_seed: Optional[int] = Field(
        default=None, description="Seed for deterministic exploration"
    )


class IntentConfiguration(BaseModel):
    """
    Configuration for intent-based execution strategy.
    """

    max_steps: int = Field(default=100, description="Step limit for goal achievement")
    use_xml_grounding: bool = Field(default=False, description="Enable structured XML analysis")


class ExecutionConfiguration(BaseModel):
    """
    Configuration for the core execution engine.
    """

    max_retries: int = Field(default=3, description="Maximum retries for physical actions")
    stability_wait: float = Field(
        default=0.5, description="Wait time after action for screen settlement"
    )


class FathomConfiguration(BaseModel):
    """
    Root configuration container for the Fathom runtime.
    Aggregates all component configurations into a single schema.
    """

    llm: LLMConfiguration = Field(default_factory=LLMConfiguration)
    device: DeviceConfiguration = Field(default_factory=DeviceConfiguration)
    engine: ExecutionConfiguration = Field(default_factory=ExecutionConfiguration)

    intent: IntentConfiguration = Field(default_factory=IntentConfiguration)
    exploration: ExplorationConfiguration = Field(default_factory=ExplorationConfiguration)
