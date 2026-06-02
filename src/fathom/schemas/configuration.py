from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Set, Union

from pydantic import BaseModel, Field

from fathom.constants.platform import (
    DeviceConnectionType,
    DevicePlatform,
    IOSAutomationBackend,
)
from fathom.constants.storage import StorageBackend


class LLMConfiguration(BaseModel):
    """
    Generic configuration for LLM providers.
    Supports any backend (Gemini, OpenAI, Anthropic) via provider field.
    """

    provider: Literal["gemini", "openai", "anthropic", "vertex_ai"] = Field(
        default="gemini", description="LLM provider name"
    )
    model: str = Field(default="gemini-3-flash-preview", description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="API access key")

    # Provider-specific settings (GCP/Azure/OpenAI specific)
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    location: Optional[str] = Field(default=None, description="Deployment location/region")
    credentials: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Credentials as file path (str) or JSON object (dict)",
    )

    # Gemini specific parameters
    thinking_level: Literal["minimal", "low", "medium", "high"] = Field(
        default="low",
        description="Controls reasoning depth. 'low' = faster, 'high' = deeper/slower.",
    )
    include_thoughts: bool = Field(
        default=False,
        description="Whether to include the model's reasoning process in the response.",
    )
    media_resolution: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Vision token density. 'low' is recommended for high-speed agents.",
    )

    # Common hyper-parameters
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    retry_delay: float = Field(default=1.0, description="Base retry delay in seconds")
    rate_limit_backoff: float = Field(default=5.0, description="Base backoff for rate limit errors")
    use_cache: bool = Field(default=True, description="Whether to use context caching for the LLM")

    # Extension hook for arbitrary provider settings
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )


class StorageConfiguration(BaseModel):
    """
    Configuration for artifact storage.
    """

    backends: Set[StorageBackend] = Field(
        default={StorageBackend.LOCAL},
        description="Storage backends to enable",
    )
    storage_bucket: Optional[str] = Field(
        default="drizz-dev-crawler-artifacts", description="Cloud storage bucket name"
    )
    project_id: Optional[str] = Field(
        default=None, description="Project identifier for cloud storage"
    )
    credentials: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Credentials as file path (str) or JSON object (dict)",
    )


class TapInteractionPolicy(BaseModel):
    """
    Runtime policy for tap interactions.
    """

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tap policy extension metadata",
    )


class TypeInteractionPolicy(BaseModel):
    """
    Runtime policy for text input interactions.
    """

    delay: int = Field(
        ge=0,
        default=500,
        description="Delay in milliseconds before retrying a type action when the initial attempt fails due to focus loss.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type policy extension metadata",
    )


class SwipeInteractionPolicy(BaseModel):
    """
    Runtime policy for swipe interactions.
    """

    duration: int = Field(
        default=300,
        description="Default swipe gesture duration in milliseconds.",
    )
    edge_margin_ratio: float = Field(
        ge=0.0,
        lt=0.5,
        default=0.08,
        description="Fraction of the execution region reserved as a safe edge margin.",
    )
    minimum_edge_margin: int = Field(
        ge=0,
        default=16,
        description="Smallest safe edge margin in screen pixels.",
    )
    maximum_edge_margin: int = Field(
        ge=0,
        default=64,
        description="Largest safe edge margin in screen pixels.",
    )


class ScrollInteractionPolicy(BaseModel):
    """
    Runtime policy for scroll interactions.
    """

    edge_margin_ratio: float = Field(
        ge=0.0,
        lt=0.5,
        default=0.15,
        description="Fraction of the scroll region reserved as a safe edge margin.",
    )
    minimum_edge_margin: int = Field(
        ge=0,
        default=48,
        description="Smallest safe edge margin in screen pixels.",
    )
    maximum_edge_margin: int = Field(
        ge=0,
        default=160,
        description="Largest safe edge margin in screen pixels.",
    )


class InteractionPolicyConfiguration(BaseModel):
    """
    Runtime interaction policy envelope grouped by action type.
    """

    tap: TapInteractionPolicy = Field(default_factory=TapInteractionPolicy)
    type: TypeInteractionPolicy = Field(default_factory=TypeInteractionPolicy)
    swipe: SwipeInteractionPolicy = Field(default_factory=SwipeInteractionPolicy)
    scroll: ScrollInteractionPolicy = Field(default_factory=ScrollInteractionPolicy)


class InteractionRuntimeConfiguration(BaseModel):
    """
    Runtime interaction configuration.
    """

    policy: InteractionPolicyConfiguration = Field(default_factory=InteractionPolicyConfiguration)


class ADBConfiguration(BaseModel):
    """
    Configuration for local Android interactions via ADB.
    """

    executable_path: str = Field(default="adb", description="Path to ADB executable")
    serial_number: Optional[str] = Field(
        default=None, description="Target Android device identifier"
    )
    command_timeout: float = Field(default=10.0, description="Shell command timeout in seconds")

    interaction: InteractionRuntimeConfiguration = Field(
        default_factory=InteractionRuntimeConfiguration,
        description="Interaction policy configuration for Android actions",
    )


class IOSConfiguration(BaseModel):
    """
    Configuration for local iOS simulator interactions via native Apple tooling.
    """

    executable_path: str = Field(default="xcrun", description="Path to xcrun executable")
    device_identifier: Optional[str] = Field(
        default=None, description="Target iOS simulator device identifier"
    )
    bundle_identifier: Optional[str] = Field(
        default=None, description="Default iOS bundle identifier context"
    )
    automation_backend: IOSAutomationBackend = Field(
        default=IOSAutomationBackend.XCRUN_SIMCTL,
        description="iOS automation backend strategy",
    )
    web_driver_agent_url: str = Field(
        default="http://127.0.0.1:8100",
        description="WebDriverAgent server URL for hierarchy extraction and optional fallback gestures",
    )
    web_driver_agent_bundle_identifier: Optional[str] = Field(
        default=None,
        description="Optional bundle identifier injected into WebDriverAgent session capabilities",
    )
    web_driver_agent_request_timeout_seconds: float = Field(
        default=15.0,
        description="WebDriverAgent request timeout in seconds",
    )

    command_timeout: float = Field(default=10.0, description="Shell command timeout in seconds")
    interaction: InteractionRuntimeConfiguration = Field(
        default_factory=InteractionRuntimeConfiguration,
        description="Interaction policy configuration for iOS actions",
    )


class RemoteDeviceConfiguration(BaseModel):
    """
    Configuration for remote device providers.
    """

    session_id: Optional[str] = Field(default=None, description="Remote session identifier")
    execution_id: Optional[str] = Field(
        default=None, description="Remote execution/workflow identifier"
    )
    provider_url: Optional[str] = Field(default=None, description="Remote provider endpoint")
    authentication_token: Optional[str] = Field(
        default=None, description="Access token for remote provider"
    )


class DeviceRuntimeConfiguration(BaseModel):
    """
    Platform-neutral runtime settings exposed by DevicePort implementations.
    """

    platform: DevicePlatform = Field(
        default=DevicePlatform.ANDROID,
        description="Resolved runtime platform",
    )
    identifier: Optional[str] = Field(
        default=None,
        description="Device identifier (serial number, simulator id, session id, etc.)",
    )
    command_timeout: float = Field(default=10.0, description="Command timeout in seconds")
    interaction: InteractionRuntimeConfiguration = Field(
        default_factory=InteractionRuntimeConfiguration,
        description="Runtime interaction contract and policies",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Adapter-specific runtime metadata",
    )


class DeviceConfiguration(BaseModel):
    """
    Unified configuration for local and remote device adapters.
    """

    type: DeviceConnectionType = Field(
        default=DeviceConnectionType.LOCAL,
        description="Device connection type",
    )
    platform: DevicePlatform = Field(
        default=DevicePlatform.ANDROID,
        description="Target platform when using local connection",
    )

    ios: IOSConfiguration = Field(default_factory=IOSConfiguration)
    android: ADBConfiguration = Field(default_factory=ADBConfiguration)
    remote: RemoteDeviceConfiguration = Field(default_factory=RemoteDeviceConfiguration)

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional device metadata")


class ExplorationConfiguration(BaseModel):
    """
    Configuration for application exploration strategy.
    """

    max_steps: int = Field(default=100, description="Maximum exploration depth")
    timeout: int = Field(default=300, description="Global timeout for the run")
    random_seed: Optional[int] = Field(
        default=None, description="Seed for deterministic exploration"
    )

    # Tap action boundaries
    tap_margin_x: int = Field(
        default=50,
        description="Minimum horizontal distance from screen edges to avoid system UI elements",
    )
    tap_margin_y: int = Field(
        default=100,
        description="Minimum vertical distance from screen edges to avoid status/navigation bars",
    )
    tap_max_x: int = Field(
        default=950,
        description="Maximum X coordinate for exploratory taps to stay within safe bounds",
    )
    tap_max_y: int = Field(
        default=900,
        description="Maximum Y coordinate for exploratory taps to stay within safe bounds",
    )
    tap_target_size: int = Field(
        default=50, description="Size of tap target bounding box for action generation"
    )


class IntentConfiguration(BaseModel):
    """
    Configuration for intent-based execution strategy.
    """

    max_steps: int = Field(default=100, description="Step limit for goal achievement")
    use_xml_grounding: bool = Field(default=False, description="Enable structured XML analysis")
    prompt_user_if_stuck: bool = Field(
        default=True,
        description="If True and in interactive mode, prompt the user for help when the agent detects a loop.",
    )


class QualifierConfiguration(BaseModel):
    """
    Configuration for the intent executability qualifier.
    """

    enabled: bool = Field(
        default=True,
        description="Whether the executability gate runs; False installs the permissive qualifier.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.85,
        description="Minimum model confidence on NOT_EXECUTABLE required to block a run.",
    )
    temperature: float = Field(
        ge=0.0,
        le=2.0,
        default=0.0,
        description="Sampling temperature for the qualifier LLM (0 = deterministic).",
    )
    use_cache: bool = Field(
        default=False,
        description="Whether the qualifier LLM may reuse cached content between runs.",
    )
    thinking_level: Literal["minimal", "low", "medium", "high"] = Field(
        default="low",
        description="Reasoning depth the qualifier LLM is allowed to spend per call.",
    )


class WorkflowHostPolicyConfiguration(BaseModel):
    """
    Workflow-host activity policy for one workflow type.
    """

    heartbeat_seconds: int = Field(
        default=60,
        description="Heartbeat timeout in seconds for the workflow host activity",
    )
    timeout_floor: int = Field(
        default=60,
        description="Minimum start-to-close timeout in minutes",
    )
    timeout_per_step: int = Field(
        default=2,
        description="Additional timeout budget in minutes per requested step",
    )
    timeout_overhead: int = Field(
        default=5,
        description="Fixed timeout overhead in minutes added to every run",
    )


class IntentWorkflowHostPolicyConfiguration(WorkflowHostPolicyConfiguration):
    """
    Workflow-host activity policy defaults for intent workflows.
    """

    heartbeat_seconds: int = Field(
        default=300,
        description="Heartbeat timeout in seconds for intent workflows",
    )
    timeout_floor: int = Field(
        default=60,
        description="Minimum start-to-close timeout in minutes for intent workflows",
    )


class ExplorationWorkflowHostPolicyConfiguration(WorkflowHostPolicyConfiguration):
    """
    Workflow-host activity policy defaults for exploration workflows.
    """

    timeout_floor: int = Field(
        default=120,
        description="Minimum start-to-close timeout in minutes for exploration workflows",
    )


class WorkflowHostConfiguration(BaseModel):
    """
    Workflow-host activity policies grouped by workflow type.
    """

    intent: IntentWorkflowHostPolicyConfiguration = Field(
        default_factory=IntentWorkflowHostPolicyConfiguration,
        description="Workflow-host activity policy for intent workflows",
    )
    exploration: ExplorationWorkflowHostPolicyConfiguration = Field(
        default_factory=ExplorationWorkflowHostPolicyConfiguration,
        description="Workflow-host activity policy for exploration workflows",
    )


class ExecutionConfiguration(BaseModel):
    """
    Configuration for the core execution engine.
    """

    max_retries: int = Field(default=3, description="Maximum retries for physical actions")
    stability_wait: float = Field(
        ge=0.0,
        le=1.5,
        default=0.5,
        description="Wait time after action for screen settlement (max 1.5s)",
    )
    workflow: WorkflowHostConfiguration = Field(
        default_factory=WorkflowHostConfiguration,
        description="Workflow-host execution policy",
    )


class TelemetryConfiguration(BaseModel):
    """
    Configuration for telemetry and logging adapters.
    """

    type: Literal["STRUCTLOG", "REDIS"] = Field(
        default="STRUCTLOG", description="Telemetry adapter type"
    )
    connection_string: Optional[str] = Field(
        default=None, description="Connection URL for streaming logs"
    )
    topic: Optional[str] = Field(
        default=None,
        description="Topic or channel pattern (e.g., enricher:commands:v1:logs:{session_id})",
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID for channel interpolation"
    )
    identity: Optional[str] = Field(default=None, description="Workflow identity for log routing")


class FathomConfiguration(BaseModel):
    """
    Root configuration container for the Fathom runtime.
    Aggregates all component configurations into a single schema.
    """

    llm: LLMConfiguration = Field(default_factory=LLMConfiguration)
    device: DeviceConfiguration = Field(default_factory=DeviceConfiguration)
    engine: ExecutionConfiguration = Field(default_factory=ExecutionConfiguration)
    storage: StorageConfiguration = Field(default_factory=StorageConfiguration)
    telemetry: TelemetryConfiguration = Field(default_factory=TelemetryConfiguration)

    intent: IntentConfiguration = Field(default_factory=IntentConfiguration)
    exploration: ExplorationConfiguration = Field(default_factory=ExplorationConfiguration)
    qualifier: QualifierConfiguration = Field(default_factory=QualifierConfiguration)
