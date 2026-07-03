from __future__ import annotations

from logging import getLogger
from typing import Any, ClassVar, Dict, Literal, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fathom.constants.collaboration import JobKind
from fathom.constants.llm import (
    DEFAULT_PRIORITY_FAILURE_THRESHOLD,
    DEFAULT_PRIORITY_LATENCY_THRESHOLD,
    DEFAULT_PRIORITY_RECOVERY_SUCCESSES,
    DEFAULT_PRIORITY_SLOW_THRESHOLD,
    DEFAULT_PRIORITY_WINDOW,
    InferencePriorityMode,
)
from fathom.constants.platform import (
    DeviceConnectionType,
    DevicePlatform,
    IOSAutomationBackend,
)
from fathom.constants.qualification import (
    DEFAULT_QUALIFIER_MAX_RETRIES,
    DEFAULT_QUALIFIER_MODEL,
    DEFAULT_QUALIFIER_TEMPERATURE,
    DEFAULT_QUALIFIER_THINKING_LEVEL,
    DEFAULT_QUALIFIER_TIMEOUT,
    DEFAULT_QUALIFIER_USE_CACHE,
)
from fathom.constants.scheduler import (
    JOB_SCHEDULER_DEFAULT_BATCH_SIZE,
    JOB_SCHEDULER_DEFAULT_FAILURE_BACKOFF,
    JOB_SCHEDULER_DEFAULT_LEASE,
    JOB_SCHEDULER_DEFAULT_MAX_ATTEMPTS,
    JOB_SCHEDULER_DEFAULT_POLL_INTERVAL,
    JOB_SCHEDULER_DEFAULT_RECOVERY_INTERVAL,
    JOB_SCHEDULER_DEFAULT_RETRY_BACKOFF,
    JobSchedulerKind,
)
from fathom.constants.storage import (
    INTERACTION_POSTGRES_APPLICATION_NAME,
    INTERACTION_POSTGRES_DEFAULT_DATABASE,
    INTERACTION_POSTGRES_DEFAULT_MIGRATION_MODE,
    INTERACTION_POSTGRES_DEFAULT_POOL_MAX_SIZE,
    INTERACTION_POSTGRES_DEFAULT_POOL_MIN_SIZE,
    INTERACTION_POSTGRES_DEFAULT_PORT,
    INTERACTION_POSTGRES_DEFAULT_SCHEMA,
    INTERACTION_POSTGRES_DEFAULT_SSL,
    INTERACTION_POSTGRES_DEFAULT_STATEMENT_TIMEOUT,
    INTERACTION_SLOW_QUERY_THRESHOLD,
    InteractionBackend,
    PostgresMigrationMode,
    PostgresSslMode,
    StorageBackend,
)
from fathom.schemas.artifact import PipelineConfiguration
from fathom.schemas.base.common import ThresholdConfiguration
from fathom.schemas.checkpoint import (
    CheckpointConfiguration,
    SqliteCheckpointConfiguration,
)
from fathom.schemas.escalation import EscalationPolicy
from fathom.schemas.finalization import FinalizationBudgetPolicy
from fathom.schemas.retries import RetryLimits
from fathom.schemas.swipe import SwipeRetryPolicy
from fathom.schemas.telemetry import PhaseMessage

logger = getLogger(__name__)


class AdaptivePriorityConfiguration(BaseModel):
    """
    Provider-neutral adaptive elevated-capacity inference policy.
    """

    model_config = ConfigDict(extra="forbid")

    window: int = Field(
        ge=1,
        default=DEFAULT_PRIORITY_WINDOW,
        description="Recent call outcomes retained by the adaptive selector.",
    )
    threshold: ThresholdConfiguration = Field(
        default_factory=lambda: ThresholdConfiguration(
            slows=DEFAULT_PRIORITY_SLOW_THRESHOLD,
            latency=DEFAULT_PRIORITY_LATENCY_THRESHOLD,
            failures=DEFAULT_PRIORITY_FAILURE_THRESHOLD,
            recovery=DEFAULT_PRIORITY_RECOVERY_SUCCESSES,
        ),
        description="Adaptive scale-up and recovery thresholds.",
    )


class PriorityInferenceConfiguration(BaseModel):
    """
    Provider-neutral configuration for elevated-capacity inference.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Whether the adapter may request elevated provider capacity.",
    )
    mode: InferencePriorityMode = Field(
        default=InferencePriorityMode.ALWAYS,
        description="Selection policy for elevated capacity.",
    )
    adaptive: AdaptivePriorityConfiguration = Field(
        default_factory=AdaptivePriorityConfiguration,
        description="Adaptive selector thresholds used when mode is adaptive.",
    )


class LLMConfiguration(BaseModel):
    """
    Generic configuration for LLM providers.
    Supports any backend (Gemini, OpenAI, Anthropic) via provider field.
    """

    provider: Literal["gemini", "openai", "anthropic", "vertex_ai"] = Field(
        default="gemini", description="LLM provider name"
    )
    model: str = Field(default="gemini-3.5-flash", description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="Configured API access key")

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
        default="high",
        description=(
            "Vision token density. 'high' gives Gemini more vertical resolution "
            "and is required for accurate bbox grounding on tall mobile screenshots. "
            "Lower values trade latency for spatial accuracy on the failure path; for live agents the oss-of-grounding cost dominates."
        ),
    )

    # Common hyper-parameters
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    retry_delay: float = Field(default=1.0, description="Base retry delay in seconds")
    rate_limit_backoff: float = Field(default=5.0, description="Base backoff for rate limit errors")
    use_cache: bool = Field(default=True, description="Whether to use context caching for the LLM")

    priority: PriorityInferenceConfiguration = Field(
        default_factory=PriorityInferenceConfiguration,
        description="Provider-neutral elevated-capacity inference policy.",
    )

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
        description=(
            "Storage backends to enable. Defaults to LOCAL only so unconfigured "
            "local runs do not attempt cloud uploads without ADC credentials. "
            "Deployments that want CLOUD uploads pass ``backends={LOCAL, CLOUD}`` "
            "explicitly (see ``services/crawler/manager.py``)."
        ),
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
    retry: SwipeRetryPolicy = Field(
        default_factory=SwipeRetryPolicy,
        description="Bounded coordinate-retry policy for swipes that produced no visual change.",
    )


class _DeprecatedAdaptivePolicy(BaseModel):
    """
    Inert backwards-compatibility stub for the deleted adaptive-scroll subsystem.

    The adaptive-scroll runtime was removed when the swipe coordinator replaced
    it. Older host callers (enricher / healing bridge) still construct
    ``ScrollInteractionPolicy.AdaptivePolicy(...)``; this stub accepts the
    legacy kwargs without applying any behavior so those hosts continue to
    boot while they migrate. Emit a deprecation warning per construction so
    the migration is visible.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    enabled: bool = Field(default=False, description="Ignored; adaptive scroll was removed.")
    maximum_attempts: int = Field(
        default=0,
        description="Ignored; replaced by SwipeRetryPolicy.magnitudes.",
    )
    verify: bool = Field(default=False, description="Ignored; replaced by the validation service.")
    budget: int = Field(
        default=0, description="Ignored; the swipe coordinator is unbounded by wall time."
    )
    suspicious_bottom_ratio: float = Field(
        default=0.0,
        description="Ignored; was an adaptive-scroll heuristic.",
    )

    def model_post_init(self, _context: object) -> None:
        """
        Emit a deprecation warning when the stub is instantiated.
        """

        logger.warning(
            "ScrollInteractionPolicy.AdaptivePolicy is deprecated and has no effect. "
            "Remove the 'adaptive=ScrollInteractionPolicy.AdaptivePolicy(...)' kwarg; "
            "adaptive scroll was replaced by the swipe coordinator + SwipeRetryPolicy.",
        )


class ScrollInteractionPolicy(BaseModel):
    """
    Runtime policy for scroll interactions.
    """

    AdaptivePolicy: ClassVar[Type[_DeprecatedAdaptivePolicy]] = _DeprecatedAdaptivePolicy

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
    adaptive: Optional[_DeprecatedAdaptivePolicy] = Field(
        default=None,
        description=(
            "Deprecated: accepted only so older hosts do not crash on import. "
            "Adaptive scroll behavior was replaced by the swipe coordinator."
        ),
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
    snapshot_timeout: float = Field(
        default=30.0,
        description=(
            "Maximum wall-clock seconds for a full screen + hierarchy snapshot "
            "(``get_snapshot``). Caps the gather over screencap and uiautomator "
            "dump so a wedged emulator (e.g. qcow2 backing file exhausted) "
            "surfaces as a clean DeviceError instead of an infinite await."
        ),
    )
    subprocess_cleanup_timeout: float = Field(
        default=2.0,
        description=(
            "Maximum wall-clock seconds the adapter will wait for a killed "
            "subprocess to reap before abandoning. Protects against "
            "uninterruptible-IO situations where the kernel cannot deliver "
            "SIGKILL to a process stuck waiting on a wedged emulator backing."
        ),
    )
    hierarchy_lock_timeout: float = Field(
        default=10.0,
        description=(
            "Maximum wall-clock seconds the adapter will wait to acquire the "
            "UiAutomation hierarchy lock. Prevents a leaked lock from a prior "
            "cancelled dump_hierarchy task from compounding into an indefinite "
            "wait on subsequent snapshots."
        ),
    )

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
    request_timeout: float = Field(
        default=60.0,
        description=(
            "Maximum wall-clock seconds for a single HTTP request to the "
            "remote device provider (screenshot, hierarchy dump, action "
            "dispatch). Higher than the local subprocess timeout because "
            "remote snapshots can include emulator-side capture, on-the-wire "
            "transfer, and provider-side queueing; some traffic patterns "
            "legitimately take up to ~60s end-to-end."
        ),
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

    max_steps: int = Field(default=100, ge=1, description="Step limit for goal achievement")

    retries: RetryLimits = Field(
        default_factory=RetryLimits,
        description="Per-kind retry caps; nested so new retry kinds are additive without changing call sites.",
    )

    use_xml_grounding: bool = Field(default=False, description="Enable structured XML analysis")
    prompt_user_if_stuck: bool = Field(
        default=True,
        description="If True and in interactive mode, prompt the user for help when the agent detects a loop.",
    )
    finalization: FinalizationBudgetPolicy = Field(
        default_factory=FinalizationBudgetPolicy,
        description="Post-terminal finalization timeout policy applied to history flush, graph state read, checkpointer close, and runner cleanup phases.",
    )
    checkpoint: CheckpointConfiguration = Field(
        default_factory=SqliteCheckpointConfiguration,
        description="Backend-specific checkpoint store configuration; discriminated by `backend`.",
        discriminator="backend",
    )
    escalation: EscalationPolicy = Field(
        default_factory=EscalationPolicy,
        description="Escalation gate policy controlling when HITL is permitted on stuck signals.",
    )


class InferenceConfiguration(BaseModel):
    """
    Generic LLM inference knobs. NO defaults — every consumer must supply every field via its own parent configuration's `default_factory`.
    This forces explicit intent at every use site and prevents one consumer's tuning from silently leaking into another's defaults.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Model identifier the LLM adapter will use.")
    temperature: float = Field(
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 = deterministic).",
    )
    use_cache: bool = Field(
        description="Whether the LLM may reuse cached content between calls.",
    )
    thinking_level: Literal["minimal", "low", "medium", "high"] = Field(
        description="Reasoning depth the LLM is allowed to spend per call.",
    )
    timeout: float = Field(
        ge=0.1,
        le=60.0,
        description="Per-attempt wall-clock timeout in seconds.",
    )
    max_retries: int = Field(
        ge=0,
        le=5,
        description="Retries after the initial attempt. Adapter handles backoff + jitter.",
    )


class QualifierConfiguration(BaseModel):
    """
    Configuration for the intent executability qualifier.

    Owns all qualifier-specific tuning defaults via `default_factory`. Each field of the inference block is set explicitly
    from the constants module so a future generic InferenceConfiguration consumer cannot accidentally inherit qualifier-flavored values.

    Use `QualifierConfiguration.evolve(...)` to override individual inference fields while keeping the rest at the eval-validated defaults;
    see method docstring for the rationale (avoids the verbose "respecify every field" boilerplate that strict-no-defaults forces on callers).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Whether the executability gate runs; False installs the permissive qualifier.",
    )
    inference: InferenceConfiguration = Field(
        default_factory=lambda: InferenceConfiguration(
            model=DEFAULT_QUALIFIER_MODEL,
            timeout=DEFAULT_QUALIFIER_TIMEOUT,
            use_cache=DEFAULT_QUALIFIER_USE_CACHE,
            temperature=DEFAULT_QUALIFIER_TEMPERATURE,
            max_retries=DEFAULT_QUALIFIER_MAX_RETRIES,
            thinking_level=DEFAULT_QUALIFIER_THINKING_LEVEL,
        ),
        description="Inference knobs tuned for the qualifier — defaults reflect eval results.",
    )

    @classmethod
    def evolve(cls, **inference_overrides: Any) -> "QualifierConfiguration":
        """
        Build a QualifierConfiguration replacing only the named inference fields,
        keeping all other qualifier-tuned defaults.

        Implementation routes the merged kwargs through the InferenceConfiguration constructor (not model_copy(update=...))
        so the nested schema's `extra="forbid"` policy enforces field names. Typos in `inference_overrides` raise ValidationError
        instead of being silently dropped onto the model's data dict.
        """

        base = cls()
        merged = {**base.inference.model_dump(), **inference_overrides}

        return cls(inference=InferenceConfiguration(**merged))


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
    phase: PhaseMessage = Field(
        default_factory=PhaseMessage,
        description=(
            "Client-facing phase messages and heartbeat budget; configurable per "
            "deployment so message strings can be localized without code changes."
        ),
    )


class PostgresInteractionConfiguration(BaseModel):
    """
    Configuration for the Postgres interaction adapter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dsn: Optional[str] = Field(default=None, min_length=1)
    host: Optional[str] = Field(default=None, min_length=1)
    port: int = Field(default=INTERACTION_POSTGRES_DEFAULT_PORT, ge=1, le=65535)

    user: Optional[str] = Field(default=None, min_length=1)
    password: Optional[str] = Field(default=None, min_length=1)

    database: str = Field(default=INTERACTION_POSTGRES_DEFAULT_DATABASE, min_length=1)
    schema_name: str = Field(default=INTERACTION_POSTGRES_DEFAULT_SCHEMA, min_length=1)

    pool_min_size: int = Field(default=INTERACTION_POSTGRES_DEFAULT_POOL_MIN_SIZE, ge=1)
    pool_max_size: int = Field(default=INTERACTION_POSTGRES_DEFAULT_POOL_MAX_SIZE, ge=1)
    statement_timeout: int = Field(
        ge=0,
        default=INTERACTION_POSTGRES_DEFAULT_STATEMENT_TIMEOUT,
    )

    application_name: str = Field(default=INTERACTION_POSTGRES_APPLICATION_NAME, min_length=1)
    ssl: PostgresSslMode = Field(default=INTERACTION_POSTGRES_DEFAULT_SSL)
    migration_mode: PostgresMigrationMode = Field(
        default=INTERACTION_POSTGRES_DEFAULT_MIGRATION_MODE
    )

    slow_query_threshold: int = Field(
        default=INTERACTION_SLOW_QUERY_THRESHOLD,
        ge=0,
        description="Slow-query log threshold, in milliseconds.",
    )

    @model_validator(mode="after")
    def __validate_pool_sizes(self) -> "PostgresInteractionConfiguration":
        """
        Ensure the maximum pool size is not smaller than the minimum.
        """

        if self.pool_max_size < self.pool_min_size:
            raise ValueError("pool_max_size must be greater than or equal to pool_min_size")

        return self

    @model_validator(mode="after")
    def __validate_connection_mode(self) -> "PostgresInteractionConfiguration":
        """
        Require a DSN or the discrete host, user, and password fields.
        """

        if self.dsn is not None:
            return self

        missing = [
            name
            for name, value in (
                ("host", self.host),
                ("user", self.user),
                ("password", self.password),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "PostgresInteractionConfiguration requires either `dsn` or all of "
                f"`host`, `user`, `password`; missing: {', '.join(missing)}."
            )

        return self


class NoopInteractionConfiguration(BaseModel):
    """
    Configuration for the noop interaction adapter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class InteractionStorageConfiguration(BaseModel):
    """
    Selects the interaction storage backend and its matching configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: InteractionBackend = Field(description="Selected interaction backend")

    noop: Optional[NoopInteractionConfiguration] = None
    postgres: Optional[PostgresInteractionConfiguration] = None

    @model_validator(mode="after")
    def __validate_backend(self) -> "InteractionStorageConfiguration":
        """
        Require the nested configuration that matches the selected backend.
        """

        if self.backend == InteractionBackend.POSTGRES and self.postgres is None:
            raise ValueError("backend=postgres requires postgres configuration")

        if self.backend == InteractionBackend.NOOP and self.noop is None:
            raise ValueError("backend=noop requires noop configuration")

        return self


class InProcessJobSchedulerConfiguration(BaseModel):
    """
    Configuration for the in-process durable job scheduler.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    kinds: Tuple[JobKind, ...] = Field(default_factory=tuple)

    lease: int = Field(default=JOB_SCHEDULER_DEFAULT_LEASE, gt=0)
    batch_size: int = Field(default=JOB_SCHEDULER_DEFAULT_BATCH_SIZE, gt=0)
    poll_interval: int = Field(default=JOB_SCHEDULER_DEFAULT_POLL_INTERVAL, ge=0)

    max_attempts: int = Field(default=JOB_SCHEDULER_DEFAULT_MAX_ATTEMPTS, gt=0)
    retry_backoff: int = Field(default=JOB_SCHEDULER_DEFAULT_RETRY_BACKOFF, ge=0)
    failure_backoff: int = Field(default=JOB_SCHEDULER_DEFAULT_FAILURE_BACKOFF, ge=0)
    recovery_interval: int = Field(default=JOB_SCHEDULER_DEFAULT_RECOVERY_INTERVAL, ge=0)


class NoopJobSchedulerConfiguration(BaseModel):
    """
    Configuration for disabled durable job dispatch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class JobSchedulerConfiguration(BaseModel):
    """
    Selects the durable job scheduler backend and its matching configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    noop: Optional[NoopJobSchedulerConfiguration] = None
    inprocess: Optional[InProcessJobSchedulerConfiguration] = None
    kind: JobSchedulerKind = Field(description="Selected job scheduler kind")

    @model_validator(mode="after")
    def __validate_kind(self) -> "JobSchedulerConfiguration":
        """
        Require the nested configuration that matches the selected scheduler kind.
        """

        if self.kind == JobSchedulerKind.IN_PROCESS and self.inprocess is None:
            raise ValueError("kind=inprocess requires inprocess configuration")

        if self.kind == JobSchedulerKind.NOOP and self.noop is None:
            raise ValueError("kind=noop requires noop configuration")

        return self


class FathomConfiguration(BaseModel):
    """
    Root configuration container for the Fathom runtime.
    Aggregates all component configurations into a single schema.
    """

    llm: LLMConfiguration = Field(default_factory=LLMConfiguration)
    device: DeviceConfiguration = Field(default_factory=DeviceConfiguration)
    engine: ExecutionConfiguration = Field(default_factory=ExecutionConfiguration)

    storage: StorageConfiguration = Field(default_factory=StorageConfiguration)
    artifact: PipelineConfiguration = Field(default_factory=PipelineConfiguration)
    telemetry: TelemetryConfiguration = Field(default_factory=TelemetryConfiguration)

    intent: IntentConfiguration = Field(default_factory=IntentConfiguration)
    qualifier: QualifierConfiguration = Field(default_factory=QualifierConfiguration)
    exploration: ExplorationConfiguration = Field(default_factory=ExplorationConfiguration)

    scheduler: Optional[JobSchedulerConfiguration] = None
    interaction: Optional[InteractionStorageConfiguration] = None
