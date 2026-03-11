from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from fathom.constants import ContextScope, ExecutionMode
from fathom.constants.run import SignalAdapterType, TargetKind
from fathom.schemas.configuration import (
    DeviceConfiguration,
    ExecutionConfiguration,
    ExplorationConfiguration,
    IntentConfiguration,
    LLMConfiguration,
    StorageConfiguration,
)


class RealignmentPolicy(BaseModel):
    """
    Defines course-correction behavior for an executing run.
    """

    budget: int = Field(
        default=3,
        description="Maximum allowed consecutive re-plans before the run is considered exhausted",
    )
    immediate: bool = Field(
        default=True,
        description="Whether injected guidance should force immediate re-evaluation",
    )


class IntentObjectiveConfiguration(BaseModel):
    """
    Objective definition for an intent-driven run.
    """

    mode: ExecutionMode = Field(default=ExecutionMode.INTENT)
    intent: str = Field(..., description="User goal for the run")
    package_name: Optional[str] = Field(
        default=None,
        description="Optional application identifier used for routing and storage",
    )
    max_steps: int = Field(default=100, description="Maximum steps allowed for the run")
    use_xml: bool = Field(default=True, description="Whether XML grounding should be enabled")


class ExplorationObjectiveConfiguration(BaseModel):
    """
    Objective definition for an exploration run.
    """

    mode: ExecutionMode = Field(default=ExecutionMode.EXPLORATION)

    intent: str = Field(
        default="Explore application structure",
        description="Logical exploration goal used for workflow metadata",
    )
    package_name: Optional[str] = Field(
        default=None,
        description="Optional application identifier used for routing and storage",
    )
    max_steps: int = Field(default=250, description="Maximum exploration steps")
    use_xml: bool = Field(
        default=True,
        description="Exploration runs do not use XML grounding by default",
    )


class RuntimeConfiguration(BaseModel):
    """
    Runtime control configuration for a run.
    """

    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="Unique session identifier",
    )
    execution_id: Optional[str] = Field(
        default=None,
        description="Host execution identifier for correlation and remote routing",
    )
    interactive: bool = Field(default=False, description="Enable HITL interaction")
    signal_type: SignalAdapterType = Field(
        default=SignalAdapterType.INTERACTIVE,
        description="Signal adapter strategy used by the host",
    )


class MemoryConfiguration(BaseModel):
    """
    Memory and context configuration for a run.
    """

    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation identifier for conversation-scoped memory",
    )
    context_scope: ContextScope = Field(
        default=ContextScope.EXECUTION,
        description="Scope used while hydrating runtime memory",
    )


class TargetConfiguration(BaseModel):
    """
    Execution target bound to the run.
    """

    name: str = Field(default="primary", description="Stable target name")
    kind: TargetKind = Field(default=TargetKind.DEVICE, description="Target kind")

    device_configuration: DeviceConfiguration = Field(
        default_factory=DeviceConfiguration,
        description="Device configuration used for the target",
    )


class ModelSelectionConfiguration(BaseModel):
    """
    Role-based model selection for the run.
    """

    planner_configuration: LLMConfiguration = Field(
        default_factory=LLMConfiguration,
        description="Planner model configuration",
    )
    verifier_configuration: Optional[LLMConfiguration] = Field(
        default=None,
        description="Optional verifier model configuration",
    )
    exporter_configuration: Optional[LLMConfiguration] = Field(
        default=None,
        description="Optional exporter model configuration",
    )


class ResourceConfiguration(BaseModel):
    """
    Execution resources required by a run.
    """

    targets: List[TargetConfiguration] = Field(
        default_factory=list,
        description="Execution targets available to the run",
    )
    language_model_configuration: ModelSelectionConfiguration = Field(
        default_factory=ModelSelectionConfiguration,
        description="Role-based model selection for the run",
    )
    storage_configuration: StorageConfiguration = Field(
        default_factory=StorageConfiguration,
        description="Artifact storage configuration",
    )


class InteractionConfiguration(BaseModel):
    """
    Interaction and execution policy configuration for the run.
    """

    realignment: RealignmentPolicy = Field(
        default_factory=RealignmentPolicy,
        description="Course-correction policy",
    )
    intent_configuration: IntentConfiguration = Field(
        default_factory=IntentConfiguration,
        description="Intent strategy configuration",
    )
    execution_configuration: ExecutionConfiguration = Field(
        default_factory=ExecutionConfiguration,
        description="Execution engine configuration",
    )
    exploration_configuration: ExplorationConfiguration = Field(
        default_factory=ExplorationConfiguration,
        description="Exploration strategy configuration",
    )


class TelemetryRequestConfiguration(BaseModel):
    """
    Telemetry streaming configuration for a run.
    """

    stream_connection_string: Optional[str] = Field(
        default=None,
        description="Optional telemetry stream connection string",
    )


class RunMetadata(BaseModel):
    """
    Host-provided descriptive metadata for a run.
    """

    provider_name: Optional[str] = Field(default=None, description="Host provider name")
    device_name: Optional[str] = Field(default=None, description="Host-visible target name")

    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional host metadata labels",
    )


class RunRequest(BaseModel):
    """
    Canonical host-agnostic run contract for Fathom.
    """

    objective: IntentObjectiveConfiguration | ExplorationObjectiveConfiguration
    runtime: RuntimeConfiguration = Field(default_factory=RuntimeConfiguration)

    memory: MemoryConfiguration = Field(default_factory=MemoryConfiguration)
    resources: ResourceConfiguration = Field(default_factory=ResourceConfiguration)
    interaction: InteractionConfiguration = Field(default_factory=InteractionConfiguration)
    telemetry: TelemetryRequestConfiguration = Field(default_factory=TelemetryRequestConfiguration)

    metadata: RunMetadata = Field(default_factory=RunMetadata)

    @model_validator(mode="after")
    def __validate_targets(self) -> "RunRequest":
        """
        Ensure at least one execution target is configured.
        """

        if not self.resources.targets:
            raise ValueError("RunRequest requires at least one target configuration")

        return self


class IntentRunRequest(RunRequest):
    """
    Canonical run contract for intent-driven execution.
    """

    objective: IntentObjectiveConfiguration


class ExplorationRunRequest(RunRequest):
    """
    Canonical run contract for exploration execution.
    """

    objective: ExplorationObjectiveConfiguration

    @model_validator(mode="after")
    def __validate_exploration_objective(self) -> "ExplorationRunRequest":
        """
        Keep exploration defaults deterministic.
        """

        if self.objective.mode != ExecutionMode.EXPLORATION:
            raise ValueError("ExplorationRunRequest requires EXPLORATION mode")

        return self
