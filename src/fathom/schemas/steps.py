from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import StepEvent
from fathom.schemas.actions import Action
from fathom.schemas.artifacts import StepArtifacts
from fathom.schemas.capture import Capture, CaptureRequest


class StepGoal(BaseModel):
    """
    Compact sub-goal context active when a step was recorded.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Sub-goal index active for this step")
    description: str = Field(min_length=1, description="Sub-goal text active for this step")
    directive: Optional[str] = Field(
        default=None, description="Expected action type for the active sub-goal"
    )


class Step(BaseModel):
    """
    A planned step containing an action and metadata.
    """

    model_config = ConfigDict(frozen=True)

    action: Action = Field(description="The action to be executed in this step")
    screen_hash: str = Field(description="Visual hash of the screen state before the action")

    step_number: int = Field(ge=0, description="The sequence number of this step")
    is_conditional: bool = Field(
        default=False, description="Whether this step is a recovery attempt"
    )
    condition: Optional[str] = Field(default=None, description="Optional condition for the step")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    event_type: Optional[StepEvent] = Field(
        default=None,
        description="Semantic event type for logging/export (e.g. validation).",
    )


class StepResult(BaseModel):
    """
    The outcome of an executed step.
    """

    model_config = ConfigDict(frozen=True)

    step: Step = Field(description="The original step definition")
    success: bool = Field(
        description="Whether the step is considered successful after observation/effect policy (semantic, may be vetoed)"
    )
    executed: bool = Field(
        default=False,
        description="Whether the command primitive ran without error (raw ExecutionResult.success, before any screen/effect judgement)",
    )
    capture: Optional[Capture] = Field(
        default=None, description="Value captured by a STORE command on this step, when one ran."
    )

    pre_hash: str = Field(description="Screen hash before execution")
    post_hash: str = Field(description="Screen hash after execution")
    screen_changed: bool = Field(description="Whether the screen visually changed after the action")

    duration: int = Field(ge=0, description="Execution duration in milliseconds")
    error: Optional[str] = Field(default=None, description="Error details if execution failed")

    observation: Optional[str] = Field(
        default=None, description="Semantic observation of the screen state"
    )

    generalized_target: Optional[str] = Field(
        default=None, description="Generalized description if target is dynamic or positional"
    )
    is_positional: bool = Field(
        default=False,
        description="Whether the generalized_target is a positional/ordinal reference",
    )

    artifacts: Optional[StepArtifacts] = Field(
        default=None,
        description="Optional namespaced artifacts captured during the step (screen.before, screen.after, ...)",
    )

    def to_record(
        self,
        absolute_center: Optional[List[int]] = None,
        activity: Optional[str] = None,
        goal: Optional[StepGoal] = None,
    ) -> "StepRecord":
        """
        Converts the result into a serializable record for persistence.
        """

        if self.step.action.bounds:
            box = self.step.action.bounds
            bounds = [box.x, box.y, box.x + box.width, box.y + box.height]
        else:
            bounds = None

        condition = getattr(self.step, "condition", None) or getattr(
            self.step.action, "condition", None
        )

        act = self.step.action
        return StepRecord(
            bounds=bounds,
            activity=activity,
            condition=condition,
            is_conditional=act.is_conditional,
            conditional_type=act.conditional_type,
            overlay_detected=act.overlay_detected,
            success=self.success,
            duration=self.duration,
            center=absolute_center,
            text=act.text,
            observation=self.observation,
            target=act.target,
            is_positional=self.is_positional,
            step_number=self.step.step_number,
            screen_changed=self.screen_changed,
            rationale=act.rationale,
            generalized_target=self.generalized_target,
            event_type=self.step.event_type or StepEvent.ACTION,
            action_type=act.action_type.value,
            action_description=act.to_description(),
            natural_language_target=act.natural_language_target,
            # Export-critical fields
            export_target=act.export_target,
            scroll_target=act.scroll_target,
            wait_subject=act.wait_subject,
            wait_pattern=act.wait_pattern,
            validation_subject=act.validation_subject,
            is_app_launcher=act.is_app_launcher,
            target_is_generic=act.target_is_generic,
            target_element_type=act.target_element_type,
            confidence=act.confidence,
            label_id=act.label_id,
            capture=self.capture,
            capture_request=act.capture,
            goal=goal,
            artifacts=self.artifacts,
        )


class StepRecord(BaseModel):
    """
    Persistence-optimized representation of an executed step.
    """

    model_config = ConfigDict(frozen=True)

    step_number: int = Field(ge=0, description="Step index")
    event_type: StepEvent = Field(
        default=StepEvent.ACTION,
        description="High-level event category used by logs and exporters.",
    )
    action_type: str = Field(min_length=1, description="Action category")
    target: str = Field(min_length=1, description="Target element description")

    natural_language_target: Optional[str] = Field(
        default=None, description="Human-friendly name of the target element"
    )
    generalized_target: Optional[str] = Field(
        default=None, description="Generalized description if target is dynamic or positional"
    )
    is_positional: bool = Field(
        default=False,
        description="Whether the generalized_target is a positional/ordinal reference",
    )
    text: Optional[str] = Field(default=None, description="Typed text content")
    rationale: Optional[str] = Field(default=None, description="Reasoning for the action")
    observation: Optional[str] = Field(default=None, description="Screen state observation")
    condition: Optional[str] = Field(default=None, description="Condition for IF-block wrapping")
    is_conditional: bool = Field(
        default=False, description="Whether this action is explicitly conditional"
    )
    conditional_type: Optional[Literal["blocker", "transient", "error", "optional"]] = Field(
        default=None, description="Conditional category for deterministic IF guards"
    )
    overlay_detected: bool = Field(
        default=False, description="Whether this action handled an overlay/popup blocker"
    )
    action_description: Optional[str] = Field(
        default=None, description="Human-readable NLP command"
    )

    success: bool = Field(description="Execution status")
    screen_changed: bool = Field(description="Visual transition status")
    duration: int = Field(ge=0, description="Duration in milliseconds")

    activity: Optional[str] = Field(default=None, description="Android activity at time of action")
    execution_activity: Optional[str] = Field(
        default=None, description="Pre-action activity/package the step executed on"
    )

    bounds: Optional[List[int]] = Field(
        default=None, description="Normalized [x1, y1, x2, y2] bounds"
    )
    center: Optional[List[int]] = Field(default=None, description="Absolute [x, y] coordinates")

    # Export-critical fields (VLM-provided, persisted for downstream consumers)
    export_target: Optional[str] = Field(
        default=None, description="Canonical phrase for exported test scripts"
    )
    scroll_target: Optional[str] = Field(
        default=None, description="Element or section being scrolled to find"
    )
    wait_subject: Optional[str] = Field(default=None, description="What is being waited for")
    wait_pattern: Optional[str] = Field(
        default=None, description="Wait category: ad, splash, load, search, generic"
    )
    validation_subject: Optional[str] = Field(
        default=None, description="Structured state or subject asserted by a validation action"
    )
    is_app_launcher: bool = Field(default=False, description="Whether this tap launches an app")
    target_is_generic: Optional[bool] = Field(
        default=None, description="Whether target is non-specific"
    )
    target_element_type: Optional[str] = Field(
        default=None, description="Element type/role (button, icon, etc.)"
    )
    confidence: Optional[float] = Field(
        default=None, description="VLM confidence score for this action"
    )
    label_id: Optional[str] = Field(
        default=None, description="Grounding label ID from XML manifest"
    )
    capture: Optional[Capture] = Field(
        default=None, description="Value captured by a STORE command on this step, when one ran."
    )
    capture_request: Optional[CaptureRequest] = Field(
        default=None,
        description="The STORE request (name, subject, value) the planner emitted; drives script generation.",
    )
    goal: Optional[StepGoal] = Field(
        default=None,
        description="Compact sub-goal context active when this step was recorded.",
    )
    artifacts: Optional[StepArtifacts] = Field(
        default=None,
        description="Optional namespaced artifacts captured during the step.",
    )


class StepHistory(BaseModel):
    """
    Validated payload of a run's persisted history file.
    """

    model_config = ConfigDict(extra="ignore")

    history: List[StepRecord] = Field(description="Recorded step records, in order.")
