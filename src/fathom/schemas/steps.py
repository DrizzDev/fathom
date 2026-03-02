from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Action


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
    event_type: Optional[Literal["action", "validation"]] = Field(
        default=None,
        description="Semantic event type for logging/export (e.g. validation).",
    )


class StepResult(BaseModel):
    """
    The outcome of an executed step.
    """

    model_config = ConfigDict(frozen=True)

    step: Step = Field(description="The original step definition")
    success: bool = Field(description="Whether the device execution reported success")

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

    def to_record(
        self, absolute_center: Optional[List[int]] = None, activity: Optional[str] = None
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

        return StepRecord(
            bounds=bounds,
            activity=activity,
            condition=condition,
            success=self.success,
            duration=self.duration,
            center=absolute_center,
            text=self.step.action.text,
            observation=self.observation,
            target=self.step.action.target,
            is_positional=self.is_positional,
            step_number=self.step.step_number,
            screen_changed=self.screen_changed,
            rationale=self.step.action.rationale,
            generalized_target=self.generalized_target,
            event_type=self.step.event_type or "action",
            action_type=self.step.action.action_type.value,
            action_description=self.step.action.to_description(),
            natural_language_target=self.step.action.natural_language_target,
        )


class StepRecord(BaseModel):
    """
    Persistence-optimized representation of an executed step.
    """

    model_config = ConfigDict(frozen=True)

    step_number: int = Field(ge=0, description="Step index")
    event_type: Literal["action", "validation"] = Field(
        default="action",
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
    action_description: Optional[str] = Field(
        default=None, description="Human-readable NLP command"
    )

    success: bool = Field(description="Execution status")
    screen_changed: bool = Field(description="Visual transition status")
    duration: int = Field(ge=0, description="Duration in milliseconds")

    activity: Optional[str] = Field(default=None, description="Android activity at time of action")

    bounds: Optional[List[int]] = Field(
        default=None, description="Normalized [x1, y1, x2, y2] bounds"
    )
    center: Optional[List[int]] = Field(default=None, description="Absolute [x, y] coordinates")
