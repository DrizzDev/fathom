from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from fathom.schemas.actions import Action


class Step(BaseModel):
    """A planned step to execute.

    Produced by the agent's planner and contains the action to perform.
    """

    model_config = {"frozen": True}

    action: Action = Field(description="Action to execute")
    screen_hash: str = Field(description="Hash of screen when step was planned")
    step_number: int = Field(ge=0, description="Step index in sequence")
    is_conditional: bool = Field(
        default=False,
        description="Whether this step is conditional (IF block)",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Condition expression if is_conditional is True",
    )


class StepResult(BaseModel):
    """
    Result of executing a step.
    """

    model_config = {"frozen": True}

    step: Step = Field(description="The step that was executed")
    success: bool = Field(description="Whether execution succeeded")
    screen_changed: bool = Field(description="Whether screen changed after action")
    pre_hash: str = Field(description="Screen hash before action")
    post_hash: str = Field(description="Screen hash after action")
    duration: int = Field(ge=0, description="Execution duration in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def to_record(self, absolute_center: Optional[List[int]] = None) -> "StepRecord":
        """
        Convert to a minimal record for serialization.
        """
        bbox_list = None
        if self.step.action.bbox:
            # Format as [x1, y1, x2, y2] normalized
            box = self.step.action.bbox
            bbox_list = [box.x, box.y, box.x + box.width, box.y + box.height]

        return StepRecord(
            step_number=self.step.step_number,
            action_type=self.step.action.action_type.value,
            target=self.step.action.target,
            text=self.step.action.text,
            reasoning=self.step.action.reasoning,
            success=self.success,
            screen_changed=self.screen_changed,
            duration=self.duration,
            bbox=bbox_list,
            center=absolute_center,
        )


class StepRecord(BaseModel):
    """
    Minimal step record for serialization and checkpointing.
    """

    model_config = {"frozen": True}

    step_number: int = Field(ge=0)
    action_type: str = Field(min_length=1)
    target: str = Field(min_length=1)
    text: Optional[str] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)
    success: bool
    screen_changed: bool
    duration: int = Field(ge=0, description="Duration in milliseconds")
    bbox: Optional[List[int]] = Field(default=None, description="Normalized [x1, y1, x2, y2]")
    center: Optional[List[int]] = Field(default=None, description="Absolute [x, y]")
