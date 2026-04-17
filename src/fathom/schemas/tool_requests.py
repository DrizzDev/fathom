"""
Pydantic request models for tool call validation.

Exploration-only — provides ExploreUIRequest for the explore_ui tool.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExploreUIRequest(BaseModel):
    """Request schema for explore_ui tool call (exploration mode only).

    Lean model — no delta signals, no validation, no memory updates,
    no script export.  Only the fields needed for exploration.
    """

    action: Dict[str, Any] = Field(
        description="Action object with action_type, rationale, target_name, tap_target."
    )
    assistant_message: str = Field(description="Reasoning for choosing this element")
    screen_description: str = Field(
        default="Exploration step", description="1-2 sentence screen summary"
    )
    content_exhausted: Optional[bool] = Field(
        default=False, description="True when all visible elements have been tried"
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure action dict contains required fields."""
        if not isinstance(v, dict):
            raise ValueError("action must be a dictionary")
        if "action_type" not in v:
            raise ValueError("action must contain 'action_type'")
        return v
