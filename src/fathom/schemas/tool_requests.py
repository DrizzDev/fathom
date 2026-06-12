"""
Pydantic request models for tool call validation.

Exploration-only — provides ExploreUIRequest for the explore_ui tool.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


class ScreenTranslation(BaseModel):
    """
    Structured describe_screen output: a functional view of a screen.

    Captures what is on the screen, what each element does, and what a user
    can achieve here — using stable labels, not volatile runtime data.
    """

    activity: str = Field(
        default="", alias="activity_name", description="Android activity this screen belongs to"
    )
    purpose: str = Field(
        default="",
        alias="screen_purpose",
        description="What the screen is for and the primary tasks available here",
    )
    elements: str = Field(
        default="",
        description="Each element: what it is, its stable label, and what it does or leads to",
    )
    actions: str = Field(
        default="",
        alias="achievable_actions",
        description="Concrete things a user can accomplish on this screen",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("activity", "purpose", "elements", "actions", mode="before")
    @classmethod
    def __coerce_none(cls, value: Any) -> Any:
        """Treat a missing/null field as an empty section."""
        return "" if value is None else value

    def to_markdown(self) -> str:
        """Render the translation as the rich-description markdown stored per activity."""

        parts: List[str] = []
        if self.activity:
            parts.append(f"**Activity:** `{self.activity}`")
        for heading, body in (
            ("Purpose", self.purpose),
            ("Elements", self.elements),
            ("What You Can Do", self.actions),
        ):
            if body.strip():
                parts.append(f"## {heading}\n{body}")
        return "\n\n".join(parts)
