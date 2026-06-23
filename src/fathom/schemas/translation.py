"""
Structured describe_screen output for the exploration strategy.
"""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fathom.constants.screen import ScreenCategory


class ScreenTranslation(BaseModel):
    """
    Functional view of a screen: what is on it, what each element does, and what
    a user can achieve here, using stable labels rather than volatile runtime data.
    """

    activity: str = Field(
        default="", alias="activity_name", description="Android activity this screen belongs to"
    )
    category: ScreenCategory = Field(
        default=ScreenCategory.OTHER,
        alias="screen_category",
        description="The functional kind of screen this is, used to group per-screen docs",
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
        """
        Treat a missing or null field as an empty section.
        """

        return "" if value is None else value

    @field_validator("category", mode="before")
    @classmethod
    def __coerce_category(cls, value: Any) -> ScreenCategory:
        """
        Map a missing or unrecognised category onto OTHER so an off-list value
        from the model never fails the whole translation parse.
        """

        if isinstance(value, ScreenCategory):
            return value
        try:
            return ScreenCategory(str(value).strip().lower())
        except (ValueError, AttributeError):
            return ScreenCategory.OTHER

    def to_markdown(self) -> str:
        """
        Renders the translation as the rich-description markdown stored per activity.
        """

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
