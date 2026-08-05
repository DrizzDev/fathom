from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field, field_validator

from fathom.constants import ActionType
from fathom.constants.flow import ScrollDirection, SwipeDirection
from fathom.schemas.base.common import NonBlank, SealedModel


class PressRequirement(SealedModel):
    """
    A user-requested tap or long-press on a named target.
    """

    operation: Literal[ActionType.TAP, ActionType.LONG_PRESS] = Field(
        description="Press operation the user requested."
    )
    target: NonBlank = Field(description="Element the press acts on.")


class TypeRequirement(SealedModel):
    """
    A user-requested text entry into a named target.
    """

    operation: Literal[ActionType.TYPE] = Field(description="Type operation the user requested.")
    target: NonBlank = Field(description="Field the text is entered into.")
    text: NonBlank = Field(description="Text the user asked to enter.")


class ScrollRequirement(SealedModel):
    """
    A user-requested scroll: content moves in the stated direction.
    """

    operation: Literal[ActionType.SCROLL] = Field(
        description="Scroll operation the user requested."
    )
    direction: ScrollDirection = Field(
        description="Content movement direction; the device adapter converts it to a physical gesture."
    )
    target: Optional[NonBlank] = Field(default=None, description="Optional scroll surface.")


class SwipeRequirement(SealedModel):
    """
    A user-requested swipe: the finger moves in the stated direction.
    """

    operation: Literal[ActionType.SWIPE] = Field(description="Swipe operation the user requested.")
    direction: SwipeDirection = Field(
        description="Finger movement direction, used verbatim by the device adapter."
    )
    target: Optional[NonBlank] = Field(default=None, description="Optional gesture surface.")


class WaitRequirement(SealedModel):
    """
    A user-requested wait for an observable condition, bounded in time.
    """

    operation: Literal[ActionType.WAIT] = Field(description="Wait operation the user requested.")
    condition: NonBlank = Field(description="Observable condition awaited.")
    # ``ge`` (inclusive) not ``gt``: Gemini structured output rejects the ``exclusiveMinimum`` that
    # ``gt`` emits. A positive bound is enforced by the validator below.
    bound: float = Field(ge=0.0, description="Maximum wait duration in seconds.")

    @field_validator("bound")
    @classmethod
    def __positive_bound(cls, value: float) -> float:
        """
        A wait bound must be strictly positive; a zero-second wait is not an admissible wait.
        """

        if value <= 0.0:
            raise ValueError("wait bound must be greater than zero")
        return value


class NavigationRequirement(SealedModel):
    """
    A user-requested device navigation with no target or payload.
    """

    operation: Literal[ActionType.BACK, ActionType.HOME, ActionType.HIDE_KEYBOARD] = Field(
        description="Navigation operation the user requested."
    )


CommandRequirement = Annotated[
    Union[
        PressRequirement,
        TypeRequirement,
        ScrollRequirement,
        SwipeRequirement,
        WaitRequirement,
        NavigationRequirement,
    ],
    Field(
        discriminator="operation",
        description="Canonical user-requested device operation and its typed parameters.",
    ),
]
