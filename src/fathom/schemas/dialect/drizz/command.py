from __future__ import annotations

from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import Field, model_validator

from fathom.constants.flow import ScrollDirection
from fathom.schemas.base import SealedModel
from fathom.schemas.dialect.drizz.target import Assertion, Target


class OpenAppCommand(SealedModel):
    """
    Launch an application by package name.
    """

    command: Literal["open_app"] = "open_app"
    package: str = Field(min_length=1, description="Target application package.")


class TapCommand(SealedModel):
    """
    Tap a UI target.
    """

    command: Literal["tap"] = "tap"
    target: Target = Field(description="Target to tap.")


class TypeCommand(SealedModel):
    """
    Type a value into a field.
    """

    command: Literal["type"] = "type"
    value: str = Field(min_length=1, description="Text value to enter.")
    field: Target = Field(description="Field to type into.")


class ScrollCommand(SealedModel):
    """
    Scroll in a direction, optionally until a target appears.
    """

    command: Literal["scroll"] = "scroll"
    direction: ScrollDirection = Field(description="Scroll direction.")
    until: Optional[Target] = Field(default=None, description="Target to scroll until visible.")
    container: Optional[str] = Field(default=None, description="Container scrolled inside.")
    percentage: Optional[int] = Field(
        default=None, ge=1, le=100, description="Fraction of the view to scroll by."
    )

    @model_validator(mode="after")
    def __percentage_is_exclusive(self) -> "ScrollCommand":
        """
        A percentage scroll cannot also carry an until target or a container.
        """

        if self.percentage is not None and (self.until is not None or self.container is not None):
            raise ValueError("Scroll percentage cannot combine with until or container.")

        return self


class WaitCommand(SealedModel):
    """
    Wait for a duration in seconds or until a subject appears.
    """

    command: Literal["wait"] = "wait"
    subject: Optional[str] = Field(default=None, min_length=1, description="Subject to wait for.")
    duration: Optional[int] = Field(
        default=None, ge=0, description="Wait duration in whole seconds."
    )

    @model_validator(mode="after")
    def __require_one_form(self) -> "WaitCommand":
        """
        Require at least one of duration or subject.
        """

        if self.duration is None and self.subject is None:
            raise ValueError("Wait command requires a duration or a subject.")

        return self


class BackCommand(SealedModel):
    """
    Press the device back button.
    """

    command: Literal["back"] = "back"


class KillCommand(SealedModel):
    """
    Force-close the active application.
    """

    command: Literal["kill"] = "kill"


class ClearCommand(SealedModel):
    """
    Clear the active application's data.
    """

    command: Literal["clear"] = "clear"


class MinimizeCommand(SealedModel):
    """
    Send the active application to the background.
    """

    command: Literal["minimize"] = "minimize"


class SetGpsCommand(SealedModel):
    """
    Set the device GPS coordinates.
    """

    command: Literal["set_gps"] = "set_gps"
    latitude: float = Field(description="Latitude in decimal degrees.")
    longitude: float = Field(description="Longitude in decimal degrees.")


class StoreCommand(SealedModel):
    """
    Store a captured value under a variable name.
    """

    command: Literal["store"] = "store"
    value: str = Field(min_length=1, description="Captured value to store.")
    name: str = Field(min_length=1, description="Variable name to store under.")


class ValidateCommand(SealedModel):
    """
    Assert one or more UI states.
    """

    command: Literal["validate"] = "validate"
    assertions: Tuple[Assertion, ...] = Field(min_length=1, description="Assertions to verify.")


class MapActionCommand(SealedModel):
    """
    Tap a target on a map or canvas surface.
    """

    command: Literal["map_action"] = "map_action"
    target: Target = Field(description="Map target to tap.")


LeafCommand = Annotated[
    Union[
        TapCommand,
        TypeCommand,
        WaitCommand,
        BackCommand,
        KillCommand,
        ClearCommand,
        StoreCommand,
        ScrollCommand,
        SetGpsCommand,
        OpenAppCommand,
        MinimizeCommand,
        ValidateCommand,
        MapActionCommand,
    ],
    Field(discriminator="command"),
]


class IfCommand(SealedModel):
    """
    Conditionally execute a body of leaf commands when a visibility condition holds.
    """

    command: Literal["if"] = "if"
    condition: str = Field(min_length=1, description="Visibility condition text.")
    body: Tuple[LeafCommand, ...] = Field(min_length=1, description="Commands run when true.")


DrizzCommand = Annotated[
    Union[
        IfCommand,
        TapCommand,
        TypeCommand,
        WaitCommand,
        BackCommand,
        KillCommand,
        ClearCommand,
        StoreCommand,
        SetGpsCommand,
        ScrollCommand,
        OpenAppCommand,
        MinimizeCommand,
        ValidateCommand,
        MapActionCommand,
    ],
    Field(discriminator="command"),
]
