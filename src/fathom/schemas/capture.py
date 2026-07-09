from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator, model_validator

from fathom.schemas.base import SealedModel


class CaptureRequest(SealedModel):
    """
    Intent-derived value to store under a name; carried by a planned STORE action, never a runtime prompt.
    """

    name: str = Field(min_length=1, description="Variable name the captured value is stored under.")
    subject: str = Field(
        min_length=1,
        description="What the intent asked to capture, e.g. price of the selected item.",
    )
    value: str = Field(
        min_length=1, description="Actual value read from the screen or task context."
    )

    @field_validator("name", "subject", "value")
    @classmethod
    def __strip_required_text(cls, value: str) -> str:
        """
        Normalize required text and reject whitespace-only capture payload fields.
        """

        stripped = value.strip()
        if not stripped:
            raise ValueError("Capture request fields must not be blank.")

        return stripped


class Capture(SealedModel):
    """
    Run-owned outcome of one STORE command: a named value captured from the screen, or a failure.
    """

    name: str = Field(min_length=1, description="Variable name the captured value is stored under.")
    step: int = Field(ge=0, description="Step number that produced this capture.")
    success: bool = Field(description="Whether a value was captured successfully.")
    value: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Captured value; present only on a successful capture.",
    )
    reason: Optional[str] = Field(
        default=None, min_length=1, description="Failure reason; present only on a failed capture."
    )

    @field_validator("name")
    @classmethod
    def __strip_name(cls, value: str) -> str:
        """
        Normalize the capture name and reject whitespace-only names.
        """

        stripped = value.strip()
        if not stripped:
            raise ValueError("Capture name must not be blank.")

        return stripped

    @field_validator("value", "reason")
    @classmethod
    def __strip_optional_text(cls, value: Optional[str]) -> Optional[str]:
        """
        Normalize optional capture text and reject whitespace-only values when present.
        """

        if value is None:
            return None

        stripped = value.strip()
        if not stripped:
            raise ValueError("Capture value and reason must not be blank.")

        return stripped

    @model_validator(mode="after")
    def __consistent(self) -> "Capture":
        """
        Keep success/failure explicit: a success carries a value and no reason, a failure the reverse.
        """

        if self.success and (self.value is None or self.reason is not None):
            raise ValueError("A successful capture must carry a value and no failure reason.")

        if not self.success and (self.value is not None or self.reason is None):
            raise ValueError("A failed capture must carry a failure reason and no value.")

        return self

    @classmethod
    def succeeded(cls, *, name: str, value: str, step: int) -> "Capture":
        """
        Build a successful capture of a named value.
        """

        return cls(name=name, step=step, success=True, value=value)

    @classmethod
    def failed(cls, *, name: str, reason: str, step: int) -> "Capture":
        """
        Build a failed capture carrying the failure reason.
        """

        return cls(name=name, step=step, success=False, reason=reason)
