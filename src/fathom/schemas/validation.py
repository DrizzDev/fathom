from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.constants.turn.validation import ValidationSource
from fathom.schemas.base.common import SealedModel


class Validation(SealedModel):
    """
    Canonical validation assertion; the single construction point every validation door passes through.
    """

    source: ValidationSource = Field(description="Door the validation entered through.")
    subject: str = Field(min_length=1, description="Non-empty condition the validation asserts.")

    @classmethod
    def command(cls, *, subject: Optional[str]) -> Optional["Validation"]:
        """
        Build from the execute_ui validate command's dedicated validation_subject field.
        """

        return cls.__build(subject=subject, source=ValidationSource.COMMAND)

    @classmethod
    def state(cls, *, condition: Optional[str]) -> Optional["Validation"]:
        """
        Build from the validate_state tool's condition_to_verify field.
        """

        return cls.__build(subject=condition, source=ValidationSource.STATE)

    @classmethod
    def goal(
        cls,
        *,
        subgoal: Optional[str],
        goal: Optional[str],
        screen: Optional[str],
        assertion: Optional[str] = None,
    ) -> Optional["Validation"]:
        """
        Build from the verify_goal tool, preferring the explicit assertion, then completion reasons,
        then the screen description.
        """

        for candidate in (assertion, subgoal, goal, screen):
            if (built := cls.__build(subject=candidate, source=ValidationSource.GOAL)) is not None:
                return built

        return None

    @classmethod
    def __build(cls, *, subject: Optional[str], source: ValidationSource) -> Optional["Validation"]:
        """
        Return a validation for a non-empty subject, None otherwise.
        """

        if subject is None or not (text := subject.strip()):
            return None

        return cls(subject=text, source=source)
