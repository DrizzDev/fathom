from __future__ import annotations

from typing import Final


class RepeatedFailureRejectionPromptBuilder:
    """
    Builds the rejection-history sentence fed back to the LLM on the should_avoid_action branch.
    """

    __HEADER: Final[str] = (
        "REJECTED: the proposed action {descriptor} has already failed on the current screen."
    )
    __INTERACTIVE_GUIDANCE: Final[str] = (
        "Choose a different action that advances the active sub-goal, or emit ask_user "
        "if no safe alternative exists on this screen."
    )
    __NON_INTERACTIVE_GUIDANCE: Final[str] = (
        "Choose a different action on the current screen, or navigate back (back / home) "
        "to a previous screen and try a different path. Do not re-emit the rejected action."
    )

    @classmethod
    def build(cls, *, action_descriptor: str, interactive: bool) -> str:
        """
        Return the rejection sentence sized to the current capability surface.
        """

        descriptor = action_descriptor.strip() or "(unknown)"
        guidance = cls.__INTERACTIVE_GUIDANCE if interactive else cls.__NON_INTERACTIVE_GUIDANCE

        header = cls.__HEADER.format(descriptor=repr(descriptor))
        return f"{header} {guidance}"
