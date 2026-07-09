from __future__ import annotations

from fathom.authoring.agent.prompts.base import AuthoringPrompt
from fathom.authoring.agent.prompts.repair import RepairAuthoringPrompt
from fathom.authoring.agent.prompts.run import RunAuthoringPrompt
from fathom.authoring.agent.prompts.step import StepAuthoringPrompt
from fathom.constants.authoring import AuthoringKind
from fathom.core.exceptions import InvariantViolation


class AuthoringPromptFactory:
    """
    Selects the prompt strategy for an authoring task kind.
    """

    def prompt(self, *, kind: AuthoringKind) -> AuthoringPrompt:
        """
        Return the prompt strategy for the requested authoring kind.
        """

        if kind is AuthoringKind.RUN:
            return RunAuthoringPrompt()

        if kind is AuthoringKind.STEP:
            return StepAuthoringPrompt()

        if kind is AuthoringKind.REPAIR:
            return RepairAuthoringPrompt()

        raise InvariantViolation(f"Unsupported authoring task kind '{kind.value}'.")
