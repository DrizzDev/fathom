from __future__ import annotations

from fathom.authoring.agent.prompts.base import AuthoringPrompt


class RepairAuthoringPrompt(AuthoringPrompt):
    """
    Prompt strategy for repairing an existing script or flow.
    """

    def objective(self) -> str:
        """
        Return the repair authoring objective.
        """

        return (
            "Task objective: repair the supplied script or Flow. Read every review issue, "
            "then change only the commands or fields needed to satisfy that issue. Preserve "
            "correct parts. Use evidence and artifacts to resolve ambiguity. Return a valid "
            "Flow that satisfies the dialect reference and deterministic review."
        )
