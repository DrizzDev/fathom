from __future__ import annotations

from fathom.authoring.agent.prompts.base import AuthoringPrompt


class StepAuthoringPrompt(AuthoringPrompt):
    """
    Prompt strategy for single-step command authoring.
    """

    def objective(self) -> str:
        """
        Return the single-step authoring objective.
        """

        return (
            "Task objective: author the best replayable command for the selected "
            "step. Use the selected step action data, surrounding run context, "
            "planner reasoning, screen observation, and optional screenshot or "
            "manifest artifacts when they are supplied. Produce only the command or "
            "commands that actually executed in the selected step; do not add waits, "
            "validations, taps, or follow-up actions that were not executed by that "
            "step. If the step is only an attempt within a larger episode and "
            "cannot stand alone faithfully, return a partial Flow rather than "
            "inventing missing context."
        )
