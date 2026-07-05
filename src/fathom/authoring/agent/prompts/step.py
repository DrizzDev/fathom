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
            "Task objective: author the best command for the selected step. Use the "
            "selected step action data, surrounding run context, planner reasoning, "
            "screen observation, and optional screenshot or manifest artifacts when "
            "they are supplied. Produce one replayable Flow fragment for that step "
            "only. If the step is only an attempt within a larger episode and cannot "
            "stand alone faithfully, return a partial Flow rather than inventing "
            "missing context."
        )
