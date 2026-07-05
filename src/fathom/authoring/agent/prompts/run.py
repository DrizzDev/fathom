from __future__ import annotations

from fathom.authoring.agent.prompts.base import AuthoringPrompt


class RunAuthoringPrompt(AuthoringPrompt):
    """
    Prompt strategy for whole-run script authoring.
    """

    def objective(self) -> str:
        """
        Return the whole-run authoring objective.
        """

        return (
            "Task objective: author the complete run script. Read episodes as the "
            "user-level execution structure, then produce the clean replay sequence for "
            "the whole run. Merge repeated attempts only when they serve the same episode "
            "purpose. Use step drafts, screenshots, manifests, and other artifacts when "
            "they are supplied to improve target wording and assertions. Preserve "
            "recorded command semantics and runtime values. Keep assertions that prove "
            "meaningful states. Return a complete Flow only when the evidence proves "
            "completion; otherwise return a partial Flow."
        )
