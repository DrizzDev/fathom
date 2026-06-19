from __future__ import annotations

from typing import Final, Mapping, Optional

from fathom.schemas.escalation import EscalationPrompt, StuckSource
from fathom.schemas.subgoal import SubGoal


class EscalationPromptBuilder:
    """
    Builds escalation prompts that explain why the agent is asking for help.

    Two surfaces: a short ``rationale`` written for the agent log (may carry technical context for RCA) and a
    ``question`` written for the human user (kept free of internal vocabulary like "sub-goal" or "step budget").
    """

    __RATIONALES: Final[Mapping[StuckSource, str]] = {
        StuckSource.LOOP_DETECTOR: (
            "Loop detected (screen repeating). Requesting human intervention."
        ),
        StuckSource.SUBGOAL_BUDGET: (
            "Sub-goal step budget exhausted without observable progress. "
            "Requesting human intervention."
        ),
    }
    __FALLBACK_RATIONALE: Final[str] = (
        "Agent unable to make progress. Requesting human intervention."
    )

    __QUESTION_TEMPLATES: Final[Mapping[StuckSource, str]] = {
        StuckSource.LOOP_DETECTOR: (
            "I'm having trouble making progress{step_clause}{action_clause}. "
            "Could you tell me what to do next?"
        ),
        StuckSource.SUBGOAL_BUDGET: (
            "I've tried several times{step_clause}{action_clause} "
            "but can't seem to complete it. How would you like me to proceed?"
        ),
    }
    __FALLBACK_QUESTION: Final[str] = (
        "I'm unable to make progress{step_clause}{action_clause}. How would you like me to proceed?"
    )

    __STEP_CLAUSE: Final[str] = ' on "{description}"'
    __ACTION_CLAUSE: Final[str] = " after repeatedly trying {descriptor}"

    @classmethod
    def build(
        cls,
        *,
        source: StuckSource,
        current_sub_goal: Optional[SubGoal],
        last_action_description: Optional[str],
    ) -> EscalationPrompt:
        """
        Return an :class:`EscalationPrompt` tailored to the stuck source and active step.
        """

        step_clause = cls.__step_clause(current_sub_goal=current_sub_goal)
        action_clause = cls.__action_clause(last_action_description=last_action_description)

        rationale = cls.__RATIONALES.get(source, cls.__FALLBACK_RATIONALE)
        template = cls.__QUESTION_TEMPLATES.get(source, cls.__FALLBACK_QUESTION)

        question = template.format(step_clause=step_clause, action_clause=action_clause)

        return EscalationPrompt(rationale=rationale, question=question)

    @classmethod
    def __step_clause(cls, *, current_sub_goal: Optional[SubGoal]) -> str:
        """
        Render the user-facing step context, or empty when no step is active.
        """

        if current_sub_goal is None or not current_sub_goal.description.strip():
            return ""

        return cls.__STEP_CLAUSE.format(description=current_sub_goal.description.strip())

    @classmethod
    def __action_clause(cls, *, last_action_description: Optional[str]) -> str:
        """
        Render the repeated-action clause, or empty when no descriptor is available.
        """

        if not last_action_description:
            return ""

        descriptor = last_action_description.strip()
        if not descriptor:
            return ""

        return cls.__ACTION_CLAUSE.format(descriptor=descriptor)
