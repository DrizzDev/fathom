from __future__ import annotations

from enum import StrEnum
from typing import FrozenSet

from pydantic import BaseModel, ConfigDict, Field


class EscapeCategory(StrEnum):
    """
    Typed taxonomy for the agent's structured "I cannot make safe
    forward progress" signal.

    Each value names a distinct condition the agent can observe on the
    current screen relative to the active sub-goal. The category drives
    the response: replan around the current screen, ask the human, or
    re-frame the decomposer's next pass. A single tool exposes the
    whole taxonomy so the prompt teaches *categories of escape* rather
    than one symptom per tool.

    Categories:

    - ``TARGET_NOT_AVAILABLE``: the active sub-goal names a target
      that cannot be grounded by either path — neither a matching
      element in the manifest nor a visually identifiable element on
      the current screenshot. A target that is absent from the
      manifest but visible on screen is NOT this category; the agent
      should ground it via bbox instead.
    - ``WRONG_SCREEN``: the agent is clearly on a different screen
      than the sub-goal expects (e.g. a debug overlay instead of the
      app's home, a permissions sheet instead of the search results).
    - ``PRECONDITION_NOT_MET``: the sub-goal assumes a prior state the
      agent has not reached (e.g. checkout before items in cart, edit
      profile before login).
    - ``AMBIGUOUS_TARGET``: multiple candidates (manifest labels or
      visible elements) plausibly match the named target and no safe
      disambiguation is available; the system routes this to the human
      rather than guessing.
    - ``UNSAFE_ACTION``: proceeding with the named action would be
      irreversible or destructive (delete account, charge card); the
      system routes this to the human.
    """

    WRONG_SCREEN = "wrong_screen"
    UNSAFE_ACTION = "unsafe_action"
    AMBIGUOUS_TARGET = "ambiguous_target"
    TARGET_NOT_AVAILABLE = "target_not_available"
    PRECONDITION_NOT_MET = "precondition_not_met"


# Categories whose remedy is a fresh decomposition against the current
# screen — surfaced to the recovery coordinator via the REQUEST_REPLAN
# trigger. Listed explicitly so a new category gains its routing intent
# at one well-named site instead of being inferred from naming.
REPLAN_ESCAPE_CATEGORIES: FrozenSet[EscapeCategory] = frozenset(
    {
        EscapeCategory.WRONG_SCREEN,
        EscapeCategory.TARGET_NOT_AVAILABLE,
        EscapeCategory.PRECONDITION_NOT_MET,
    }
)

# Categories whose remedy is human disambiguation rather than replan.
# The planner emits an ASK_USER action with the escape detail as the
# question text so the human sees what the agent could not resolve.
HUMAN_ESCAPE_CATEGORIES: FrozenSet[EscapeCategory] = frozenset(
    {
        EscapeCategory.UNSAFE_ACTION,
        EscapeCategory.AMBIGUOUS_TARGET,
    }
)


class EscapeReport(BaseModel):
    """
    Structured payload the agent emits via the ``request_replan`` tool
    when it cannot make safe forward progress on the active sub-goal.

    Carries both the typed ``category`` (drives routing) and a free-text
    ``detail`` (surfaced to the decomposer preamble or to the human in
    the HITL prompt). The category is mandatory; an empty escape report
    is not meaningful and would defeat the typed contract.
    """

    model_config = ConfigDict(frozen=True)

    category: EscapeCategory = Field(
        description=(
            "Typed reason the agent cannot proceed. Drives whether the "
            "system replans against the current screen or escalates to the human."
        )
    )
    detail: str = Field(
        min_length=1,
        description=(
            "One-sentence explanation in the agent's own words. Used "
            "as the decomposer preamble (replan path) or the human prompt body (HITL path)."
        ),
    )

    def routes_to_replan(self) -> bool:
        """
        Whether this report should dispatch the recovery coordinator with the ``REQUEST_REPLAN`` trigger.
        """

        return self.category in REPLAN_ESCAPE_CATEGORIES

    def routes_to_human(self) -> bool:
        """
        Whether this report should surface to the human via the ASK_USER path instead of replanning.
        """

        return self.category in HUMAN_ESCAPE_CATEGORIES
