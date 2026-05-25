from __future__ import annotations

from typing import Sequence

import pytest

from fathom.core.services.decomposer import IntentDecomposer
from fathom.interfaces.llm import LLMPort
from fathom.schemas.subgoal import SubGoal

pytestmark = pytest.mark.release

VARO_ONBOARDING_INTENT = (
    "I want to launch the Varo app and login with "
    "dev+test+Ilu+z2O5@varomoney.com and password Password1! "
    'I then want you to step through the onboarding wizard and get to the main "Home" screen.'
)


class DecompositionAssertions:
    """
    Assertion helpers for live decomposition invariants.
    """

    @staticmethod
    def joined_descriptions(*, sub_goals: Sequence[SubGoal]) -> str:
        """
        Return normalized sub-goal descriptions for invariant matching.
        """

        return " ".join(goal.description for goal in sub_goals).casefold()

    @staticmethod
    def assert_contains_in_order(*, text: str, phrases: Sequence[str]) -> None:
        """
        Assert all phrases appear in order within text.
        """

        cursor = -1
        for phrase in phrases:
            position = text.find(phrase.casefold())
            assert position > cursor, f"Expected {phrase!r} after offset {cursor}: {text}"
            cursor = position


class TestIntentDecomposer:
    """
    Live LLM checks for IntentDecomposer.
    """

    async def test_varo_onboarding_intent_preserves_required_steps(self, live_llm: LLMPort) -> None:
        """
        Varo onboarding must decompose into ordered, executable sub-goals.
        """

        decomposer = IntentDecomposer(llm=live_llm)

        sub_goals = await decomposer.decompose(intent=VARO_ONBOARDING_INTENT)
        descriptions = DecompositionAssertions.joined_descriptions(sub_goals=sub_goals)

        assert len(sub_goals) >= 3
        assert "dev+test+ilu+z2o5@varomoney.com" in descriptions
        assert "password1!" in descriptions
        assert "varo" in descriptions
        assert "home" in descriptions
        assert "onboarding" in descriptions or "wizard" in descriptions
        DecompositionAssertions.assert_contains_in_order(
            text=descriptions,
            phrases=("launch", "login", "password", "home"),
        )

    async def test_compound_scroll_and_select_intent_stays_atomic(self, live_llm: LLMPort) -> None:
        """
        Compound navigation-to-action wording must not be over-split.
        """

        decomposer = IntentDecomposer(llm=live_llm)

        sub_goals = await decomposer.decompose(
            intent="Scroll to labs section and select any category"
        )
        descriptions = DecompositionAssertions.joined_descriptions(sub_goals=sub_goals)

        assert len(sub_goals) == 1
        assert "scroll to labs section" in descriptions
        assert "select any category" in descriptions
