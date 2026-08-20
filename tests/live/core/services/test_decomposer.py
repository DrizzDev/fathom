from __future__ import annotations

from typing import Sequence

import pytest

from fathom.constants import ActionType
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.translation import ProposalTranslator
from fathom.interfaces.llm import LLMPort
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.success import CaptureSuccess, CommandSuccess, ObservedSuccess

pytestmark = pytest.mark.release

BANKING_ONBOARDING_INTENT = (
    "I want to launch the banking app and login with "
    "dev+test+Ilu+z2O5@example.com and password Password1! "
    'I then want you to step through the onboarding wizard and get to the main "Home" screen.'
)

CONDITIONAL_STORE_INTENT = (
    "Open Shopping and then, Search for Ghar soaps, check whether customer rating is >= 4.2 "
    "or not if it is, store the price of selected item as item_price and then proceed to buy "
    "until login screen"
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

        return " ".join(goal.objective for goal in sub_goals).casefold()

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

    @staticmethod
    def index_of_directive(*, sub_goals: Sequence[SubGoal], directive: ActionType) -> int:
        """
        Return the first sub-goal index whose canonical success matches the requested directive.
        """

        for index, sub_goal in enumerate(sub_goals):
            success = sub_goal.success
            if directive is ActionType.STORE and isinstance(success, CaptureSuccess):
                return index
            if directive is ActionType.VALIDATE and isinstance(success, ObservedSuccess):
                return index
            if isinstance(success, CommandSuccess) and success.requirement.operation is directive:
                return index

        raise AssertionError(f"Missing directive {directive!r}: {sub_goals!r}")


class TestIntentDecomposer:
    """
    Live LLM checks for IntentDecomposer.
    """

    async def test_banking_onboarding_intent_preserves_required_steps(self, llm: LLMPort) -> None:
        """
        Banking onboarding must decompose into ordered, executable sub-goals.
        """

        decomposer = IntentDecomposer(
            llm=llm, translator=ProposalTranslator(catalog=CommandCatalogProvider().build())
        )

        sub_goals = await decomposer.decompose(intent=BANKING_ONBOARDING_INTENT)
        descriptions = DecompositionAssertions.joined_descriptions(sub_goals=sub_goals)

        assert len(sub_goals) >= 3
        assert "dev+test+ilu+z2o5@example.com" in descriptions
        assert "password1!" in descriptions
        assert "banking" in descriptions
        assert "home" in descriptions
        assert "onboarding" in descriptions or "wizard" in descriptions
        DecompositionAssertions.assert_contains_in_order(
            text=descriptions,
            phrases=("launch", "login", "password", "home"),
        )

    async def test_compound_scroll_and_select_intent_stays_atomic(self, llm: LLMPort) -> None:
        """
        Compound navigation-to-action wording must not be over-split.
        """

        decomposer = IntentDecomposer(
            llm=llm, translator=ProposalTranslator(catalog=CommandCatalogProvider().build())
        )

        sub_goals = await decomposer.decompose(
            intent="Scroll to labs section and select any category"
        )
        descriptions = DecompositionAssertions.joined_descriptions(sub_goals=sub_goals)

        assert len(sub_goals) == 1
        assert "scroll to labs section" in descriptions
        assert "select any category" in descriptions

    async def test_conditional_store_intent_separates_validation_store_and_followup(
        self, llm: LLMPort
    ) -> None:
        """
        A checked precondition before STORE must become VALIDATE, STORE, then follow-up.
        """

        decomposer = IntentDecomposer(
            llm=llm, translator=ProposalTranslator(catalog=CommandCatalogProvider().build())
        )

        sub_goals = await decomposer.decompose(intent=CONDITIONAL_STORE_INTENT)
        descriptions = DecompositionAssertions.joined_descriptions(sub_goals=sub_goals)

        validate_index = DecompositionAssertions.index_of_directive(
            sub_goals=sub_goals, directive=ActionType.VALIDATE
        )
        store_index = DecompositionAssertions.index_of_directive(
            sub_goals=sub_goals, directive=ActionType.STORE
        )
        tap_index = DecompositionAssertions.index_of_directive(
            sub_goals=sub_goals[store_index + 1 :], directive=ActionType.TAP
        )

        assert validate_index < store_index
        assert store_index < store_index + 1 + tap_index
        assert "customer rating" in descriptions
        assert "item_price" in descriptions
        assert "proceed to buy" in descriptions or "login screen" in descriptions

        store_description = sub_goals[store_index].objective.casefold()
        assert "check whether customer rating" not in store_description
        assert "store" in store_description or "capture" in store_description
