from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.healing.orchestrator import HealingOrchestrator
from fathom.schemas.actions import Bounds, CoordinateSystem
from fathom.schemas.budgets import HealingBudget
from fathom.schemas.configuration import IntentConfiguration
from fathom.schemas.healing import HealingDecisionKind, HealingRequest
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.perception import KeyboardConfiguration, PerceptionConfiguration
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.supervision import BlockReason
from fathom.schemas.tasks import ExecutionTask, ExecutionTaskState, TaskAttemptState


class HealingOrchestratorTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers deterministic healing decisions.
    """

    @staticmethod
    def __budget() -> HealingBudget:
        """
        Build a small healing budget.
        """

        return HealingBudget(task=2, run=5)

    @staticmethod
    def __task() -> ExecutionTask:
        """
        Build a minimal active task.
        """

        return ExecutionTask(
            identifier="task:1",
            objective="close overlay",
            criterion="overlay is gone",
            state=ExecutionTaskState.ACTIVE,
            attempts=TaskAttemptState(count=0, limit=5),
        )

    @staticmethod
    def __candidate(*, identifier: str) -> PerceivedElement:
        """
        Build a tappable CTA candidate.
        """

        return PerceivedElement(
            identifier=identifier,
            bounds=Bounds(
                x=100,
                y=400,
                width=200,
                height=80,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            text=None,
            parent=None,
            tappable=True,
            confidence=0.9,
            role=ElementRole.BUTTON,
            source=ElementSource.VISION,
        )

    @classmethod
    def __screen(cls, *, calls_to_action: tuple[PerceivedElement, ...]) -> ScreenObservation:
        """
        Build a screen observation with visible CTA candidates.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                xml_hash="0" * 16,
                visual_hash="0" * 16,
                interaction_hash="0" * 16,
            ),
            elements=calls_to_action,
            calls_to_action=calls_to_action,
            keyboard=KeyboardObservation(visible=False),
        )

    async def test_target_unresolved_single_cta_is_tapped(self) -> None:
        """
        A screenshot-only unresolved target can heal by tapping the only CTA.
        """

        orchestrator = HealingOrchestrator()
        candidate = self.__candidate(identifier="cv_1")

        decision = await orchestrator.decide(
            request=HealingRequest(
                task=self.__task(),
                reason=BlockReason.TARGET_UNRESOLVED,
                screen=self.__screen(calls_to_action=(candidate,)),
            ),
            run_used=0,
            task_used=0,
            budget=self.__budget(),
        )

        self.assertIsNotNone(decision.action)
        self.assertEqual(decision.kind, HealingDecisionKind.TRY_ACTION)

        assert decision.action is not None
        self.assertEqual(decision.action.label_id, "cv_1")
        self.assertEqual(decision.action.action_type, ActionType.TAP)

    async def test_target_unresolved_multiple_ctas_fails_bounded(self) -> None:
        """
        Multiple CTAs are ambiguous and must not be guessed deterministically.
        """

        orchestrator = HealingOrchestrator()

        decision = await orchestrator.decide(
            request=HealingRequest(
                task=self.__task(),
                screen=self.__screen(
                    calls_to_action=(
                        self.__candidate(identifier="cv_1"),
                        self.__candidate(identifier="cv_2"),
                    )
                ),
                reason=BlockReason.TARGET_UNRESOLVED,
            ),
            run_used=0,
            task_used=0,
            budget=self.__budget(),
        )

        self.assertEqual(decision.kind, HealingDecisionKind.FAIL_BOUNDED)

    async def test_keyboard_block_is_disabled_by_default(self) -> None:
        """
        Keyboard healing is off by default while scroll stability is prioritized.
        """

        orchestrator = HealingOrchestrator()

        decision = await orchestrator.decide(
            request=HealingRequest(
                task=self.__task(),
                reason=BlockReason.KEYBOARD_OCCLUDING,
                screen=self.__screen(calls_to_action=()),
            ),
            run_used=0,
            task_used=0,
            budget=self.__budget(),
        )

        self.assertIsNone(decision.action)
        self.assertEqual(decision.kind, HealingDecisionKind.FAIL_BOUNDED)

    async def test_keyboard_block_heals_with_hide_keyboard_when_enabled(self) -> None:
        """
        Keyboard healing can still be explicitly enabled for controlled runs.
        """

        orchestrator = HealingOrchestrator(
            perception_configuration=PerceptionConfiguration(
                keyboard=KeyboardConfiguration(enabled=True)
            ),
            runtime_policy=IntentConfiguration.RuntimePolicyConfiguration(
                keyboard=IntentConfiguration.KeyboardRuntimeConfiguration(allow_recovery=True)
            ),
        )

        decision = await orchestrator.decide(
            request=HealingRequest(
                task=self.__task(),
                reason=BlockReason.KEYBOARD_OCCLUDING,
                screen=self.__screen(calls_to_action=()),
            ),
            run_used=0,
            task_used=0,
            budget=self.__budget(),
        )

        self.assertIsNotNone(decision.action)
        self.assertEqual(decision.kind, HealingDecisionKind.TRY_ACTION)

        assert decision.action is not None
        self.assertEqual(decision.action.action_type, ActionType.HIDE_KEYBOARD)

    async def test_repeated_no_effect_does_not_mutate_scroll_into_cta_tap(self) -> None:
        """
        Repeated ineffective scrolls must not heal by tapping unrelated visible CTAs.
        """

        orchestrator = HealingOrchestrator()

        decision = await orchestrator.decide(
            request=HealingRequest(
                task=self.__task().model_copy(
                    update={"objective": "scroll until Asha Tiffin is visible"}
                ),
                reason=BlockReason.REPEATED_NO_EFFECT,
                screen=self.__screen(calls_to_action=(self.__candidate(identifier="cv_1"),)),
            ),
            run_used=0,
            task_used=0,
            budget=self.__budget(),
        )

        self.assertIsNone(decision.action)
        self.assertEqual(decision.kind, HealingDecisionKind.FAIL_BOUNDED)
