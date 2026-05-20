from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.healing.orchestrator import HealingOrchestrator
from fathom.schemas.actions import Bounds, CoordinateSystem
from fathom.schemas.budgets import HealingBudget
from fathom.schemas.healing import HealingDecisionKind, HealingRequest
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
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
            source=ElementSource.VISION,
            role=ElementRole.BUTTON,
            confidence=0.9,
            text=None,
            tappable=True,
            parent=None,
        )

    @classmethod
    def __screen(cls, *, calls_to_action: tuple[PerceivedElement, ...]) -> ScreenObservation:
        """
        Build a screen observation with visible CTA candidates.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="0" * 16,
                interaction_hash="0" * 16,
            ),
            elements=calls_to_action,
            keyboard=KeyboardObservation(visible=False),
            calls_to_action=calls_to_action,
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
                screen=self.__screen(calls_to_action=(candidate,)),
                reason=BlockReason.TARGET_UNRESOLVED,
            ),
            budget=self.__budget(),
            task_used=0,
            run_used=0,
        )

        self.assertEqual(decision.kind, HealingDecisionKind.TRY_ACTION)
        self.assertIsNotNone(decision.action)
        assert decision.action is not None
        self.assertEqual(decision.action.action_type, ActionType.TAP)
        self.assertEqual(decision.action.label_id, "cv_1")

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
            budget=self.__budget(),
            task_used=0,
            run_used=0,
        )

        self.assertEqual(decision.kind, HealingDecisionKind.FAIL_BOUNDED)

    async def test_keyboard_block_heals_with_hide_keyboard(self) -> None:
        """
        Keyboard occlusion heals with a deterministic HIDE_KEYBOARD action.
        """

        orchestrator = HealingOrchestrator()

        decision = await orchestrator.decide(
            request=HealingRequest(
                task=self.__task(),
                screen=self.__screen(calls_to_action=()),
                reason=BlockReason.KEYBOARD_OCCLUDING,
            ),
            budget=self.__budget(),
            task_used=0,
            run_used=0,
        )

        self.assertEqual(decision.kind, HealingDecisionKind.TRY_ACTION)
        self.assertIsNotNone(decision.action)
        assert decision.action is not None
        self.assertEqual(decision.action.action_type, ActionType.HIDE_KEYBOARD)
