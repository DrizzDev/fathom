from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.supervision.policies import (
    BudgetPolicy,
    KeyboardPolicy,
    OverlayPolicy,
    RepetitionPolicy,
    SafetyPolicy,
    ScrollPolicy,
    TargetPolicy,
)
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import (
    KeyboardObservation,
    OverlayObservation,
    ScreenObservation,
)
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.supervision import BlockReason
from fathom.schemas.tasks import (
    ExecutionTask,
    ExecutionTaskState,
    TaskAttemptState,
)


def _hashes() -> ScreenHashBundle:
    """
    Return a minimal screen hash bundle for fixture observations.
    """

    return ScreenHashBundle(
        visual_hash="0" * 16,
        xml_hash="a" * 16,
        interaction_hash="b" * 16,
    )


def _observation(
    *,
    keyboard_visible: bool = False,
    overlays: tuple = (),
    scroll: tuple = (),
    calls_to_action: tuple = (),
) -> ScreenObservation:
    """
    Build a minimal ScreenObservation with the supplied feature flags.
    """

    return ScreenObservation(
        activity="screen",
        hashes=_hashes(),
        elements=(),
        keyboard=KeyboardObservation(visible=keyboard_visible),
        overlays=overlays,
        scroll=scroll,
        calls_to_action=calls_to_action,
    )


def _action(
    *,
    action_type: ActionType = ActionType.TAP,
    target: str = "Continue",
    rationale: str = "fixture action",
) -> Action:
    """
    Build a minimal Action for policy fixtures.
    """

    return Action(
        action_type=action_type,
        target=target,
        rationale=rationale,
        confidence=0.9,
    )


def _resolved(*, bounds: Bounds = None) -> LocalizationResult:
    """
    Build a resolved LocalizationResult.
    """

    return LocalizationResult(
        status=LocalizationStatus.RESOLVED,
        confidence=1.0,
        bounds=bounds,
    )


def _task(*, count: int = 0, limit: int = 5) -> ExecutionTask:
    """
    Build an ExecutionTask with the supplied attempt accounting.
    """

    return ExecutionTask(
        identifier="task:0",
        objective="objective",
        criterion="criterion",
        state=ExecutionTaskState.ACTIVE,
        attempts=TaskAttemptState(count=count, limit=limit),
    )


class TargetPolicyTest(unittest.TestCase):
    """
    Pins for TargetPolicy block decisions across localization statuses.
    """

    def test_resolved_returns_none(self) -> None:
        """
        Resolved localization must not produce a block reason.
        """

        self.assertIsNone(TargetPolicy().evaluate(localization=_resolved()))

    def test_unresolved_returns_target_unresolved(self) -> None:
        """
        Unresolved localization must surface TARGET_UNRESOLVED.
        """

        verdict = TargetPolicy().evaluate(
            localization=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                confidence=0.0,
            )
        )

        self.assertEqual(verdict, BlockReason.TARGET_UNRESOLVED)

    def test_ambiguous_returns_target_ambiguous(self) -> None:
        """
        Ambiguous localization must surface TARGET_AMBIGUOUS.
        """

        verdict = TargetPolicy().evaluate(
            localization=LocalizationResult(
                status=LocalizationStatus.AMBIGUOUS,
                confidence=0.4,
            )
        )

        self.assertEqual(verdict, BlockReason.TARGET_AMBIGUOUS)


class RepetitionPolicyTest(unittest.TestCase):
    """
    Pins for RepetitionPolicy floor semantics on consecutive no-progress counts.
    """

    def test_below_floor_returns_none(self) -> None:
        """
        Counts below the recovery threshold do not block.
        """

        self.assertIsNone(RepetitionPolicy().evaluate(count=1))

    def test_at_floor_returns_repeated_no_effect(self) -> None:
        """
        Counts at the recovery threshold surface REPEATED_NO_EFFECT.
        """

        self.assertEqual(
            RepetitionPolicy().evaluate(count=3),
            BlockReason.REPEATED_NO_EFFECT,
        )


class KeyboardPolicyTest(unittest.TestCase):
    """
    Pins for KeyboardPolicy keyboard+swipe interaction.
    """

    def test_keyboard_hidden_allows_swipe(self) -> None:
        """
        Swipe with keyboard hidden must not be blocked.
        """

        self.assertIsNone(
            KeyboardPolicy().evaluate(
                action=_action(action_type=ActionType.SWIPE_UP),
                observation=_observation(keyboard_visible=False),
            )
        )

    def test_keyboard_visible_blocks_swipe(self) -> None:
        """
        Swipe with keyboard visible must surface KEYBOARD_OCCLUDING.
        """

        self.assertEqual(
            KeyboardPolicy().evaluate(
                action=_action(action_type=ActionType.SWIPE_UP),
                observation=_observation(keyboard_visible=True),
            ),
            BlockReason.KEYBOARD_OCCLUDING,
        )

    def test_keyboard_visible_allows_tap(self) -> None:
        """
        Non-swipe actions are not blocked by an open keyboard.
        """

        self.assertIsNone(
            KeyboardPolicy().evaluate(
                action=_action(action_type=ActionType.TAP),
                observation=_observation(keyboard_visible=True),
            )
        )


class ScrollPolicyTest(unittest.TestCase):
    """
    Pins for ScrollPolicy blind-scroll detection.
    """

    def test_non_swipe_returns_none(self) -> None:
        """
        Non-swipe actions are outside scroll-policy scope.
        """

        self.assertIsNone(
            ScrollPolicy().evaluate(
                action=_action(action_type=ActionType.TAP),
                observation=_observation(),
                no_progress=5,
            )
        )

    def test_blind_swipe_blocks_when_no_scroll_evidence(self) -> None:
        """
        Swipes with no scroll evidence and prior no-progress are blocked.
        """

        self.assertEqual(
            ScrollPolicy().evaluate(
                action=_action(action_type=ActionType.SWIPE_UP),
                observation=_observation(),
                no_progress=1,
            ),
            BlockReason.NON_SCROLLABLE_SURFACE,
        )

    def test_first_swipe_returns_none(self) -> None:
        """
        The first swipe attempt without prior no-progress is allowed.
        """

        self.assertIsNone(
            ScrollPolicy().evaluate(
                action=_action(action_type=ActionType.SWIPE_UP),
                observation=_observation(),
                no_progress=0,
            )
        )


class OverlayPolicyTest(unittest.TestCase):
    """
    Pins for OverlayPolicy overlay-behind blocking.
    """

    def test_no_overlays_returns_none(self) -> None:
        """
        With no overlays the policy must not block.
        """

        self.assertIsNone(
            OverlayPolicy().evaluate(
                action=_action(),
                observation=_observation(),
                localization=_resolved(
                    bounds=Bounds(
                        x=10,
                        y=10,
                        width=20,
                        height=20,
                        coordinate_system="pixel",
                        source="model",
                    )
                ),
            )
        )

    def test_overlay_with_target_outside_blocks(self) -> None:
        """
        Overlay present but action bounds outside overlay candidate must block.
        """

        overlay = OverlayObservation(
            bounds=Bounds(
                x=0,
                y=0,
                width=100,
                height=100,
                coordinate_system="pixel",
                source="model",
            ),
            visible=True,
            candidates=(),
        )

        verdict = OverlayPolicy().evaluate(
            action=_action(action_type=ActionType.TAP),
            observation=_observation(overlays=(overlay,)),
            localization=_resolved(
                bounds=Bounds(
                    x=500,
                    y=500,
                    width=20,
                    height=20,
                    coordinate_system="pixel",
                    source="model",
                )
            ),
        )

        self.assertEqual(verdict, BlockReason.OVERLAY_STILL_PRESENT)


class BudgetPolicyTest(unittest.TestCase):
    """
    Pins for BudgetPolicy attempt-budget enforcement.
    """

    def test_no_task_returns_none(self) -> None:
        """
        Missing task means no budget enforcement applies.
        """

        self.assertIsNone(BudgetPolicy().evaluate(task=None))

    def test_under_budget_returns_none(self) -> None:
        """
        Tasks with attempts below limit are allowed to continue.
        """

        self.assertIsNone(BudgetPolicy().evaluate(task=_task(count=2, limit=5)))

    def test_over_budget_blocks(self) -> None:
        """
        Tasks that reached their attempt cap surface TASK_BUDGET_EXCEEDED.
        """

        self.assertEqual(
            BudgetPolicy().evaluate(task=_task(count=5, limit=5)),
            BlockReason.TASK_BUDGET_EXCEEDED,
        )


class SafetyPolicyTest(unittest.TestCase):
    """
    Pins for SafetyPolicy keyword vocabulary matching.
    """

    def test_benign_action_returns_none(self) -> None:
        """
        Actions with safe vocabulary must not be blocked.
        """

        self.assertIsNone(
            SafetyPolicy().evaluate(action=_action(target="Continue", rationale="ok"))
        )

    def test_unsafe_keyword_in_target_blocks(self) -> None:
        """
        Actions whose target contains an unsafe keyword are blocked.
        """

        self.assertEqual(
            SafetyPolicy().evaluate(action=_action(target="factory reset", rationale="safe")),
            BlockReason.UNSAFE_ACTION,
        )

    def test_unsafe_keyword_in_rationale_blocks(self) -> None:
        """
        Actions whose rationale contains an unsafe keyword are blocked.
        """

        self.assertEqual(
            SafetyPolicy().evaluate(
                action=_action(target="Settings", rationale="this will wipe the device")
            ),
            BlockReason.UNSAFE_ACTION,
        )
