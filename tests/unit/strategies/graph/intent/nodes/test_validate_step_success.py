"""
Regression coverage for the validation step-success contract.

A VALIDATE executes on ``ExecutionChannel.OBSERVATION`` and promises no screen change,
so an unchanged screen is not evidence it failed. ObserveNode once judged it by the
device rule, recorded ``success=False`` for a validation that confirmed what was already
true, and CompletionNode skips sub-goal evaluation on a failed step — so the sub-goal the
validation was proving could never be discharged and the run re-validated until the stuck
detector ended it.

These tests pin both halves of that contract: a successfully executed validation reaches
the completion evaluation, and reaching it is not the same as passing it. Negative,
unclear and failed validations must still not advance a sub-goal, or the fix would buy
progress by lowering the bar for it.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from fathom.constants import ActionExecutionKind, ActionType
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.agent.loop import LoopDetector
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.screens import ScreenState
from fathom.strategies.graph.intent.nodes.observe import ObserveNode


def _node() -> ObserveNode:
    """
    An ObserveNode whose provider context exposes the real command catalog.
    """

    provider = MagicMock(name="IntentNodeProvider")
    provider.context.catalog = CommandCatalogProvider().build()
    return ObserveNode(provider=provider)


def _no_progress() -> ActionEffect:
    """
    The effect reading a validation produces: nothing moved, because nothing should.
    """

    return ActionEffect(status=ActionEffectStatus.NO_PROGRESS, visual_progress=0.0, phash_distance=0)


def _step_success(
    *,
    action_type: ActionType,
    execution_success: bool,
    effect: ActionEffect,
    kind: ActionExecutionKind,
) -> bool:
    """
    Call the private step-success rule under test.
    """

    return _node()._ObserveNode__step_success(  # noqa: SLF001
        action_type=action_type,
        action_effect=effect,
        action_execution_kind=kind,
        execution_success=execution_success,
    )


class ValidateStepSuccessTest(unittest.TestCase):
    """
    A validation is judged by whether it ran, not by whether the screen moved.
    """

    def test_successful_validate_survives_an_unchanged_screen(self) -> None:
        """
        The regression: a validation that ran must record success so CompletionNode
        evaluates the sub-goal instead of skipping it on ``not step_result.success``.
        """

        self.assertTrue(
            _step_success(
                action_type=ActionType.VALIDATE,
                execution_success=True,
                effect=_no_progress(),
                kind=ActionExecutionKind.DEVICE,
            )
        )

    def test_failed_validate_execution_stays_unsuccessful(self) -> None:
        """
        Only the execution result decides. A validation whose handler failed must not be
        rescued by living on a non-interactive channel.
        """

        self.assertFalse(
            _step_success(
                action_type=ActionType.VALIDATE,
                execution_success=False,
                effect=_no_progress(),
                kind=ActionExecutionKind.DEVICE,
            )
        )

    def test_every_non_interactive_command_survives_an_unchanged_screen(self) -> None:
        """
        VALIDATE was not special. WAIT, STORE, the memory commands and COMPLETE share the
        non-interactive channels and were misjudged by the same rule.
        """

        catalog = CommandCatalogProvider().build()
        for action_type in (
            ActionType.VALIDATE,
            ActionType.WAIT,
            ActionType.STORE,
            ActionType.SAVE_MEMORY,
            ActionType.RETRIEVE_MEMORY,
            ActionType.COMPLETE,
        ):
            with self.subTest(action_type=action_type):
                self.assertTrue(
                    catalog.is_non_interactive(action_type=action_type),
                    "catalog no longer declares this command non-interactive",
                )
                self.assertTrue(
                    _step_success(
                        action_type=action_type,
                        execution_success=True,
                        effect=_no_progress(),
                        kind=ActionExecutionKind.DEVICE,
                    )
                )

    def test_interactive_commands_still_fail_on_no_progress(self) -> None:
        """
        The device rule must survive intact: a tap or gesture that moved nothing failed.

        Without this the fix would silence the signal the stuck detector depends on.
        """

        for action_type in (ActionType.TAP, ActionType.SWIPE_DOWN, ActionType.TYPE):
            with self.subTest(action_type=action_type):
                self.assertFalse(
                    _step_success(
                        action_type=action_type,
                        execution_success=True,
                        effect=_no_progress(),
                        kind=ActionExecutionKind.DEVICE,
                    )
                )

    def test_ask_user_behaviour_is_unchanged(self) -> None:
        """
        The control channel behaved correctly before this change and must be untouched.
        """

        self.assertTrue(
            _step_success(
                action_type=ActionType.ASK_USER,
                execution_success=True,
                effect=_no_progress(),
                kind=ActionExecutionKind.CONTROL,
            )
        )
        self.assertFalse(
            _step_success(
                action_type=ActionType.ASK_USER,
                execution_success=False,
                effect=_no_progress(),
                kind=ActionExecutionKind.CONTROL,
            )
        )


class ValidationDoesNotGrantCompletionTest(unittest.TestCase):
    """
    Reaching the completion evaluation is not the same as passing it.

    The fix lets a validation be evaluated. These tests pin that evaluation still
    requires confirming evidence, so a run cannot advance by validating its way forward.
    """

    def test_validate_is_declared_on_the_observation_channel(self) -> None:
        """
        The taxonomy the fix defers to. If VALIDATE ever moves off a non-interactive
        channel, ObserveNode would judge it by the device rule again and the deadlock
        would return without any change to ObserveNode itself.
        """

        catalog = CommandCatalogProvider().build()
        self.assertTrue(catalog.is_non_interactive(action_type=ActionType.VALIDATE))
        self.assertFalse(catalog.is_control(action_type=ActionType.VALIDATE))

    def test_validation_completion_requires_verified_evidence(self) -> None:
        """
        VALIDATE completes on CLAIM_VERIFIED, so a claim alone does not discharge the
        sub-goal; the completion policy must still see the evidence hold.
        """

        catalog = CommandCatalogProvider().build()
        profile = catalog.profile(action_type=ActionType.VALIDATE)
        self.assertEqual(profile.completion.value, "CLAIM_VERIFIED")


class RepeatedValidationRemainsBoundedTest(unittest.TestCase):
    """
    Validations no longer terminate a run by failing, so something else must bound them.

    Before the fix a stuck agent died at the loop-detector threshold because each
    validation recorded a failure. Now they record success, and the repetition detector
    is what stands between an inconclusive run and its whole step budget. If this guard
    ever weakens, the fix turns a run that died in five steps into one that burns forty.
    """

    @staticmethod
    def __screen(*, visual_hash: str) -> ScreenState:
        """
        One screen reading. A re-validated screen does not change, so the hash repeats.
        """

        return ScreenState(
            activity="com.android.settings",
            timestamp=0,
            activity_hash="a" * 16,
            visual_hash=visual_hash,
        )

    def test_repeated_identical_validations_trip_the_loop_detector(self) -> None:
        """
        The same validation on an unchanging screen, repeated past the threshold, is
        still detected as stuck.
        """

        detector = LoopDetector(threshold=3, window_size=10)
        for _ in range(6):
            detector.record(
                screen=self.__screen(visual_hash="f" * 16),
                action_type=ActionType.VALIDATE.value,
                action_description="Validate Settings app is open",
            )

        self.assertTrue(
            detector.is_stuck(),
            "a run that only ever re-validates the same screen must remain bounded",
        )

    def test_varied_actions_on_changing_screens_do_not_trip_the_detector(self) -> None:
        """
        A run making real progress must not be cut short by the same guard.
        """

        # Hashes far apart in Hamming distance: sequential values like 0001/0002 are
        # near-duplicates and the detector is right to call those a loop.
        screens = ("0000000000000000", "ffffffffffffffff", "0f0f0f0f0f0f0f0f")

        detector = LoopDetector(threshold=3, window_size=10)
        for visual_hash, description in zip(
            screens, ("Tap Network", "Tap Airplane mode", "Validate airplane mode on")
        ):
            detector.record(
                screen=self.__screen(visual_hash=visual_hash),
                action_type=ActionType.TAP.value,
                action_description=description,
            )

        self.assertFalse(detector.is_stuck())


if __name__ == "__main__":
    unittest.main()
