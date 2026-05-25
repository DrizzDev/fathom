from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fathom.constants import ActionType
from fathom.constants.screen import ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD
from fathom.constants.state import IntentStateKey, PlanMetadataKey
from fathom.schemas.actions import Action
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.results import PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenDiff
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.effect import PostAction


class PostActionChangedTest(unittest.TestCase):
    """
    Pins :meth:`PostAction.changed` — the "did the screen change?" oracle.

    The check prefers the typed diff signal when available and falls
    back to a pHash hamming-distance comparison only when no diff is
    present. The pins cover all four states: diff says effect, diff
    says no-effect, no diff with identical hashes, and no diff with
    distant hashes.
    """

    @staticmethod
    def __diff(*, action_had_effect: bool) -> ScreenDiff:
        """
        :class:`ScreenDiff` fixture parameterised on the
        ``action_had_effect`` flag. Other fields are driven coherently
        so the diff looks realistic regardless of which branch the
        test exercises.
        """

        return ScreenDiff(
            phash_distance=10 if action_had_effect else 0,
            ssim_score=0.5 if action_had_effect else 1.0,
            content_pixel_diff_ratio=0.5 if action_had_effect else 0.0,
            xml_hash_changed=action_had_effect,
            interaction_hash_changed=action_had_effect,
            activity_changed=False,
        )

    def test_diff_with_effect_reports_change(self) -> None:
        """
        When the diff reports effect, hashes are ignored and change is true.
        """

        self.assertTrue(
            PostAction.changed(
                screen_diff=self.__diff(action_had_effect=True),
                pre_hash="0" * 16,
                post_hash="0" * 16,
                threshold=ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD,
            ),
        )

    def test_diff_without_effect_reports_no_change(self) -> None:
        """
        When the diff reports no effect, change is false regardless of hash distance.
        """

        self.assertFalse(
            PostAction.changed(
                screen_diff=self.__diff(action_had_effect=False),
                pre_hash="0" * 16,
                post_hash="f" * 16,
                threshold=ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD,
            ),
        )

    def test_missing_diff_falls_back_to_hash_distance(self) -> None:
        """
        Without a diff, identical hashes yield no change.
        """

        self.assertFalse(
            PostAction.changed(
                screen_diff=None,
                pre_hash="0" * 16,
                post_hash="0" * 16,
                threshold=ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD,
            ),
        )

    def test_missing_diff_with_distant_hashes_reports_change(self) -> None:
        """
        Without a diff, hashes whose hamming distance exceeds the threshold yield change.
        """

        self.assertTrue(
            PostAction.changed(
                screen_diff=None,
                pre_hash="0" * 16,
                post_hash="f" * 16,
                threshold=ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD,
            ),
        )


class PostActionPlanObservationTest(unittest.TestCase):
    """
    Pins :meth:`PostAction.plan_observation` metadata decoding.

    The planner may emit a freeform observation string in its metadata
    so it persists into :class:`StepResult.observation`. The extractor
    must defend against three malformed states: missing plan, non-
    :class:`PlanResult` plan slot, and non-string observation payload.
    Each must return ``None`` rather than leaking the raw value.
    """

    @staticmethod
    def __plan(*, observation: object) -> PlanResult:
        """
        :class:`PlanResult` fixture carrying the supplied observation
        under :attr:`PlanMetadataKey.OBSERVATION`. Parameterised on
        ``observation`` so each test can drive a different shape
        (string, dict, etc.) through the extractor.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="x",
            rationale="t",
            confidence=1.0,
        )
        step = Step(
            action=action,
            event_type="action",
            condition="x",
            screen_hash="0" * 16,
            step_number=1,
        )
        return PlanResult(
            step=step,
            metadata={PlanMetadataKey.OBSERVATION.value: observation},
            should_retry=False,
            is_complete=False,
            reason="t",
        )

    def test_returns_observation_string_when_present(self) -> None:
        """
        A string observation in plan metadata is returned verbatim.
        """

        state = {IntentStateKey.PLAN: self.__plan(observation="post-action observation text")}

        self.assertEqual(
            PostAction.plan_observation(state=state),  # type: ignore[arg-type]
            "post-action observation text",
        )

    def test_returns_none_when_observation_not_string(self) -> None:
        """
        A non-string observation value yields None instead of leaking the raw payload.
        """

        state = {IntentStateKey.PLAN: self.__plan(observation={"not": "a string"})}

        self.assertIsNone(PostAction.plan_observation(state=state))  # type: ignore[arg-type]

    def test_returns_none_when_plan_absent(self) -> None:
        """
        State without a PLAN entry yields None.
        """

        self.assertIsNone(PostAction.plan_observation(state={}))  # type: ignore[arg-type]

    def test_returns_none_when_plan_not_planresult(self) -> None:
        """
        State whose PLAN slot holds a non-PlanResult value yields None.
        """

        state = {IntentStateKey.PLAN: "raw string"}

        self.assertIsNone(PostAction.plan_observation(state=state))  # type: ignore[arg-type]


class PostActionEffectFromTest(unittest.TestCase):
    """
    Pins :meth:`PostAction.effect_from`.

    The translator now derives effect status from screen diff only; no
    outcome classifier or supervisor verdict is allowed to rewrite it.
    """

    @staticmethod
    def __diff() -> ScreenDiff:
        """
        Build a benign screen diff fixture.
        """

        return ScreenDiff(
            phash_distance=5,
            ssim_score=0.9,
            content_pixel_diff_ratio=0.05,
            xml_hash_changed=False,
            interaction_hash_changed=False,
            activity_changed=False,
        )

    def test_effect_is_derived_from_diff_only(self) -> None:
        """
        A benign diff stays uncertain.
        """

        effect = PostAction.effect_from(diff=self.__diff())

        self.assertEqual(effect.status, ActionEffectStatus.UNCERTAIN)


class PostActionCancellationTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins cancellation behavior during post-action observation.
    """

    @staticmethod
    def __capture(*, image: bytes = b"png") -> ScreenCapture:
        """
        Build a screen capture fixture.
        """

        return ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=image,
            timestamp=1,
        )

    @staticmethod
    def __execution_context() -> ExecutionContext:
        """
        Build the execution context consumed by :class:`PostAction`.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="Search bar",
            rationale="Tap search.",
            confidence=1.0,
        )
        step = Step(
            action=action,
            screen_hash="0" * 16,
            step_number=1,
        )
        return ExecutionContext(
            step=step,
            capture=PostActionCancellationTest.__capture(),
            localization=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                confidence=0.0,
            ),
            package="app",
        )

    async def test_cancellation_after_capture_skips_diff_and_observation(self) -> None:
        """
        Once cancellation is visible after capture, expensive perception stages are skipped.
        """

        context = SimpleNamespace(
            configuration=SimpleNamespace(engine=SimpleNamespace(stability_wait=0)),
            perception_port=SimpleNamespace(capture=AsyncMock(return_value=self.__capture())),
            use_xml=True,
            is_cancelled=True,
            workflow_id="wf",
        )
        observer = Mock()
        comparator = Mock()
        post_action = PostAction(
            context=context,
            observer=observer,
            comparator=comparator,
        )

        observation, screen_diff, post_hash, post_activity, artifacts = await post_action.observe(
            context=self.__execution_context(),
        )

        self.assertIsNone(observation)
        self.assertIsNone(screen_diff)
        self.assertEqual(post_hash, "0000000000000000")
        self.assertEqual(post_activity, "app")
        self.assertIsNone(artifacts)
        comparator.compare.assert_not_called()
        observer.observe.assert_not_called()
