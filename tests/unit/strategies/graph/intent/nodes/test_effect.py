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
from fathom.schemas.screens import ScreenCapture, ScreenDiff, ScreenState
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


class PostActionArtifactEnvelopeTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the before/after screen artifact envelope emitted after a completed action.
    """

    @staticmethod
    def __screen_state() -> ScreenState:
        """
        Build a minimal screen state fixture for post-action comparison.
        """

        return ScreenState(
            activity="app",
            timestamp=1,
            activity_hash="activity",
            visual_hash="0" * 16,
            xml_hash="xml",
            interaction_hash="interaction",
        )

    @staticmethod
    def __capture(
        *,
        storage_id: str | None = None,
        timestamp: int = 1,
    ) -> ScreenCapture:
        """
        Build a screen capture fixture, optionally carrying a persisted artifact id.
        """

        metadata = {"storage_id": storage_id} if storage_id else {}
        return ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=b"png",
            timestamp=timestamp,
            metadata=metadata,
        )

    @staticmethod
    def __execution_context(*, capture: ScreenCapture) -> ExecutionContext:
        """
        Build the execution context consumed by :class:`PostAction`.
        """

        action = Action(
            action_type=ActionType.SWIPE_UP,
            target="Main content",
            rationale="Scroll main content.",
            confidence=1.0,
        )
        step = Step(
            action=action,
            screen_hash="0" * 16,
            step_number=7,
        )
        return ExecutionContext(
            step=step,
            capture=capture,
            pre_screen=PostActionArtifactEnvelopeTest.__screen_state(),
            localization=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                confidence=0.0,
            ),
            package="app",
        )

    async def test_observe_emits_before_and_after_artifacts_for_report_mapping(self) -> None:
        """
        A persisted pre-capture plus post-capture save produces both screen artifact sides.
        """

        saved_metadata = {}

        async def save(*, data: bytes, metadata: dict) -> str:
            """
            Capture post-action storage metadata and return a storage URI.
            """

            _ = data
            saved_metadata.update(metadata)
            return "cloud://after-screen"

        post_capture = self.__capture(timestamp=2)
        context = SimpleNamespace(
            configuration=SimpleNamespace(engine=SimpleNamespace(stability_wait=0)),
            perception_port=SimpleNamespace(capture=AsyncMock(return_value=post_capture)),
            device=SimpleNamespace(get_current_package=AsyncMock(return_value="app")),
            hierarchy=SimpleNamespace(extract_elements=Mock(return_value=[])),
            storage=SimpleNamespace(save=AsyncMock(side_effect=save)),
            telemetry=SimpleNamespace(warning=AsyncMock()),
            use_xml=False,
            is_cancelled=False,
            workflow_id="workflow-1",
        )
        observer = Mock()
        observer.resolve_capture_hashes.return_value = SimpleNamespace(
            visual_hash="f" * 16,
            xml_hash="xml-after",
            interaction_hash="interaction-after",
        )
        observer.build_screen_state.return_value = self.__screen_state()
        observer.observe = AsyncMock(return_value=None)
        comparator = Mock()
        comparator.compare.return_value = ScreenDiff(
            phash_distance=16,
            xml_hash_changed=True,
            interaction_hash_changed=True,
            activity_changed=False,
        )
        post_action = PostAction(
            context=context,
            observer=observer,
            comparator=comparator,
        )

        _, _, _, _, artifacts = await post_action.observe(
            context=self.__execution_context(
                capture=self.__capture(storage_id="cloud://before-screen"),
            ),
        )

        self.assertIsNotNone(artifacts)
        assert artifacts is not None
        self.assertIsNotNone(artifacts.screen)
        assert artifacts.screen is not None
        self.assertIsNotNone(artifacts.screen.before)
        self.assertIsNotNone(artifacts.screen.after)
        assert artifacts.screen.before is not None
        assert artifacts.screen.after is not None
        self.assertEqual(artifacts.screen.before.uri, "cloud://before-screen")
        self.assertEqual(artifacts.screen.after.uri, "cloud://after-screen")
        self.assertEqual(saved_metadata["category"], "screenshot")
        self.assertEqual(saved_metadata["phase"], "post_action")
        self.assertEqual(saved_metadata["session_id"], "workflow-1")
        self.assertEqual(saved_metadata["step_number"], 7)
