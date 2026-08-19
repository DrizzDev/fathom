from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock

from fathom.constants import ActionType
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.constants.screen import ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD
from fathom.constants.state import IntentStateKey
from fathom.schemas.actions import Action
from fathom.schemas.artifact import ArtifactRecord, ScreenshotPayload
from fathom.schemas.artifacts import ScreenArtifact
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.results import (
    ActionTraceAttempt,
    ActionTraceEvent,
    PlanContext,
    PlanResult,
    TraceEmission,
)
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
        from the plan context observation. Parameterised on
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
            context=PlanContext(observation=observation),
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
        Once cancellation is visible after capture, expensive perception
        stages are skipped, but the before artifact still carries the
        capture bytes so the caller can render the pre-action frame.
        """

        context = SimpleNamespace(
            configuration=SimpleNamespace(
                engine=SimpleNamespace(stability_wait=0, transition_grace_period=0)
            ),
            device=SimpleNamespace(
                capture_screen=AsyncMock(return_value=b"post"),
                get_current_package=AsyncMock(return_value="app"),
            ),
            perception_port=SimpleNamespace(capture=AsyncMock(return_value=self.__capture())),
            storage=SimpleNamespace(backend=None),
            telemetry=AsyncMock(),
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
        self.assertIsNotNone(artifacts)
        self.assertIsNotNone(artifacts.screen.before)
        self.assertEqual(artifacts.screen.before.image, b"png")
        self.assertIsNone(artifacts.screen.after)
        comparator.compare.assert_not_called()
        observer.observe.assert_not_called()

    async def test_post_action_observation_timeout_degrades_without_hanging(self) -> None:
        """
        A hung post-action perception capture returns degraded evidence instead of wedging the run.
        """

        async def capture_slowly() -> ScreenCapture:
            """
            Simulate a wedged perception adapter.
            """

            await asyncio.sleep(delay=1.0)
            return self.__capture()

        context = SimpleNamespace(
            configuration=SimpleNamespace(
                engine=SimpleNamespace(stability_wait=0, transition_grace_period=0),
                device=SimpleNamespace(
                    type=DeviceConnectionType.LOCAL,
                    platform=DevicePlatform.ANDROID,
                    android=SimpleNamespace(snapshot_timeout=0.01),
                ),
            ),
            device=SimpleNamespace(
                configuration=None,
                capture_screen=AsyncMock(return_value=b"post"),
                get_current_package=AsyncMock(return_value="app"),
            ),
            perception_port=SimpleNamespace(capture=AsyncMock(side_effect=capture_slowly)),
            storage=SimpleNamespace(backend=None),
            telemetry=SimpleNamespace(warning=AsyncMock()),
            use_xml=True,
            is_cancelled=False,
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
        self.assertIsNone(artifacts)
        self.assertEqual(post_hash, "0000000000000000")
        self.assertEqual(post_activity, "app")
        context.telemetry.warning.assert_awaited_once()


def _capture(
    *,
    image: bytes = b"raw-png",
    screenshot_uri: Optional[str] = None,
    annotated_image: Optional[bytes] = None,
    annotated_uri: Optional[str] = None,
) -> ScreenCapture:
    """
    Build a configurable :class:`ScreenCapture` for the three builder tests.
    """

    return ScreenCapture(
        width=1080,
        height=2400,
        activity="com.test.app",
        image=image,
        annotated_image=annotated_image,
        screenshot_uri=screenshot_uri,
        annotated_uri=annotated_uri,
        timestamp=1_714_200_000_000,
        metadata={},
    )


class PostActionArtifactBuildersTest(unittest.IsolatedAsyncioTestCase):
    """
    Pin the three artifact builders consumed by :class:`PostAction`; before/annotated read URI fields
    populated upstream, after stages the post-action capture through the pipeline and stamps the URI back.
    """

    def __post_action(self, *, pipeline) -> PostAction:
        """
        Build a PostAction with a mocked context exposing only the surfaces the artifact path reads.
        """

        context = MagicMock()
        context.workflow_id = "wf-1"
        context.artifact_pipeline = pipeline
        context.agent_state.step_count = 4
        context.storage = MagicMock()
        context.storage.backend = None
        return PostAction(
            context=context,
            observer=MagicMock(),
            comparator=MagicMock(),
        )

    def test_before_carries_capture_bytes_and_uri(self) -> None:
        """
        The before artifact carries the raw capture bytes verbatim and the URI stamped by PerceptionService.
        """

        post_action = self.__post_action(pipeline=None)
        capture = _capture(image=b"pre-png", screenshot_uri="gs://b/before.png")

        artifact = post_action._PostAction__build_before(capture=capture, visual_hash="h-b")  # type: ignore[attr-defined]

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.image, b"pre-png")
        self.assertEqual(artifact.uri, "gs://b/before.png")

    def test_before_returns_none_when_capture_is_empty(self) -> None:
        """
        A capture without bytes or URI yields no before artifact.
        """

        post_action = self.__post_action(pipeline=None)
        capture = _capture(image=b"", screenshot_uri=None)

        artifact = post_action._PostAction__build_before(capture=capture)  # type: ignore[attr-defined]

        self.assertIsNone(artifact)

    def test_annotated_carries_annotated_bytes_and_uri(self) -> None:
        """
        The annotated artifact carries the annotated bytes rendered by HierarchyService and the matching URI.
        """

        post_action = self.__post_action(pipeline=None)
        capture = _capture(annotated_image=b"anno-png", annotated_uri="gs://b/anno.png")

        artifact = post_action._PostAction__build_annotated(capture=capture, visual_hash="h-a")  # type: ignore[attr-defined]

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.image, b"anno-png")
        self.assertEqual(artifact.uri, "gs://b/anno.png")

    def test_annotated_returns_none_when_annotation_missing(self) -> None:
        """
        Without annotated bytes or URI, no annotated artifact is produced.
        """

        post_action = self.__post_action(pipeline=None)
        capture = _capture(annotated_image=None, annotated_uri=None)

        artifact = post_action._PostAction__build_annotated(capture=capture)  # type: ignore[attr-defined]

        self.assertIsNone(artifact)

    async def test_after_emits_screenshot_payload_and_stamps_uri(self) -> None:
        """
        Post-action artifact carries ``step_count + 1`` (1-based step number).
        """

        pipeline = MagicMock()
        pipeline.emit = AsyncMock(return_value=Path("/efs/staged/after.png"))
        post_action = self.__post_action(pipeline=pipeline)
        capture = _capture(image=b"post-png")

        artifact = await post_action._PostAction__build_after(  # type: ignore[attr-defined]
            capture=capture,
            package_name="com.test.app",
            visual_hash="hash-after",
        )

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.image, b"post-png")
        self.assertEqual(artifact.uri, "/efs/staged/after.png")
        pipeline.emit.assert_awaited_once()
        record = pipeline.emit.call_args.kwargs["record"]
        self.assertIsInstance(record, ArtifactRecord)
        self.assertIsInstance(record.payload, ScreenshotPayload)
        self.assertEqual(record.session_id, "wf-1")
        self.assertEqual(record.package_name, "com.test.app")
        self.assertEqual(record.step_number, 5)

    async def test_after_carries_bytes_when_pipeline_missing(self) -> None:
        """
        Pipeline-less embeddings still produce an after artifact carrying the capture bytes with no URI.
        """

        post_action = self.__post_action(pipeline=None)
        capture = _capture(image=b"post-png")

        artifact = await post_action._PostAction__build_after(  # type: ignore[attr-defined]
            capture=capture,
            package_name="com.test.app",
            visual_hash=None,
        )

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.image, b"post-png")
        self.assertIsNone(artifact.uri)

    async def test_after_carries_bytes_when_pipeline_emit_raises(self) -> None:
        """
        A pipeline emit failure is swallowed; the after artifact still ships capture bytes for the report.
        """

        pipeline = MagicMock()
        pipeline.emit = AsyncMock(side_effect=RuntimeError("pipeline broker down"))
        post_action = self.__post_action(pipeline=pipeline)
        capture = _capture(image=b"post-png")

        artifact = await post_action._PostAction__build_after(  # type: ignore[attr-defined]
            capture=capture,
            package_name="com.test.app",
            visual_hash=None,
        )

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.image, b"post-png")
        self.assertIsNone(artifact.uri)
        pipeline.emit.assert_awaited_once()

    async def test_after_returns_none_when_capture_has_no_bytes(self) -> None:
        """
        A capture that yielded no bytes produces no after artifact.
        """

        post_action = self.__post_action(pipeline=None)
        capture = _capture(image=b"")

        artifact = await post_action._PostAction__build_after(  # type: ignore[attr-defined]
            capture=capture,
            package_name="com.test.app",
            visual_hash=None,
        )

        self.assertIsNone(artifact)


class PostActionBuildTracesTest(unittest.TestCase):
    """
    Pin __build_traces; maps executor emissions to bundle artifacts while preserving order and dropping
    emissions whose pipeline staging produced no handle.
    """

    @staticmethod
    def __post_action() -> PostAction:
        """
        Build a PostAction with mocked dependencies sufficient for invoking __build_traces directly.
        """

        context = MagicMock()
        context.workflow_id = "wf-1"
        context.storage = MagicMock()
        context.storage.backend = None
        return PostAction(
            context=context,
            observer=MagicMock(),
            comparator=MagicMock(),
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Build a ScreenCapture fixture shared by every trace event emitted in this class.
        """

        return ScreenCapture(
            width=1080,
            height=2400,
            activity="com.test.app",
            image=b"png",
            timestamp=1_714_200_000_000,
        )

    def test_no_emissions_returns_none(self) -> None:
        """
        An empty emissions tuple means no trace path ran; the helper returns None to mark the absent slot.
        """

        result = self.__post_action()._PostAction__build_traces(emissions=())  # type: ignore[attr-defined]

        self.assertIsNone(result)

    def test_emissions_map_to_ordered_artifact_tuple(self) -> None:
        """
        Each emission with a staged artifact contributes one entry to the returned tuple in original order.
        """

        emissions = (
            TraceEmission(
                event=ActionTraceEvent(
                    capture=self.__capture(),
                    coords=(10, 20),
                    attempt=ActionTraceAttempt(index=0),
                ),
                artifact=ScreenArtifact(uri="cdn://trace-0"),
            ),
            TraceEmission(
                event=ActionTraceEvent(
                    capture=self.__capture(),
                    coords=(30, 40),
                    attempt=ActionTraceAttempt(index=1),
                ),
                artifact=ScreenArtifact(uri="cdn://trace-1"),
            ),
        )

        result = self.__post_action()._PostAction__build_traces(emissions=emissions)  # type: ignore[attr-defined]

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([artifact.uri for artifact in result], ["cdn://trace-0", "cdn://trace-1"])

    def test_emissions_without_artifact_are_dropped(self) -> None:
        """
        Emissions whose pipeline staging returned no handle drop out so consumers iterate only on real artifacts.
        """

        emissions = (
            TraceEmission(
                event=ActionTraceEvent(capture=self.__capture(), coords=(10, 20)),
                artifact=ScreenArtifact(uri="cdn://trace-0"),
            ),
            TraceEmission(
                event=ActionTraceEvent(capture=self.__capture(), coords=(30, 40)),
                artifact=None,
            ),
        )

        result = self.__post_action()._PostAction__build_traces(emissions=emissions)  # type: ignore[attr-defined]

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].uri, "cdn://trace-0")

    def test_all_emissions_without_artifact_yield_empty_tuple(self) -> None:
        """
        When the trace path ran but every emission failed to stage, the helper returns an empty tuple, not None.
        Caller distinguishes "ran-empty" from "never-ran" by tuple-vs-None.
        """

        emissions = (
            TraceEmission(
                event=ActionTraceEvent(capture=self.__capture(), coords=(10, 20)),
                artifact=None,
            ),
        )

        result = self.__post_action()._PostAction__build_traces(emissions=emissions)  # type: ignore[attr-defined]

        self.assertEqual(result, ())


class PostActionCompareEnrichmentSkipTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the post-action enrichment skip in __compare: the heavy
    ``observer.observe`` call is intentionally bypassed; downstream code
    falls back to the pre-action observation on the next ANALYZE turn.
    """

    @staticmethod
    def __capture(*, image: bytes = b"post-bytes") -> ScreenCapture:
        """
        Build a minimal ScreenCapture for the post-action observation step.
        """

        return ScreenCapture(
            width=1080,
            height=2400,
            activity="com.test.app",
            image=image,
            timestamp=1_714_200_000_000,
        )

    @staticmethod
    def __screen_diff() -> ScreenDiff:
        """
        Build a benign ScreenDiff fixture so the comparator returns a valid object.
        """

        return ScreenDiff(
            phash_distance=0,
            ssim_score=1.0,
            content_pixel_diff_ratio=0.0,
            xml_hash_changed=False,
            interaction_hash_changed=False,
            activity_changed=False,
        )

    def __build_post_action(self) -> tuple:
        """
        Build a PostAction wired against MagicMocks; return (post_action, observer, context).
        """

        from fathom.schemas.screens import ScreenHashBundle, ScreenState

        context = MagicMock()
        context.workflow_id = "wf-1"
        context.is_cancelled = False
        context.use_xml = False
        context.artifact_pipeline = None
        context.storage = MagicMock()
        context.storage.backend = None
        context.agent_state.step_count = 1

        post_capture = self.__capture()
        context.perception_port.capture = AsyncMock(return_value=post_capture)

        observer = MagicMock()
        observer.resolve_capture_hashes = MagicMock(
            return_value=ScreenHashBundle(
                visual_hash="vh",
                xml_hash="xh",
                interaction_hash="ih",
            )
        )
        observer.build_screen_state = MagicMock(
            return_value=ScreenState(
                activity="com.test.app",
                timestamp=1_714_200_000_000,
                activity_hash="ah",
                visual_hash="vh",
                xml_hash="xh",
                interaction_hash="ih",
            )
        )
        observer.observe = AsyncMock()

        comparator = MagicMock()
        comparator.compare = MagicMock(return_value=self.__screen_diff())

        post_action = PostAction(
            context=context,
            observer=observer,
            comparator=comparator,
        )
        return post_action, observer, context

    def __execution_context(self) -> ExecutionContext:
        """
        Build an execution context that is ineligible for settle recapture.
        """

        return ExecutionContext(
            package="com.test.app",
            step=Step(
                step_number=1,
                screen_hash="0",
                action=Action(action_type=ActionType.TAP, rationale="tap"),
            ),
            capture=self.__capture(image=b"before"),
            localization=LocalizationResult(status=LocalizationStatus.RESOLVED, confidence=1.0),
            execution_result=None,
        )

    async def test_observer_observe_is_not_called_in_post_action_path(self) -> None:
        """
        The post-action compare path bypasses observer.observe entirely so OCR/icon enrichment never re-runs.
        """

        post_action, observer, _ = self.__build_post_action()

        await post_action._PostAction__compare(  # type: ignore[attr-defined]
            context=self.__execution_context(),
            package_name="com.test.app",
            before_capture=self.__capture(image=b"before"),
            before_state=None,
            trace_emissions=(),
        )

        observer.observe.assert_not_called()

    async def test_post_observation_is_none(self) -> None:
        """
        The returned PostActionObservation carries no observation, leaving the next turn's GROUND to rebuild it.
        """

        post_action, _, _ = self.__build_post_action()

        result = await post_action._PostAction__compare(  # type: ignore[attr-defined]
            context=self.__execution_context(),
            package_name="com.test.app",
            before_capture=self.__capture(image=b"before"),
            before_state=None,
            trace_emissions=(),
        )

        self.assertIsNone(result.observation)

    async def test_enrichment_skipped_event_logged(self) -> None:
        """
        An ``observe.post.enrichment_skipped`` event is emitted with the documented reason.
        """

        post_action, _, _ = self.__build_post_action()

        with self.assertLogs(
            "fathom.strategies.graph.intent.nodes.effect", level="INFO"
        ) as captured:
            await post_action._PostAction__compare(  # type: ignore[attr-defined]
                context=self.__execution_context(),
                package_name="com.test.app",
                before_capture=self.__capture(image=b"before"),
                before_state=None,
                trace_emissions=(),
            )

        skip_events = [
            record
            for record in captured.records
            if record.__dict__.get("event") == "observe.post.enrichment_skipped"
        ]
        self.assertEqual(len(skip_events), 1)
        self.assertEqual(skip_events[0].__dict__["reason"], "ground_rebuilds_observation_next_turn")
