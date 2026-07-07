from __future__ import annotations

import asyncio
import time
from io import BytesIO
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image

from fathom.constants import ActionType
from fathom.constants.execution import POST_ACTION_OBSERVATION_TIMEOUT_SECONDS
from fathom.constants.perception import ACTION_REGION_HALF_SIDE, ACTION_REGION_STATIC_HAMMING_FLOOR
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.constants.screen import ZERO_HASH
from fathom.constants.state import IntentStateKey, PlanMetadataKey
from fathom.constants.storage import StorageBackend
from fathom.core.perception.hashing import VisualHashEngine
from fathom.core.services.settlement import ScreenSettlementService
from fathom.schemas.artifact import ArtifactRecord, ScreenshotPayload
from fathom.schemas.artifacts import (
    ScreenArtifact,
    ScreenArtifactBundle,
    StepArtifacts,
)
from fathom.schemas.effect import ActionEffect
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.observation import PostActionObservation, ScreenObservation
from fathom.schemas.results import PlanResult, TraceEmission
from fathom.schemas.screens import ScreenCapture, ScreenDiff, ScreenState
from fathom.schemas.settlement import (
    PostActionScreen,
    PreActionScreen,
    ScreenSettlementEvidence,
)
from fathom.schemas.ui import LabeledElement
from fathom.strategies.graph.intent.nodes.observer import ScreenObserver
from fathom.strategies.graph.state import IntentGraphState

if TYPE_CHECKING:
    from fathom.core.services.comparator import ScreenComparator
    from fathom.strategies.graph.context import GraphContext

logger = getLogger(__name__)


class PostAction:
    """
    Captures and classifies post-action evidence (screen, diff, effect).
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        observer: ScreenObserver,
        comparator: ScreenComparator,
    ) -> None:
        """
        Initialize with the graph context, observer helper, and screen comparator.
        """

        self.__context = context
        self.__observer = observer
        self.__comparator = comparator
        self.__settlement = ScreenSettlementService(
            state=observer,
            device=context.device,
            comparison=comparator,
            configuration=context.configuration,
        )

    async def observe(
        self,
        *,
        context: ExecutionContext,
    ) -> Tuple[
        Optional[ScreenObservation],
        Optional[ScreenDiff],
        str,
        str,
        Optional[StepArtifacts],
    ]:
        """
        Capture the post-action screen and return observation, diff, hash, package, and artifacts.

        The fifth tuple element is the :class:`StepArtifacts` envelope
        (screen.before / screen.after) populated when the
        :class:`StoragePort` accepts the post-action capture; callers
        propagate it onto :class:`StepResult` so STEP_COMPLETED
        telemetry can reference persisted screen URIs.
        """

        await self.__settlement.pause()

        pre_hash = context.pre_screen.visual_hash if context.pre_screen is not None else ZERO_HASH
        trace_emissions = (
            context.execution_result.trace_emissions if context.execution_result is not None else ()
        )

        try:
            comparison = await asyncio.wait_for(
                self.__compare(
                    context=context,
                    package_name=context.package,
                    before_capture=context.capture,
                    before_state=context.pre_screen,
                    trace_emissions=trace_emissions,
                ),
                timeout=self.__observation_timeout(),
            )
        except asyncio.TimeoutError as exception:
            logger.warning(
                "Post-action observation timed out",
                extra={
                    **self.__log_context(),
                    "event": "observe.post.timeout",
                    "timeout.seconds": self.__observation_timeout(),
                    "error.kind": type(exception).__name__,
                },
            )
            await self.__context.telemetry.warning(
                "Observe: Post-action observation timed out"
            )
            return None, None, pre_hash, context.package, None
        except Exception as exception:
            await self.__context.telemetry.warning(
                f"Observe: Failed to capture post-screen: {exception}"
            )
            return None, None, pre_hash, context.package, None

        observation = comparison.observation
        screen_diff = comparison.screen_diff
        post_hash = comparison.post_visual_hash or pre_hash

        try:
            post_activity = await self.__context.device.get_current_package() or context.package
        except Exception:
            post_activity = context.package

        return observation, screen_diff, post_hash, post_activity, comparison.artifacts

    def __observation_timeout(self) -> float:
        """
        Return the maximum wall-clock seconds for one post-action observation pass.
        """

        configuration = getattr(self.__context, "configuration", None)
        device = getattr(configuration, "device", None)

        if device is None:
            runtime = getattr(self.__context.device, "configuration", None)
            timeout = getattr(runtime, "command_timeout", None)
            return float(timeout or POST_ACTION_OBSERVATION_TIMEOUT_SECONDS)

        if device.type == DeviceConnectionType.REMOTE:
            return float(device.remote.request_timeout)

        if device.platform == DevicePlatform.IOS:
            return float(device.ios.command_timeout)

        return float(device.android.snapshot_timeout)

    @staticmethod
    def effect_from(*, diff: Optional[ScreenDiff]) -> ActionEffect:
        """
        Build diagnostic post-action effect data from the screen diff only.
        """

        return ActionEffect.from_screen_diff(diff=diff)

    @staticmethod
    def changed(
        *,
        screen_diff: Optional[ScreenDiff],
        pre_hash: str,
        post_hash: str,
        threshold: int,
    ) -> bool:
        """
        Return whether the screen changed after the executed action.
        """

        if screen_diff is not None:
            return screen_diff.action_had_effect

        return ScreenState.hamming_distance(left_hash=pre_hash, right_hash=post_hash) > threshold

    def log_diff(
        self,
        *,
        action_effect: ActionEffect,
        screen_diff: Optional[ScreenDiff],
    ) -> None:
        """
        Emit a structured log entry describing the observed screen diff.
        """

        if screen_diff is None:
            return

        logger.info(
            "Screen diff observed",
            extra={
                **self.__log_context(),
                "event": "observe.screen.diff",
                "ssim.score": screen_diff.ssim_score,
                "effect.status": action_effect.status.value,
                "effect.had": screen_diff.action_had_effect,
                "phash.distance": screen_diff.phash_distance,
                "activity.changed": screen_diff.activity_changed,
                "xml.hash.changed": screen_diff.xml_hash_changed,
                "changed.regions": len(screen_diff.changed_regions),
                "effect.visual_progress": action_effect.visual_progress,
                "scroll.translation": str(screen_diff.scroll_translation),
                "content.diff.ratio": screen_diff.content_pixel_diff_ratio,
                "effect.signal.expected": action_effect.signal_counts.expected,
                "effect.signal.progress": action_effect.signal_counts.progress,
                "interaction.hash.changed": screen_diff.interaction_hash_changed,
                "effect.signal.no_progress": action_effect.signal_counts.no_progress,
            },
        )

    @staticmethod
    def plan_observation(*, state: IntentGraphState) -> Optional[str]:
        """
        Return the plan-emitted observation string when present.
        """

        plan = state.get(IntentStateKey.PLAN)

        if isinstance(plan, PlanResult):
            value = plan.metadata.get(PlanMetadataKey.OBSERVATION.value)
            return value if isinstance(value, str) else None

        return None

    async def __compare(
        self,
        *,
        package_name: str,
        context: ExecutionContext,
        before_capture: ScreenCapture,
        before_state: Optional[ScreenState],
        trace_emissions: Tuple[TraceEmission, ...],
    ) -> PostActionObservation:
        """
        Capture the post-action screen, compare it to the pre-action state, and compose the artifact bundle.
        Before/annotated URIs ride on ``before_capture``; after URI comes from the pipeline; traces map from
        ``trace_emissions`` collected by the executor on this turn.
        """

        before_visual_hash = before_state.visual_hash if before_state is not None else None

        before = self.__build_before(capture=before_capture, visual_hash=before_visual_hash)
        annotated = self.__build_annotated(capture=before_capture, visual_hash=before_visual_hash)

        traces = self.__build_traces(emissions=trace_emissions)

        capture_start = time.time()
        post_capture = await self.__context.perception_port.capture()
        logger.info(
            "Post-action capture completed",
            extra={
                **self.__log_context(),
                "event": "observe.post.capture",
                "duration.ms": int((time.time() - capture_start) * 1000),
                "image.bytes": len(post_capture.image) if post_capture.image else 0,
            },
        )

        if self.__context.is_cancelled:
            return PostActionObservation(
                artifacts=self.__compose_step_artifacts(
                    before=before, after=None, annotated=annotated, traces=traces
                ),
            )

        if not post_capture.image:
            return PostActionObservation(
                artifacts=self.__compose_step_artifacts(
                    before=before, after=None, annotated=annotated, traces=traces
                ),
            )

        self.__observe_action_region(
            after_image=post_capture.image,
            trace_emissions=trace_emissions,
            before_image=before_capture.image,
        )

        elements_start = time.time()
        post_elements = self.__elements(capture=post_capture)
        logger.info(
            "Post-action elements extracted",
            extra={
                **self.__log_context(),
                "event": "observe.post.elements",
                "duration.ms": int((time.time() - elements_start) * 1000),
                "elements.count": len(post_elements),
            },
        )

        if self.__context.is_cancelled:
            return PostActionObservation(
                artifacts=self.__compose_step_artifacts(
                    before=before, after=None, annotated=annotated, traces=traces
                ),
            )

        hash_start = time.time()
        post_hashes = self.__observer.resolve_capture_hashes(
            capture=post_capture,
            elements=post_elements,
        )
        logger.info(
            "Post-action hashes computed",
            extra={
                **self.__log_context(),
                "event": "observe.post.hashes",
                "duration.ms": int((time.time() - hash_start) * 1000),
            },
        )

        if self.__context.is_cancelled:
            return PostActionObservation(
                post_visual_hash=post_hashes.visual_hash,
                artifacts=self.__compose_step_artifacts(
                    before=before, after=None, annotated=annotated, traces=traces
                ),
            )

        after_state = self.__observer.build_screen_state(
            capture=post_capture,
            xml_hash=post_hashes.xml_hash,
            visual_hash=post_hashes.visual_hash,
            interaction_hash=post_hashes.interaction_hash,
        )

        diff_start = time.time()
        screen_diff = await asyncio.to_thread(
            self.__comparator.compare,
            after=post_capture,
            before=before_capture,
            after_state=after_state,
            before_state=before_state,
        )
        logger.info(
            "Post-action diff computed",
            extra={
                **self.__log_context(),
                "event": "observe.post.diff",
                "duration.ms": int((time.time() - diff_start) * 1000),
            },
        )

        settled = await self.__settlement.compare(
            evidence=ScreenSettlementEvidence(
                execution=context,
                workflow_id=self.__context.workflow_id,
                before=PreActionScreen(capture=before_capture, state=before_state),
                after=PostActionScreen(
                    diff=screen_diff,
                    hashes=post_hashes,
                    capture=post_capture,
                ),
            )
        )
        screen_diff = settled.diff
        post_hashes = settled.hashes
        post_capture = settled.capture

        if self.__context.is_cancelled:
            return PostActionObservation(
                screen_diff=screen_diff,
                post_visual_hash=post_hashes.visual_hash,
                artifacts=self.__compose_step_artifacts(
                    before=before, after=None, annotated=annotated, traces=traces
                ),
            )

        # Post-action enrichment (OCR + icon + ensemble) was previously
        # re-run here on every turn; the next GROUND call rebuilds the same
        # observation from a fresh capture before any planner reads it, so
        # the only downstream consumers of the post-action enrichment were
        # history and debug artefacts. Skipping the call drops a full
        # OCR/icon round-trip per step. Pre-action observation flows
        # through as the fallback on the next ANALYZE turn.
        post_observation: Optional[ScreenObservation] = None
        logger.info(
            "Post-action enrichment skipped",
            extra={
                **self.__log_context(),
                "event": "observe.post.enrichment_skipped",
                "reason": "ground_rebuilds_observation_next_turn",
            },
        )

        if self.__context.is_cancelled:
            return PostActionObservation(
                screen_diff=screen_diff,
                post_visual_hash=post_hashes.visual_hash,
                artifacts=self.__compose_step_artifacts(
                    before=before, after=None, annotated=annotated, traces=traces
                ),
            )

        after = await self.__build_after(
            capture=post_capture,
            package_name=package_name,
            visual_hash=post_hashes.visual_hash,
        )

        return PostActionObservation(
            screen_diff=screen_diff,
            observation=post_observation,
            post_visual_hash=post_hashes.visual_hash,
            artifacts=self.__compose_step_artifacts(
                before=before, after=after, annotated=annotated, traces=traces
            ),
        )

    def __elements(self, *, capture: ScreenCapture) -> List[LabeledElement]:
        """
        Extract post-action interactive elements when XML is available.
        """

        if not self.__context.use_xml or not capture.xml_content:
            return []

        return self.__context.hierarchy.extract_elements(
            screen=capture,
            xml=capture.xml_content,
            action_type=ActionType.TAP,
        )

    def __observe_action_region(
        self,
        *,
        after_image: bytes,
        before_image: bytes,
        trace_emissions: Tuple[TraceEmission, ...],
    ) -> None:
        """
        Fire-and-forget: pHash the region around each dispatched tap centroid pre/post and log the result.
        """

        if not before_image or not after_image or not trace_emissions:
            return

        coords = self.__tap_centroids(trace_emissions=trace_emissions)
        if not coords:
            return

        log_context = self.__log_context()

        asyncio.create_task(
            self.__action_region_worker(
                coords=coords,
                log_context=log_context,
                after_image=after_image,
                before_image=before_image,
            ),
            name="effect.action_region.observer",
        )

    @staticmethod
    def __tap_centroids(
        *,
        trace_emissions: Tuple[TraceEmission, ...],
    ) -> Tuple[Tuple[int, int], ...]:
        """
        Project every emitted trace event whose coords look like a 2-point tap into ``(x, y)`` pairs.
        """

        centroids: List[Tuple[int, int]] = []

        for emission in trace_emissions:
            coords = emission.event.coords
            if len(coords) < 2:
                continue

            centroids.append((int(coords[0]), int(coords[1])))

        return tuple(centroids)

    async def __action_region_worker(
        self,
        *,
        after_image: bytes,
        before_image: bytes,
        log_context: Dict[str, Any],
        coords: Tuple[Tuple[int, int], ...],
    ) -> None:
        """
        Crop the region around every centroid in both captures, pHash each, log the Hamming distance.
        """

        try:
            await asyncio.to_thread(
                self.__compute_and_log_action_regions,
                coords=coords,
                log_context=log_context,
                after_image=after_image,
                before_image=before_image,
            )
        except Exception as exception:  # noqa: BLE001 - observation must never raise
            logger.warning(
                "Action region observation failed",
                extra={
                    **log_context,
                    "event": "effect.bbox.failed",
                    "error.kind": type(exception).__name__,
                },
            )

    @staticmethod
    def __compute_and_log_action_regions(
        *,
        after_image: bytes,
        before_image: bytes,
        log_context: Dict[str, Any],
        coords: Tuple[Tuple[int, int], ...],
    ) -> None:
        """
        CPU-bound pHash crop + compare loop intentionally executed on a worker thread.
        """

        engine = VisualHashEngine()

        after = Image.open(BytesIO(after_image)).convert("RGB")
        before = Image.open(BytesIO(before_image)).convert("RGB")

        width, height = before.size
        after_width, after_height = after.size

        for index, (centroid_x, centroid_y) in enumerate(coords):
            half = ACTION_REGION_HALF_SIDE

            x0 = max(0, centroid_x - half)
            y0 = max(0, centroid_y - half)
            x1 = min(width, centroid_x + half)
            y1 = min(height, centroid_y + half)

            after_x1 = min(after_width, centroid_x + half)
            after_y1 = min(after_height, centroid_y + half)

            if x1 - x0 <= 0 or y1 - y0 <= 0 or after_x1 - x0 <= 0 or after_y1 - y0 <= 0:
                continue

            before_crop = before.crop((x0, y0, x1, y1))
            after_crop = after.crop((x0, y0, after_x1, after_y1))

            after_buffer = BytesIO()
            before_buffer = BytesIO()

            after_crop.save(after_buffer, format="PNG")
            before_crop.save(before_buffer, format="PNG")

            pre_hash = engine.hash(image=before_buffer.getvalue())
            post_hash = engine.hash(image=after_buffer.getvalue())

            distance = ScreenState.hamming_distance(left_hash=pre_hash, right_hash=post_hash)
            static = distance <= ACTION_REGION_STATIC_HAMMING_FLOOR

            logger.info(
                "Action region pHash comparison",
                extra={
                    **log_context,
                    "event": "effect.bbox.observed",
                    "trace.index": index,
                    "region.static": static,
                    "centroid.x": centroid_x,
                    "centroid.y": centroid_y,
                    "region.half_side": half,
                    "region.pre_hash": pre_hash,
                    "region.post_hash": post_hash,
                    "region.hamming.distance": distance,
                    "region.hamming.floor": ACTION_REGION_STATIC_HAMMING_FLOOR,
                },
            )

    def __build_before(
        self,
        *,
        capture: ScreenCapture,
        visual_hash: Optional[str] = None,
    ) -> Optional[ScreenArtifact]:
        """
        Build the pre-action :class:`ScreenArtifact` from the raw screen
        bytes carried on ``capture.image``, with the optional storage URI stamped earlier by :class:`PerceptionService`.
        """

        if not capture.image and not capture.screenshot_uri:
            return None

        return ScreenArtifact(
            width=capture.width,
            height=capture.height,
            visual_hash=visual_hash,
            image=capture.image or None,
            captured_at=capture.timestamp,
            storage_backend=self.__resolve_storage_backend(),
            uri=capture.screenshot_uri if capture.screenshot_uri else None,
        )

    def __build_annotated(
        self,
        *,
        capture: ScreenCapture,
        visual_hash: Optional[str] = None,
    ) -> Optional[ScreenArtifact]:
        """
        Build the annotated :class:`ScreenArtifact` from the annotated
        bytes on ``capture.annotated_image``, with the optional storage
        URI stamped earlier by :class:`HierarchyService`.
        """

        if not capture.annotated_image and not capture.annotated_uri:
            return None

        return ScreenArtifact(
            width=capture.width,
            height=capture.height,
            visual_hash=visual_hash,
            image=capture.annotated_image,
            captured_at=capture.timestamp,
            storage_backend=self.__resolve_storage_backend(),
            uri=capture.annotated_uri if capture.annotated_uri else None,
        )

    async def __build_after(
        self,
        *,
        package_name: str,
        capture: ScreenCapture,
        visual_hash: Optional[str],
    ) -> Optional[ScreenArtifact]:
        """
        Build the post-action :class:`ScreenArtifact` from the captured
        bytes and stamp the storage URI from the artifact pipeline's
        staged path when a pipeline is wired. Returns ``None`` only when the capture itself yielded no bytes.
        """

        if not capture.image:
            return None

        uri = await self.__stage_post_action(capture=capture, package_name=package_name)

        return ScreenArtifact(
            uri=uri,
            image=capture.image,
            width=capture.width,
            height=capture.height,
            visual_hash=visual_hash,
            captured_at=capture.timestamp,
            storage_backend=self.__resolve_storage_backend(),
        )

    async def __stage_post_action(
        self,
        *,
        package_name: str,
        capture: ScreenCapture,
    ) -> Optional[str]:
        """
        Hand the post-action capture to the artifact pipeline and surface
        the staged path so the after artifact can carry a storage URI
        alongside the bytes. Returns ``None`` when the pipeline is un-wired or emission fails.
        """

        if (pipeline := self.__context.artifact_pipeline) is None:
            return None

        try:
            staged_path = await pipeline.emit(
                record=ArtifactRecord(
                    package_name=package_name,
                    created=int(time.time() * 1000),
                    session_id=self.__context.workflow_id,
                    payload=ScreenshotPayload(capture=capture),
                    step_number=self.__context.agent_state.step_count + 1,
                ),
            )
        except Exception as exception:
            logger.warning(
                "Failed to emit post-action capture to pipeline",
                extra={
                    **self.__log_context(),
                    "error.message": str(exception),
                    "event": "observe.pipeline.emit_failed",
                },
            )
            return None

        return str(staged_path) if staged_path is not None else None

    @staticmethod
    def __compose_step_artifacts(
        *,
        after: Optional[ScreenArtifact],
        before: Optional[ScreenArtifact],
        annotated: Optional[ScreenArtifact],
        traces: Optional[Tuple[ScreenArtifact, ...]],
    ) -> Optional[StepArtifacts]:
        """
        Build the StepArtifacts envelope, or None when every slot is empty.
        """

        if before is None and after is None and annotated is None and not traces:
            return None

        return StepArtifacts(
            screen=ScreenArtifactBundle(
                after=after,
                before=before,
                traces=traces,
                annotated=annotated,
            ),
        )

    def __build_traces(
        self,
        *,
        emissions: Tuple[TraceEmission, ...],
    ) -> Optional[Tuple[ScreenArtifact, ...]]:
        """
        Project trace emissions into the bundle's traces tuple, returning None when the executor ran no trace path.
        Emissions whose pipeline staging failed are dropped so callers iterate only over artifacts with handles.
        """

        if not emissions:
            return None

        artifacts = tuple(
            emission.artifact for emission in emissions if emission.artifact is not None
        )
        return artifacts

    def __resolve_storage_backend(self) -> StorageBackend:
        """
        Read the live storage adapter's declared backend, falling back
        to :class:`StorageBackend.LOCAL` when the adapter does not
        advertise one.
        """

        backend = getattr(self.__context.storage, "backend", None)

        if isinstance(backend, StorageBackend):
            return backend

        return StorageBackend.LOCAL

    def __log_context(self) -> Dict[str, Any]:
        """
        Return shared structured-logging context for post-action entries.
        """

        return {
            "component": "graph.intent.effect",
            "workflow.id": self.__context.workflow_id,
        }
