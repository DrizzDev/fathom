from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from fathom.constants import ActionType
from fathom.constants.screen import ZERO_HASH
from fathom.constants.state import IntentStateKey, PlanMetadataKey
from fathom.core.services.comparator import ScreenComparator
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.observation import PostActionObservation, ScreenObservation
from fathom.schemas.outcomes import OutcomeStatus
from fathom.schemas.results import PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenDiff, ScreenState
from fathom.schemas.ui import LabeledElement
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes.observer import ScreenObserver
from fathom.strategies.graph.state import IntentGraphState
from fathom.utils.wait import stability_wait

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

    async def observe(
        self,
        *,
        context: ExecutionContext,
    ) -> Tuple[Optional[ScreenObservation], Optional[ScreenDiff], str, str]:
        """
        Capture the post-action screen and return observation, diff, hash, and package.
        """

        await stability_wait(self.__context.configuration)

        pre_hash = context.pre_screen.visual_hash if context.pre_screen is not None else ZERO_HASH
        try:
            comparison = await self.__compare(
                before_capture=context.capture,
                before_state=context.pre_screen,
            )
        except Exception as exception:
            await self.__context.telemetry.warning(
                f"Observe: Failed to capture post-screen: {exception}"
            )
            return None, None, pre_hash, context.package

        observation = comparison.observation
        screen_diff = comparison.screen_diff
        post_hash = comparison.post_visual_hash or pre_hash
        try:
            post_activity = await self.__context.device.get_current_package() or context.package
        except Exception:
            post_activity = context.package
        return observation, screen_diff, post_hash, post_activity

    @staticmethod
    def effect_from(
        *,
        status: OutcomeStatus,
        diff: Optional[ScreenDiff],
    ) -> ActionEffect:
        """
        Map an OutcomeStatus onto the ActionEffect status so prompt telemetry
        agrees with the action-aware outcome contract even when the screen
        diff alone is ambiguous.
        """

        effect = ActionEffect.from_screen_diff(diff=diff)

        if status == OutcomeStatus.NO_EFFECT:
            return effect.model_copy(update={"status": ActionEffectStatus.NO_PROGRESS})

        if status == OutcomeStatus.EFFECTIVE:
            return effect.model_copy(update={"status": ActionEffectStatus.PROGRESS})

        return effect

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
        screen_diff: Optional[ScreenDiff],
        action_effect: ActionEffect,
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
                "phash.distance": screen_diff.phash_distance,
                "ssim.score": screen_diff.ssim_score,
                "content.diff.ratio": screen_diff.content_pixel_diff_ratio,
                "changed.regions": len(screen_diff.changed_regions),
                "scroll.translation": str(screen_diff.scroll_translation),
                "effect.status": action_effect.status.value,
                "effect.visual_progress": action_effect.visual_progress,
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
        before_capture: ScreenCapture,
        before_state: Optional[ScreenState],
    ) -> PostActionObservation:
        """
        Capture the post-action screen and compare it to the pre-action state.
        """

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

        if not post_capture.image:
            return PostActionObservation()

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

        post_observation = await self.__observer.observe(
            capture=post_capture,
            hashes=post_hashes,
            elements=post_elements,
        )

        return PostActionObservation(
            screen_diff=screen_diff,
            observation=post_observation,
            post_visual_hash=post_hashes.visual_hash,
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

    def __log_context(self) -> Dict[str, Any]:
        """
        Return shared structured-logging context for post-action entries.
        """

        return {
            "component": "graph.intent.effect",
            "workflow.id": self.__context.workflow_id,
        }
