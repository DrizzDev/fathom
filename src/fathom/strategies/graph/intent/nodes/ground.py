from __future__ import annotations

import asyncio
import logging
import time
from typing import cast

from fathom.constants import ActionType
from fathom.constants.messages import GROUNDING_FAILURE_MESSAGE
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.exceptions import FathomError
from fathom.core.services.manifest import ManifestMerger
from fathom.processing.annotator import ImageAnnotator
from fathom.schemas.hierarchy import HierarchyProcessingResult
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class GroundNode:
    """
    GROUND graph node; captures and observes the current screen.
    """

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider

    async def __call__(self, state: IntentGraphState) -> IntentGraphState:
        """
        Run the GROUND node handler.
        """

        return await self.run(state=state)

    async def run(self, *, state: IntentGraphState) -> IntentGraphState:
        """
        Capture the screen and update state.

        ERROR BOUNDARY: All exceptions are caught and converted to terminal states
        to ensure graph execution completes gracefully even on device failures.
        """

        logger.info(
            "Starting grounding node",
            extra={
                "component": "graph.intent.ground",
                "event": "ground.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )
        logger.info(
            f"Current step count: {self.__provider.context.agent_state.step_count}",
            extra={
                "component": "graph.intent.ground",
                "event": "ground.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )
        logger.info(
            f"Incoming state has planned_step: {state.get(IntentStateKey.PLANNED_STEP) is not None}",
            extra={
                "component": "graph.intent.ground",
                "event": "ground.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        # ERROR BOUNDARY: Wrap entire node in try/except
        try:
            if await self.__provider.is_cancelled():
                logger.warning(
                    "Execution cancelled",
                    extra={
                        "component": "graph.intent.ground",
                        "event": "ground.log",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )
                self.__provider.context.agent_state.mark_complete(
                    reason=CompletionReason.CANCELLED.value
                )

                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                    },
                )

            # Check max steps BEFORE planning to avoid planning actions we can't execute
            if self.__provider.context.agent_state.step_count >= self.__provider.context.max_steps:
                logger.warning(
                    f"Max steps ({self.__provider.context.max_steps}) reached. "
                    f"Current step count: {self.__provider.context.agent_state.step_count}"
                )
                self.__provider.context.agent_state.mark_complete(
                    reason=CompletionReason.MAX_STEPS.value
                )

                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.MAX_STEPS.value,
                    },
                )

            await self.__provider.context.phase.grounding(
                intent=self.__provider.context.intent,
            )

            start_time = time.time()

            # 1. Capture State (Screenshot + optional hierarchy)
            screen = await self.__provider.context.perception.perceive(
                session_id=self.__provider.context.workflow_id,
                step_number=self.__provider.context.agent_state.step_count + 1,
            )

            if not screen.image:
                await self.__provider.context.telemetry.error(
                    "Ground: Empty screenshot captured",
                    step=self.__provider.context.agent_state.step_count + 1,
                )
                logger.error(
                    "Empty screenshot captured",
                    extra={
                        "event": "ground.log",
                        "component": "graph.intent.ground",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )
                self.__provider.context.agent_state.mark_complete(
                    reason=CompletionReason.FAILED.value
                )
                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.CAPTURE: None,
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    },
                )

            # 2. Capture Dimensions (Independent hardware metadata)
            width = screen.width
            height = screen.height
            logger.info(f"Device dimension is {height=}x{width=}")

            # Validate dimensions
            if width <= 0 or height <= 0:
                await self.__provider.context.telemetry.error(
                    f"Ground: Invalid dimensions {width}x{height}",
                    step=self.__provider.context.agent_state.step_count + 1,
                )
                logger.error(
                    f"Invalid dimensions {width}x{height}",
                    extra={
                        "event": "ground.log",
                        "component": "graph.intent.ground",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )
                self.__provider.context.agent_state.mark_complete(
                    reason=CompletionReason.FAILED.value
                )
                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.CAPTURE: None,
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    },
                )

            activity = screen.activity

            raw_screen = screen
            # XML Dump if enabled
            xml = raw_screen.xml_content

            elements = None

            logger.info(
                f"[GROUND] Config use_xml={self.__provider.context.use_xml}, xml_content present={xml is not None}"
            )

            if self.__provider.context.use_xml and xml:
                hierarchy_dump_duration = raw_screen.metadata.get("hierarchy_dump_duration")
                if isinstance(hierarchy_dump_duration, (int, float)):
                    self.__provider.context.metrics.record(
                        operation="hierarchy_dump",
                        duration=float(hierarchy_dump_duration),
                    )

                process_start = time.time()
                hierarchy_result = await self.__provider.context.hierarchy.process_xml_and_screen(
                    xml=xml,
                    screen=raw_screen,
                    package_name=activity,
                    action_type=ActionType.TAP,
                    session_id=self.__provider.context.workflow_id,
                    path_manager=self.__provider.context.path_manager,
                    step_number=self.__provider.context.agent_state.step_count + 1,
                )
                self.__provider.context.metrics.record(
                    operation="hierarchy_processing", duration=time.time() - process_start
                )

                elements = hierarchy_result.label_map
                if hierarchy_result.annotated_capture is not None:
                    screen = hierarchy_result.annotated_capture
            else:
                hierarchy_result = HierarchyProcessingResult()

            capture_hashes = self.__provider.observer.resolve_capture_hashes(
                capture=raw_screen,
                elements=hierarchy_result.labeled_elements,
            )

            screen_state = self.__provider.observer.build_screen_state(
                capture=raw_screen,
                xml_hash=capture_hashes.xml_hash,
                visual_hash=capture_hashes.visual_hash,
                interaction_hash=capture_hashes.interaction_hash,
            )
            screen_observation = await self.__provider.observer.observe(
                capture=raw_screen,
                hashes=capture_hashes,
                elements=hierarchy_result.labeled_elements,
            )
            # Append text-bearing perception elements (OCR / icon / vision)
            # onto the planner manifest. This must run even when XML is
            # unavailable: OCR exists specifically to provide label anchors in
            # that failure mode.
            merge_result = ManifestMerger.merge(
                label_map=elements or {},
                observation=screen_observation,
            )
            elements = merge_result.label_map

            # Whenever the merger actually appended perception entries, draw
            # the same numeric labels onto the LLM-facing image. If XML did not
            # produce an annotated image, start from the raw screenshot.
            if merge_result.appended:
                base_image = screen.annotated_image or screen.image
                overlaid_bytes = ImageAnnotator.overlay_perception_boxes(
                    image_bytes=base_image,
                    entries=list(merge_result.appended),
                )
                if overlaid_bytes and overlaid_bytes is not screen.annotated_image:
                    screen = screen.model_copy(
                        update={"annotated_image": overlaid_bytes},
                    )
            screen = screen.model_copy(update={"state": screen_state})

            is_new_screen = self.__provider.context.agent_state.update_screen(screen=screen_state)
            self.__provider.context.agent_state.runtime.screen.update(
                screen=screen_state,
                observation=screen_observation,
            )

            duration = time.time() - start_time
            self.__provider.context.metrics.record(operation="screenshot", duration=duration)

            logger.info(
                f"Screen captured: hash={capture_hashes.visual_hash}, activity={activity}, is_new={is_new_screen}, elements={len(elements) if elements else 0}",
                extra={
                    "event": "ground.log",
                    "component": "graph.intent.ground",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            logger.info(
                f"Grounding completed in {duration:.2f}s",
                extra={
                    "event": "ground.log",
                    "component": "graph.intent.ground",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            logger.info(
                "-> Transitioning to ANALYZE",
                extra={
                    "event": "ground.log",
                    "component": "graph.intent.ground",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )

            # Reset per-step fields
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.ANALYSIS: None,
                    CommonStateKey.CAPTURE: screen,
                    IntentStateKey.XML_CONTENT: xml,
                    CommonStateKey.STEP_RESULT: None,
                    IntentStateKey.ELEMENTS: elements,
                    IntentStateKey.PLANNED_STEP: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.SCREEN_STATE: screen_state,
                    CommonStateKey.GROUNDING_DURATION: duration,
                    CommonStateKey.IS_NEW_SCREEN: is_new_screen,
                    CommonStateKey.SCREEN_OBSERVATION: screen_observation,
                },
            )

            # Persist sub-goal state to graph for checkpoint recovery
            self.__provider.persistence.persist(result=result)

            return result

        except asyncio.CancelledError:
            # CancelledError is the cooperative cancellation signal; it
            # must propagate so the LangGraph task tree unwinds cleanly.
            raise
        except Exception as exception:
            logger.exception(
                f"Grounding failed: {exception}",
                extra={
                    "event": "ground.log",
                    "component": "graph.intent.ground",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            display_error = (
                exception.display(fallback=GROUNDING_FAILURE_MESSAGE)
                if isinstance(exception, FathomError)
                else GROUNDING_FAILURE_MESSAGE
            )
            await self.__provider.context.telemetry.error(display_error)
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)

            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.CAPTURE: None,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                },
            )

            # Persist failure state
            self.__provider.persistence.persist(result=result)

            return result
