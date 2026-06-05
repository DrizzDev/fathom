from __future__ import annotations

import asyncio
import io
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fathom.constants.localization import LocalizationGridScale
from fathom.constants.runtime import MILLISECONDS_PER_SECOND
from fathom.core.prompts.localization import VisionLocalizationPrompt
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.localization import TargetLocalizerPort
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.localization import (
    LocalizationProposal,
    VisionLocalizationConfiguration,
    VisionLocalizationPayload,
)
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class GeminiVisionLocalizer(TargetLocalizerPort):
    """
    Ensemble member that asks Gemini for a normalized bounding rectangle.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        workflow_id: Optional[str] = None,
        prompt: Optional[VisionLocalizationPrompt] = None,
        configuration: Optional[VisionLocalizationConfiguration] = None,
    ) -> None:
        """
        Initialize the member with its LLM port, prompt builder, timeout + retry policy, and run context.
        """

        self.__llm = llm
        self.__workflow_id = workflow_id
        self.__prompt = prompt if prompt is not None else VisionLocalizationPrompt()
        self.__structured_output = StructuredOutput(payload=VisionLocalizationPayload)
        self.__configuration = (
            configuration if configuration is not None else VisionLocalizationConfiguration()
        )

    @property
    def name(self) -> str:
        """
        Stable identifier used in logs and ensemble proposals.
        """

        return "gemini.vision"

    async def locate(
        self,
        *,
        action: Action,
        capture: ScreenCapture,
        budget: LocalizationBudget,
        observation: ScreenObservation,
    ) -> Optional[LocalizationProposal]:
        """
        Run the vision-LLM locate request inside a bounded retry loop.
        """

        _ = observation
        _ = budget

        if not (target := self.__target_text(action=action)):
            return None

        context = self.__context(target=target, activity=capture.activity)
        return await self.__attempt_with_retries(target=target, capture=capture, context=context)

    async def __attempt_with_retries(
        self,
        *,
        target: str,
        capture: ScreenCapture,
        context: Dict[str, Any],
    ) -> Optional[LocalizationProposal]:
        """
        Retry the vision call on timeout using the configured policy.
        """

        attempts = self.__configuration.retry.attempts
        timeout = self.__configuration.timeout / MILLISECONDS_PER_SECOND

        retrying = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                min=0,
                multiplier=timeout,
                exp_base=self.__configuration.retry.backoff,
            ),
            retry=retry_if_exception_type(asyncio.TimeoutError),
        )

        try:
            async for attempt in retrying:
                with attempt:
                    proposal = await self.__bounded_call(
                        target=target, capture=capture, context=context, attempt=attempt
                    )
            return proposal
        except asyncio.TimeoutError:
            logger.warning(
                "Vision localizer exhausted attempt budget",
                extra={
                    **context,
                    "attempt.budget": attempts,
                    "event": "localizer.vision.exhausted",
                },
            )
        return None

    async def __bounded_call(
        self,
        *,
        target: str,
        attempt: Any,
        capture: ScreenCapture,
        context: Dict[str, Any],
    ) -> Optional[LocalizationProposal]:
        """
        Wrap one Gemini call in a per-attempt timeout and log start/timeout.
        """

        attempt_index = attempt.retry_state.attempt_number - 1
        timeout = self.__configuration.timeout / MILLISECONDS_PER_SECOND

        logger.info(
            "Vision localizer call started",
            extra={
                **context,
                "timeout.seconds": timeout,
                "attempt.index": attempt_index,
                "event": "localizer.vision.started",
            },
        )
        try:
            return await asyncio.wait_for(
                self.__single_call(target=target, capture=capture, context=context),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Vision localizer attempt timed out",
                extra={
                    **context,
                    "timeout.seconds": timeout,
                    "attempt.index": attempt_index,
                    "event": "localizer.vision.timeout",
                },
            )
            raise

    async def __single_call(
        self,
        *,
        target: str,
        capture: ScreenCapture,
        context: Dict[str, Any],
    ) -> Optional[LocalizationProposal]:
        """
        Execute one Gemini call and project the payload into pixel bounds.
        """

        result = await self.__llm.generate(
            use_cache=False,
            structured_output=self.__structured_output,
            system_instruction=self.__prompt.SYSTEM_INSTRUCTION,
            prompt=self.__prompt.build(target=target, image=capture.image),
        )
        if (payload := self.__try_parse(content=result.content, context=context)) is None:
            return None

        if payload.refused:
            logger.info(
                "Vision localizer reported target not visible",
                extra={**context, "event": "localizer.vision.refusal"},
            )
            return None

        bounds = self.__bounds_from_payload(
            payload=payload, width=capture.width, height=capture.height
        )
        if bounds is None:
            logger.warning(
                "Vision localizer payload yielded zero-area bounds after projection",
                extra={
                    **context,
                    "payload.raw": self.__payload_log(payload=payload),
                    "event": "localizer.vision.payload.invalid",
                },
            )
            return None

        pixel_width, pixel_height = self.__pixel_dimensions(image=capture.image)

        logger.info(
            "Vision localizer proposal returned",
            extra={
                **context,
                "confidence": payload.confidence,
                "payload.system": (
                    f"normalized.{LocalizationGridScale.MINIMUM}.{LocalizationGridScale.MAXIMUM}"
                ),
                "bounds.system": bounds.system.value,
                "event": "localizer.vision.completed",
                "conversion": "normalized_to_logical",
                "payload.raw": self.__payload_log(payload=payload),
                "capture.pixel": {"width": pixel_width, "height": pixel_height},
                "capture.logical": {"width": capture.width, "height": capture.height},
                "bounds.resolved": {
                    "x": bounds.x,
                    "y": bounds.y,
                    "width": bounds.width,
                    "height": bounds.height,
                },
            },
        )
        return LocalizationProposal(
            bounds=bounds,
            source=self.name,
            confidence=payload.confidence,
            rationale=f"Gemini vision proposal for target '{target}'.",
        )

    def __try_parse(
        self,
        *,
        content: str,
        context: Dict[str, Any],
    ) -> Optional[VisionLocalizationPayload]:
        """
        Validate the JSON response against the published schema.
        """

        if not (stripped := (content or "").strip()):
            return None

        try:
            return VisionLocalizationPayload.model_validate_json(stripped)
        except ValidationError:
            logger.warning(
                "Vision localizer response did not match the published schema",
                extra={
                    **context,
                    "response.length": len(stripped),
                    "response.preview": stripped[:512],
                    "event": "localizer.vision.payload.invalid",
                },
            )
            return None

    @staticmethod
    def __bounds_from_payload(
        *,
        width: int,
        height: int,
        payload: VisionLocalizationPayload,
    ) -> Optional[Bounds]:
        """
        Project the grid bbox onto the capture's logical canvas.
        """

        x_min = (payload.x1 * width) // LocalizationGridScale.MAXIMUM
        y_min = (payload.y1 * height) // LocalizationGridScale.MAXIMUM
        x_max = (payload.x2 * width) // LocalizationGridScale.MAXIMUM
        y_max = (payload.y2 * height) // LocalizationGridScale.MAXIMUM

        bound_width = x_max - x_min
        bound_height = y_max - y_min

        if bound_width <= 0 or bound_height <= 0:
            return None

        return Bounds(
            x=x_min,
            y=y_min,
            width=bound_width,
            height=bound_height,
            source=CoordinateSource.MODEL,
            coordinate_system=CoordinateSystem.LOGICAL,
        )

    @staticmethod
    def __pixel_dimensions(*, image: bytes) -> Tuple[Optional[int], Optional[int]]:
        """
        Decode the capture PNG header and return its pixel dimensions.
        """

        if not image:
            return None, None

        try:
            with Image.open(io.BytesIO(image)) as decoded:
                return decoded.width, decoded.height
        except (OSError, ValueError):
            return None, None

    @staticmethod
    def __payload_log(*, payload: VisionLocalizationPayload) -> Dict[str, Any]:
        """
        Build a log-safe snapshot of the payload edges.
        """

        return {
            "x1": payload.x1,
            "y1": payload.y1,
            "x2": payload.x2,
            "y2": payload.y2,
            "confidence": payload.confidence,
        }

    @staticmethod
    def __target_text(*, action: Action) -> str:
        """
        Return the semantic target text from the action.
        """

        return (
            action.natural_language_target
            or action.export_target
            or action.script_target
            or action.target
            or ""
        ).strip()

    def __context(self, *, target: str, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for this invocation.
        """

        return {
            "member": self.name,
            "activity": activity,
            "target": target[:80],
            "workflow.id": self.__workflow_id,
            "component": "adapter.localizer.vision",
        }
