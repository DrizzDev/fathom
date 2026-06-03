from __future__ import annotations

import io
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from pydantic import ValidationError

from fathom.constants.localization import LocalizationGridScale
from fathom.core.prompts.localization import VisionLocalizationPrompt
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.localization import TargetLocalizerPort
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.localization import LocalizationProposal, VisionLocalizationPayload
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
    ) -> None:
        """
        Initialize the member with an injected LLM port, prompt builder, and run context.
        """

        self.__llm = llm
        self.__workflow_id = workflow_id
        self.__prompt = prompt if prompt is not None else VisionLocalizationPrompt()
        self.__structured_output = StructuredOutput(payload=VisionLocalizationPayload)

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
        Issue a single Gemini call and parse a normalized bounding rectangle.
        """

        _ = observation
        _ = budget

        if not (target := self.__target_text(action=action)):
            return None

        log_context = self.__log_context(target=target, activity=capture.activity)
        logger.info(
            "Vision localizer call started",
            extra={**log_context, "event": "localizer.vision.started"},
        )

        result = await self.__llm.generate(
            prompt=self.__prompt.build(target=target, image=capture.image),
            use_cache=False,
            system_instruction=self.__prompt.SYSTEM_INSTRUCTION,
            structured_output=self.__structured_output,
        )

        if (payload := self.__try_parse(content=result.content, context=log_context)) is None:
            return None

        if payload.refused:
            logger.info(
                "Vision localizer reported target not visible",
                extra={**log_context, "event": "localizer.vision.refusal"},
            )
            return None

        bounds = self.__bounds_from_payload(
            payload=payload,
            width=capture.width,
            height=capture.height,
        )
        if bounds is None:
            logger.warning(
                "Vision localizer payload yielded zero-area bounds after projection",
                extra={
                    **log_context,
                    "payload.raw": self.__payload_log(payload=payload),
                    "event": "localizer.vision.payload.invalid",
                },
            )
            return None

        pixel_width, pixel_height = self.__pixel_dimensions(image=capture.image)

        logger.info(
            "Vision localizer proposal returned",
            extra={
                **log_context,
                "confidence": payload.confidence,
                "payload.system": (
                    f"normalized.{LocalizationGridScale.MINIMUM}.{LocalizationGridScale.MAXIMUM}"
                ),
                "bounds.system": bounds.system.value,
                "event": "localizer.vision.completed",
                "conversion": "normalized_to_logical",
                "capture.pixel": {"width": pixel_width, "height": pixel_height},
                "capture.logical": {"width": capture.width, "height": capture.height},
                "payload.raw": self.__payload_log(payload=payload),
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
        Validate the JSON response against the published schema and return None on failure.
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
                    "event": "localizer.vision.payload.invalid",
                    "response.preview": stripped[:512],
                },
            )
            return None

    @staticmethod
    def __bounds_from_payload(
        *,
        payload: VisionLocalizationPayload,
        width: int,
        height: int,
    ) -> Optional[Bounds]:
        """
        Project the grid bbox onto the capture's logical canvas and stamp it LOGICAL/MODEL.
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

    def __log_context(self, *, target: str, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for this localizer invocation.
        """

        return {
            "member": self.name,
            "activity": activity,
            "target": target[:80],
            "workflow.id": self.__workflow_id,
            "component": "adapter.localizer.vision",
        }
