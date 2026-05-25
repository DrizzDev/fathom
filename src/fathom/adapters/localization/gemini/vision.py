from __future__ import annotations

import json
from logging import getLogger
from typing import Any, Dict, Optional

from fathom.core.prompts.localization import VisionLocalizationPrompt
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.localization import TargetLocalizerPort
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import LocalizationProposal
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class GeminiVisionLocalizer(TargetLocalizerPort):
    """
    Ensemble member that asks Gemini for normalized bounding-box coordinates.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        prompt: Optional[VisionLocalizationPrompt] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the member with an injected LLM port, prompt builder, and run context.
        """

        self.__llm = llm
        self.__prompt = prompt if prompt is not None else VisionLocalizationPrompt()
        self.__workflow_id = workflow_id

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
        Issue a single Gemini call and parse a normalized bounding box.
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
            prompt=self.__prompt.build(
                target=target,
                image=capture.image,
                image_width=capture.width,
                image_height=capture.height,
            ),
            use_cache=False,
            system_instruction=self.__prompt.SYSTEM_INSTRUCTION,
        )

        if (payload := self.__parse_payload(content=result.content, context=log_context)) is None:
            logger.warning(
                "Vision localizer payload unparseable",
                extra={
                    **log_context,
                    "event": "localizer.vision.payload.unparseable",
                    "response.preview": (result.content or "")[:512],
                },
            )
            return None

        if self.__is_refusal(payload=payload):
            logger.info(
                "Vision localizer reported target not visible",
                extra={**log_context, "event": "localizer.vision.refusal"},
            )
            return None

        if (
            bounds := self.__bounds_from_payload(
                payload=payload, width=capture.width, height=capture.height
            )
        ) is None:
            logger.warning(
                "Vision localizer payload missing bounds",
                extra={
                    **log_context,
                    "event": "localizer.vision.payload.invalid",
                    "payload.raw": payload,
                },
            )
            return None

        confidence = self.__confidence(payload=payload)
        logger.info(
            "Vision localizer proposal returned",
            extra={
                **log_context,
                "event": "localizer.vision.completed",
                "confidence": confidence,
                "payload.raw": {
                    "x": payload.get("x"),
                    "y": payload.get("y"),
                    "width": payload.get("width"),
                    "height": payload.get("height"),
                },
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
            confidence=confidence,
            rationale=f"Gemini vision proposal for target '{target}'.",
        )

    def __parse_payload(
        self,
        *,
        content: str,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Parse the model's JSON response and return None on malformed output.
        """

        if not (stripped := content.strip()):
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning(
                "Vision localizer response was not valid JSON",
                extra={
                    **context,
                    "event": "localizer.vision.payload.invalid_json",
                    "response.length": len(stripped),
                },
            )
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def __bounds_from_payload(
        self,
        *,
        payload: Dict[str, Any],
        width: int,
        height: int,
    ) -> Optional[Bounds]:
        """
        Convert normalized payload coordinates into pixel-space Bounds.
        """

        try:
            x = float(payload["x"])
            y = float(payload["y"])
            box_width = float(payload["width"])
            box_height = float(payload["height"])
        except (KeyError, TypeError, ValueError):
            return None

        x_min = max(0.0, min(1.0, x)) * width
        y_min = max(0.0, min(1.0, y)) * height
        bound_width = int(max(0.0, min(1.0, box_width)) * width)
        bound_height = int(max(0.0, min(1.0, box_height)) * height)

        if bound_width <= 0 or bound_height <= 0:
            return None

        return Bounds(
            x=int(x_min),
            y=int(y_min),
            width=bound_width,
            height=bound_height,
            source=CoordinateSource.MODEL,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    @staticmethod
    def __is_refusal(*, payload: Dict[str, Any]) -> bool:
        """
        Detect the refusal-protocol payload (all-zero coordinates and confidence).
        """

        for key in ("x", "y", "width", "height"):
            if float(payload.get(key, -1.0) or 0.0) != 0.0:
                return False
        return float(payload.get("confidence", -1.0) or 0.0) == 0.0

    @staticmethod
    def __confidence(*, payload: Dict[str, Any]) -> float:
        """
        Extract and clamp the model-reported confidence value.
        """

        raw = payload.get("confidence", 0.5)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, value))

    def __target_text(self, *, action: Action) -> str:
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
            "component": "adapter.localizer.vision",
            "workflow.id": self.__workflow_id,
            "activity": activity,
            "target": target[:80],
            "member": self.name,
        }
