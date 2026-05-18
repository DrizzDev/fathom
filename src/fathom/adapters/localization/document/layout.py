from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional

from fathom.interfaces.localization import TargetLocalizerPort
from fathom.schemas.actions import Action
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import LocalizationProposal
from fathom.schemas.observation import ElementSource, PerceivedElement, ScreenObservation
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class DocumentAiLayoutLocalizer(TargetLocalizerPort):
    """
    Ensemble member that resolves targets against OCR tokens already in the observation.
    """

    def __init__(self, *, workflow_id: Optional[str] = None) -> None:
        """
        Initialize the layout localizer with optional run context.
        """

        self.__workflow_id = workflow_id

    @property
    def name(self) -> str:
        """
        Stable identifier used in logs and ensemble proposals.
        """

        return "document.ai.layout"

    async def locate(
        self,
        *,
        action: Action,
        capture: ScreenCapture,
        budget: LocalizationBudget,
        observation: ScreenObservation,
    ) -> Optional[LocalizationProposal]:
        """
        Return a proposal when an OCR token's text matches the action target.
        """

        _ = budget

        if not (target := self.__target_text(action=action)):
            return None

        context = self.__log_context(target=target, activity=capture.activity)

        # Surface the implicit OCR dependency explicitly. When no OCR-sourced
        # elements survived perception (OCR disabled, timed out, or returned
        # nothing) the localizer cannot contribute — log a structured
        # degradation event so the ensemble disagreement signal is
        # explainable instead of silently mysterious.
        if not self.__has_ocr_elements(observation=observation):
            logger.info(
                "Layout localizer degraded: observation carries no OCR elements",
                extra={**context, "event": "localizer.layout.degraded"},
            )
            return None

        if (match := self.__match_token(target=target, observation=observation)) is None:
            logger.info(
                "Layout localizer found no OCR match",
                extra={**context, "event": "localizer.layout.miss"},
            )
            return None

        logger.info(
            "Layout localizer matched OCR token",
            extra={
                **context,
                "event": "localizer.layout.match",
                "confidence": match.confidence,
            },
        )
        return LocalizationProposal(
            bounds=match.bounds,
            confidence=match.confidence,
            source=self.name,
            rationale=f"OCR token '{match.text}' matched action target.",
        )

    @staticmethod
    def __has_ocr_elements(*, observation: ScreenObservation) -> bool:
        """
        Return whether the observation carries any OCR-sourced element.

        The layout localizer matches strictly against OCR tokens. When
        OCR is disabled, times out, or returns no usable tokens, the
        observation will have zero OCR elements — the localizer must
        surface that degradation rather than masquerading as a normal miss.
        """

        return any(element.source == ElementSource.OCR for element in observation.elements)

    def __match_token(
        self,
        *,
        target: str,
        observation: ScreenObservation,
    ) -> Optional[PerceivedElement]:
        """
        Return the first OCR-sourced element whose normalized text equals the target.
        """

        for element in observation.elements:
            if element.source != ElementSource.OCR:
                continue
            if not element.text:
                continue
            if self.__normalize(value=element.text) == target:
                return element
        return None

    def __target_text(self, *, action: Action) -> str:
        """
        Return the normalized semantic target text from the action.
        """

        return self.__normalize(
            value=(
                action.natural_language_target
                or action.export_target
                or action.script_target
                or action.target
                or ""
            )
        )

    @staticmethod
    def __normalize(*, value: str) -> str:
        """
        Normalize text for exact matching against OCR token surface forms.
        """

        return " ".join(value.strip().lower().split())

    def __log_context(self, *, target: str, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for this localizer invocation.
        """

        return {
            "component": "adapter.localizer.layout",
            "workflow.id": self.__workflow_id,
            "activity": activity,
            "target": target[:80],
            "member": self.name,
        }
