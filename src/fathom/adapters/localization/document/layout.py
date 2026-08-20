from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional

from fathom.core.localization.matcher import OcrPhraseMatcher
from fathom.interfaces.localization import TargetLocalizerPort
from fathom.schemas.actions import Action
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import LocalizationProposal
from fathom.schemas.observation import ElementSource, ScreenObservation
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class DocumentAiLayoutLocalizer(TargetLocalizerPort):
    """
    Ensemble member that resolves targets against OCR phrases in the observation.
    """

    def __init__(
        self,
        *,
        workflow_id: Optional[str] = None,
        matcher: Optional[OcrPhraseMatcher] = None,
    ) -> None:
        """
        Bind optional run context and an OCR phrase matcher, defaulting to a fresh matcher.
        """

        self.__workflow_id = workflow_id
        self.__matcher = matcher if matcher is not None else OcrPhraseMatcher()

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
        Return a proposal when an OCR phrase's text matches the action target.
        """

        _ = budget

        target = self.__target_text(action=action)
        if not target:
            return None

        context = self.__log_context(target=target, activity=capture.activity)

        if not self.__has_ocr_elements(observation=observation):
            logger.info(
                "Layout localizer degraded: observation carries no OCR elements",
                extra={**context, "event": "localizer.layout.degraded"},
            )
            return None

        match = self.__matcher.find_best_match(target=target, elements=observation.elements)
        if match is None:
            logger.info(
                "Layout localizer found no phrase match above threshold",
                extra={**context, "event": "localizer.layout.miss"},
            )
            return None

        logger.info(
            "Layout localizer matched OCR phrase",
            extra={
                **context,
                "event": "localizer.layout.match",
                "match.text": match.text,
                "match.score": match.score,
                "match.confidence": match.confidence,
                "match.token_count": match.token_count,
            },
        )
        return LocalizationProposal(
            source=self.name,
            bounds=match.bounds,
            confidence=match.confidence,
            rationale=f"OCR phrase '{match.text}' matched action target.",
        )

    @staticmethod
    def __has_ocr_elements(*, observation: ScreenObservation) -> bool:
        """
        Return whether the observation carries any OCR-sourced element.
        """

        return any(element.source == ElementSource.OCR for element in observation.elements)

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
            "component": "adapter.localizer.layout",
            "workflow.id": self.__workflow_id,
            "activity": activity,
            "target": target[:80],
            "member": self.name,
        }
