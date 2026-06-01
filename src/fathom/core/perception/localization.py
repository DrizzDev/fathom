from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional

from fathom.constants import SPATIAL_ACTION_TYPES, SWIPE_ACTIONS
from fathom.constants.perception import MODEL_BOUNDS_MINIMUM_IOU
from fathom.core.localization.ensemble import EnsembleLocalizerService
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import (
    LocalizationCandidate,
    LocalizationProposal,
    LocalizationResult,
    LocalizationStatus,
    Point,
)
from fathom.schemas.observation import ElementSource, PerceivedElement, ScreenObservation
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class TargetLocalizationService:
    """
    Resolves semantic action targets into executable coordinates.
    """

    def __init__(
        self,
        *,
        workflow_id: Optional[str] = None,
        ensemble: Optional[EnsembleLocalizerService] = None,
    ) -> None:
        """
        Initialize the localizer with an optional ensemble layer and run context.
        """

        self.__ensemble = ensemble
        self.__workflow_id = workflow_id

    async def localize(
        self,
        *,
        image: bytes,
        action: Action,
        budget: LocalizationBudget,
        observation: ScreenObservation,
        capture: Optional[ScreenCapture] = None,
    ) -> LocalizationResult:
        """
        Localize one action against the current screen observation.
        """

        _ = image

        if action.action_type not in SPATIAL_ACTION_TYPES:
            result = LocalizationResult(
                confidence=1.0,
                status=LocalizationStatus.RESOLVED,
                reason="Non-spatial action does not require target coordinates.",
            )
            self.__log_localization_result(
                action=action,
                result=result,
                method="non_spatial",
                activity=observation.activity,
            )
            return result

        if action.action_type.value in SWIPE_ACTIONS and not action.label_id:
            result = LocalizationResult(
                confidence=1.0,
                status=LocalizationStatus.RESOLVED,
                reason="Gesture action can execute without a specific element target.",
            )
            self.__log_localization_result(
                action=action,
                result=result,
                method="gesture_without_label",
                activity=observation.activity,
            )
            return result

        if action.label_id:
            label_result = self.__by_identifier(action=action, observation=observation)
            self.__log_localization_result(
                action=action,
                method="label_id",
                result=label_result,
                activity=observation.activity,
            )
            if label_result.status == LocalizationStatus.RESOLVED:
                return label_result

        target_result = self.__by_target_identifier(action=action, observation=observation)
        self.__log_localization_result(
            action=action,
            result=target_result,
            method="target_identifier",
            activity=observation.activity,
        )
        if target_result.status == LocalizationStatus.RESOLVED:
            return target_result

        text_result = self.__by_exact_text(action=action, observation=observation)
        self.__log_localization_result(
            action=action,
            result=text_result,
            method="exact_text",
            activity=observation.activity,
        )
        if text_result.status == LocalizationStatus.RESOLVED:
            return text_result

        model_result = self.__by_model_bounds(action=action, observation=observation)
        self.__log_localization_result(
            action=action,
            result=model_result,
            method="model_bounds",
            activity=observation.activity,
        )
        if model_result.status == LocalizationStatus.RESOLVED:
            return model_result

        ensemble_result = await self.__by_ensemble(
            action=action,
            budget=budget,
            capture=capture,
            observation=observation,
        )
        self.__log_localization_result(
            action=action,
            method="ensemble",
            result=ensemble_result,
            activity=observation.activity,
        )
        if ensemble_result.status == LocalizationStatus.RESOLVED:
            return ensemble_result

        candidates = tuple(
            LocalizationCandidate(
                element=element,
                score=element.confidence,
                reason="Visible tappable candidate.",
                point=self.__center(bounds=element.bounds),
            )
            for element in observation.elements
            if element.tappable
        )

        result = LocalizationResult(
            confidence=0.0,
            candidates=candidates,
            status=LocalizationStatus.UNRESOLVED,
            reason="No perceived element matched the semantic target.",
        )
        self.__log_localization_result(
            action=action,
            result=result,
            method="candidate_fallback",
            activity=observation.activity,
        )
        return result

    async def __by_ensemble(
        self,
        *,
        action: Action,
        budget: LocalizationBudget,
        observation: ScreenObservation,
        capture: Optional[ScreenCapture],
    ) -> LocalizationResult:
        """
        Consult the ensemble layer when in-process passes did not resolve.
        """

        if self.__ensemble is None or capture is None:
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Ensemble layer unavailable for this localization request.",
            )

        if (
            proposal := await self.__ensemble.locate(
                action=action,
                budget=budget,
                capture=capture,
                observation=observation,
            )
        ) is None:
            logger.info(
                "Localization fell through ensemble without consensus",
                extra={
                    "event": "localization.ensemble.miss",
                    **self.__log_context(action=action, activity=capture.activity),
                },
            )
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Ensemble did not produce a consensus proposal.",
            )

        logger.info(
            "Localization resolved via ensemble consensus",
            extra={
                "consensus.source": proposal.source,
                "event": "localization.ensemble.resolved",
                "consensus.confidence": proposal.confidence,
                **self.__log_context(action=action, activity=capture.activity),
            },
        )
        return self.__from_proposal(proposal=proposal)

    def __log_context(self, *, action: Action, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for localization entries.
        """

        return {
            "activity": activity,
            "component": "core.localization",
            "workflow.id": self.__workflow_id,
            "action.type": action.action_type.value,
            "action.target": (action.target or "")[:80],
        }

    def __log_localization_result(
        self,
        *,
        method: str,
        activity: str,
        action: Action,
        result: LocalizationResult,
    ) -> None:
        """
        Emit one structured record for each localization decision point.

        These records are intentionally verbose enough to reconstruct a
        wrong-label RCA from logs alone: action target, planner metadata,
        method, selected element text/role/source/bounds, candidate count,
        and the final bounds that would reach execution.
        """

        selected = self.__selected_element(result=result)
        logger.info(
            "Localization method evaluated",
            extra={
                **self.__log_context(action=action, activity=activity),
                "event": "localization.method.evaluated",
                "localization.method": method,
                "localization.reason": result.reason,
                "localization.status": result.status.value,
                "localization.confidence": result.confidence,
                "localization.source": result.source.value if result.source else None,
                "localization.bounds": self.__bounds_snapshot(bounds=result.bounds),
                "localization.point": (
                    {"x": result.point.x, "y": result.point.y} if result.point else None
                ),
                "candidate.count": len(result.candidates),
                "candidate.preview": self.__candidate_preview(result=result),
                "action.label_id": action.label_id,
                "action.natural_language_target": (
                    (action.natural_language_target or "")[:120]
                    if action.natural_language_target
                    else None
                ),
                "action.target_is_generic": action.target_is_generic,
                "action.target_element_type": action.target_element_type,
                "action.bounds": self.__bounds_snapshot(bounds=action.bounds),
                "selected.element": self.__element_snapshot(element=selected),
            },
        )

    @staticmethod
    def __selected_element(*, result: LocalizationResult) -> Optional[PerceivedElement]:
        """
        Return the first candidate element when the localization selected one.
        """

        for candidate in result.candidates:
            if candidate.element is not None:
                return candidate.element

        return None

    @classmethod
    def __candidate_preview(cls, *, result: LocalizationResult) -> tuple[Dict[str, Any], ...]:
        """
        Return a compact preview of the first few localization candidates.
        """

        return tuple(
            {
                "score": candidate.score,
                "reason": candidate.reason,
                "point": cls.__point_snapshot(point=candidate.point),
                "element": cls.__element_snapshot(element=candidate.element),
            }
            for candidate in result.candidates[:3]
        )

    @staticmethod
    def __point_snapshot(*, point: Optional[Point]) -> Optional[Dict[str, int]]:
        """
        Return a log-safe point snapshot.
        """

        if point is None:
            return None

        return {"x": point.x, "y": point.y}

    @staticmethod
    def __element_snapshot(*, element: Optional[PerceivedElement]) -> Optional[Dict[str, Any]]:
        """
        Return a stable log-safe snapshot for a perceived element.
        """

        if element is None:
            return None

        return {
            "axis": element.axis,
            "kind": element.kind,
            "parent": element.parent,
            "role": element.role.value,
            "label_id": element.label_id,
            "tappable": element.tappable,
            "source": element.source.value,
            "scrollable": element.scrollable,
            "confidence": element.confidence,
            "identifier": element.identifier,
            "text": (element.text or "")[:120],
            "bounds": TargetLocalizationService.__bounds_snapshot(bounds=element.bounds),
        }

    @staticmethod
    def __bounds_snapshot(*, bounds: Optional[Bounds]) -> Optional[Dict[str, Any]]:
        """
        Return bounds in a consistent log shape.
        """

        if bounds is None:
            return None

        return {
            "x": bounds.x,
            "y": bounds.y,
            "width": bounds.width,
            "height": bounds.height,
            "system": bounds.system.value,
            "source": bounds.source.value if bounds.source else None,
        }

    def __from_proposal(self, *, proposal: LocalizationProposal) -> LocalizationResult:
        """
        Convert an ensemble proposal into a resolved LocalizationResult.
        """

        return LocalizationResult(
            bounds=proposal.bounds,
            source=ElementSource.MODEL,
            confidence=proposal.confidence,
            status=LocalizationStatus.RESOLVED,
            point=self.__center(bounds=proposal.bounds),
            candidates=(
                LocalizationCandidate(
                    element=None,
                    score=proposal.confidence,
                    point=self.__center(bounds=proposal.bounds),
                    reason=proposal.rationale or "Ensemble consensus proposal.",
                ),
            ),
            reason=f"Resolved via ensemble consensus: {proposal.source}.",
        )

    def __by_identifier(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
    ) -> LocalizationResult:
        """
        Resolve by explicit manifest identifier.
        """

        for element in observation.elements:
            if element.identifier == action.label_id:
                return self.__resolved(element=element, reason="Matched explicit label identifier.")

        return LocalizationResult(
            confidence=0.0,
            status=LocalizationStatus.UNRESOLVED,
            reason="Explicit label identifier was not present in the observation.",
        )

    def __by_target_identifier(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
    ) -> LocalizationResult:
        """
        Resolve when the model names a runtime observation identifier.
        """

        target = self.__target_text(action=action)

        if not target:
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Action has no target identifier.",
            )

        for element in observation.elements:
            if self.__normalize(value=element.identifier) == target:
                return self.__resolved(
                    element=element,
                    reason="Matched runtime observation identifier.",
                )

        return LocalizationResult(
            confidence=0.0,
            status=LocalizationStatus.UNRESOLVED,
            reason="No element matched the runtime observation identifier.",
        )

    def __by_exact_text(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
    ) -> LocalizationResult:
        """
        Resolve by exact normalized visible text.
        """

        target = self.__target_text(action=action)

        if not target:
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Action has no semantic text target.",
            )

        matches = tuple(
            element
            for element in observation.elements
            if element.text and self.__normalize(value=element.text) == target
        )
        if len(matches) == 1:
            return self.__resolved(element=matches[0], reason="Matched exact visible text.")

        if len(matches) > 1:
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.AMBIGUOUS,
                candidates=tuple(
                    LocalizationCandidate(
                        element=element,
                        score=element.confidence,
                        reason="Exact text match.",
                        point=self.__center(bounds=element.bounds),
                    )
                    for element in matches
                ),
                reason="Multiple elements matched the exact visible text.",
            )

        return LocalizationResult(
            confidence=0.0,
            status=LocalizationStatus.UNRESOLVED,
            reason="No element matched the exact visible text.",
        )

    def __by_model_bounds(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
    ) -> LocalizationResult:
        """
        Resolve model bounds only when they overlap perceived evidence.
        """

        if action.bounds is None:
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Action did not include model bounds.",
            )

        best_score = 0.0
        best_element: Optional[PerceivedElement] = None

        for element in observation.elements:
            score = self.__iou(first=action.bounds, second=element.bounds)
            if score > best_score:
                best_score = score
                best_element = element

        if best_element is None or best_score < MODEL_BOUNDS_MINIMUM_IOU:
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Model bounds did not overlap perceived screen evidence.",
            )

        return LocalizationResult(
            bounds=best_element.bounds,
            source=ElementSource.MODEL,
            status=LocalizationStatus.RESOLVED,
            point=self.__center(bounds=best_element.bounds),
            confidence=min(best_element.confidence, best_score),
            candidates=(
                LocalizationCandidate(
                    score=best_score,
                    element=best_element,
                    point=self.__center(bounds=best_element.bounds),
                    reason="Model bounds overlapped perceived element.",
                ),
            ),
            reason="Model bounds reconciled with perceived screen evidence.",
        )

    def __resolved(self, *, element: PerceivedElement, reason: str) -> LocalizationResult:
        """
        Build a resolved localization result from a perceived element.
        """

        return LocalizationResult(
            bounds=element.bounds,
            source=element.source,
            confidence=element.confidence,
            status=LocalizationStatus.RESOLVED,
            point=self.__center(bounds=element.bounds),
            candidates=(
                LocalizationCandidate(
                    reason=reason,
                    element=element,
                    score=element.confidence,
                    point=self.__center(bounds=element.bounds),
                ),
            ),
            reason=reason,
        )

    def __target_text(self, *, action: Action) -> str:
        """
        Return the normalized semantic target text.
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
        Normalize text for exact matching.
        """

        return " ".join(value.strip().lower().split())

    @staticmethod
    def __center(*, bounds: Bounds) -> Point:
        """
        Return the center point of bounds.
        """

        return Point(x=bounds.center_x, y=bounds.center_y)

    @staticmethod
    def __iou(*, first: Bounds, second: Bounds) -> float:
        """
        Return intersection-over-union for two bounds.
        """

        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        first_area = first.width * first.height
        second_area = second.width * second.height
        union = first_area + second_area - intersection
        if union <= 0:
            return 0.0

        return float(intersection / union)
