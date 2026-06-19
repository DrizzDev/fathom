from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional

from fathom.constants import SPATIAL_ACTION_TYPES, SWIPE_ACTIONS
from fathom.core.localization.ensemble import EnsembleLocalizerService
from fathom.core.localization.matcher import OcrPhraseMatcher
from fathom.core.localization.regional import RegionalEvidenceMatcher
from fathom.schemas.actions import Action, Bounds, CoordinateSource
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import (
    LocalizationCandidate,
    LocalizationProposal,
    LocalizationResult,
    LocalizationStatus,
    PhraseMatch,
    Point,
    RegionalEvidenceProposal,
    RegionalEvidenceVerdict,
)
from fathom.schemas.observation import ElementSource, PerceivedElement, ScreenObservation
from fathom.schemas.resolution import UnresolvedKind
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
        phrase_matcher: Optional[OcrPhraseMatcher] = None,
        ensemble: Optional[EnsembleLocalizerService] = None,
        regional_matcher: Optional[RegionalEvidenceMatcher] = None,
    ) -> None:
        """
        Initialize the localizer with optional ensemble layer, phrase + regional
        matchers, and run context. Defaults instantiate the Domain matchers so
        unit tests and minimal compositions need not wire them explicitly.
        """

        self.__ensemble = ensemble
        self.__workflow_id = workflow_id
        self.__phrase_matcher = phrase_matcher if phrase_matcher is not None else OcrPhraseMatcher()

        self.__regional_matcher = (
            regional_matcher
            if regional_matcher is not None
            else RegionalEvidenceMatcher(phrase_matcher=self.__phrase_matcher)
        )

    async def localize(
        self,
        *,
        image: bytes,
        action: Action,
        budget: LocalizationBudget,
        observation: ScreenObservation,
        capture: Optional[ScreenCapture] = None,
        snap_outcome: Optional[UnresolvedKind] = None,
    ) -> LocalizationResult:
        """
        Resolve an action's target into executable coordinates against the screen observation.
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

        label_unreliable = snap_outcome is UnresolvedKind.GENERIC_CONTAINER

        if action.label_id and not label_unreliable:
            label_result = self.__by_identifier(action=action, observation=observation)
            self.__log_localization_result(
                action=action,
                method="label_id",
                result=label_result,
                activity=observation.activity,
            )
            if label_result.status == LocalizationStatus.RESOLVED:
                return label_result

        elif action.label_id and label_unreliable and snap_outcome is not None:
            logger.info(
                "Localization label_id stage skipped on unreliable upstream snap",
                extra={
                    **self.__log_context(action=action, activity=observation.activity),
                    "event": "localization.label_id.skipped",
                    "snap.outcome": snap_outcome.value,
                },
            )

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

        regional_result = self.__by_regional_evidence(action=action, observation=observation)

        self.__log_localization_result(
            action=action,
            result=regional_result,
            method="regional_evidence",
            activity=observation.activity,
        )

        if regional_result.status == LocalizationStatus.RESOLVED:
            return regional_result

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

        blind_result = self.__by_blind_model_bounds(action=action)
        self.__log_localization_result(
            action=action,
            result=blind_result,
            method="blind_model_bounds",
            activity=observation.activity,
        )

        if blind_result.status == LocalizationStatus.RESOLVED:
            logger.info(
                "Localization resolved via planner-supplied bbox fallback",
                extra={
                    **self.__log_context(action=action, activity=observation.activity),
                    "event": "localization.blind_model_bounds.fired",
                    "localization.confidence": blind_result.confidence,
                    "localization.bounds": self.__bounds_snapshot(bounds=blind_result.bounds),
                },
            )
            return blind_result

        result = LocalizationResult(
            confidence=0.0,
            status=LocalizationStatus.UNRESOLVED,
            reason=(
                "Every cascade stage abstained and the planner did not supply "
                "fallback bounds; supervise must escalate."
            ),
        )
        self.__log_localization_result(
            action=action,
            result=result,
            method="unresolved",
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

    def __log_phrase_fallback(
        self,
        *,
        target: str,
        action: Action,
        phrase: Optional[PhraseMatch],
        observation: ScreenObservation,
    ) -> None:
        """
        Emit a structured record for one phrase-fallback evaluation inside ``__by_exact_text``.
        """

        eligible_ocr = sum(
            1
            for element in observation.elements
            if element.source == ElementSource.OCR and element.text
        )

        logger.info(
            "Phrase fallback evaluated",
            extra={
                **self.__log_context(action=action, activity=observation.activity),
                "event": "localization.phrase_fallback.evaluated",
                "target.normalized": target[:120],
                "phrase.matched": phrase is not None,
                "observation.ocr_token_count": eligible_ocr,
                "target.word_count": len(target.split()) if target else 0,
                "observation.total_element_count": len(observation.elements),
                "phrase.score": phrase.score if phrase is not None else None,
                "phrase.text": (phrase.text[:120] if phrase is not None else None),
                "phrase.confidence": (phrase.confidence if phrase is not None else None),
                "phrase.token_count": phrase.token_count if phrase is not None else None,
                "phrase.bounds": (
                    self.__bounds_snapshot(bounds=phrase.bounds) if phrase is not None else None
                ),
            },
        )

    def __log_regional_verdict(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
        verdict: RegionalEvidenceVerdict,
    ) -> None:
        """
        Emit a structured record carrying every gate metric and the verdict's decision.
        """

        configuration = self.__regional_matcher.configuration
        proposed_bounds = verdict.proposal.bounds if verdict.proposal is not None else None
        logger.info(
            "Regional evidence evaluated",
            extra={
                **self.__log_context(action=action, activity=observation.activity),
                "event": "localization.regional_evidence.evaluated",
                "regional.resolved": verdict.resolved,
                "regional.decision": verdict.decision.value,
                "regional.metrics.iou": verdict.metrics.iou,
                "regional.thresholds.iou": configuration.iou,
                "regional.metrics.fused": verdict.metrics.fused,
                "regional.thresholds.floor": configuration.floor,
                "regional.metrics.recall": verdict.metrics.recall,
                "regional.thresholds.recall": configuration.recall,
                "regional.metrics.density": verdict.metrics.density,
                "regional.thresholds.density": configuration.density,
                "regional.metrics.containment": verdict.metrics.containment,
                "regional.cluster.token_count": verdict.cluster_token_count,
                "regional.thresholds.containment": configuration.containment,
                "regional.in_region_token_count": verdict.in_region_token_count,
                "regional.action.bounds": self.__bounds_snapshot(bounds=action.bounds),
                "regional.proposed.bounds": self.__bounds_snapshot(bounds=proposed_bounds),
                "regional.cluster.phrase": (verdict.phrase[:120] if verdict.phrase else None),
                "regional.observation.ocr_token_count": sum(
                    1 for element in observation.elements if element.source == ElementSource.OCR
                ),
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
        Convert an ensemble proposal into a resolved result tagged ``VISION``.
        """

        bounds = proposal.bounds.model_copy(update={"source": CoordinateSource.VISION})

        return LocalizationResult(
            bounds=bounds,
            source=ElementSource.MODEL,
            confidence=proposal.confidence,
            status=LocalizationStatus.RESOLVED,
            point=self.__center(bounds=bounds),
            candidates=(
                LocalizationCandidate(
                    element=None,
                    score=proposal.confidence,
                    point=self.__center(bounds=bounds),
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
        Resolve by exact normalized text, falling back to a row-clustered phrase match.
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

        phrase = self.__phrase_matcher.find_best_match(
            target=target,
            elements=observation.elements,
        )
        self.__log_phrase_fallback(
            phrase=phrase,
            action=action,
            target=target,
            observation=observation,
        )
        if phrase is not None:
            return self.__from_phrase_match(phrase=phrase)

        return LocalizationResult(
            confidence=0.0,
            status=LocalizationStatus.UNRESOLVED,
            reason="No element matched the exact visible text or row-clustered phrase.",
        )

    def __by_regional_evidence(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
    ) -> LocalizationResult:
        """
        Resolve by fusing planner bounds with OCR phrase evidence inside the region.
        """

        if action.bounds is None:
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Action did not include model bounds.",
            )

        target = self.__target_text(action=action)
        if not target:
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Action has no semantic text target for regional evidence.",
            )

        verdict = self.__regional_matcher.evaluate(
            target=target,
            bounds=action.bounds,
            elements=observation.elements,
        )
        self.__log_regional_verdict(
            action=action,
            verdict=verdict,
            observation=observation,
        )
        if verdict.proposal is not None:
            return self.__from_regional_proposal(proposal=verdict.proposal)

        return LocalizationResult(
            confidence=0.0,
            status=LocalizationStatus.UNRESOLVED,
            reason=f"Regional evidence abstained: {verdict.decision.value}.",
        )

    def __by_blind_model_bounds(self, *, action: Action) -> LocalizationResult:
        """
        Dispatch planner bounds verbatim when every other stage abstained.
        """

        if action.bounds is None:
            logger.warning(
                "Blind dispatch skipped because action carries no model bounds",
                extra={
                    **self.__log_context(action=action, activity=""),
                    "event": "localization.blind_model_bounds.skipped",
                    "skip.reason": "no_model_bounds",
                },
            )
            return LocalizationResult(
                confidence=0.0,
                status=LocalizationStatus.UNRESOLVED,
                reason="Action did not include model bounds for blind dispatch.",
            )

        bounds = action.bounds.model_copy(update={"source": CoordinateSource.MODEL})
        confidence = action.confidence if 0.0 <= action.confidence <= 1.0 else 0.0
        center = self.__center(bounds=bounds)

        logger.warning(
            "Blind dispatch of planner-emitted bounds",
            extra={
                **self.__log_context(action=action, activity=""),
                "event": "localization.blind_model_bounds.dispatched",
                "blind.confidence": confidence,
                "blind.point": {"x": center.x, "y": center.y},
                "blind.bounds": self.__bounds_snapshot(bounds=bounds),
            },
        )
        return LocalizationResult(
            point=center,
            bounds=bounds,
            source=ElementSource.MODEL,
            confidence=confidence,
            status=LocalizationStatus.RESOLVED,
            candidates=(
                LocalizationCandidate(
                    point=center,
                    element=None,
                    score=confidence,
                    reason="Blind dispatch of planner-emitted bounds; no corroboration available.",
                ),
            ),
            reason=(
                "Dispatching planner-emitted bounds without corroboration; "
                "all text and vision methods abstained."
            ),
        )

    def __from_phrase_match(self, *, phrase: PhraseMatch) -> LocalizationResult:
        """
        Convert a phrase cluster into a resolved result tagged ``OCR``.
        """

        bounds = phrase.bounds.model_copy(update={"source": CoordinateSource.OCR})
        return LocalizationResult(
            bounds=bounds,
            source=ElementSource.OCR,
            confidence=phrase.confidence,
            status=LocalizationStatus.RESOLVED,
            point=self.__center(bounds=bounds),
            candidates=(
                LocalizationCandidate(
                    element=None,
                    score=phrase.score,
                    point=self.__center(bounds=bounds),
                    reason=(
                        f"Phrase '{phrase.text[:40]}' matched target via row-cluster "
                        f"(F1={phrase.score:.2f}, tokens={phrase.token_count})."
                    ),
                ),
            ),
            reason=(
                "Resolved by OCR phrase row-cluster fallback "
                f"(text='{phrase.text[:60]}', F1={phrase.score:.2f})."
            ),
        )

    def __from_regional_proposal(self, *, proposal: RegionalEvidenceProposal) -> LocalizationResult:
        """
        Convert a regional-evidence proposal into a resolved result tagged ``MODEL_GROUNDED``.
        """

        bounds = proposal.bounds.model_copy(update={"source": CoordinateSource.MODEL_GROUNDED})
        return LocalizationResult(
            bounds=bounds,
            source=ElementSource.MODEL,
            confidence=proposal.score,
            status=LocalizationStatus.RESOLVED,
            point=self.__center(bounds=bounds),
            candidates=(
                LocalizationCandidate(
                    element=None,
                    score=proposal.score,
                    point=self.__center(bounds=bounds),
                    reason=(
                        f"Regional evidence: phrase='{proposal.phrase[:40]}' "
                        f"recall={proposal.recall:.2f} density={proposal.density:.2f} "
                        f"containment={proposal.containment:.2f} iou={proposal.iou:.2f}"
                    ),
                ),
            ),
            reason=(
                "Resolved by regional evidence (LLM bounds + OCR phrase corroboration); "
                f"score={proposal.score:.2f}."
            ),
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
