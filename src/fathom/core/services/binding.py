from __future__ import annotations

from typing import Dict, Final, List, Optional, Tuple

from fathom.constants.turn.binding import (
    BindingLimit,
    BindingOrigin,
    BindingState,
    BindingThreshold,
)
from fathom.schemas.actions import Action, Bounds, CoordinateSource
from fathom.schemas.binding import Binding
from fathom.schemas.localization import LocalizationResult
from fathom.schemas.observation import ElementSource, PerceivedElement


class Binder:
    """
    Produces the typed grounding result for a spatial action by re-anchoring matched bounds to the interactive node.
    """

    __ORIGINS: Final[Dict[ElementSource, BindingOrigin]] = {
        ElementSource.OCR: BindingOrigin.OCR,
        ElementSource.CV: BindingOrigin.VISION,
        ElementSource.ICON: BindingOrigin.VISION,
        ElementSource.MODEL: BindingOrigin.VISION,
        ElementSource.VISION: BindingOrigin.VISION,
        ElementSource.XML: BindingOrigin.HIERARCHY,
        ElementSource.ACCESSIBILITY: BindingOrigin.ACCESSIBILITY,
    }
    __COORDINATES: Final[Dict[CoordinateSource, BindingOrigin]] = {
        CoordinateSource.OCR: BindingOrigin.OCR,
        CoordinateSource.MODEL: BindingOrigin.VISION,
        CoordinateSource.XML: BindingOrigin.HIERARCHY,
        CoordinateSource.VIEWPORT: BindingOrigin.VISION,
    }

    def bind(
        self,
        *,
        action: Action,
        elements: Tuple[PerceivedElement, ...],
        localization: Optional[LocalizationResult] = None,
    ) -> Binding:
        """
        Return the grounding result for the action's resolved target.
        """

        if (bounds := action.bounds) is None:
            return Binding(
                state=BindingState.MISSING,
                confidence=0.0,
                evidence=("action carries no bounds",),
            )

        if (matched := self.__match(action=action, elements=elements)) is None:
            return self.__perceptual(bounds=bounds, localization=localization)

        return self.__anchored(matched=matched, elements=elements)

    def __anchored(
        self,
        *,
        matched: PerceivedElement,
        elements: Tuple[PerceivedElement, ...],
    ) -> Binding:
        """
        Ground a hierarchy-matched target onto the element that actually receives the action.
        """

        if self.__actionable(element=matched):
            return Binding(
                bounds=matched.bounds,
                state=BindingState.BOUND,
                anchor=matched.identifier,
                confidence=matched.confidence,
                origin=self.__ORIGINS[matched.source],
                evidence=(f"matched element {matched.identifier} is interactive",),
            )

        candidates = self.__within(container=matched, elements=elements)
        if len(candidates) == 1:
            return self.__re_anchored(matched=matched, anchor=candidates[0])

        if len(candidates) > 1:
            return self.__contested(matched=matched, candidates=candidates)

        if (enclosing := self.__enclosing(target=matched, elements=elements)) is not None:
            return Binding(
                state=BindingState.BOUND,
                origin=self.__ORIGINS[matched.source],
                confidence=matched.confidence,
                bounds=matched.bounds,
                anchor=enclosing.identifier,
                evidence=(
                    f"interactive ancestor {enclosing.identifier} receives actions "
                    f"landing on {matched.identifier}",
                ),
            )

        return Binding(
            state=BindingState.INFERRED,
            origin=self.__ORIGINS[matched.source],
            confidence=matched.confidence,
            bounds=matched.bounds,
            evidence=(f"no interactive relative found for {matched.identifier}",),
        )

    def __re_anchored(self, *, matched: PerceivedElement, anchor: PerceivedElement) -> Binding:
        """
        Bind onto the single interactive descendant inside a non-interactive container.
        """

        origin = (
            BindingOrigin.HYBRID
            if anchor.source is not matched.source
            else self.__ORIGINS[anchor.source]
        )

        return Binding(
            state=BindingState.BOUND,
            origin=origin,
            confidence=anchor.confidence,
            bounds=anchor.bounds,
            anchor=anchor.identifier,
            evidence=(
                f"re-anchored from container {matched.identifier} to interactive "
                f"descendant {anchor.identifier}",
            ),
        )

    def __contested(
        self,
        *,
        matched: PerceivedElement,
        candidates: List[PerceivedElement],
    ) -> Binding:
        """
        Report multiple interactive descendants competing for one container target.
        """

        preview = ", ".join(
            candidate.identifier for candidate in candidates[: BindingLimit.EVIDENCE_PREVIEW]
        )

        return Binding(
            state=BindingState.CONTESTED,
            origin=self.__ORIGINS[matched.source],
            confidence=matched.confidence,
            bounds=matched.bounds,
            evidence=(
                f"container {matched.identifier} holds {len(candidates)} interactive "
                f"descendants: {preview}",
            ),
        )

    def __perceptual(
        self,
        *,
        bounds: Bounds,
        localization: Optional[LocalizationResult],
    ) -> Binding:
        """
        Ground a target resolved outside the hierarchy manifest (vision or OCR localization).
        """

        if localization is not None and localization.source is not None:
            origin = self.__ORIGINS[localization.source]
            confidence = localization.confidence
        else:
            origin = self.__COORDINATES.get(
                bounds.source or CoordinateSource.MODEL,
                BindingOrigin.VISION,
            )
            confidence = 0.0

        state = (
            BindingState.BOUND
            if confidence >= BindingThreshold.CONFIDENCE_FLOOR
            else BindingState.INFERRED
        )

        return Binding(
            state=state,
            origin=origin,
            confidence=confidence,
            bounds=bounds,
            evidence=(f"perceptual grounding via {origin.value} at confidence {confidence:.2f}",),
        )

    def __within(
        self,
        *,
        container: PerceivedElement,
        elements: Tuple[PerceivedElement, ...],
    ) -> List[PerceivedElement]:
        """
        Return interactive elements strictly inside the container's bounds.
        """

        return [
            element
            for element in elements
            if element.identifier != container.identifier
            and self.__actionable(element=element)
            and self.__contains(outer=container.bounds, inner=element.bounds)
            and self.__area(bounds=element.bounds) < self.__area(bounds=container.bounds)
        ]

    def __enclosing(
        self,
        *,
        target: PerceivedElement,
        elements: Tuple[PerceivedElement, ...],
    ) -> Optional[PerceivedElement]:
        """
        Return the smallest interactive element enclosing the target's bounds.
        """

        enclosing = [
            element
            for element in elements
            if element.identifier != target.identifier
            and self.__actionable(element=element)
            and self.__contains(outer=element.bounds, inner=target.bounds)
        ]
        if not enclosing:
            return None

        return min(enclosing, key=lambda element: self.__area(bounds=element.bounds))

    @staticmethod
    def __match(
        *,
        action: Action,
        elements: Tuple[PerceivedElement, ...],
    ) -> Optional[PerceivedElement]:
        """
        Return the manifest element the action's label resolved against.
        """

        if not action.label_id:
            return None

        return next(
            (element for element in elements if element.label_id == action.label_id),
            None,
        )

    @staticmethod
    def __actionable(*, element: PerceivedElement) -> bool:
        """
        Return whether an element receives actions; declared clickability beats role inference.
        """

        if element.interactive is not None:
            return element.interactive

        return element.tappable

    @staticmethod
    def __contains(*, outer: Bounds, inner: Bounds) -> bool:
        """
        Return whether the inner bounds sit fully inside the outer bounds.
        """

        return (
            inner.x >= outer.x
            and inner.y >= outer.y
            and inner.x + inner.width <= outer.x + outer.width
            and inner.y + inner.height <= outer.y + outer.height
        )

    @staticmethod
    def __area(*, bounds: Bounds) -> int:
        """
        Return the pixel area of the bounds.
        """

        return bounds.width * bounds.height
