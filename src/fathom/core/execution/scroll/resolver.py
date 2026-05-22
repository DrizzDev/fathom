from __future__ import annotations

from fathom.constants.scroll import ScrollEvidenceSource
from fathom.interfaces.command import CommandScopeResolvePort
from fathom.schemas.actions import Bounds, CoordinateSource
from fathom.schemas.command import CommandAnchor, CommandScope
from fathom.schemas.observation import ScreenObservation, ScrollRegion
from fathom.schemas.scroll import ScrollScope
from fathom.utils.coordinates import CoordinateConverter


class ScrollScopeResolver(CommandScopeResolvePort):
    """
    Resolves one scroll anchor into one container-scoped execution target.
    """

    async def resolve(
        self,
        *,
        anchor: CommandAnchor,
        observation: ScreenObservation,
        fallback: CommandScope,
        converter: CoordinateConverter,
    ) -> ScrollScope:
        """
        Resolve the most likely scroll scope for the current command.
        """

        desired_axis = fallback.axis or "vertical"
        candidates = tuple(
            self.__scope_from_region(region=region, converter=converter)
            for region in observation.scroll
            if self.__is_axis_compatible(region=region, desired_axis=desired_axis)
        )
        if not candidates:
            return self.__scroll_scope(scope=fallback)

        if (matched_scope := self.__exact_match(candidates=candidates, anchor=anchor)) is not None:
            return matched_scope

        explicit = tuple(
            candidate for candidate in candidates if candidate.manifest_label_id is not None
        )
        if explicit:
            explicit_scored = self.__rank_candidates(
                candidates=explicit,
                observation=observation,
                anchor=anchor,
            )
            if explicit_scored and explicit_scored[0][0] > 0:
                return explicit_scored[0][1]

        scored = self.__rank_candidates(
            candidates=candidates,
            observation=observation,
            anchor=anchor,
        )
        return scored[0][1] if scored and scored[0][0] > 0 else self.__scroll_scope(scope=fallback)

    @staticmethod
    def __is_axis_compatible(*, region: ScrollRegion, desired_axis: str) -> bool:
        """
        Accept only scopes compatible with the requested movement axis.
        """

        axis = (region.axis or "vertical").lower()
        desired = (desired_axis or "vertical").lower()
        return axis == desired or axis == "unknown"

    def __exact_match(
        self,
        *,
        candidates: tuple[ScrollScope, ...],
        anchor: CommandAnchor,
    ) -> ScrollScope | None:
        """
        Prefer an exact anchor match before heuristic scoring.
        """

        if anchor.observation_region_id:
            matched = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.observation_region_id == anchor.observation_region_id
                ),
                None,
            )
            if matched is not None:
                return matched

        if anchor.manifest_label_id:
            matched = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.manifest_label_id == anchor.manifest_label_id
                ),
                None,
            )
            if matched is not None:
                return matched

        return None

    def __rank_candidates(
        self,
        *,
        candidates: tuple[ScrollScope, ...],
        observation: ScreenObservation,
        anchor: CommandAnchor,
    ) -> list[tuple[int, ScrollScope]]:
        """
        Score and sort candidate scopes.
        """

        return sorted(
            (
                (
                    self.__score_candidate(
                        candidate=candidate,
                        observation=observation,
                        anchor=anchor,
                    ),
                    candidate,
                )
                for candidate in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )

    def __score_candidate(
        self,
        *,
        candidate: ScrollScope,
        observation: ScreenObservation,
        anchor: CommandAnchor,
    ) -> int:
        """
        Score one scroll scope using anchor containment and scope shape.
        """

        score = 0
        if candidate.kind.value in {"viewport", "list", "container"}:
            score += 40
        if anchor.bounds is not None:
            if self.__contains(first=candidate.bounds, second=anchor.bounds):
                score += 120
            score += int(self.__overlap_ratio(first=candidate.bounds, second=anchor.bounds) * 100.0)
        if anchor.manifest_label_id:
            element = next(
                (
                    item
                    for item in observation.elements
                    if item.label_id == anchor.manifest_label_id
                    or item.identifier == anchor.manifest_label_id
                ),
                None,
            )
            if element is not None and self.__contains(
                first=candidate.bounds, second=element.bounds
            ):
                score += 150
        if anchor.target:
            lowered = anchor.target.lower()
            if any(token in lowered for token in ("main", "page", "screen", "scrollable")):
                score += 40
        score += int(candidate.confidence * 100.0)
        score += int(candidate.bounds.height / 20)
        score -= int(candidate.bounds.y / 40)
        return score

    def __scope_from_region(
        self,
        *,
        region: ScrollRegion,
        converter: CoordinateConverter,
    ) -> ScrollScope:
        """
        Convert one observation region into one typed scroll scope.
        """

        bounds = region.bounds
        return ScrollScope(
            identifier=region.identifier or region.observation_region_id or "scroll_region",
            kind=region.kind,
            bounds=bounds,
            region=converter.region_from_bounds(
                bounds=bounds,
                source=bounds.source or CoordinateSource.VIEWPORT,
            ),
            axis=region.axis,
            confidence=region.confidence,
            source=region.source or ScrollEvidenceSource.SURFACE,
            manifest_label_id=region.manifest_label_id,
            observation_region_id=region.observation_region_id,
        )

    @staticmethod
    def __scroll_scope(*, scope: CommandScope) -> ScrollScope:
        """
        Convert a generic scope into one scroll scope.
        """

        return ScrollScope(
            identifier=scope.identifier,
            kind=scope.kind,
            bounds=scope.bounds,
            region=scope.region,
            axis=scope.axis,
            confidence=scope.confidence,
            source=ScrollEvidenceSource.SURFACE,
            manifest_label_id=None,
            observation_region_id=None,
        )

    @staticmethod
    def __contains(*, first: Bounds, second: Bounds) -> bool:
        """
        Return whether the first bounds fully contains the second.
        """

        return (
            first.x <= second.x
            and first.y <= second.y
            and first.x + first.width >= second.x + second.width
            and first.y + first.height >= second.y + second.height
        )

    @staticmethod
    def __overlap_ratio(*, first: Bounds, second: Bounds) -> float:
        """
        Return a simple overlap ratio between two bounds.
        """

        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        denominator = max(1, second.width * second.height)
        return intersection / denominator
