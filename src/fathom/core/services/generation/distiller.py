from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from fathom.constants.flow import EvidenceMarker
from fathom.schemas.generation import Distillation
from fathom.schemas.steps import StepRecord


class Distiller:
    """
    Drops planner recovery steps and collapses only non-semantic consecutive scroll thrash.
    """

    __SCROLL = "scroll"
    __SWIPE_PREFIX = "swipe"
    __VALIDATION = "validation"
    __PARTIAL_REASON = "No successful goal validation was recorded; the run did not complete."
    __LOOP_REASON = "Recovery thrash distilled and no successful goal validation was recorded."

    def distill(self, *, records: Sequence[StepRecord]) -> Distillation:
        """
        Drop recovery steps, collapse consecutive scrolls between recoveries, and flag partial runs.
        """

        recovery = {
            index for index, record in enumerate(records) if self.__is_recovery(record=record)
        }
        collapse_regions = self.__scroll_regions(recovery=recovery)

        kept: List[StepRecord] = []
        discarded: List[int] = [records[index].step_number for index in recovery]
        last_scroll_region: Optional[int] = None

        for index, record in enumerate(records):
            if index in recovery:
                continue

            region = collapse_regions.get(index)

            if (
                region is not None
                and self.__is_scroll(record=record)
                and last_scroll_region == region
            ):
                discarded.append(record.step_number)
                continue

            kept.append(record)
            last_scroll_region = (
                region if region is not None and self.__is_scroll(record=record) else None
            )

        partial = not any(self.__is_success_validation(record=record) for record in kept)
        reason = (self.__LOOP_REASON if recovery else self.__PARTIAL_REASON) if partial else None

        return Distillation(
            records=tuple(kept),
            discarded=tuple(sorted(discarded)),
            partial=partial,
            reason=reason,
        )

    def __scroll_regions(self, *, recovery: Set[int]) -> Dict[int, int]:
        """
        Map each position strictly between two adjacent recovery markers to its interval id.
        """

        markers = sorted(recovery)
        regions: Dict[int, int] = {}

        if len(markers) < 2:
            return regions

        for region, left in enumerate(markers[:-1]):
            right = markers[region + 1]
            for index in range(left + 1, right):
                regions[index] = region

        return regions

    def __is_recovery(self, *, record: StepRecord) -> bool:
        """
        Recognise a step the planner took only to escape a loop or stuck screen.
        """

        if record.condition == EvidenceMarker.RECOVERY:
            return True

        rationale = record.rationale or ""
        return rationale.startswith(EvidenceMarker.LOOP_RATIONALE)

    def __is_success_validation(self, *, record: StepRecord) -> bool:
        """
        Recognise a recorded goal validation that actually succeeded.
        """

        return record.event_type == self.__VALIDATION and record.success

    def __is_scroll(self, *, record: StepRecord) -> bool:
        """
        Recognise a scroll or swipe gesture.
        """

        return record.action_type == self.__SCROLL or record.action_type.startswith(
            self.__SWIPE_PREFIX
        )
