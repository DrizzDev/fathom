from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from fathom.constants.flow import EvidenceMarker
from fathom.schemas.generation import Distillation, ScrollCollapseState
from fathom.schemas.steps import StepRecord


class Distiller:
    """
    Drops planner recovery steps and collapses only non-semantic consecutive scroll thrash.
    """

    __SCROLL = "scroll"
    __SWIPE_PREFIX = "swipe"
    __VALIDATION = "validation"
    __DISMISSAL_ACTION = "tap"
    __PARTIAL_REASON = "No successful goal validation was recorded; the run did not complete."
    __LOOP_REASON = "Recovery thrash distilled and no successful goal validation was recorded."
    __DISMISSAL_TARGETS = frozenset({"close", "dismiss", "x"})
    __DISMISSAL_TERMS = frozenset({"close", "closing", "dismiss", "dismissing", "overlay", "blocker"})
    __INTENTIONAL_DISMISSAL_TERMS = frozenset({"close", "dismiss"})

    def distill(self, *, records: Sequence[StepRecord]) -> Distillation:
        """
        Drop recovery steps, collapse consecutive scrolls between recoveries, and flag partial runs.
        """

        recovery = {
            index for index, record in enumerate(records) if self.__is_recovery(record=record)
        }
        collapse_regions = self.__scroll_regions(recovery=recovery)

        kept: List[StepRecord] = []
        collapse_state = ScrollCollapseState()
        discarded: List[int] = [records[index].step_number for index in recovery]

        for index, record in enumerate(records):
            if index in recovery:
                continue

            region = collapse_regions.get(index)
            command = self.__collapse_command(record=record)

            if collapse_state.repeats(command=command, region=region):
                discarded.append(record.step_number)
                continue

            kept.append(record)
            collapse_state = collapse_state.advance(command=command, region=region)

        trailing = self.__trailing_no_progress(records=kept)
        if trailing:
            kept = kept[: -len(trailing)]
            discarded.extend(record.step_number for record in trailing)

        partial = not any(self.__is_success_validation(record=record) for record in kept)
        reason = (self.__LOOP_REASON if recovery else self.__PARTIAL_REASON) if partial else None

        return Distillation(
            reason=reason,
            partial=partial,
            records=tuple(kept),
            discarded=tuple(sorted(discarded)),
        )

    def __scroll_regions(self, *, recovery: Set[int]) -> Dict[int, int]:
        """
        Map each position strictly between two adjacent recovery markers to its interval id.
        """

        regions: Dict[int, int] = {}
        markers = sorted(recovery)

        if len(markers) < 2:
            return regions

        for region, left in enumerate(markers[:-1]):
            right = markers[region + 1]
            for index in range(left + 1, right):
                regions[index] = region

        return regions

    def __is_recovery(self, *, record: StepRecord) -> bool:
        """
        Recognize a step the planner took only to escape a loop or stuck screen.
        """

        if record.condition == EvidenceMarker.RECOVERY:
            return True

        if self.__is_surface_dismissal(record=record):
            return True

        rationale = record.rationale or ""
        return rationale.startswith(EvidenceMarker.LOOP_RATIONALE)

    def __is_surface_dismissal(self, *, record: StepRecord) -> bool:
        """
        Recognize dismissals of incidental surfaces that are not the active user goal.
        """

        if record.action_type != self.__DISMISSAL_ACTION:
            return False

        target = (record.export_target or record.natural_language_target or record.target).lower()
        if target not in self.__DISMISSAL_TARGETS:
            return False

        goal = (record.goal.description if record.goal is not None else "").lower()
        if any(term in goal for term in self.__INTENTIONAL_DISMISSAL_TERMS):
            return False

        rationale = f"{record.rationale or ''} {record.observation or ''}".lower()
        return any(term in rationale for term in self.__DISMISSAL_TERMS)

    def __collapse_command(self, *, record: StepRecord) -> Optional[str]:
        """
        Return the command family eligible for recovery-region collapse.
        """

        if self.__is_scroll(record=record):
            return self.__SCROLL

        return None

    def __trailing_no_progress(self, *, records: List[StepRecord]) -> Tuple[StepRecord, ...]:
        """
        Return trailing successful steps that recorded no screen change and no semantic value.
        """

        trailing: List[StepRecord] = []

        for record in reversed(records):
            if not self.__no_progress(record=record):
                break

            trailing.append(record)

        return tuple(reversed(trailing))

    def __no_progress(self, *, record: StepRecord) -> bool:
        """
        Return whether a step recorded no durable script value or visible progress.
        """

        return (
            record.success
            and record.capture is None
            and not record.screen_changed
            and record.event_type != self.__VALIDATION
            and self.__is_recovery(record=record)
        )

    def __is_success_validation(self, *, record: StepRecord) -> bool:
        """
        Recognize a recorded goal validation that actually succeeded.
        """

        return record.event_type == self.__VALIDATION and record.success

    def __is_scroll(self, *, record: StepRecord) -> bool:
        """
        Recognize a scroll or swipe gesture.
        """

        return record.action_type == self.__SCROLL or record.action_type.startswith(
            self.__SWIPE_PREFIX
        )
