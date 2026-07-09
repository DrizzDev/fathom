from __future__ import annotations

from typing import List, Optional, Set, Tuple

from fathom.schemas.flow import StepTarget, TargetAnchors, TargetClaim, TargetStructure
from fathom.schemas.steps import StepRecord


class TargetEvidenceBuilder:
    """
    Builds target evidence that separates evidence anchors from planner claims.
    """

    def build(self, *, record: StepRecord, export: Optional[str]) -> StepTarget:
        """
        Build the target view for one recorded execution step.
        """

        anchors = self.__anchors(record=record, export=export)
        claim = self.__claim(record=record, anchors=anchors)

        return StepTarget(
            export=export,
            claim=claim,
            anchors=anchors,
            scroll=record.scroll_target,
            positional=record.is_positional,
            element=record.target_element_type,
            name=record.natural_language_target,
            generalized=record.generalized_target,
            structure=TargetStructure(role=record.target_element_type),
        )

    def __anchors(self, *, record: StepRecord, export: Optional[str]) -> TargetAnchors:
        """
        Return anchors supplied by structured evidence channels.
        """

        accessibility = self.__unique(
            values=(
                export,
                record.validation_subject,
            )
        )

        return TargetAnchors(
            visual=self.__visual(record=record),
            accessibility=accessibility,
        )

    def __claim(self, *, record: StepRecord, anchors: TargetAnchors) -> TargetClaim:
        """
        Return the planner target claim and whether an anchor confirms it.
        """

        text = self.__first(
            values=(
                record.natural_language_target,
                record.target,
                record.scroll_target,
            )
        )
        if text is None:
            return TargetClaim()

        return TargetClaim(
            text=text,
            verified=self.__matches_anchor(text=text, anchors=anchors),
        )

    def __visual(self, *, record: StepRecord) -> Tuple[str, ...]:
        """
        Return visual anchors that are explicitly captured as values.
        """

        if record.capture is None or not record.capture.success or record.capture.value is None:
            return ()

        return self.__unique(values=(record.capture.value,))

    def __matches_anchor(self, *, text: str, anchors: TargetAnchors) -> bool:
        """
        Return whether text exactly matches a verified anchor after light normalization.
        """

        normalized = self.__normalize(value=text)
        if normalized is None:
            return False

        return normalized in {
            anchor
            for value in (*anchors.visual, *anchors.accessibility)
            if (anchor := self.__normalize(value=value)) is not None
        }

    def __unique(self, *, values: Tuple[Optional[str], ...]) -> Tuple[str, ...]:
        """
        Return non-empty values in first-seen order.
        """

        unique: List[str] = []
        seen: Set[str] = set()

        for value in values:
            cleaned = self.__clean(value=value)
            if cleaned is None:
                continue

            key = self.__normalize(value=cleaned)
            if key is None or key in seen:
                continue

            seen.add(key)
            unique.append(cleaned)

        return tuple(unique)

    def __first(self, *, values: Tuple[Optional[str], ...]) -> Optional[str]:
        """
        Return the first non-empty value.
        """

        return next(
            (cleaned for value in values if (cleaned := self.__clean(value=value)) is not None),
            None,
        )

    @staticmethod
    def __clean(*, value: Optional[str]) -> Optional[str]:
        """
        Return a stripped string when one is available.
        """

        if value is None:
            return None

        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def __normalize(*, value: str) -> Optional[str]:
        """
        Normalize text for exact anchor matching.
        """

        normalized = " ".join(value.casefold().split())
        return normalized or None
