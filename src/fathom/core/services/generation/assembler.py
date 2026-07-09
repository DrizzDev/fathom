from __future__ import annotations

from typing import Optional, Tuple

from fathom.constants import ActionType, StepEvent
from fathom.constants.state import RunOutcome
from fathom.core.exceptions import InvariantViolation
from fathom.core.services.generation.target import TargetEvidenceBuilder
from fathom.schemas.flow import (
    CompletionAssertion,
    Evidence,
    EvidenceStep,
    StepCapture,
    StepGuard,
    StepLaunch,
    StepOutcome,
    StepWait,
)
from fathom.schemas.generation import LaunchMarker, NormalizedEntry, NormalizedTrace
from fathom.schemas.steps import StepRecord


class EvidenceAssembler:
    """
    Maps a normalized workflow trace into the Evidence aggregate, including launch markers.
    """

    __LAUNCH = "launch"

    def __init__(self, *, target_builder: Optional[TargetEvidenceBuilder] = None) -> None:
        """
        Bind collaborators used to assemble evidence.
        """

        self.__target = target_builder or TargetEvidenceBuilder()

    def assemble(
        self,
        *,
        intent: str,
        goal: str,
        package: str,
        trace: NormalizedTrace,
        partial: bool = False,
        discarded: Tuple[int, ...] = (),
        reason: Optional[str] = None,
        assertions: Tuple[CompletionAssertion, ...] = (),
        outcome: RunOutcome = RunOutcome.COMPLETED,
    ) -> Evidence:
        """
        Build the evidence aggregate from a normalized trace of launches and step records.
        """

        completed = bool(assertions)
        return Evidence(
            goal=goal,
            intent=intent,
            outcome=outcome,
            assertions=assertions,
            reason=None if completed else reason,
            partial=False if completed else partial,
            discarded=discarded,
            package=self.__package(trace=trace, fallback=package),
            steps=tuple(self.__entry(entry=entry) for entry in trace.entries),
        )

    def __entry(self, *, entry: NormalizedEntry) -> EvidenceStep:
        """
        Map one normalized entry into an evidence step.
        """

        if entry.launch is not None:
            return self.__launch(marker=entry.launch)

        if entry.record is not None:
            return self.__step(record=entry.record)

        raise InvariantViolation("Normalized entry must contain a launch or record.")

    @staticmethod
    def __package(*, trace: NormalizedTrace, fallback: str) -> str:
        """
        Return the first launch package when present, otherwise the supplied package.
        """

        return next(
            (entry.launch.package for entry in trace.entries if entry.launch is not None),
            fallback,
        )

    def __launch(self, *, marker: LaunchMarker) -> EvidenceStep:
        """
        Map a synthesised launch marker into a launch evidence step.
        """

        index = min(marker.source_steps) if marker.source_steps else 0
        return EvidenceStep(
            index=index,
            event=self.__LAUNCH,
            action=self.__LAUNCH,
            launch=StepLaunch(
                package=marker.package,
                provenance=marker.provenance,
                source_steps=marker.source_steps,
            ),
        )

    def __step(self, *, record: StepRecord) -> EvidenceStep:
        """
        Map one recorded step record into an evidence step.
        """

        return EvidenceStep(
            text=record.text,
            goal=record.goal,
            event=record.event_type,
            index=record.step_number,
            action=record.action_type,
            rationale=record.rationale,
            artifacts=record.artifacts,
            observation=record.observation,
            target=self.__target.build(record=record, export=self.__target_export(record=record)),
            wait=StepWait(subject=record.wait_subject, pattern=record.wait_pattern),
            guard=StepGuard(
                condition=record.condition,
                kind=record.conditional_type,
                overlay=record.overlay_detected,
                conditional=record.is_conditional,
            ),
            outcome=StepOutcome(
                success=record.success,
                duration=record.duration,
                changed=record.screen_changed,
            ),
            capture=self.__capture(record=record),
        )

    @classmethod
    def __target_export(cls, *, record: StepRecord) -> Optional[str]:
        """
        Return the script assertion target, preferring structured validation subjects.
        """

        if cls.__has_validation_subject(record=record):
            return record.validation_subject

        return record.export_target

    @classmethod
    def __has_validation_subject(cls, *, record: StepRecord) -> bool:
        """
        Return whether the record carries a structured validation assertion.
        """

        if not record.validation_subject:
            return False

        return (
            record.event_type == StepEvent.VALIDATION
            or record.action_type == ActionType.VALIDATE.value
        )

    @staticmethod
    def __capture(*, record: StepRecord) -> Optional[StepCapture]:
        """
        Combine the persisted STORE request and outcome into capture evidence, if this step stored.
        """

        request = record.capture_request
        outcome = record.capture
        if request is None and outcome is None:
            return None

        if record.action_type != ActionType.STORE.value:
            raise InvariantViolation(
                f"Step {record.step_number} carries capture evidence on non-STORE action "
                f"'{record.action_type}'."
            )

        if request is None:
            raise InvariantViolation(
                f"Step {record.step_number} carries a capture outcome without a STORE request."
            )

        if outcome is None:
            raise InvariantViolation(
                f"Step {record.step_number} carries a STORE request without a capture outcome."
            )

        if request.name != outcome.name:
            raise InvariantViolation(
                f"Step {record.step_number} capture name mismatch: request='{request.name}', "
                f"outcome='{outcome.name}'."
            )

        if outcome.step != record.step_number:
            raise InvariantViolation(
                f"Step {record.step_number} capture outcome references step {outcome.step}."
            )

        if outcome.success and outcome.value != request.value:
            raise InvariantViolation(
                f"Step {record.step_number} capture value mismatch for '{request.name}'."
            )

        return StepCapture(
            name=request.name,
            value=outcome.value,
            reason=outcome.reason,
            subject=request.subject,
            success=outcome.success,
        )
