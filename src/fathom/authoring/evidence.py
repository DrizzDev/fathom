from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from fathom.constants.authoring import AuthoringArtifactKind, AuthoringArtifactRole
from fathom.schemas.artifacts import ScreenArtifact
from fathom.schemas.authoring import (
    AuthoringArtifactReference,
    AuthoringCapture,
    AuthoringCommand,
    AuthoringEpisode,
    AuthoringEvidence,
    AuthoringNarrative,
    AuthoringRun,
    AuthoringScreen,
    AuthoringStep,
    AuthoringTarget,
    AuthoringValidation,
    RepairAuthoringEvidence,
    RunAuthoringEvidence,
    StepAuthoringEvidence,
)
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.flow import Evidence, EvidenceStep, Flow, Report
from fathom.schemas.steps import StepGoal


class AuthoringEvidenceBuilder:
    """
    Builds authoring task evidence views from normalized execution evidence.
    """

    __VALIDATION = "validation"
    __MANIFEST_SUFFIXES: Tuple[str, ...] = (".json", ".xml")
    __IMAGE_SUFFIXES: Tuple[str, ...] = (".gif", ".jpg", ".jpeg", ".png", ".webp")

    def build_run(
        self, *, evidence: Evidence, drafts: Tuple[AuthoringDraft, ...] = ()
    ) -> AuthoringEvidence:
        """
        Build whole-run authoring evidence from existing normalized evidence.
        """

        return AuthoringEvidence(
            run=RunAuthoringEvidence(
                drafts=drafts,
                source=evidence,
                run=self.__run(evidence=evidence),
                steps=self.__steps(evidence=evidence),
                episodes=self.__episodes(evidence=evidence),
                artifacts=self.__artifacts(evidence=evidence),
                assertions=evidence.assertions,
            )
        )

    def build_step(self, *, evidence: Evidence, step_index: int) -> AuthoringEvidence:
        """
        Build single-step authoring evidence from existing normalized evidence.
        """

        return AuthoringEvidence(
            step=StepAuthoringEvidence(
                source=evidence,
                step_index=step_index,
                run=self.__run(evidence=evidence),
                step=self.__selected_step(evidence=evidence, step_index=step_index),
                artifacts=self.__artifacts(evidence=evidence, step_index=step_index),
            )
        )

    def build_repair(
        self,
        *,
        flow: Optional[Flow] = None,
        script: Optional[str] = None,
        review: Optional[Report] = None,
        evidence: Optional[Evidence] = None,
    ) -> AuthoringEvidence:
        """
        Build repair authoring evidence from an existing script, flow, review, and optional run evidence.
        """

        references = self.__artifacts(evidence=evidence) if evidence is not None else ()

        return AuthoringEvidence(
            repair=RepairAuthoringEvidence(
                flow=flow,
                script=script,
                review=review,
                source=evidence,
                artifacts=references,
            )
        )

    @staticmethod
    def __run(*, evidence: Evidence) -> AuthoringRun:
        """
        Build run-level authoring facts.
        """

        return AuthoringRun(
            goal=evidence.goal,
            reason=evidence.reason,
            intent=evidence.intent,
            package=evidence.package,
            partial=evidence.partial,
            discarded=evidence.discarded,
        )

    def __steps(self, *, evidence: Evidence) -> Tuple[AuthoringStep, ...]:
        """
        Build ordered authoring step views.
        """

        return tuple(self.__step(step=step) for step in evidence.steps)

    def __selected_step(self, *, evidence: Evidence, step_index: int) -> AuthoringStep:
        """
        Return the authoring view for a selected evidence step.
        """

        for step in evidence.steps:
            if step.index == step_index:
                return self.__step(step=step)

        raise ValueError(f"Evidence does not contain step {step_index}.")

    def __step(self, *, step: EvidenceStep) -> AuthoringStep:
        """
        Build the authoring-owned view of one execution step.
        """

        return AuthoringStep(
            goal=step.goal,
            text=step.text,
            index=step.index,
            command=AuthoringCommand(
                event=step.event,
                action=step.action,
                success=step.outcome.success,
            ),
            screen=AuthoringScreen(
                changed=step.outcome.changed,
                duration=step.outcome.duration,
            ),
            target=AuthoringTarget(
                name=step.target.name,
                export=step.target.export,
                scroll=step.target.scroll,
                element=step.target.element,
                positional=step.target.positional,
                generalized=step.target.generalized,
            ),
            narrative=AuthoringNarrative(
                reasoning=step.rationale,
                observation=step.observation,
            ),
            capture=self.__capture(step=step),
            validation=self.__validation(step=step),
            artifacts=self.__step_references(step=step),
        )

    @staticmethod
    def __capture(*, step: EvidenceStep) -> Optional[AuthoringCapture]:
        """
        Build authoring capture facts when the step recorded a STORE capture.
        """

        if step.capture is None:
            return None

        return AuthoringCapture(
            name=step.capture.name,
            value=step.capture.value,
            reason=step.capture.reason,
            subject=step.capture.subject,
            success=step.capture.success,
        )

    @staticmethod
    def __validation(*, step: EvidenceStep) -> Optional[AuthoringValidation]:
        """
        Build authoring validation facts when the step recorded a validation subject.
        """

        if step.event != AuthoringEvidenceBuilder.__VALIDATION:
            return None

        subject = step.target.export or step.target.name or step.target.generalized
        if subject is None:
            return None

        return AuthoringValidation(
            subject=subject,
            pattern=step.wait.pattern,
        )

    def __episodes(self, *, evidence: Evidence) -> Tuple[AuthoringEpisode, ...]:
        """
        Group evidence steps by their recorded sub-goal.
        """

        goals: Dict[int, StepGoal] = {}
        steps: Dict[int, List[int]] = {}

        for step in evidence.steps:
            if step.goal is None:
                continue

            goals[step.goal.index] = step.goal
            steps.setdefault(step.goal.index, []).append(step.index)

        return tuple(
            AuthoringEpisode(goal=goals[index], steps=tuple(steps[index]))
            for index in sorted(steps)
        )

    def __artifacts(
        self, *, evidence: Evidence, step_index: Optional[int] = None
    ) -> Tuple[AuthoringArtifactReference, ...]:
        """
        Collect artifact references exposed by the normalized evidence.
        """

        references: List[AuthoringArtifactReference] = []

        for artifact in evidence.artifacts:
            references.append(self.__artifact_reference(uri=artifact))

        for step in evidence.steps:
            if step_index is not None and step.index != step_index:
                continue

            references.extend(self.__step_references(step=step))

        return tuple(references)

    @staticmethod
    def __artifact_reference(*, uri: str) -> AuthoringArtifactReference:
        """
        Classify a run-level artifact reference by its path suffix.
        """

        suffix = uri.lower().rsplit(".", 1)
        extension = f".{suffix[-1]}" if len(suffix) == 2 else ""

        if extension in AuthoringEvidenceBuilder.__IMAGE_SUFFIXES:
            kind = AuthoringArtifactKind.IMAGE
            role = AuthoringArtifactRole.CONTEXT

        elif extension in AuthoringEvidenceBuilder.__MANIFEST_SUFFIXES:
            role = AuthoringArtifactRole.TREE
            kind = AuthoringArtifactKind.MANIFEST

        else:
            role = AuthoringArtifactRole.LOG
            kind = AuthoringArtifactKind.TEXT

        return AuthoringArtifactReference(uri=uri, kind=kind, role=role)

    @staticmethod
    def __has_after_artifact(*, step: EvidenceStep) -> bool:
        """
        Return whether the step already exposes a structured post-action screen artifact.
        """

        return (
            step.artifacts is not None
            and step.artifacts.screen is not None
            and step.artifacts.screen.after is not None
            and step.artifacts.screen.after.uri is not None
        )

    def __step_references(self, *, step: EvidenceStep) -> Tuple[AuthoringArtifactReference, ...]:
        """
        Collect all artifact references attached to one step.
        """

        references: List[AuthoringArtifactReference] = []

        if step.screenshot and not self.__has_after_artifact(step=step):
            references.append(
                AuthoringArtifactReference(
                    uri=step.screenshot,
                    step_index=step.index,
                    kind=AuthoringArtifactKind.IMAGE,
                    role=AuthoringArtifactRole.AFTER,
                )
            )

        references.extend(self.__step_artifacts(step=step))
        return tuple(references)

    def __step_artifacts(self, *, step: EvidenceStep) -> Tuple[AuthoringArtifactReference, ...]:
        """
        Convert structured step artifacts into authoring artifact references.
        """

        if step.artifacts is None or step.artifacts.screen is None:
            return ()

        screen = step.artifacts.screen
        references: List[AuthoringArtifactReference] = []

        if screen.before is not None:
            self.__append_screen(
                step_index=step.index,
                references=references,
                artifact=screen.before,
                role=AuthoringArtifactRole.BEFORE,
            )

        if screen.after is not None:
            self.__append_screen(
                references=references,
                artifact=screen.after,
                step_index=step.index,
                role=AuthoringArtifactRole.AFTER,
            )

        if screen.annotated is not None:
            self.__append_screen(
                references=references,
                step_index=step.index,
                artifact=screen.annotated,
                role=AuthoringArtifactRole.ANNOTATED,
            )

        for trace in screen.traces or ():
            self.__append_screen(
                artifact=trace,
                references=references,
                step_index=step.index,
                kind=AuthoringArtifactKind.TRACE,
                role=AuthoringArtifactRole.TRACE,
            )

        return tuple(references)

    @staticmethod
    def __append_screen(
        *,
        step_index: int,
        artifact: ScreenArtifact,
        role: AuthoringArtifactRole,
        references: List[AuthoringArtifactReference],
        kind: AuthoringArtifactKind = AuthoringArtifactKind.IMAGE,
    ) -> None:
        """
        Append a screen artifact reference when it has a URI.
        """

        if artifact.uri is None:
            return

        references.append(
            AuthoringArtifactReference(
                kind=kind,
                role=role,
                uri=artifact.uri,
                step_index=step_index,
                mime=artifact.mime_type,
            )
        )
