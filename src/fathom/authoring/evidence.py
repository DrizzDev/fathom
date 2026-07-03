from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from fathom.constants.authoring import AuthoringArtifactKind, AuthoringArtifactRole
from fathom.schemas.artifacts import ScreenArtifact
from fathom.schemas.authoring import (
    AuthoringArtifactReference,
    AuthoringEpisode,
    AuthoringEvidence,
    RepairAuthoringEvidence,
    RunAuthoringEvidence,
    StepAuthoringEvidence,
)
from fathom.schemas.flow import Evidence, EvidenceStep, Flow, Report
from fathom.schemas.steps import StepGoal


class AuthoringEvidenceBuilder:
    """
    Builds authoring task evidence views from normalized execution evidence.
    """

    def build_run(self, *, evidence: Evidence) -> AuthoringEvidence:
        """
        Build whole-run authoring evidence from existing normalized evidence.
        """

        return AuthoringEvidence(
            run=RunAuthoringEvidence(
                evidence=evidence,
                episodes=self.__episodes(evidence=evidence),
                artifacts=self.__artifacts(evidence=evidence),
            )
        )

    def build_step(self, *, evidence: Evidence, step_index: int) -> AuthoringEvidence:
        """
        Build single-step authoring evidence from existing normalized evidence.
        """

        return AuthoringEvidence(
            step=StepAuthoringEvidence(
                evidence=evidence,
                step_index=step_index,
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
                evidence=evidence,
                artifacts=references,
            )
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
            references.append(
                AuthoringArtifactReference(
                    uri=artifact,
                    kind=AuthoringArtifactKind.TEXT,
                    role=AuthoringArtifactRole.OTHER,
                )
            )

        for step in evidence.steps:
            if step_index is not None and step.index != step_index:
                continue

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
                role=AuthoringArtifactRole.OTHER,
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
