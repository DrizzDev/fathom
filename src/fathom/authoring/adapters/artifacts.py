from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from fathom.constants.authoring import AuthoringArtifactKind, AuthoringArtifactRole
from fathom.interfaces.authoring import AuthoringArtifactProvider
from fathom.interfaces.llm import PromptPart
from fathom.schemas.authoring import AuthoringArtifactReference, AuthoringTask
from fathom.schemas.authoring.configuration import AuthoringArtifactConfiguration
from fathom.schemas.authoring.evidence import AuthoringStep, RunAuthoringEvidence


class FileAuthoringArtifactProvider(AuthoringArtifactProvider):
    """
    Resolves local authoring artifact references into bounded model prompt parts.
    """

    def __init__(self, *, configuration: AuthoringArtifactConfiguration) -> None:
        """
        Bind artifact payload limits for model requests.
        """

        self.__configuration = configuration

    def build(self, *, task: AuthoringTask) -> Tuple[PromptPart, ...]:
        """
        Return prompt parts containing selected artifact payloads for an authoring task.
        """

        text_count = 0
        image_count = 0
        parts: List[PromptPart] = []

        for reference in self.__references(task=task):
            path = self.__path(reference=reference)
            if path is None or not path.exists() or not path.is_file():
                continue

            if self.__image(reference=reference):
                if image_count >= self.__configuration.max_images:
                    continue

                payload = path.read_bytes()
                if not payload:
                    continue

                parts.append({"text": self.__label(reference=reference)})
                parts.append(payload)
                image_count += 1
                continue

            if not self.__text(reference=reference):
                continue

            if text_count >= self.__configuration.max_text_artifacts:
                continue

            content = self.__read_text(path=path)
            if not content:
                continue

            parts.append({"text": self.__text_part(reference=reference, content=content)})
            text_count += 1

        return tuple(parts)

    def __references(self, *, task: AuthoringTask) -> Tuple[AuthoringArtifactReference, ...]:
        """
        Return artifact references for the task-specific evidence view.
        """

        if task.evidence.step is not None:
            return task.evidence.step.artifacts

        if task.evidence.run is not None:
            return self.__run_references(evidence=task.evidence.run)

        if task.evidence.repair is not None:
            return task.evidence.repair.artifacts

        return ()

    def __run_references(
        self, *, evidence: RunAuthoringEvidence
    ) -> Tuple[AuthoringArtifactReference, ...]:
        """
        Prioritize bounded run artifacts toward screens that explain authored commands.
        """

        step_priority = {
            step.index: priority
            for priority, step in enumerate(self.__run_steps_by_relevance(evidence=evidence))
        }

        return tuple(
            sorted(
                evidence.artifacts,
                key=lambda reference: self.__run_reference_priority(
                    reference=reference,
                    step_priority=step_priority,
                ),
            )
        )

    def __run_steps_by_relevance(
        self, *, evidence: RunAuthoringEvidence
    ) -> Tuple[AuthoringStep, ...]:
        """
        Return run steps ordered by how useful their screens are for final authoring.
        """

        remaining: Dict[int, AuthoringStep] = {step.index: step for step in evidence.steps}
        ordered: List[AuthoringStep] = []

        for selector in (
            self.__capture_step,
            self.__validation_step,
            self.__unconfirmed_target_step,
        ):
            for step in evidence.steps:
                if step.index in remaining and selector(step=step):
                    ordered.append(remaining.pop(step.index))

        if evidence.steps:
            terminal = evidence.steps[-1]
            if terminal.index in remaining:
                ordered.append(remaining.pop(terminal.index))

        ordered.extend(step for step in evidence.steps if step.index in remaining)
        return tuple(ordered)

    @staticmethod
    def __capture_step(*, step: AuthoringStep) -> bool:
        """
        Return whether the step carries STORE capture evidence.
        """

        return step.capture is not None

    @staticmethod
    def __validation_step(*, step: AuthoringStep) -> bool:
        """
        Return whether the step carries validation evidence.
        """

        return step.validation is not None

    @staticmethod
    def __unconfirmed_target_step(*, step: AuthoringStep) -> bool:
        """
        Return whether the step has a planner target claim not confirmed by evidence channels.
        """

        return step.target.claim.text is not None and not step.target.claim.verified

    def __run_reference_priority(
        self,
        *,
        reference: AuthoringArtifactReference,
        step_priority: Dict[int, int],
    ) -> Tuple[int, int, int, str]:
        """
        Sort run references by evidence relevance, screen role, then stable URI.
        """

        step_rank = self.__step_rank(reference=reference, step_priority=step_priority)
        return (
            0 if self.__image(reference=reference) else 1,
            step_rank,
            self.__role_rank(reference=reference),
            reference.uri,
        )

    @staticmethod
    def __step_rank(
        *,
        reference: AuthoringArtifactReference,
        step_priority: Dict[int, int],
    ) -> int:
        """
        Return the relevance rank for the reference's step, or the end rank for run artifacts.
        """

        if reference.step_index is None:
            return len(step_priority)

        return step_priority.get(reference.step_index, len(step_priority))

    @staticmethod
    def __role_rank(*, reference: AuthoringArtifactReference) -> int:
        """
        Prefer annotated and post-action screenshots over pre-action screens.
        """

        ranks: Dict[AuthoringArtifactRole, int] = {
            AuthoringArtifactRole.ANNOTATED: 0,
            AuthoringArtifactRole.AFTER: 1,
            AuthoringArtifactRole.BEFORE: 2,
            AuthoringArtifactRole.TRACE: 3,
        }
        return ranks.get(reference.role, 4)

    @staticmethod
    def __path(*, reference: AuthoringArtifactReference) -> Optional[Path]:
        """
        Resolve a local artifact URI to a path when the artifact is locally readable.
        """

        parsed = urlparse(reference.uri)
        if parsed.scheme and parsed.scheme != "file":
            return None

        raw = unquote(parsed.path if parsed.scheme == "file" else reference.uri)
        if not raw:
            return None

        return Path(raw)

    def __image(self, *, reference: AuthoringArtifactReference) -> bool:
        """
        Return whether the reference should be attached as image bytes.
        """

        return self.__configuration.include_images and reference.kind in (
            AuthoringArtifactKind.IMAGE,
            AuthoringArtifactKind.TRACE,
        )

    def __text(self, *, reference: AuthoringArtifactReference) -> bool:
        """
        Return whether the reference should be embedded as text.
        """

        if reference.kind is AuthoringArtifactKind.MANIFEST:
            return self.__configuration.include_manifests

        return self.__configuration.include_text and reference.kind is AuthoringArtifactKind.TEXT

    def __read_text(self, *, path: Path) -> str:
        """
        Read a bounded text artifact payload.
        """

        limit = self.__configuration.max_text_characters
        if limit == 0:
            return ""

        return path.read_text(encoding="utf-8", errors="replace")[:limit]

    @staticmethod
    def __label(*, reference: AuthoringArtifactReference) -> str:
        """
        Return a short label inserted before a binary artifact part.
        """

        step = reference.step_index if reference.step_index is not None else "run"
        return f"Artifact kind={reference.kind.value} role={reference.role.value} step={step}"

    def __text_part(self, *, reference: AuthoringArtifactReference, content: str) -> str:
        """
        Return a bounded text artifact prompt part with source metadata.
        """

        truncated = len(content) >= self.__configuration.max_text_characters
        suffix = "\n[truncated]" if truncated else ""

        return "\n".join((self.__label(reference=reference), "```", content, f"```{suffix}"))
