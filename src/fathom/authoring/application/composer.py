from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.constants.generation import ScriptSource
from fathom.schemas.authoring import AuthoringBaseline, AuthoringBaselineCommand
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.flow import Evidence
from fathom.schemas.generation import GenerationResult, ScriptReview


class StepDraftComposer:
    """
    Composes reviewed step authoring drafts into an execution-ordered script fallback.
    """

    def compose(
        self,
        *,
        evidence: Evidence,
        drafts: Tuple[AuthoringDraft, ...],
        baseline: Optional[AuthoringBaseline] = None,
        completion_validation: Tuple[str, ...] = (),
        require_completion_validation: bool = False,
    ) -> Optional[GenerationResult]:
        """
        Return a script from generated step drafts, filling missing steps from baseline lines.
        """

        indexed = self.__indexed_drafts(drafts=drafts)
        lines = self.__composed_lines(
            evidence=evidence,
            drafts=indexed,
            baseline=baseline,
        )
        if completion_validation:
            lines = (*lines, *completion_validation)

        text = "\n".join(lines)
        if not text:
            return None

        reason = evidence.reason
        partial = evidence.partial or (require_completion_validation and not completion_validation)

        if require_completion_validation and not completion_validation:
            reason = (
                reason or "Completion assertions could not be rendered into a terminal validation."
            )

        return GenerationResult(
            text=text,
            attempts=1,
            source=ScriptSource.STEP_DRAFTS,
            review=ScriptReview(
                reason=reason,
                partial=partial,
                discarded=evidence.discarded,
            ),
        )

    def __composed_lines(
        self,
        *,
        evidence: Evidence,
        drafts: Dict[int, AuthoringDraft],
        baseline: Optional[AuthoringBaseline],
    ) -> Tuple[str, ...]:
        """
        Compose lines by evidence identity, preserving merged baseline commands.
        """

        if baseline is None or not baseline.commands:
            return self.__draft_lines(evidence=evidence, drafts=drafts, covered=set())

        lines: List[str] = []
        covered: Set[int] = set()

        for command in baseline.commands:
            source_steps = tuple(step for step in command.source_steps if step not in covered)
            if not source_steps:
                continue

            if self.__merged(command=command):
                lines.append(command.text)
                covered.update(command.source_steps)
                continue

            step = source_steps[0]
            draft = drafts.get(step)

            if draft is not None and draft.artifact is not None:
                lines.extend(self.__lines(content=draft.artifact.content))
            else:
                lines.append(command.text)

            covered.add(step)

        lines.extend(self.__draft_lines(evidence=evidence, drafts=drafts, covered=covered))
        return tuple(lines)

    @classmethod
    def __indexed_drafts(cls, *, drafts: Tuple[AuthoringDraft, ...]) -> Dict[int, AuthoringDraft]:
        """
        Return latest generated step drafts keyed by evidence step index.
        """

        indexed: Dict[int, AuthoringDraft] = {}

        for draft in drafts:
            if not cls.__usable(draft=draft):
                continue

            if draft.step_index is not None:
                indexed[draft.step_index] = draft

        return indexed

    @staticmethod
    def __usable(*, draft: AuthoringDraft) -> bool:
        """
        Return whether a draft can contribute rendered script text.
        """

        return (
            draft.kind is AuthoringKind.STEP
            and draft.artifact is not None
            and draft.step_index is not None
            and bool(draft.artifact.content.strip())
            and draft.status is AuthoringStatus.GENERATED
        )

    @staticmethod
    def __lines(*, content: str) -> Tuple[str, ...]:
        """
        Return non-empty lines from one draft artifact.
        """

        return tuple(line.strip() for line in content.splitlines() if line.strip())

    def __draft_lines(
        self,
        *,
        covered: Set[int],
        evidence: Evidence,
        drafts: Dict[int, AuthoringDraft],
    ) -> Tuple[str, ...]:
        """
        Return draft lines for evidence steps not already represented.
        """

        lines: List[str] = []

        for step in evidence.steps:
            if step.index in covered:
                continue

            draft = drafts.get(step.index)
            if draft is not None and draft.artifact is not None:
                lines.extend(self.__lines(content=draft.artifact.content))

        return tuple(lines)

    @staticmethod
    def __merged(*, command: AuthoringBaselineCommand) -> bool:
        """
        Return whether one baseline command represents multiple evidence steps.
        """

        return len(command.source_steps) > 1
