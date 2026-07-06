from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.constants.generation import ScriptSource
from fathom.schemas.authoring import AuthoringBaseline
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
    ) -> Optional[GenerationResult]:
        """
        Return a script from generated step drafts, filling missing steps from baseline lines.
        """

        lines: List[str] = []
        indexed = self.__indexed_drafts(drafts=drafts)
        baseline_lines = self.__baseline_lines(baseline=baseline)

        for position, step in enumerate(evidence.steps):
            draft = indexed.get(step.index)
            if draft is not None and draft.artifact is not None:
                lines.extend(self.__lines(content=draft.artifact.content))
                continue

            if position < len(baseline_lines):
                lines.append(baseline_lines[position])

        text = "\n".join(lines).strip()
        if not text:
            return None

        return GenerationResult(
            text=text,
            attempts=1,
            source=ScriptSource.STEP_DRAFTS,
            review=ScriptReview(
                reason=evidence.reason,
                partial=evidence.partial,
                discarded=evidence.discarded,
            ),
        )

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

    @classmethod
    def __baseline_lines(cls, *, baseline: Optional[AuthoringBaseline]) -> Tuple[str, ...]:
        """
        Return non-empty baseline lines used when a step draft is missing.
        """

        if baseline is None:
            return ()

        return cls.__lines(content=baseline.content)
