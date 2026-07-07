from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from fathom.constants import StepEvent
from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.constants.generation import ScriptCommandRole, ScriptSource
from fathom.schemas.authoring import AuthoringBaseline, AuthoringBaselineCommand
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.flow import Evidence, EvidenceStep, Issue
from fathom.schemas.generation import (
    CompletionValidation,
    GenerationResult,
    ScriptCommand,
    ScriptLineage,
    ScriptReview,
)


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
        completion: Optional[CompletionValidation] = None,
    ) -> Optional[GenerationResult]:
        """
        Return a script from generated step drafts, filling missing steps from baseline lines.
        """

        indexed = self.__indexed_drafts(drafts=drafts)
        completion = completion or CompletionValidation()

        commands = self.__commands(drafts=indexed, evidence=evidence, baseline=baseline)
        commands = self.__with_completion(
            commands=commands, completion=completion, evidence=evidence
        )
        missing = self.__missing_steps(commands=commands, evidence=evidence)

        lines = self.__lines(commands=commands)

        text = "\n".join(lines)
        if not text:
            return None

        reason = evidence.reason
        partial = evidence.partial or completion.missing or bool(missing)

        if completion.missing:
            reason = (
                reason or "Completion assertions could not be rendered into a terminal validation."
            )
        elif missing:
            reason = reason or (
                "Some executed steps were omitted because no grounded script command was available."
            )

        return GenerationResult(
            text=text,
            attempts=1,
            source=ScriptSource.STEP_DRAFTS,
            review=ScriptReview(
                reason=reason,
                partial=partial,
                commands=commands,
                discarded=evidence.discarded,
                lineage=self.__lineage(commands=commands),
                advisories=self.__advisories(commands=commands, drafts=indexed),
            ),
        )

    @staticmethod
    def __lines(*, commands: Tuple[ScriptCommand, ...]) -> Tuple[str, ...]:
        """
        Return rendered script lines from composed command metadata.
        """

        return tuple(command.text for command in commands if not command.structural)

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
    def __content_lines(*, content: str) -> Tuple[str, ...]:
        """
        Return non-empty lines from one draft artifact.
        """

        return tuple(line.strip() for line in content.splitlines() if line.strip())

    def __commands(
        self,
        *,
        evidence: Evidence,
        drafts: Dict[int, AuthoringDraft],
        baseline: Optional[AuthoringBaseline],
    ) -> Tuple[ScriptCommand, ...]:
        """
        Return command provenance for the composed draft artifact.
        """

        covered: Set[int] = set()
        commands: List[ScriptCommand] = []

        if baseline is not None:
            for command in baseline.commands:
                if command.structural:
                    continue

                source_steps = tuple(step for step in command.source_steps if step not in covered)
                if not source_steps and command.role is not ScriptCommandRole.LAUNCH:
                    continue

                if len(source_steps) != len(command.source_steps):
                    continue

                if self.__stable(command=command):
                    commands.append(self.__baseline_command(command=command))
                    covered.update(command.source_steps)
                    continue

                step_index = source_steps[0]
                draft = drafts.get(step_index)

                if draft is not None and draft.artifact is not None:
                    commands.extend(self.__draft_commands(draft=draft))
                else:
                    commands.append(self.__baseline_command(command=command))

                covered.add(step_index)

        for step in evidence.steps:
            if step.launch is not None:
                continue

            if step.index in covered:
                continue

            if self.__guarded_step(step=step):
                continue

            draft = drafts.get(step.index)
            if draft is not None and draft.artifact is not None:
                commands.extend(self.__draft_commands(draft=draft))

        return tuple(commands)

    @staticmethod
    def __guarded_step(*, step: EvidenceStep) -> bool:
        """
        Return whether a step needs IF structure to remain replayable.
        """

        return step.guard.conditional

    def __with_completion(
        self,
        *,
        evidence: Evidence,
        completion: CompletionValidation,
        commands: Tuple[ScriptCommand, ...],
    ) -> Tuple[ScriptCommand, ...]:
        """
        Return commands with required terminal validation added or provenance-upgraded once.
        """

        if not completion.lines:
            return commands

        commands = self.__without_redundant_validation(commands=commands, evidence=evidence)

        matched: Set[str] = set()
        updated: List[ScriptCommand] = []

        for command in commands:
            if command.text not in completion.lines:
                updated.append(command)
                continue

            matched.add(command.text)
            updated.append(
                command.model_copy(
                    update={
                        "source_steps": self.__merged_steps(
                            first=command.source_steps,
                            second=completion.source_steps,
                        ),
                        "verified_by": self.__merged_labels(
                            first=command.verified_by,
                            second=completion.verified_by,
                        ),
                    }
                )
            )

        for validation in completion.lines:
            if validation in matched:
                continue

            updated.append(
                ScriptCommand(
                    text=validation,
                    verified_by=completion.verified_by,
                    source_steps=completion.source_steps,
                )
            )

        return tuple(updated)

    @staticmethod
    def __without_redundant_validation(
        *, commands: Tuple[ScriptCommand, ...], evidence: Evidence
    ) -> Tuple[ScriptCommand, ...]:
        """
        Drop runtime terminal-state validations replaced by completion assertions.
        """

        changed = tuple(step.index for step in evidence.steps if step.outcome.changed)
        if not changed:
            return commands

        terminal_validations = {
            step.index
            for step in evidence.steps
            if step.index > changed[-1] and step.event == StepEvent.VALIDATION
        }
        if not terminal_validations:
            return commands

        return tuple(
            command
            for command in commands
            if not (
                command.source_steps and set(command.source_steps).issubset(terminal_validations)
            )
        )

    @staticmethod
    def __merged_steps(*, first: Tuple[int, ...], second: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Return source steps merged without duplicates while preserving order.
        """

        merged: List[int] = []

        for value in (*first, *second):
            if value not in merged:
                merged.append(value)

        return tuple(merged)

    @staticmethod
    def __merged_labels(*, first: Tuple[str, ...], second: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Return provenance labels merged without duplicates while preserving order.
        """

        merged: List[str] = []

        for value in (*first, *second):
            if value not in merged:
                merged.append(value)

        return tuple(merged)

    def __draft_commands(self, *, draft: AuthoringDraft) -> Tuple[ScriptCommand, ...]:
        """
        Return command metadata for one draft, falling back to its step index when needed.
        """

        if draft.artifact is None:
            return ()

        if draft.artifact.commands:
            lines = tuple(command.text for command in draft.artifact.commands)
        else:
            lines = self.__content_lines(content=draft.artifact.content)

        commands: List[ScriptCommand] = []
        source_steps = (draft.step_index,) if draft.step_index is not None else ()

        for index, line in enumerate(lines):
            commands.append(
                ScriptCommand(
                    text=line,
                    screen_authored=True,
                    source_steps=source_steps if index == 0 else (),
                )
            )

        return tuple(commands)

    @staticmethod
    def __baseline_command(*, command: AuthoringBaselineCommand) -> ScriptCommand:
        """
        Return evidence-verified command metadata for one deterministic baseline command.
        """

        return ScriptCommand(
            text=command.text,
            role=command.role,
            structural=command.structural,
            source_steps=command.source_steps,
            verified_by=("execution",) if command.source_steps else (),
        )

    @staticmethod
    def __lineage(*, commands: Tuple[ScriptCommand, ...]) -> Tuple[ScriptLineage, ...]:
        """
        Return lineage for the composed commands in output order.
        """

        lineage: List[ScriptLineage] = []

        for index, command in enumerate(commands):
            lineage.append(
                ScriptLineage(
                    node_index=index,
                    verified_by=command.verified_by,
                    source_steps=command.source_steps,
                    screen_authored=command.screen_authored,
                )
            )

        return tuple(lineage)

    @staticmethod
    def __advisories(
        *,
        drafts: Dict[int, AuthoringDraft],
        commands: Tuple[ScriptCommand, ...],
    ) -> Tuple[Issue, ...]:
        """
        Return advisories from drafts represented in the composed artifact.
        """

        advisories: List[Issue] = []
        selected = {step for command in commands for step in command.source_steps if step in drafts}

        for step in sorted(selected):
            draft = drafts[step]
            if draft.artifact is not None:
                advisories.extend(draft.artifact.advisories)

        return tuple(advisories)

    @staticmethod
    def __missing_steps(
        *, commands: Tuple[ScriptCommand, ...], evidence: Evidence
    ) -> Tuple[int, ...]:
        """
        Return non-launch evidence steps not represented by the composed commands.
        """

        represented = {step for command in commands for step in command.source_steps}
        return tuple(
            step.index
            for step in evidence.steps
            if step.launch is None and step.index not in represented
        )

    @staticmethod
    def __stable(*, command: AuthoringBaselineCommand) -> bool:
        """
        Return whether a baseline command must not be replaced by a step draft.
        """

        return (
            command.role is ScriptCommandRole.LAUNCH
            or command.role is ScriptCommandRole.BRANCH
            or len(command.source_steps) > 1
        )
