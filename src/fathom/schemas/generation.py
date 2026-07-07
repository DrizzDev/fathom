from __future__ import annotations

from typing import Optional, Tuple

from pydantic import Field, model_validator

from fathom.constants.flow import LaunchProvenance
from fathom.constants.generation import ScriptSource, ScriptStatus, SkipReason
from fathom.schemas.base import SealedModel
from fathom.schemas.flow import Flow, Issue
from fathom.schemas.steps import StepRecord


class LaunchMarker(SealedModel):
    """
    A deterministically synthesised app launch injected into the normalized trace.
    """

    package: str = Field(min_length=1, description="Real app package to launch.")
    provenance: LaunchProvenance = Field(description="Why the launch was synthesised.")
    source_steps: Tuple[int, ...] = Field(
        default_factory=tuple, description="Collapsed launcher step numbers grounding the launch."
    )


class NormalizedEntry(SealedModel):
    """
    One ordered entry of a normalized trace: either a launch or a kept step record.
    """

    record: Optional[StepRecord] = Field(default=None, description="Step, if this is a step.")
    launch: Optional[LaunchMarker] = Field(default=None, description="Launch, if this is a launch.")

    @model_validator(mode="after")
    def __exactly_one(self) -> "NormalizedEntry":
        """
        Require exactly one of launch or record.
        """

        if (self.launch is None) == (self.record is None):
            raise ValueError("A normalized entry must hold exactly one of launch or record.")

        return self


class NormalizedTrace(SealedModel):
    """
    The ordered, launch-normalized workflow trace fed into evidence assembly.
    """

    entries: Tuple[NormalizedEntry, ...] = Field(
        default_factory=tuple, description="Launches and kept steps in execution order."
    )


class Distillation(SealedModel):
    """
    Outcome of distilling raw step records: the kept records plus what was dropped and why.
    """

    records: Tuple[StepRecord, ...] = Field(description="Records kept for generation, in order.")
    discarded: Tuple[int, ...] = Field(
        default_factory=tuple, description="Step numbers dropped as recovery or loop thrash."
    )
    partial: bool = Field(
        default=False, description="True when no successful goal validation survived distillation."
    )
    reason: Optional[str] = Field(default=None, description="Why the run was marked partial.")


class ScrollCollapseState(SealedModel):
    """
    Tracks the active repeated-command region during recovery distillation.
    """

    command: Optional[str] = Field(
        default=None, description="Command family currently eligible for collapse."
    )
    region: Optional[int] = Field(
        default=None, description="Recovery interval currently being collapsed."
    )

    def repeats(self, *, command: Optional[str], region: Optional[int]) -> bool:
        """
        Return whether the incoming command repeats the active collapse state.
        """

        return (
            command is not None
            and region is not None
            and (self.command == command and self.region == region)
        )

    def advance(self, *, command: Optional[str], region: Optional[int]) -> "ScrollCollapseState":
        """
        Return the next collapse state after a kept command.
        """

        if command is None or region is None:
            return ScrollCollapseState()

        return self.model_copy(update={"command": command, "region": region})


class ScriptLineage(SealedModel):
    """
    Evidence provenance for one rendered script node.
    """

    node_index: int = Field(ge=0, description="Rendered node index in flattened script order.")
    source_steps: Tuple[int, ...] = Field(
        default_factory=tuple, description="Evidence step numbers cited by the node."
    )
    verified_by: Tuple[str, ...] = Field(
        default_factory=tuple, description="Evidence channels that supported the node."
    )
    screen_authored: bool = Field(
        default=False,
        description="Whether the node used an unconfirmed screen-authored planner claim.",
    )


class ScriptCommand(SealedModel):
    """
    Rendered command text paired with the flow node provenance that produced it.
    """

    text: str = Field(min_length=1, description="Rendered command text for one flow node.")
    source_steps: Tuple[int, ...] = Field(
        default_factory=tuple, description="Evidence steps represented by this command."
    )
    verified_by: Tuple[str, ...] = Field(
        default_factory=tuple, description="Evidence channels that verified the command text."
    )
    screen_authored: bool = Field(
        default=False, description="Whether the command text was authored from screen context."
    )
    structural: bool = Field(
        default=False,
        description="Whether the text is syntax structure; reserved for structural renderers.",
    )


class CompletionValidation(SealedModel):
    """
    Rendered terminal validation required to prove completed fallback scripts.
    """

    required: bool = Field(
        default=False, description="Whether fallback composition must include terminal validation."
    )
    lines: Tuple[str, ...] = Field(
        default_factory=tuple, description="Rendered terminal validation lines."
    )
    source_steps: Tuple[int, ...] = Field(
        default_factory=tuple, description="Evidence steps grounding terminal validation lines."
    )
    verified_by: Tuple[str, ...] = Field(
        default=("completion_assertion",),
        description="Evidence channels that verified terminal validation lines.",
    )

    @property
    def missing(self) -> bool:
        """
        Return whether required terminal validation is unavailable.
        """

        return self.required and not self.lines


class ScriptReview(SealedModel):
    """
    Shared review state of a script-generation outcome.
    """

    partial: bool = Field(default=False, description="Whether the script needs review.")
    reason: Optional[str] = Field(default=None, description="Why the run is partial, when it is.")

    discarded: Tuple[int, ...] = Field(
        default_factory=tuple, description="Step numbers dropped during distillation."
    )
    advisories: Tuple[Issue, ...] = Field(
        default_factory=tuple,
        description="Non-blocking authoring quality notes for review and retry guidance.",
    )
    lineage: Tuple[ScriptLineage, ...] = Field(
        default_factory=tuple,
        description="Per-node evidence provenance labels for authored script review.",
    )
    commands: Tuple[ScriptCommand, ...] = Field(
        default_factory=tuple,
        description="Rendered commands with evidence provenance, keyed by source steps.",
    )


class SkippedStep(SealedModel):
    """
    An evidence step the deterministic projector dropped, with the reason it was not scripted.
    """

    index: int = Field(ge=0, description="Evidence step index that was dropped.")
    action: str = Field(min_length=1, description="Recorded action type of the dropped step.")
    reason: SkipReason = Field(description="Why the step was dropped from the baseline flow.")


class ProjectionReport(SealedModel):
    """
    Outcome of deterministic projection: the flow and the evidence steps it could not faithfully render.
    """

    flow: Flow = Field(description="The projected target-neutral flow.")
    skipped: Tuple[SkippedStep, ...] = Field(
        default_factory=tuple, description="Evidence steps dropped during projection, with reasons."
    )


class ScriptFileMetadata(SealedModel):
    """
    Sidecar metadata describing a persisted script file's review state.
    """

    status: ScriptStatus = Field(
        default=ScriptStatus.GENERATED,
        description="Whether a script was produced or generation failed.",
    )
    source: ScriptSource = Field(
        default=ScriptSource.QUALITY, description="Which generation path produced the artifact."
    )
    issues: Tuple[Issue, ...] = Field(
        default_factory=tuple, description="Blocking or review issues recorded for this artifact."
    )
    review: ScriptReview = Field(
        default_factory=ScriptReview, description="Partiality, reason, and dropped steps."
    )
    skipped: Tuple[SkippedStep, ...] = Field(
        default_factory=tuple,
        description="Evidence steps the projector could not faithfully script.",
    )


class BaselineArtifact(SealedModel):
    """
    A produced deterministic baseline script and its sidecar metadata, ready to persist.
    """

    text: Optional[str] = Field(
        default=None,
        description="Rendered baseline script when generated; None when generation failed.",
    )
    metadata: ScriptFileMetadata = Field(
        description="Sidecar metadata: source, status, issues, review, and skipped diagnostics."
    )


class GenerationFailure(SealedModel):
    """
    Outcome of a failed script generation: the blocking issues and review context, never empty success.
    """

    issues: Tuple[Issue, ...] = Field(
        min_length=1, description="Blocking fidelity or syntax issues that prevented a script."
    )
    review: ScriptReview = Field(
        default_factory=ScriptReview, description="Partiality, reason, and dropped steps."
    )


class GenerationResult(SealedModel):
    """
    Outcome of a successful script generation: the rendered, validated script and its review state.
    """

    text: str = Field(min_length=1, description="Rendered, validated script text.")
    attempts: int = Field(ge=1, description="Generation attempts made, including repairs.")
    source: ScriptSource = Field(
        default=ScriptSource.QUALITY, description="Generation path that produced the script."
    )
    review: ScriptReview = Field(
        default_factory=ScriptReview, description="Partiality, reason, and dropped steps."
    )
