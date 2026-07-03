from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import Field

from fathom.constants.flow import LaunchProvenance
from fathom.schemas.base import SealedModel
from fathom.schemas.flow import Evidence, EvidenceStep, StepLaunch, StepTarget
from fathom.schemas.steps import StepGoal


class PromptTarget(SealedModel):
    """
    Compact UI target view exposed to an authoring prompt.
    """

    export: Optional[str] = Field(default=None, description="Canonical target phrase.")
    name: Optional[str] = Field(default=None, description="Raw on-screen target label.")
    generalized: Optional[str] = Field(default=None, description="Variable target descriptor.")
    scroll: Optional[str] = Field(default=None, description="Target the step scrolled toward.")
    element: Optional[str] = Field(default=None, description="Recorded UI role for the target.")
    positional: Optional[bool] = Field(default=None, description="True when order disambiguates.")

    @classmethod
    def from_target(cls, *, target: StepTarget) -> Optional["PromptTarget"]:
        """
        Build a target packet only when the recorded step has target data.
        """

        packet = cls(
            name=target.name,
            export=target.export,
            scroll=target.scroll,
            element=target.element,
            generalized=target.generalized,
            positional=True if target.positional else None,
        )

        if packet.model_dump(exclude_none=True):
            return packet

        return None


class PromptWait(SealedModel):
    """
    Compact wait data exposed to an authoring prompt.
    """

    subject: Optional[str] = Field(default=None, description="Recorded wait subject.")
    pattern: Optional[str] = Field(default=None, description="Recorded wait category.")


class PromptGuard(SealedModel):
    """
    Compact conditional guard exposed to an authoring prompt.
    """

    conditional: bool = Field(description="Whether the step ran under a condition.")
    condition: Optional[str] = Field(default=None, description="Recorded condition text.")
    overlay: Optional[bool] = Field(default=None, description="True when handling an overlay.")


class PromptCapture(SealedModel):
    """
    Compact capture data exposed to an authoring prompt.
    """

    success: bool = Field(description="Whether capture succeeded.")
    name: str = Field(min_length=1, description="Variable name requested by capture.")
    subject: str = Field(min_length=1, description="What the intent requested to capture.")
    value: Optional[str] = Field(default=None, description="Captured value when successful.")
    reason: Optional[str] = Field(default=None, description="Failure reason when capture failed.")


class PromptLaunch(SealedModel):
    """
    Compact launch marker exposed to an authoring prompt.
    """

    package: str = Field(min_length=1, description="Package the script must launch.")
    provenance: LaunchProvenance = Field(description="How the launch marker was derived.")
    source_steps: Tuple[int, ...] = Field(
        default_factory=tuple, description="Step numbers grounding the launch marker."
    )

    @classmethod
    def from_launch(cls, *, launch: StepLaunch) -> "PromptLaunch":
        """
        Build the prompt view for one deterministic launch marker.
        """

        return cls(
            package=launch.package,
            provenance=launch.provenance,
            source_steps=launch.source_steps,
        )


class PromptStep(SealedModel):
    """
    Compact execution step exposed to an authoring prompt.
    """

    step_id: int = Field(ge=0, description="Recorded step index.")
    event: str = Field(min_length=1, description="Recorded event category.")
    action: str = Field(min_length=1, description="Recorded action type.")
    success: bool = Field(description="Whether execution reported success.")

    target: Optional[PromptTarget] = Field(default=None, description="Recorded target data.")
    typed_text: Optional[str] = Field(default=None, description="Text typed by this step.")

    wait: Optional[PromptWait] = Field(default=None, description="Recorded wait data.")
    guard: Optional[PromptGuard] = Field(default=None, description="Recorded condition data.")
    capture: Optional[PromptCapture] = Field(default=None, description="Recorded capture data.")

    rationale: Optional[str] = Field(default=None, description="Planner rationale.")
    observation: Optional[str] = Field(default=None, description="Planner observation.")
    launch: Optional[PromptLaunch] = Field(default=None, description="Recorded launch marker.")

    @classmethod
    def from_step(cls, *, step: EvidenceStep) -> "PromptStep":
        """
        Build the compact prompt view for one evidence step.
        """

        return cls(
            wait=cls.__wait(step=step),
            guard=cls.__guard(step=step),
            launch=PromptLaunch.from_launch(launch=step.launch)
            if step.launch is not None
            else None,
            capture=cls.__capture(step=step),
            event=step.event,
            step_id=step.index,
            action=step.action,
            typed_text=step.text,
            rationale=step.rationale,
            success=step.outcome.success,
            observation=step.observation,
            target=PromptTarget.from_target(target=step.target),
        )

    @staticmethod
    def __wait(*, step: EvidenceStep) -> Optional[PromptWait]:
        """
        Return wait data only when the step recorded it.
        """

        if not step.wait.subject and not step.wait.pattern:
            return None

        return PromptWait(subject=step.wait.subject, pattern=step.wait.pattern)

    @staticmethod
    def __guard(*, step: EvidenceStep) -> Optional[PromptGuard]:
        """
        Return conditional guard data only when the step recorded it.
        """

        if not step.guard.conditional and not step.guard.condition and not step.guard.overlay:
            return None

        return PromptGuard(
            condition=step.guard.condition,
            conditional=step.guard.conditional,
            overlay=True if step.guard.overlay else None,
        )

    @staticmethod
    def __capture(*, step: EvidenceStep) -> Optional[PromptCapture]:
        """
        Return capture data only when the step recorded it.
        """

        if step.capture is None:
            return None

        return PromptCapture(
            name=step.capture.name,
            value=step.capture.value,
            reason=step.capture.reason,
            subject=step.capture.subject,
            success=step.capture.success,
        )


class PromptGoal(SealedModel):
    """
    Compact sub-goal context for an authoring episode.
    """

    index: int = Field(ge=0, description="Sub-goal index that owns the episode.")
    directive: Optional[str] = Field(default=None, description="Expected action type.")
    description: str = Field(min_length=1, description="Sub-goal intent for this episode.")

    @classmethod
    def from_goal(cls, *, goal: StepGoal) -> "PromptGoal":
        """
        Build prompt goal context from persisted step goal context.
        """

        return cls(index=goal.index, description=goal.description, directive=goal.directive)


class PromptEpisode(SealedModel):
    """
    Contiguous recorded steps that belong to one sub-goal context.
    """

    goal: Optional[PromptGoal] = Field(
        default=None, description="Sub-goal context shared by the episode's steps."
    )
    steps: Tuple[PromptStep, ...] = Field(min_length=1, description="Steps in the episode.")


class PromptRun(SealedModel):
    """
    Compact run-level context exposed to an authoring prompt.
    """

    __VALIDATION = "validation"

    intent: str = Field(description="User intent for the run.")
    goal: str = Field(description="Goal-state description for the run.")

    package: str = Field(description="Target application package.")
    partial: bool = Field(description="Whether the run is partial.")
    partial_reason: Optional[str] = Field(default=None, description="Why the run is partial.")

    discarded_steps: Optional[Tuple[int, ...]] = Field(
        default=None, description="Step numbers discarded before authoring."
    )
    launches: Optional[Tuple[PromptLaunch, ...]] = Field(
        default=None, description="Launch markers present in the run."
    )
    successful_validations: Optional[Tuple[int, ...]] = Field(
        default=None, description="Successful validation step ids."
    )
    captures: Optional[Tuple[PromptCapture, ...]] = Field(
        default=None, description="Captures recorded by capture steps."
    )
    final_observation: Optional[str] = Field(
        default=None, description="Last recorded observation in the run."
    )

    @classmethod
    def from_evidence(cls, *, evidence: Evidence) -> "PromptRun":
        """
        Build compact run-level context from full evidence.
        """

        launches = tuple(
            PromptLaunch.from_launch(launch=step.launch)
            for step in evidence.steps
            if step.launch is not None
        )
        validations = tuple(
            step.index
            for step in evidence.steps
            if step.event == cls.__VALIDATION and step.outcome.success
        )
        captures = tuple(
            PromptCapture(
                name=step.capture.name,
                value=step.capture.value,
                reason=step.capture.reason,
                subject=step.capture.subject,
                success=step.capture.success,
            )
            for step in evidence.steps
            if step.capture is not None
        )
        final = next(
            (step.observation for step in reversed(evidence.steps) if step.observation), None
        )

        return cls(
            goal=evidence.goal,
            intent=evidence.intent,
            final_observation=final,
            package=evidence.package,
            partial=evidence.partial,
            launches=launches or None,
            captures=captures or None,
            partial_reason=evidence.reason,
            discarded_steps=evidence.discarded or None,
            successful_validations=validations or None,
        )


class PromptEvidence(SealedModel):
    """
    Compact, typed evidence packet sent to an authoring prompt.
    """

    run: PromptRun = Field(description="Run-level authoring context.")
    episodes: Tuple[PromptEpisode, ...] = Field(description="Compact execution episodes.")

    @classmethod
    def from_evidence(cls, *, evidence: Evidence) -> "PromptEvidence":
        """
        Build the prompt packet from full authoring evidence.
        """

        return cls(
            episodes=cls.__episodes(steps=evidence.steps),
            run=PromptRun.from_evidence(evidence=evidence),
        )

    @classmethod
    def __episodes(cls, *, steps: Tuple[EvidenceStep, ...]) -> Tuple[PromptEpisode, ...]:
        """
        Group contiguous evidence steps by persisted sub-goal context.
        """

        episodes: List[PromptEpisode] = []
        current_steps: List[PromptStep] = []
        current_goal: Optional[StepGoal] = None

        for step in steps:
            if current_steps and not cls.__same_goal(left=current_goal, right=step.goal):
                episodes.append(cls.__episode(goal=current_goal, steps=current_steps))
                current_steps = []

            current_goal = step.goal
            current_steps.append(PromptStep.from_step(step=step))

        if current_steps:
            episodes.append(cls.__episode(goal=current_goal, steps=current_steps))

        return tuple(episodes)

    @staticmethod
    def __same_goal(*, left: Optional[StepGoal], right: Optional[StepGoal]) -> bool:
        """
        Return whether two persisted goal contexts describe the same episode.
        """

        if left is None or right is None:
            return False

        return left.index == right.index

    @staticmethod
    def __episode(*, goal: Optional[StepGoal], steps: List[PromptStep]) -> PromptEpisode:
        """
        Build one prompt episode from accumulated steps.
        """

        return PromptEpisode(
            steps=tuple(steps),
            goal=PromptGoal.from_goal(goal=goal) if goal is not None else None,
        )
