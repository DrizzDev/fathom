from __future__ import annotations

from typing import Dict, Tuple

from pydantic import Field

from fathom.schemas.base.common import SealedModel


class StepTiming(SealedModel):
    """
    Per-step compute breakdown in milliseconds; planner rides inside analyze and vision inside record.
    """

    step: int = Field(description="Zero-based index of the agent step this timing describes.")
    subgoal: int = Field(description="Active sub-goal index during the step, or -1 when none.")

    ground: float = Field(ge=0.0, description="GROUND node wall time in milliseconds.")
    analyze: float = Field(ge=0.0, description="ANALYZE node wall time in milliseconds.")

    planner: float = Field(
        ge=0.0, description="Planner LLM sub-duration within ANALYZE in milliseconds."
    )
    vision: float = Field(
        ge=0.0, description="Completion vision-call sub-duration within RECORD in milliseconds."
    )

    supervise: float = Field(ge=0.0, description="SUPERVISE node wall time in milliseconds.")
    execute: float = Field(ge=0.0, description="EXECUTE device-action wall time in milliseconds.")

    observe: float = Field(ge=0.0, description="OBSERVE post-action wall time in milliseconds.")
    record: float = Field(ge=0.0, description="RECORD and persist wall time in milliseconds.")

    wait: float = Field(
        ge=0.0,
        description="Human wait in milliseconds (ask_user/interrupt), excluded from compute.",
    )
    compute: float = Field(
        ge=0.0, description="Sum of node compute phases in milliseconds, excluding the wait."
    )
    total: float = Field(ge=0.0, description="Compute plus wait for the step in milliseconds.")

    def to_event(self) -> Dict[str, float]:
        """
        Project the step timing to dotted-key structured-log fields.
        """

        return {
            "step.number": self.step,
            "sub_goal.index": self.subgoal,
            "timing.ground": self.ground,
            "timing.analyze": self.analyze,
            "timing.planner": self.planner,
            "timing.vision": self.vision,
            "timing.record": self.record,
            "timing.execute": self.execute,
            "timing.observe": self.observe,
            "timing.supervise": self.supervise,
            "timing.wait": self.wait,
            "timing.compute": self.compute,
            "timing.total": self.total,
        }


class PhaseRollup(SealedModel):
    """
    Aggregate total, per-step mean, and share of agent compute for one timing phase across the run.
    """

    total: float = Field(
        ge=0.0, description="Summed duration of the phase over all steps in milliseconds."
    )
    mean: float = Field(ge=0.0, description="Mean per-step duration of the phase in milliseconds.")
    share: float = Field(ge=0.0, description="Phase total as a percentage of run agent compute.")


class Usage(SealedModel):
    """
    Call count and total duration for one LLM call kind over the run.
    """

    calls: int = Field(ge=0, description="Number of steps that issued this LLM call.")
    duration: float = Field(
        ge=0.0, description="Total time for this LLM call kind over the run in milliseconds."
    )


class RunTimingSummary(SealedModel):
    """
    Run-level rollup: per-phase totals/means, agent compute versus wait, and planner versus vision split.
    """

    steps: int = Field(ge=0, description="Number of committed step timings in the run.")
    wall: float = Field(
        ge=0.0, description="Total step wall time in milliseconds, compute plus wait."
    )
    compute: float = Field(
        ge=0.0, description="Total agent compute time over the run in milliseconds."
    )
    wait: float = Field(ge=0.0, description="Total wait time over the run in milliseconds.")
    planner: Usage = Field(description="Planner LLM call count and total duration over the run.")
    vision: Usage = Field(
        description="Completion vision call count and total duration over the run."
    )
    phases: Dict[str, PhaseRollup] = Field(description="Per-phase rollups keyed by phase name.")

    def to_event(self) -> Dict[str, object]:
        """
        Project the run summary to a structured-log payload.
        """

        return {
            "timing.wall": self.wall,
            "timing.wait": self.wait,
            "timing.steps": self.steps,
            "timing.compute": self.compute,
            "timing.vision.calls": self.vision.calls,
            "timing.planner.calls": self.planner.calls,
            "timing.vision.duration": self.vision.duration,
            "timing.planner.duration": self.planner.duration,
            "timing.phases": {name: rollup.model_dump() for name, rollup in self.phases.items()},
        }


__all__: Tuple[str, ...] = ("StepTiming", "PhaseRollup", "Usage", "RunTimingSummary")
