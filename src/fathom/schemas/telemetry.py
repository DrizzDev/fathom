from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IntentMessage(BaseModel):
    """
    Client-facing strings for intent-lifecycle phases.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    qualifying: str = Field(
        default="Reading your request...",
        description="Shown while the qualifier evaluates whether the intent is executable.",
    )
    decomposing: str = Field(
        default="Breaking this into steps...",
        description="Shown while the decomposer derives sub-goals from the intent.",
    )
    derived: str = Field(
        default="Got it - here is the plan",
        description="Shown once the decomposer has emitted its sub-goal list.",
    )


class StepMessage(BaseModel):
    """
    Client-facing strings for per-step lifecycle phases (one per turn).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    grounding: str = Field(
        default="Looking at the screen...",
        description="Shown while the agent captures and grounds the screen for the next step.",
    )


class HeartbeatBudget(BaseModel):
    """
    Threshold beyond which a phase emits PHASE_HEARTBEAT events, and the
    maximum number of beats one phase may emit before the pulse stops on its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float = Field(
        ge=0.5,
        le=60.0,
        default=4.0,
        description=(
            "Seconds a phase may run before the client receives a heartbeat. "
            "Heartbeats keep the client aware that work is still in progress."
        ),
    )
    limit: int = Field(
        ge=1,
        le=600,
        default=60,
        description=(
            "Maximum number of heartbeats one phase may emit before the pulse "
            "stops on its own. Bounds the background loop so it can never run "
            "forever even if the phase forgets to close it."
        ),
    )
    message: str = Field(
        default="Still working...",
        description="Heartbeat message rendered by the client for long-running phases.",
    )
    script_finalization: str = Field(
        default="Finalizing the script...",
        description="Heartbeat message rendered while a cancelled run finalizes its partial script.",
    )


class PhaseMessage(BaseModel):
    """
    Aggregate of client-facing phase strings sourced from configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: StepMessage = Field(default_factory=StepMessage)
    intent: IntentMessage = Field(default_factory=IntentMessage)
    heartbeat: HeartbeatBudget = Field(default_factory=HeartbeatBudget)
