from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StuckSource(StrEnum):
    """
    Canonical sources of an escalation-worthy "stuck" signal.

    SUBGOAL_BUDGET fires when the active sub-goal exhausts its per-sub-goal
    step budget. LOOP_DETECTOR fires when the screen-level loop detector
    classifies the recent window as stuck.

    Global ``max_steps`` exhaustion is not modelled here — the analyze node
    terminates the workflow with ``CompletionReason.MAX_STEPS`` before the
    planner runs, so it is unreachable as an escalation source.
    """

    LOOP_DETECTOR = "loop_detector"
    SUBGOAL_BUDGET = "subgoal_budget"


class EscalationReason(StrEnum):
    """
    Stable identifier for the decision that produced an escalation outcome.

    DISABLED applies when the policy is turned off entirely. DEFERRAL_LIMIT
    fires when the per-sub-goal deferral count has exceeded its cap (the
    escape valve). SUBGOAL_BUDGET escalates whenever budget is exhausted.
    ACTIVE_STALL escalates when ``since_progress`` contains a non-passive
    no-progress turn. PASSIVE_LIMIT escalates when the consecutive passive
    no-progress count has exceeded the configured tolerance. PASSIVE_RUN
    defers escalation because the contributing tail is purely passive and
    below tolerance.
    """

    DISABLED = "disabled"
    DEFERRAL_LIMIT = "deferral_limit"
    SUBGOAL_BUDGET = "subgoal_budget"
    ACTIVE_STALL = "active_stall"
    PASSIVE_LIMIT = "passive_limit"
    PASSIVE_RUN = "passive_run"


class EscalationPolicy(BaseModel):
    """
    Tunables consumed by :class:`EscalationGate`.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=True,
        description="Master kill switch. When False, every stuck signal escalates immediately.",
    )
    deferral_limit: int = Field(
        default=2,
        ge=0,
        le=10,
        description=(
            "Per sub-goal: number of consecutive deferrals tolerated. When "
            "the count exceeds this value the next stuck signal escalates "
            "regardless of source — escape-valve semantics."
        ),
    )
    passive_tolerance: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Number of consecutive passive (validate-only) no-progress turns "
            "tolerated in the contributing tail of the loop window. Escalation "
            "fires when the count exceeds this value (default 3 -> 4th triggers)."
        ),
    )


class EscalationDecision(BaseModel):
    """
    Outcome of one :class:`EscalationGate` decision.
    """

    model_config = ConfigDict(frozen=True)

    allow: bool = Field(description="True when escalation is permitted now, False when deferred.")
    reason: EscalationReason = Field(description="Stable identifier explaining the decision.")
    stuck_source: StuckSource = Field(
        description="Source that triggered the escalation evaluation."
    )
    deferrals: int = Field(
        ge=0,
        description="Per-sub-goal consecutive-deferral count observed at decision time.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Human-readable detail for logs and telemetry; not consumed by control flow.",
    )


class EscalationPrompt(BaseModel):
    """
    Rationale + user-facing question pair emitted when the planner escalates to HITL.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1, description="User-facing question dispatched as ask_user.")
    rationale: str = Field(min_length=1, description="Short reasoning text shown to the agent log.")
