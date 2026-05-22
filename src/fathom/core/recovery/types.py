from __future__ import annotations

from enum import StrEnum
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Action
from fathom.schemas.escape import EscapeReport
from fathom.schemas.localization import LocalizationCandidate
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.subgoal import ExecutionContract, SubGoal
from fathom.schemas.supervision import BlockReason


class RecoveryTrigger(StrEnum):
    """
    Enumerates the points at which the graph asks the recovery coordinator
    for help. New triggers can be added without touching existing strategies.

    Each value is a distinct stuck signal:

    - ``ACTION_BLOCKED``: the planner determined the previously-planned
      approach is unreachable from the current screen.
    - ``VERIFY_REJECTED``: final verification refused the run as complete.
    - ``LOOP_DETECTED``: the loop detector observed repetition across
      screens or actions strong enough to indicate the agent is cycling.
    - ``NO_PROGRESS``: the action-effect classifier marked the last N
      actions as producing no visual progress.
    - ``TARGET_UNRESOLVED``: the resolution layer could not map the
      agent's named target to a tappable manifest element or screen
      coordinate.
    - ``SUBGOAL_BUDGET_EXCEEDED``: the active sub-goal exhausted its
      per-sub-goal step budget without its success criterion being met.
    - ``REQUEST_REPLAN``: the agent emitted a structured
      :class:`fathom.schemas.escape.EscapeReport` declaring it cannot
      make safe forward progress on the active sub-goal; the
      ``escape_report.category`` on :class:`RecoveryRequest` drives
      per-category framing in the decomposer preamble.
    """

    NO_PROGRESS = "NO_PROGRESS"
    LOOP_DETECTED = "LOOP_DETECTED"
    REQUEST_REPLAN = "REQUEST_REPLAN"
    ACTION_BLOCKED = "ACTION_BLOCKED"
    VERIFY_REJECTED = "VERIFY_REJECTED"
    TARGET_UNRESOLVED = "TARGET_UNRESOLVED"
    SUBGOAL_BUDGET_EXCEEDED = "SUBGOAL_BUDGET_EXCEEDED"


class RecoveryRequest(BaseModel):
    """
    Frozen input handed to a recovery strategy when its trigger fires.
    Carries enough context to reason about the stuck state without
    granting access to mutable agent state.
    """

    trigger: RecoveryTrigger = Field(description="Trigger that produced this request")
    capture: ScreenCapture = Field(description="Current screen capture for visual grounding")

    reason: str = Field(description="Free-text failure reason from the graph node")
    hint: Optional[str] = Field(
        default=None, description="Optional concrete hint such as the blocked action"
    )

    pending_sub_goals: List[str] = Field(description="Remaining unfinished sub-goal descriptions")
    stuck_sub_goal: str = Field(
        description="Description of the sub-goal the agent failed to complete"
    )
    strict_mode: bool = Field(
        default=False,
        description="Whether strict execution constraints are active for this run.",
    )
    execution_contract: ExecutionContract = Field(
        default_factory=ExecutionContract,
        description="Structured execution constraints carried by the active sub-goal.",
    )

    recent_actions: List[str] = Field(
        default_factory=list, description="Most-recent action descriptors, oldest first"
    )

    escape_report: Optional[EscapeReport] = Field(
        default=None,
        description=(
            "Structured escape signal forwarded from the planner when ``trigger`` is ``REQUEST_REPLAN``. "
            "Strategies consult ``escape_report.category`` to vary framing per category."
        ),
    )

    block_reason: Optional[BlockReason] = Field(
        default=None,
        description="Supervisor block reason that initiated recovery, when present.",
    )

    observation: Optional[ScreenObservation] = Field(
        default=None,
        description="Current screen observation; consumed by mechanical recovery strategies.",
    )

    candidates: List[LocalizationCandidate] = Field(
        default_factory=list,
        description="Localization candidates for alternative-target recovery.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class ReplanOutcome(BaseModel):
    """
    Replace the pending sub-goal tail with re-decomposed sub-goals.
    """

    summary: str = Field(description="Short human-readable summary for telemetry/logs")
    kind: Literal["replan"] = Field(default="replan", description="Outcome discriminator")
    new_sub_goals: List[SubGoal] = Field(description="Replacement sub-goals in execution order")

    model_config = ConfigDict(frozen=True)


class TryActionOutcome(BaseModel):
    """
    Retry execution with an alternative action proposed by recovery.
    """

    kind: Literal["try_action"] = Field(default="try_action", description="Outcome discriminator")
    summary: str = Field(description="Short human-readable summary for telemetry/logs")
    action: Action = Field(description="Alternative action the coordinator must execute next")

    model_config = ConfigDict(frozen=True)


class EscalateOutcome(BaseModel):
    """
    Escalate the stuck state to the human operator.
    """

    kind: Literal["escalate"] = Field(default="escalate", description="Outcome discriminator")
    summary: str = Field(description="Short human-readable summary for telemetry/logs")
    question: str = Field(description="Question surfaced to the human operator")

    model_config = ConfigDict(frozen=True)


class BoundedFailureOutcome(BaseModel):
    """
    Terminate the run with a structured diagnostic.
    """

    kind: Literal["bounded_failure"] = Field(
        default="bounded_failure", description="Outcome discriminator"
    )
    summary: str = Field(description="Short human-readable summary for telemetry/logs")
    diagnostic: str = Field(description="Actionable failure diagnostic for telemetry and audit")

    model_config = ConfigDict(frozen=True)


class NoopOutcome(BaseModel):
    """
    Strategy declined to act; coordinator falls through to the next strategy.
    """

    kind: Literal["noop"] = Field(default="noop", description="Outcome discriminator")
    summary: str = Field(description="Short human-readable summary for telemetry/logs")

    model_config = ConfigDict(frozen=True)


RecoveryOutcome = Annotated[
    Union[
        NoopOutcome,
        ReplanOutcome,
        EscalateOutcome,
        TryActionOutcome,
        BoundedFailureOutcome,
    ],
    Field(discriminator="kind"),
]
