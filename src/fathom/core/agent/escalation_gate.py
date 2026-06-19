from __future__ import annotations

from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.escalation import (
    EscalationDecision,
    EscalationPolicy,
    EscalationReason,
    StuckSource,
)
from fathom.schemas.loop import LoopEvidence
from fathom.schemas.vision import ActionKind


class EscalationGate:
    """
    Stateless predicate over typed escalation evidence.

    Decides whether a stuck signal warrants escalation right now without
    mutating any input. The gate reads :attr:`LoopEvidence.since_progress`
    (the slice after the most recent PROGRESS turn) and never inspects older
    turns — that bounds the decision to evidence that materially contributed
    to the current stuck classification, closing the false-positive class
    where one unrelated historical no-progress turn would otherwise unlock HITL.

    Decision rules, in priority order:

    1. Policy disabled -> allow (preserves pre-gate behaviour for kill-switch).
    2. Per-sub-goal deferrals exceed the configured limit -> allow
       (escape valve: deferral cannot hide a genuinely stuck flow forever).
    3. Source is ``SUBGOAL_BUDGET`` -> allow (hard budget signal, validates or not).
    4. Source is ``LOOP_DETECTOR``:
         a. Trailing passive NO_PROGRESS count exceeds tolerance -> allow.
         b. ``since_progress`` contains any active NO_PROGRESS turn -> allow.
         c. Otherwise -> defer (the loop is passive-only and within tolerance).
    """

    def __init__(self, *, policy: EscalationPolicy) -> None:
        """
        Bind the gate to its policy.
        """

        self.__policy = policy

    def decide(
        self,
        *,
        source: StuckSource,
        evidence: LoopEvidence,
        deferrals: int,
    ) -> EscalationDecision:
        """
        Compute the decision for a single evaluation.
        """

        if not self.__policy.enabled:
            return EscalationDecision(
                allow=True,
                reason=EscalationReason.DISABLED,
                stuck_source=source,
                deferrals=deferrals,
                message="Escalation policy is disabled; escalating unconditionally.",
            )

        if deferrals > self.__policy.deferral_limit:
            return EscalationDecision(
                allow=True,
                reason=EscalationReason.DEFERRAL_LIMIT,
                stuck_source=source,
                deferrals=deferrals,
                message=(
                    f"Escape valve: deferrals {deferrals} exceed "
                    f"limit {self.__policy.deferral_limit}."
                ),
            )

        if source is StuckSource.SUBGOAL_BUDGET:
            return EscalationDecision(
                allow=True,
                reason=EscalationReason.SUBGOAL_BUDGET,
                stuck_source=source,
                deferrals=deferrals,
                message="Sub-goal budget exhausted; escalating regardless of action mix.",
            )

        passive_run = self.__passive_run(evidence=evidence)

        if passive_run > self.__policy.passive_tolerance:
            return EscalationDecision(
                allow=True,
                reason=EscalationReason.PASSIVE_LIMIT,
                stuck_source=source,
                deferrals=deferrals,
                message=(
                    f"Passive no-progress run {passive_run} exceeds "
                    f"tolerance {self.__policy.passive_tolerance}."
                ),
            )

        if self.__has_active_stall(evidence=evidence):
            return EscalationDecision(
                allow=True,
                reason=EscalationReason.ACTIVE_STALL,
                stuck_source=source,
                deferrals=deferrals,
                message=(
                    "Contributing tail contains an active no-progress turn; "
                    "the loop is not purely passive."
                ),
            )

        if passive_run == 0:
            # No positive evidence of a passive-only pattern in the contributing
            # tail. The loop detector classified the window as stuck for some
            # other reason (screen repetition without recorded effects, action
            # repetition with UNCERTAIN effects). Per the reviewer's "be
            # conservative about not suppressing forever, not about fabricating
            # no-progress evidence" rule, allow escalation rather than defer
            # on absent evidence.
            return EscalationDecision(
                allow=True,
                reason=EscalationReason.ACTIVE_STALL,
                stuck_source=source,
                deferrals=deferrals,
                message=(
                    "No passive no-progress evidence in the contributing tail; "
                    "treating the stuck signal as an active stall."
                ),
            )

        return EscalationDecision(
            allow=False,
            reason=EscalationReason.PASSIVE_RUN,
            stuck_source=source,
            deferrals=deferrals,
            message=(
                f"Passive loop within tolerance "
                f"({passive_run} <= {self.__policy.passive_tolerance}); "
                "deferring escalation and falling through to re-plan."
            ),
        )

    @staticmethod
    def __passive_run(*, evidence: LoopEvidence) -> int:
        """
        Count trailing consecutive passive (VALIDATION) turns with NO_PROGRESS.

        Walks backwards through ``since_progress``. Stops at the first turn
        that is not ``(VALIDATION AND NO_PROGRESS)``; UNCERTAIN turns on
        VALIDATION are pass-through (do not count, do not break) so the
        streak does not get artificially clipped by an inconclusive metric.
        """

        count = 0
        for turn in reversed(evidence.since_progress):
            if (
                turn.action_kind is ActionKind.VALIDATION
                and turn.effect_status is ActionEffectStatus.NO_PROGRESS
            ):
                count += 1
                continue

            if (
                turn.action_kind is ActionKind.VALIDATION
                and turn.effect_status is ActionEffectStatus.UNCERTAIN
            ):
                continue

            break

        return count

    @staticmethod
    def __has_active_stall(*, evidence: LoopEvidence) -> bool:
        """
        Whether any non-passive turn with NO_PROGRESS lives in ``since_progress``.

        ``since_progress`` already excludes turns before the most recent
        PROGRESS effect, so a True result here means the current stuck signal
        is not purely passive and escalation is warranted.
        """

        return any(
            turn.action_kind is not ActionKind.VALIDATION
            and turn.effect_status is ActionEffectStatus.NO_PROGRESS
            for turn in evidence.since_progress
        )
