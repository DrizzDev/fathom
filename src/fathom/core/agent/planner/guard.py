from __future__ import annotations

from typing import List, Optional

from fathom.constants import ActionType
from fathom.constants.planner import GuardOutcome, PlannerEventKind
from fathom.core.agent.state import AgentState
from fathom.core.runtime.identity import TargetIdentity
from fathom.schemas.actions import Action
from fathom.schemas.planner import GoalRef, GuardDecision, GuardEvent, PlannerEvent
from fathom.schemas.results import AnalysisResult
from fathom.schemas.supervision import BlockReason


class ActionGuard:
    """
    Adjudicates whether a proposed action repeats a failed or already-succeeded action, as a pure verdict.
    """

    def evaluate(
        self, *, state: AgentState, action: Action, analysis: AnalysisResult
    ) -> GuardDecision:
        """
        Run each guard rule in order and return the first blocking verdict, aggregating every rule's events.
        """

        events: List[PlannerEvent] = []
        descriptor = action.to_description()

        if state.should_avoid_action(action=action):
            events.append(self.__blocked_event(state=state, descriptor=descriptor))
            return GuardDecision(
                outcome=GuardOutcome.SILENT_REJECTION, action=descriptor, events=tuple(events)
            )

        reason = self.__current_screen_repeat_reason(action=action, analysis=analysis)
        if (
            reason is not None
            and state.operator_directive is not None
            and state.has_active_directive
            and state.directive_matches(action=action)
        ):
            events.append(
                GuardEvent(
                    kind=PlannerEventKind.GUARD_BYPASSED,
                    action=descriptor,
                    goal=self.__goal(state=state),
                )
            )
            reason = None

        if reason is not None:
            events.append(
                self.__blocked_event(
                    state=state,
                    descriptor=descriptor,
                    block_reason=BlockReason.REPEATED_CURRENT_SCREEN_ACTION,
                )
            )
            return GuardDecision(
                outcome=GuardOutcome.CURRENT_SCREEN_REPEAT,
                action=descriptor,
                reason=reason,
                block=BlockReason.REPEATED_CURRENT_SCREEN_ACTION,
                events=tuple(events),
            )

        if state.is_action_repeating_on_screen(action=action) and not (
            state.has_active_directive and state.directive_matches(action=action)
        ):
            events.append(self.__blocked_event(state=state, descriptor=descriptor))
            return GuardDecision(
                outcome=GuardOutcome.REPEATING_ON_SCREEN, action=descriptor, events=tuple(events)
            )

        return GuardDecision(outcome=GuardOutcome.ALLOW, events=tuple(events))

    def __blocked_event(
        self, *, state: AgentState, descriptor: str, block_reason: Optional[BlockReason] = None
    ) -> GuardEvent:
        """
        Build the action-blocked event for a guarded action.
        """

        return GuardEvent(
            kind=PlannerEventKind.ACTION_BLOCKED,
            action=descriptor,
            block_reason=block_reason,
            goal=self.__goal(state=state),
        )

    @staticmethod
    def __goal(*, state: AgentState) -> Optional[GoalRef]:
        """
        Reference the active sub-goal by index, when one is active.
        """

        current = state.get_current_sub_goal()
        return GoalRef(index=current.index) if current is not None else None

    @classmethod
    def __current_screen_repeat_reason(
        cls, *, action: Action, analysis: AnalysisResult
    ) -> Optional[str]:
        """
        Return a block reason when the action repeats a successful current-screen action.
        """

        if action.action_type in {
            ActionType.WAIT,
            ActionType.ASK_USER,
            ActionType.VALIDATE,
            ActionType.COMPLETE,
        }:
            return None

        history = analysis.metadata.get("current_workflow_screen_actions")

        if not isinstance(history, list):
            return None

        for entry in history:
            if not isinstance(entry, dict):
                continue

            if entry.get("success") is not True:
                continue

            previous_action_type = str(entry.get("action") or entry.get("type") or "").lower()
            if previous_action_type and previous_action_type != action.action_type.value:
                continue

            previous_target = str(entry.get("target") or "")
            if not previous_target:
                continue

            if cls.__describes_same_target(action=action, previous=previous_target):
                return (
                    f"Action {action.to_description()!r} already succeeded on the current "
                    "screen during this workflow."
                )

        return None

    @staticmethod
    def __describes_same_target(*, action: Action, previous: str) -> bool:
        """
        Return whether a planned action points at a previously handled target.
        """

        candidates = (
            action.natural_language_target,
            action.target,
            action.script_target,
            action.export_target,
        )
        return any(
            candidate is not None
            and TargetIdentity.describes_same_target(
                previous=previous,
                replacement=candidate,
            )
            for candidate in candidates
        )
