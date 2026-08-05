from __future__ import annotations

from typing import List, Optional

from fathom.constants import ActionType
from fathom.constants.planner import EscalationPath, PlannerEventKind
from fathom.constants.retries import RetryBranch, RetryKind
from fathom.constants.state import CompletionReason
from fathom.core.agent.escalation_gate import EscalationGate
from fathom.core.agent.planner.factory import PlanStepFactory
from fathom.core.agent.planner.feedback import RetryFeedback
from fathom.core.agent.state import AgentState
from fathom.core.agent.stuck_source import StuckSourceResolver
from fathom.core.context.manager import ContextManager
from fathom.core.prompts.escalation import EscalationPromptBuilder
from fathom.schemas.actions import Action
from fathom.schemas.escalation import EscalationDecision, StuckSource
from fathom.schemas.planner import EscalationEvent, GoalRef, PlannerEvent
from fathom.schemas.results import AnalysisResult, PlanDecision, PlannerRetry, PlanResult
from fathom.schemas.screens import ScreenCapture

# Generic recovery hint injected when the escalation gate defers HITL. The
# model is told what the graph observed and what behaviors are still safe;
# we deliberately avoid any app-specific suggestions.
ESCALATION_DEFERRED_GUIDANCE: str = (
    "The graph detected a possible loop, but recent no-progress actions were "
    "validation checks. Validation actions are expected not to change the screen. "
    "Do not ask the user yet. Re-evaluate the active sub-goal against the current "
    "screen. If the validation condition is visible, emit action_type='validate' "
    "with sub_goal_completed=true. If the current screen already satisfies a "
    "skipped or nonexistent action step, validate the resulting state and mark "
    "the sub-goal complete. Only ask_user if required information is unavailable "
    "or the screen contradicts the sub-goal and no safe action or validation is possible."
)


class EscalationOrchestrator:
    """
    Coordinates HITL escalation: synthesizing ASK_USER, gating a model-emitted ASK_USER, and autonomous recovery.
    """

    def __init__(
        self,
        *,
        gate: EscalationGate,
        resolver: StuckSourceResolver,
        factory: PlanStepFactory,
    ) -> None:
        """
        Bind the escalation gate, stuck-source resolver, and step factory the orchestration sequences.
        """

        self.__gate = gate
        self.__resolver = resolver
        self.__factory = factory

    async def before_vision(
        self,
        *,
        state: AgentState,
        context_manager: ContextManager,
        capture: ScreenCapture,
        interactive_mode: bool,
        prompt_if_stuck: bool,
    ) -> PlanDecision:
        """
        Resolve pre-vision escalation or autonomous recovery, returning a decision plus its events.
        """

        if interactive_mode and prompt_if_stuck:
            return await self.__synthesize(
                state=state, context_manager=context_manager, capture=capture
            )

        if state.is_stuck:
            recovery_action = state.get_recovery_action()
            if recovery_action:
                return PlanDecision(
                    plan=self.__factory.plan_result(
                        capture=capture,
                        is_recovery=True,
                        action=recovery_action,
                        step_number=state.step_count,
                    )
                )

        return PlanDecision()

    async def gate_model_ask_user(
        self,
        *,
        state: AgentState,
        context_manager: ContextManager,
        capture: ScreenCapture,
        analysis: AnalysisResult,
        action: Action,
        interactive_mode: bool,
    ) -> PlanDecision:
        """
        Gate a model-emitted ASK_USER through the escalation policy, returning a decision plus its events.

        When no escalation source is active the model is asking for legitimate external information
        (credentials, OTP, ambiguity) and is allowed through unchanged.
        """

        if action.action_type != ActionType.ASK_USER:
            return PlanDecision()

        if not interactive_mode:
            return PlanDecision(
                plan=self.__substitute_with_recovery(
                    state=state, capture=capture, analysis=analysis
                )
            )

        events: List[PlannerEvent] = []
        llm_source = self.__resolver.resolve(agent_state=state)
        if llm_source is not None:
            events.append(self.__detected(state=state, path=EscalationPath.LLM_TOOL, source=llm_source))

        llm_decision = self.__decide_model_ask_user(state=state)
        if llm_decision is None:
            events.append(
                EscalationEvent(kind=PlannerEventKind.ASK_USER_EMITTED, path=EscalationPath.LLM_TOOL)
            )
            return PlanDecision(events=tuple(events))

        if not llm_decision.allow:
            state.record_deferral()
            await context_manager.inject_user_guidance(
                step=state.step_count, guidance=ESCALATION_DEFERRED_GUIDANCE
            )
            events.append(
                EscalationEvent(
                    kind=PlannerEventKind.ESCALATION_DEFERRED,
                    path=EscalationPath.LLM_TOOL,
                    reason=llm_decision.reason,
                    stuck_source=llm_decision.stuck_source,
                    deferrals=state.deferral_count,
                )
            )
            return PlanDecision(
                plan=RetryFeedback.result(
                    retry=PlannerRetry(
                        kind=RetryKind.ESCALATION_DEFERRED,
                        branch=RetryBranch.ESCALATION_DEFERRED,
                    ),
                    reason=f"Escalation deferred: {llm_decision.reason.value}",
                    analysis=analysis,
                ),
                events=tuple(events),
            )

        state.clear_deferrals()
        events.append(
            EscalationEvent(
                kind=PlannerEventKind.ESCALATION_ALLOWED,
                path=EscalationPath.LLM_TOOL,
                reason=llm_decision.reason,
                stuck_source=llm_decision.stuck_source,
                deferrals=llm_decision.deferrals,
            )
        )
        events.append(
            EscalationEvent(
                kind=PlannerEventKind.ASK_USER_EMITTED,
                path=EscalationPath.LLM_TOOL,
                reason=llm_decision.reason,
                stuck_source=llm_decision.stuck_source,
            )
        )
        return PlanDecision(events=tuple(events))

    async def __synthesize(
        self,
        *,
        state: AgentState,
        context_manager: ContextManager,
        capture: ScreenCapture,
    ) -> PlanDecision:
        """
        Synthesize an ASK_USER step when the gate allows, else defer with guidance and fall through.
        """

        escalation_source = self.__resolver.resolve(agent_state=state)
        if escalation_source is None:
            return PlanDecision()

        # If we already have guidance, proceed straight to analysis.
        if context_manager.get_user_guidance():
            return PlanDecision()

        decision = self.__gate.decide(
            evidence=state.loop_evidence(),
            source=escalation_source,
            deferrals=state.deferral_count,
        )

        events: List[PlannerEvent] = [
            self.__detected(
                state=state, path=EscalationPath.PLANNER_SYNTHESIZED, source=escalation_source
            )
        ]

        if decision.allow:
            state.clear_deferrals()
            events.append(
                EscalationEvent(
                    kind=PlannerEventKind.ESCALATION_ALLOWED,
                    path=EscalationPath.PLANNER_SYNTHESIZED,
                    reason=decision.reason,
                    stuck_source=decision.stuck_source,
                    deferrals=decision.deferrals,
                )
            )
            events.append(
                EscalationEvent(
                    kind=PlannerEventKind.ASK_USER_EMITTED,
                    path=EscalationPath.PLANNER_SYNTHESIZED,
                    reason=decision.reason,
                    stuck_source=decision.stuck_source,
                )
            )
            prompt = EscalationPromptBuilder.build(
                source=escalation_source,
                current_sub_goal=state.get_current_sub_goal(),
                last_action_description=state.last_action_description,
            )
            return PlanDecision(
                plan=PlanResult(
                    step=self.__factory.step(
                        action=Action(
                            confidence=1.0,
                            target="Request user assistance",
                            action_type=ActionType.ASK_USER,
                            rationale=prompt.rationale,
                            text=prompt.question,
                        ),
                        capture=capture,
                        is_recovery=True,
                        step_number=state.step_count,
                    ),
                    is_complete=False,
                    reason=CompletionReason.INTERVENTION_REQUIRED.value,
                ),
                events=tuple(events),
            )

        # Defer escalation: record the deferral, inject generic recovery guidance for the upcoming
        # analysis, and fall through so the model gets another chance to plan a legitimate action.
        state.record_deferral()
        await context_manager.inject_user_guidance(
            step=state.step_count, guidance=ESCALATION_DEFERRED_GUIDANCE
        )
        events.append(
            EscalationEvent(
                kind=PlannerEventKind.ESCALATION_DEFERRED,
                path=EscalationPath.PLANNER_SYNTHESIZED,
                reason=decision.reason,
                stuck_source=decision.stuck_source,
                deferrals=state.deferral_count,
                goal=self.__goal(state=state),
            )
        )
        return PlanDecision(events=tuple(events))

    def __detected(
        self, *, state: AgentState, path: EscalationPath, source: StuckSource
    ) -> EscalationEvent:
        """
        Build the escalation-detected event carrying the gate's pre-decision context.
        """

        return EscalationEvent(
            kind=PlannerEventKind.ESCALATION_DETECTED,
            path=path,
            stuck_source=source,
            deferrals=state.deferral_count,
            goal=self.__goal(state=state),
        )

    @staticmethod
    def __goal(*, state: AgentState) -> Optional[GoalRef]:
        """
        Reference the active sub-goal by index, when one is active.
        """

        current = state.get_current_sub_goal()
        return GoalRef(index=current.index) if current is not None else None

    def __substitute_with_recovery(
        self,
        *,
        state: AgentState,
        capture: ScreenCapture,
        analysis: AnalysisResult,
    ) -> PlanResult:
        """
        Substitute ASK_USER with the next recovery ladder rung; terminate when exhausted.
        """

        if (recovery_action := state.get_recovery_action()) is None:
            state.mark_complete(reason=CompletionReason.INTERVENTION_REQUIRED.value)
            return PlanResult(
                step=None,
                is_complete=True,
                metrics=analysis.metrics,
                memories=analysis.memories,
                reason=CompletionReason.INTERVENTION_REQUIRED.value,
            )

        return self.__factory.plan_result(
            capture=capture,
            is_recovery=True,
            action=recovery_action,
            metrics=analysis.metrics,
            memories=analysis.memories,
            step_number=state.step_count,
        )

    def __decide_model_ask_user(self, *, state: AgentState) -> Optional[EscalationDecision]:
        """
        Gate a model-emitted ASK_USER through the escalation policy; None when no stuck source is active.
        """

        escalation_source: Optional[StuckSource] = self.__resolver.resolve(agent_state=state)
        if escalation_source is None:
            return None

        return self.__gate.decide(
            source=escalation_source,
            evidence=state.loop_evidence(),
            deferrals=state.deferral_count,
        )
