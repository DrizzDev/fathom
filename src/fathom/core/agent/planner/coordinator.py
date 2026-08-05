from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.constants.planner import GuardOutcome
from fathom.core.agent.action import ActionBuilder
from fathom.core.agent.command import CommandGate
from fathom.core.agent.escalation_gate import EscalationGate
from fathom.core.agent.planner.admission import RequirementAdmitter
from fathom.core.agent.planner.escalation import EscalationOrchestrator
from fathom.core.agent.planner.factory import PlanStepFactory
from fathom.core.agent.planner.feedback import RejectionFeedback, RetryFeedback
from fathom.core.agent.planner.guard import ActionGuard
from fathom.core.agent.planner.materializer import CommandMaterializer
from fathom.core.agent.planner.scope import ToolScopeResolver
from fathom.core.agent.planner.selection import ActionSelector
from fathom.core.agent.planner.terminal import TerminalPolicy
from fathom.core.agent.planner.vision_turn import VisionTurn
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.agent.stuck_source import StuckSourceResolver
from fathom.core.agent.tools import DEFAULT_TOOL_POLICIES, ToolScope
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.context.manager import ContextManager
from fathom.core.exceptions import ToolValidationError
from fathom.core.services.vision import VisionService
from fathom.schemas.escalation import EscalationPolicy
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.planner import CommandRejectedEvent, PlannerEvent
from fathom.schemas.results import PlanContext, PlanResult, PlanTurn
from fathom.schemas.screens import ScreenCapture


class StepPlanner:
    """
    Plans and prepares steps for execution.
    """

    def __init__(
        self,
        vision_tool: VisionService,
        *,
        tool_scope: Optional[ToolScope] = None,
        escalation_policy: Optional[EscalationPolicy] = None,
        escalation_gate: Optional[EscalationGate] = None,
        stuck_source_resolver: Optional[StuckSourceResolver] = None,
        command_gate: Optional[CommandGate] = None,
        action_builder: Optional[ActionBuilder] = None,
    ) -> None:
        self.__vision = vision_tool
        self.__factory = PlanStepFactory()
        self.__admitter = RequirementAdmitter()
        self.__scope = ToolScopeResolver(
            tool_scope=tool_scope or ToolScope(policies=DEFAULT_TOOL_POLICIES)
        )
        self.__guard = ActionGuard()
        self.__rejection = RejectionFeedback(vision=self.__vision)
        self.__escalation = EscalationOrchestrator(
            gate=escalation_gate or EscalationGate(policy=escalation_policy or EscalationPolicy()),
            resolver=stuck_source_resolver or StuckSourceResolver(),
            factory=self.__factory,
        )
        self.__vision_turn = VisionTurn(vision=self.__vision)
        self.__materializer = CommandMaterializer(
            command_gate=command_gate or CommandGate(catalog=CommandCatalogProvider().build()),
            action_builder=action_builder or ActionBuilder(),
        )

    @property
    def vision_tool(self) -> VisionService:
        """
        Returns the underlying vision tool.
        """

        return self.__vision

    async def plan_step(
        self,
        state: AgentState,
        reasoner: Reasoner,
        capture: ScreenCapture,
        context_manager: ContextManager,
        *,
        screen_width: int,
        screen_height: int,
        use_xml: bool = True,
        prompt_if_stuck: bool = False,
        elements: Optional[Dict[str, Any]] = None,
        screen_observation: Optional[ScreenObservation] = None,
    ) -> PlanTurn:
        """
        Plan the next step, returning the planned outcome and the turn's observability events.
        """

        events: List[PlannerEvent] = []
        interactive_mode = state.capabilities.hitl.enabled

        if not state.can_continue:
            terminal_reason = TerminalPolicy.reason(state=state)
            if not state.is_complete:
                state.mark_complete(reason=terminal_reason)

            return PlanTurn(plan=PlanResult(step=None, is_complete=True, reason=terminal_reason))

        current_tracking_note = state.tracking_note

        recovery = await self.__escalation.before_vision(
            state=state,
            capture=capture,
            context_manager=context_manager,
            interactive_mode=interactive_mode,
            prompt_if_stuck=prompt_if_stuck,
        )
        events.extend(recovery.events)
        if recovery.plan is not None:
            return PlanTurn(plan=recovery.plan, events=tuple(events))

        tools, scope_event = self.__scope.resolve(
            state=state, current_sub_goal=state.get_current_sub_goal()
        )
        events.append(scope_event)

        analysis = await self.__vision_turn.analyze(
            state=state,
            capture=capture,
            context_manager=context_manager,
            tools=tools,
            screen_width=screen_width,
            screen_height=screen_height,
            use_xml=use_xml,
            elements=elements,
            screen_observation=screen_observation,
            tracking_note=current_tracking_note,
        )
        try:
            analysis = self.__materializer.materialize(analysis=analysis)
        except ToolValidationError as exception:
            events.append(CommandRejectedEvent(reason=exception.feedback.message))
            return PlanTurn(
                plan=self.__rejection.reject_command(
                    state=state, analysis=analysis, feedback=exception.feedback
                ),
                events=tuple(events),
            )

        if analysis.action is None:
            tool_response = analysis.tool_response
            if tool_response is not None and tool_response.has_non_command_parts:
                return PlanTurn(
                    plan=RetryFeedback.no_action(
                        analysis=analysis,
                        reason=analysis.reasoning or "Tool response routed",
                        updates=tool_response.updates,
                        data=tool_response.data,
                        artifacts=tool_response.artifacts,
                        diagnostics=tool_response.diagnostics,
                    ),
                    events=tuple(events),
                )

            return PlanTurn(
                plan=RetryFeedback.no_action(
                    analysis=analysis,
                    reason="No executable command was produced by the tool response",
                ),
                events=tuple(events),
            )

        # Use-bounded signals: rejection history and verifier feedback are one-turn channels.
        # Human guidance ages through a short TTL so a missed HITL instruction survives another
        # planner turn without becoming a permanent stale imperative.
        state.clear_rejection_history()
        context_manager.consume_user_guidance()
        context_manager.clear_verifier_feedback()

        if analysis.content_exhausted:
            state.reset_loop_detector()
            # Do not mark_complete here: content_exhausted means "no more content on this list/feed",
            # not "task done". Marking complete would cause early exits; fall through and plan next.

        action = ActionSelector.select(state=state, reasoner=reasoner, analysis=analysis)

        gated = await self.__escalation.gate_model_ask_user(
            state=state,
            action=action,
            capture=capture,
            analysis=analysis,
            context_manager=context_manager,
            interactive_mode=interactive_mode,
        )
        events.extend(gated.events)
        if gated.plan is not None:
            return PlanTurn(plan=gated.plan, events=tuple(events))

        guard_decision = self.__guard.evaluate(state=state, action=action, analysis=analysis)
        events.extend(guard_decision.events)
        if guard_decision.outcome is not GuardOutcome.ALLOW:
            return PlanTurn(
                plan=self.__rejection.reject(
                    decision=guard_decision,
                    state=state,
                    action=action,
                    analysis=analysis,
                    interactive_mode=interactive_mode,
                ),
                events=tuple(events),
            )

        # Record action for sub-goal trace verification
        if state.has_sub_goals() and action.action_type not in {
            ActionType.WAIT,
            ActionType.ASK_USER,
        }:
            state.record_sub_goal_action()

        return PlanTurn(
            plan=self.__factory.plan_result(
                action=action,
                capture=capture,
                metrics=analysis.metrics,
                memories=analysis.memories,
                step_number=state.step_count,
                requirement=self.__admitter.admit(
                    current_sub_goal=state.get_current_sub_goal(), action=action
                ),
                context=PlanContext(analysis=analysis, observation=analysis.screen_description),
                updates=(
                    analysis.tool_response.updates if analysis.tool_response is not None else ()
                ),
                data=analysis.tool_response.data if analysis.tool_response is not None else (),
                artifacts=(
                    analysis.tool_response.artifacts if analysis.tool_response is not None else ()
                ),
                diagnostics=(
                    analysis.tool_response.diagnostics if analysis.tool_response is not None else ()
                ),
            ),
            events=tuple(events),
        )
