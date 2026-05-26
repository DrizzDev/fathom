from __future__ import annotations

import hashlib
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, cast

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason, PlanMetadataKey
from fathom.core.agent.escalation_gate import EscalationGate
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.agent.stuck_source import StuckSourceResolver
from fathom.core.context.manager import ContextManager
from fathom.core.runtime.identity import TargetIdentity
from fathom.core.services.vision import SubGoalContext, VisionService
from fathom.schemas.actions import Action
from fathom.schemas.escalation import EscalationDecision, EscalationPolicy, StuckSource
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.schemas.supervision import BlockReason

# Generic recovery hint injected when the escalation gate defers HITL. The
# model is told what the graph observed and what behaviours are still safe;
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

logger = getLogger(name=__name__)


class StepPlanner:
    """
    Plans and prepares steps for execution.
    """

    def __init__(
        self,
        vision_tool: VisionService,
        *,
        escalation_policy: Optional[EscalationPolicy] = None,
        escalation_gate: Optional[EscalationGate] = None,
        stuck_source_resolver: Optional[StuckSourceResolver] = None,
    ) -> None:
        self.__vision = vision_tool
        self.__escalation_gate = escalation_gate or EscalationGate(
            policy=escalation_policy or EscalationPolicy(),
        )
        self.__stuck_source_resolver = stuck_source_resolver or StuckSourceResolver()

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
        interactive_mode: bool = False,
        elements: Optional[Dict[str, Any]] = None,
        screen_observation: Optional[ScreenObservation] = None,
    ) -> PlanResult:
        """
        Plan the next step based on current state.
        """

        if not state.can_continue_with(interactive_mode=interactive_mode):
            if state.is_complete:
                return PlanResult(
                    step=None, is_complete=True, reason=CompletionReason.SUCCESS.value
                )

            # TERMINAL FAIL: If we reach here, it means max_steps or budgets are hit.
            # We must stop immediately to avoid infinite HITL loops.
            return PlanResult(
                step=None,
                is_complete=True,
                reason=CompletionReason.STUCK.value
                if state.is_stuck
                else CompletionReason.FAILED.value,
            )

        # Default tracking note
        current_tracking_note = state.tracking_note

        # IMMEDIATE RECOVERY
        # NATIVE INTERCEPT: Yield to HITL if enabled before attempting aggressive auto-recovery.
        # The HITL path widens the trigger to ``StuckSourceResolver`` so that sub-goal budget
        # exhaustion can also reach the gate. The autonomous-recovery path below stays scoped
        # to the original ``state.is_stuck`` signal to preserve existing non-interactive behaviour.
        if interactive_mode and prompt_if_stuck:
            escalation_source = self.__stuck_source_resolver.resolve(agent_state=state)

            if escalation_source is not None:
                # If we have guidance, proceed to analysis.
                if context_manager.get_user_guidance():
                    # Pass through to analysis
                    pass
                else:
                    evidence = state.loop_evidence()
                    decision = self.__escalation_gate.decide(
                        source=escalation_source,
                        evidence=evidence,
                        deferrals=state.deferral_count,
                    )

                    self.__log_escalation_detected(
                        path="planner_synthesized",
                        state=state,
                        evidence=evidence,
                        source=escalation_source,
                    )

                    if decision.allow:
                        state.clear_deferrals()
                        logger.info(
                            "Escalation allowed by gate; emitting ASK_USER",
                            extra={
                                "component": "core.agent.planner",
                                "event": "planner.escalation.allowed",
                                "escalation.path": "planner_synthesized",
                                "escalation.reason": decision.reason.value,
                                "escalation.stuck_source": decision.stuck_source.value,
                                "escalation.deferrals_before": decision.deferrals,
                            },
                        )
                        logger.info(
                            "ASK_USER materialized by planner",
                            extra={
                                "component": "core.agent.planner",
                                "event": "planner.ask_user.emitted",
                                "escalation.path": "planner_synthesized",
                                "escalation.reason": decision.reason.value,
                                "escalation.stuck_source": decision.stuck_source.value,
                            },
                        )
                        return PlanResult(
                            step=self.__build_step(
                                action=Action(
                                    confidence=1.0,
                                    target="Request user assistance",
                                    action_type=ActionType.ASK_USER,
                                    rationale="Loop detected (Screen repeating). Requesting human intervention.",
                                    text="I have detected a loop and I'm repeating the same screen state. How should I proceed to break this cycle?",
                                ),
                                capture=capture,
                                is_recovery=True,
                                step_number=state.step_count,
                            ),
                            is_complete=False,
                            reason=CompletionReason.INTERVENTION_REQUIRED.value,
                            metadata={
                                "escalation.path": "planner_synthesized",
                                "escalation.reason": decision.reason.value,
                                "escalation.stuck_source": decision.stuck_source.value,
                            },
                        )

                    # Defer escalation: record the deferral, inject generic
                    # recovery guidance for the upcoming analysis, and fall
                    # through so the model gets another chance to plan a
                    # legitimate action this turn.
                    state.record_deferral()
                    await context_manager.inject_user_guidance(
                        guidance=ESCALATION_DEFERRED_GUIDANCE,
                        step=state.step_count,
                    )
                    logger.info(
                        "Escalation deferred by gate; injecting recovery guidance",
                        extra={
                            "component": "core.agent.planner",
                            "event": "planner.escalation.deferred",
                            "escalation.reason": decision.reason.value,
                            "escalation.stuck_source": decision.stuck_source.value,
                            "escalation.deferrals_after": state.deferral_count,
                            "sub_goal.index": (
                                current_sub_goal.index
                                if (current_sub_goal := state.get_current_sub_goal())
                                else None
                            ),
                        },
                    )

        # Autonomous Recovery (non-interactive mode preserves original is_stuck trigger).
        elif state.is_stuck:
            recovery_action = state.get_recovery_action()
            if recovery_action:
                return self.__build_plan_result(
                    capture=capture,
                    is_recovery=True,
                    action=recovery_action,
                    step_number=state.step_count,
                )

        # Build minimal sub-goal context for vision (avoid passing AgentState into VisionService)
        sub_goal_info: Optional[SubGoalContext] = None
        current_sub_goal = state.get_current_sub_goal()
        if current_sub_goal and state.has_sub_goals():
            current_idx, total = state.get_sub_goal_progress()
            sub_goal_info = {
                "index": current_idx,
                "total": total,
                "description": current_sub_goal.description,
            }

        analysis = await self.__vision.analyze(
            use_xml=use_xml,
            capture=capture,
            elements=elements,
            intent=state.intent,
            is_stuck=state.is_stuck,
            screen_width=screen_width,
            screen_height=screen_height,
            sub_goal_info=sub_goal_info,
            context_manager=context_manager,
            last_action=state.last_action_type,
            tracking_note=current_tracking_note,
            loop_observation=state.build_loop_observation(),
            screen_observation=screen_observation,
            prior_rejection_history=state.rejection_history,
            visual_hash=self.__compute_simple_hash(capture=capture),
            failures=cast("List[str]", state.build_context().get("relevant_failures", [])),
        )

        # Use-bounded signals: rejection history and verifier feedback
        # are one-turn channels. Human guidance ages through a short TTL
        # so a missed HITL instruction survives another planner turn
        # without becoming a permanent stale imperative.
        state.clear_rejection_history()
        context_manager.consume_user_guidance()
        context_manager.clear_verifier_feedback()

        if analysis.content_exhausted:
            state.reset_loop_detector()
            # Do not mark_complete here: content_exhausted means "no more content on this list/feed",
            # not "task done". Marking complete would cause early exits; fall through and plan next.

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        # Gate LLM-emitted ASK_USER through the same escalation policy so the
        # tool-call path cannot route around the deterministic-branch decision.
        # When no escalation source is active the model is asking for legitimate
        # external information (credentials, OTP, ambiguity) and is allowed
        # through unchanged.
        if action.action_type == ActionType.ASK_USER:
            llm_source = self.__stuck_source_resolver.resolve(agent_state=state)
            if llm_source is not None:
                self.__log_escalation_detected(
                    path="llm_tool",
                    state=state,
                    evidence=state.loop_evidence(),
                    source=llm_source,
                )

            llm_decision = self.__decide_llm_emitted_ask_user(state=state)
            if llm_decision is None:
                logger.info(
                    "LLM-emitted ask_user passed through (no stuck source)",
                    extra={
                        "component": "core.agent.planner",
                        "event": "planner.ask_user.emitted",
                        "escalation.path": "llm_tool",
                        "escalation.gated": False,
                    },
                )
            if llm_decision is not None and not llm_decision.allow:
                state.record_deferral()
                await context_manager.inject_user_guidance(
                    guidance=ESCALATION_DEFERRED_GUIDANCE,
                    step=state.step_count,
                )
                logger.info(
                    "Escalation deferred on LLM-emitted ask_user; retrying after guidance",
                    extra={
                        "component": "core.agent.planner",
                        "event": "planner.escalation.deferred",
                        "escalation.path": "llm_tool",
                        "escalation.reason": llm_decision.reason.value,
                        "escalation.stuck_source": llm_decision.stuck_source.value,
                        "escalation.deferrals_after": state.deferral_count,
                        "route.next": "ground",
                        "plan.should_retry": True,
                    },
                )
                return PlanResult(
                    step=None,
                    is_complete=False,
                    should_retry=True,
                    metrics=analysis.metrics,
                    memories=analysis.memories,
                    reason=f"Escalation deferred: {llm_decision.reason.value}",
                    metadata={
                        **(analysis.metadata or {}),
                        "escalation.path": "llm_tool",
                        "escalation.suppressed": True,
                        "escalation.reason": llm_decision.reason.value,
                        "escalation.stuck_source": llm_decision.stuck_source.value,
                    },
                )
            if llm_decision is not None and llm_decision.allow:
                state.clear_deferrals()
                logger.info(
                    "LLM-emitted ask_user allowed by escalation gate",
                    extra={
                        "component": "core.agent.planner",
                        "event": "planner.escalation.allowed",
                        "escalation.path": "llm_tool",
                        "escalation.reason": llm_decision.reason.value,
                        "escalation.stuck_source": llm_decision.stuck_source.value,
                        "escalation.deferrals_before": llm_decision.deferrals,
                    },
                )
                logger.info(
                    "ASK_USER materialized from LLM tool call",
                    extra={
                        "component": "core.agent.planner",
                        "event": "planner.ask_user.emitted",
                        "escalation.path": "llm_tool",
                        "escalation.gated": True,
                        "escalation.reason": llm_decision.reason.value,
                        "escalation.stuck_source": llm_decision.stuck_source.value,
                    },
                )

        if state.should_avoid_action(action=action):
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason=CompletionReason.FAILED.value,
            )

        current_screen_repeat = self.__current_screen_repeat_reason(
            action=action,
            analysis=analysis,
        )
        if current_screen_repeat is not None:
            logger.warning(
                "[Planner] Blocking repeated current-screen action: %s",
                current_screen_repeat,
            )
            state.record_blocked_action(
                action=action,
                reason=current_screen_repeat,
                block_reason=BlockReason.REPEATED_CURRENT_SCREEN_ACTION,
            )
            state.set_rejection_history(
                self.__vision.build_rejection_history_from_analysis(
                    analysis=analysis,
                    rejection_reason=(
                        f"REJECTED: {current_screen_repeat} Choose a different action "
                        "that advances the active sub-goal on the current screen, or ask "
                        "the user if the screen contradicts the sub-goal."
                    ),
                )
            )
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                memories=analysis.memories,
                reason=CompletionReason.ACTION_BLOCKED.value,
                metadata={
                    **(analysis.metadata or {}),
                    "blocked_action": action.to_description(),
                    "block_reason": BlockReason.REPEATED_CURRENT_SCREEN_ACTION.value,
                },
            )

        # Check if same tap/type action repeated 3+ times on the same screen.
        # Swipes/scrolls are excluded since they legitimately repeat.
        if state.is_action_repeating_on_screen(action=action):
            repeated_desc = action.to_description()
            logger.warning(
                "[Planner] Action '%s' repeated 3+ times on same screen — blocking repeat.",
                repeated_desc[:60],
            )
            # Record as failure so it appears in relevant_failures context for the LLM,
            # preventing the model from proposing the same ineffective action again.
            state.record_repeated_action_failure(action=action)

            # Store the rejected analysis metadata so vision.analyze() can build
            # a multi-turn conversation history on the next GROUND→ANALYZE cycle.
            # The LLM will see its own rejected tool call as a prior model turn.
            state.set_rejection_history(
                self.__vision.build_rejection_history_from_analysis(
                    analysis=analysis,
                    rejection_reason=(
                        f"REJECTED: '{repeated_desc}' was repeated 3+ times on the same screen "
                        "without progress. You MUST choose a completely different action."
                    ),
                )
            )

            # Signal the blocked action via PlanResult so the graph node can
            # handle guidance injection (planner should not mutate context state).
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                memories=analysis.memories,
                reason=CompletionReason.ACTION_BLOCKED.value,
                metadata={
                    **(analysis.metadata or {}),
                    "blocked_action": repeated_desc,
                },
            )

        # ── Intent completion check (non-sub-goal path only) ──
        # When sub-goals are defined, completion is driven entirely by the RECORD node
        # after each action executes. The planner must not short-circuit here — doing so
        # would return step=None and the proposed action would never reach EXECUTE.
        if not state.has_sub_goals():
            current_sub_goal = state.get_current_sub_goal()
            current_sub_goal_text = current_sub_goal.description if current_sub_goal else None
            completion = reasoner.analyze_completion(
                analysis=analysis,
                screen_description=capture.activity,
                current_sub_goal=current_sub_goal_text,
            )

            if completion.is_complete:
                state.mark_complete(reason=completion.evidence)

                return PlanResult(
                    step=None,
                    is_complete=True,
                    metrics=analysis.metrics,
                    reason=analysis.reasoning,
                    metadata=analysis.metadata,
                    memories=analysis.memories,
                )

        # Record action for sub-goal trace verification
        if state.has_sub_goals() and action.action_type not in {
            ActionType.WAIT,
            ActionType.ASK_USER,
        }:
            state.record_sub_goal_action()

        step_metadata: Dict[str, Any] = {}
        if action.action_type is ActionType.VALIDATE and (
            analysis.is_sub_goal_complete or analysis.is_goal_complete
        ):
            step_metadata["terminal_validation_candidate"] = True

        return self.__build_plan_result(
            action=action,
            capture=capture,
            metrics=analysis.metrics,
            memories=analysis.memories,
            step_number=state.step_count,
            step_metadata=step_metadata,
            metadata={
                **(analysis.metadata or {}),
                PlanMetadataKey.OBSERVATION.value: analysis.screen_description,
                # Pass analysis to RECORD node for post-execution sub-goal completion check.
                PlanMetadataKey.ANALYSIS.value: analysis,
            },
        )

    def __select_action(
        self,
        state: AgentState,
        reasoner: Reasoner,
        analysis: AnalysisResult,
    ) -> Action:
        """
        Return the best action from the analysis result.
        """

        context = state.build_context()
        failures_raw = context.get("relevant_failures", [])
        failures = failures_raw if isinstance(failures_raw, list) else []

        return reasoner.select_best_action(
            primary=analysis.action,
            alternatives=analysis.alternatives,
            failed_actions={str(failure) for failure in failures},
        )

    def __log_escalation_detected(
        self,
        *,
        path: str,
        state: AgentState,
        evidence: "Any",
        source: StuckSource,
    ) -> None:
        """
        Emit a structured ``planner.escalation.detected`` event with the
        full field set the escalation gate is about to evaluate against.
        """

        current = state.get_current_sub_goal()
        recent_effects = [turn.effect_status.value for turn in evidence.since_progress]

        logger.info(
            "Escalation evaluation entered",
            extra={
                "component": "core.agent.planner",
                "event": "planner.escalation.detected",
                "escalation.path": path,
                "escalation.stuck_source": source.value,
                "escalation.deferrals_before": state.deferral_count,
                "loop.stuck": evidence.stuck,
                "loop.reason": evidence.reason.value,
                "loop.since_progress.count": len(evidence.since_progress),
                "loop.since_progress.effects": recent_effects,
                "loop.recent.count": len(evidence.recent),
                "sub_goal.index": current.index if current else None,
                "sub_goal.action_count": state.current_sub_goal_action_count,
                "sub_goal.max_steps": current.max_steps if current else None,
            },
        )

    def __decide_llm_emitted_ask_user(
        self,
        *,
        state: AgentState,
    ) -> Optional[EscalationDecision]:
        """
        Gate an LLM-emitted ``ASK_USER`` action through the escalation policy.

        Returns the gate's decision when a stuck source is active, signalling
        the caller to either allow (clear deferrals, keep the action) or
        defer (record the deferral, inject guidance, and request a re-plan).

        Returns ``None`` when no source is active — the model is asking for
        legitimate external information (credentials, OTP, ambiguity) and the
        request is unrelated to the loop/budget pattern this gate protects.
        """

        escalation_source: Optional[StuckSource] = self.__stuck_source_resolver.resolve(
            agent_state=state,
        )

        if escalation_source is None:
            return None

        return self.__escalation_gate.decide(
            source=escalation_source,
            evidence=state.loop_evidence(),
            deferrals=state.deferral_count,
        )

    @staticmethod
    def __current_screen_repeat_reason(
        *,
        action: Action,
        analysis: AnalysisResult,
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
            if StepPlanner.__describes_same_target(action=action, previous=previous_target):
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

    def __build_plan_result(
        self,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        *,
        memories: int = 0,
        is_recovery: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        step_metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> PlanResult:
        """
        Return a PlanResult with the given action and metadata.
        """

        step: Step = self.__build_step(
            action=action,
            capture=capture,
            is_recovery=is_recovery,
            step_number=step_number,
            event_type=(metadata or {}).get("event_type"),
            metadata=step_metadata,
        )

        return PlanResult(
            step=step,
            is_complete=False,
            memories=memories,
            metrics=metrics or {},
            metadata=metadata or {},
            is_valid_action=action.is_valid,
            validation_reasoning=action.validation_reason,
            reason=action.rationale or ("Step planned" if not is_recovery else "Recovery step"),
        )

    def __build_step(
        self,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        is_recovery: bool = False,
        event_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Step:
        """
        Helper to construct a Step object.
        """

        screen_hash: str = self.__compute_simple_hash(capture=capture)

        validated_event_type: Optional[Literal["action", "validation"]] = None

        if event_type == "action":
            validated_event_type = "action"

        elif event_type == "validation":
            validated_event_type = "validation"

        return Step(
            action=action,
            screen_hash=screen_hash,
            step_number=step_number,
            is_conditional=is_recovery,
            event_type=validated_event_type,
            condition="recovery" if is_recovery else None,
            metadata=metadata or {},
        )

    def __compute_simple_hash(self, *, capture: ScreenCapture) -> str:
        """
        Return a stable screen identity for the current capture.
        """

        if capture.state is not None and capture.state.visual_hash:
            return capture.state.visual_hash[:16]

        logger.warning("Screen capture state missing visual_hash; falling back to activity hash")

        return hashlib.md5(
            capture.activity.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:16]
