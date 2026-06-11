from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Optional, Tuple, cast

from fathom.constants.observability import CompletionEvent
from fathom.constants.runtime import DEFAULT_VERIFICATION_REJECTION_LIMIT
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey, VerifyMode
from fathom.core.exceptions import InvariantViolation
from fathom.core.prompts.templates import (
    SUBGOAL_VERIFICATION_SYSTEM,
    VERIFICATION_SYSTEM,
    build_intent_verification_user_prompt,
    build_subgoal_verification_user_prompt,
)
from fathom.schemas.artifact import ArtifactRecord, VerificationPayload
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.tasks import ExecutionTaskState
from fathom.strategies.graph.intent.verification import VerificationModePolicy
from fathom.strategies.graph.state import IntentGraphState
from fathom.utils.parsing import strip_code_fences

if TYPE_CHECKING:
    from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider

logger = logging.getLogger(__name__)

MAX_VERIFICATION_EVIDENCE_CHARS = 500
DEFAULT_ACCEPTED_VERIFICATION_REASON = "Verifier accepted completion without detailed rationale."


class VerifyNode:
    """
    VERIFY graph node; checks final intent satisfaction.
    """

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider
        self.__mode_policy = VerificationModePolicy()

    async def __call__(self, state: IntentGraphState) -> IntentGraphState:
        """
        Run the VERIFY node handler.
        """

        return await self.run(state=state)

    async def run(self, *, state: IntentGraphState) -> IntentGraphState:
        """
        Explicitly verify if the intent is truly complete by capturing the screen and asking the LLM.
        If verification fails, it adds negative feedback and routes back to the main loop.
        """

        self.__provider.persistence.restore(state=state)

        logger.info(
            "Verify node started",
            extra={
                "component": "graph.intent.verify",
                "event": "verify.started",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        if await self.__provider.is_cancelled():
            logger.warning(
                "Execution cancelled",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(
                reason=CompletionReason.CANCELLED.value
            )

            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    CommonStateKey.IS_COMPLETE: True,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        agent_state = self.__provider.context.agent_state
        current_sub_goal = agent_state.get_current_sub_goal()
        try:
            mode = self.__mode_policy.mode_for_verify(state=state, agent_state=agent_state)
            self.__assert_mode_invariant(mode=mode, current_sub_goal=current_sub_goal)
        except InvariantViolation as exception:
            return self.__fail_invariant(exception=exception)

        start_time = time.time()

        # 1. Capture the latest screen state
        try:
            capture = await self.__provider.context.perception.perceive(
                session_id=self.__provider.context.workflow_id,
                step_number=self.__provider.context.agent_state.step_count,
            )

            if not capture.image:
                logger.warning(
                    "Failed to capture screen for verification",
                    extra={
                        "component": "graph.intent.verify",
                        "event": "verify.log",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )
                self.__provider.context.agent_state.mark_complete(
                    reason=CompletionReason.FAILED.value
                )

                result = cast(
                    "IntentGraphState",
                    {
                        IntentStateKey.VERIFY_MODE: None,
                        IntentStateKey.SHOULD_RETRY: False,
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    },
                )
                self.__provider.persistence.persist(result=result)
                return result
        except asyncio.CancelledError:
            # Cooperative cancellation must propagate so the graph
            # unwinds; do not absorb it into a FAILED completion.
            raise
        except Exception as exception:
            logger.exception(
                "Screen capture failed",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.capture.failed",
                    "error.message": str(exception),
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)

            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        # 2. Construct binary validation prompt from explicit VERIFY mode.
        user_guidance = [
            guidance.content
            for guidance in self.__provider.context.context_manager.get_user_guidance()
        ]
        system_prompt, user_prompt = self.__prompt(
            mode=mode,
            current_sub_goal=current_sub_goal,
            user_guidance=user_guidance,
        )

        # 3. Ask the LLM
        try:
            llm_result = await self.__provider.context.llm.generate(
                use_cache=False,
                system_instruction=system_prompt,
                prompt=[user_prompt, capture.image],
            )

            text = strip_code_fences(llm_result.content)
            data = json.loads(text)
            is_truly_complete = bool(data.get("is_complete", False))
            reason = str(data.get("reason", "Verification failed without specific reason."))

        except asyncio.CancelledError:
            # Cooperative cancellation must propagate so the graph
            # unwinds; do not absorb it as a verification failure.
            raise
        except Exception as exception:
            diagnostic = f"LLM verification failed: {exception}"[:500]
            logger.exception(
                "LLM verification failed",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.llm.failed",
                    "error.message": str(exception),
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    CommonStateKey.FAILURE_DIAGNOSTIC: diagnostic,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        duration = time.time() - start_time
        logger.info(
            "Verification finished",
            extra={
                "component": "graph.intent.verify",
                "event": "verify.finished",
                "verdict.reason": reason,
                "verdict.complete": is_truly_complete,
                "duration.seconds": round(duration, 4),
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        # Verification artefact emission suppressed; no downstream reader today.
        # await self.__emit_verification_artifact(capture=capture, complete=is_truly_complete, reason=reason)

        if (
            is_truly_complete
            and mode in {VerifyMode.SUB_GOAL, VerifyMode.PENDING_FINAL_COMMIT}
            and current_sub_goal is not None
        ):
            result = self.__commit_acceptance(current_sub_goal=current_sub_goal, reason=reason)
            self.__provider.persistence.persist(result=result)
            return result

        if is_truly_complete:
            self.__provider.context.agent_state.clear_verification_loop()
            self.__provider.context.agent_state.mark_complete(
                reason=self.__accepted_reason(reason=reason)
            )

            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: self.__accepted_reason(reason=reason),
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        if self.__provider.context.agent_state.step_count >= self.__provider.context.max_steps:
            logger.warning(
                "Verification rejected after max steps reached; terminating workflow",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.max.steps.terminated",
                    "max_steps": self.__provider.context.max_steps,
                    "workflow.id": self.__provider.context.workflow_id,
                    "step_count": self.__provider.context.agent_state.step_count,
                },
            )
            self.__provider.context.agent_state.mark_complete(
                reason=CompletionReason.MAX_STEPS.value
            )
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.MAX_STEPS.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        loop_state = self.__provider.context.agent_state.record_verify_rejection(
            screen=self.__verification_loop_screen(capture=capture),
            activity=capture.activity,
        )
        if loop_state.consecutive_rejections >= DEFAULT_VERIFICATION_REJECTION_LIMIT:
            logger.warning(
                "Verifier rejected completion repeatedly on the same step/screen; terminating",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.loop.terminated",
                    "verify.mode": mode.value,
                    "limit": DEFAULT_VERIFICATION_REJECTION_LIMIT,
                    "workflow.id": self.__provider.context.workflow_id,
                    "consecutive.rejections": loop_state.consecutive_rejections,
                },
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.STUCK.value)
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.STUCK.value,
                    CommonStateKey.FAILURE_DIAGNOSTIC: (
                        f"Verification failed {loop_state.consecutive_rejections} times "
                        "without a new recorded action."
                    ),
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        self.__provider.context.agent_state.reset_completion()
        feedback = f"Verification failed: {reason}"
        logger.warning(
            "Verification rejected completion",
            extra={
                "component": "graph.intent.verify",
                "event": "verify.rejected",
                "feedback": feedback,
                "verify.mode": mode.value,
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        # Route verifier rejection through the typed verifier-feedback
        # channel so the next planner iteration sees it as system feedback
        # — distinct from real user instructions.
        await self.__provider.context.context_manager.inject_verifier_feedback(
            feedback=feedback, step=self.__provider.context.agent_state.step_count
        )

        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.PLAN: None,
                IntentStateKey.VERIFY_MODE: None,
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.SHOULD_RETRY: True,
                IntentStateKey.PLANNED_STEP: None,
                CommonStateKey.COMPLETION_REASON: None,
            },
        )
        self.__provider.persistence.persist(result=result)
        return result

    def __assert_mode_invariant(
        self,
        *,
        mode: VerifyMode,
        current_sub_goal: Optional[SubGoal],
    ) -> None:
        """
        Validate VERIFY mode contracts that cannot be inferred safely.
        """

        if (
            mode in {VerifyMode.SUB_GOAL, VerifyMode.PENDING_FINAL_COMMIT}
            and current_sub_goal is None
        ):
            raise InvariantViolation(f"{mode.value} VERIFY entered without an active sub-goal.")

        if (
            mode is VerifyMode.PENDING_FINAL_COMMIT
            and not self.__provider.context.agent_state.has_active_final_sub_goal()
        ):
            raise InvariantViolation(
                "PENDING_FINAL_COMMIT VERIFY entered while active sub-goal is not final."
            )

        if (
            mode is VerifyMode.SUB_GOAL
            and self.__provider.context.agent_state.has_active_final_sub_goal()
        ):
            raise InvariantViolation(
                "SUB_GOAL VERIFY entered for the active final sub-goal; "
                "use PENDING_FINAL_COMMIT to verify full intent before final commit."
            )

        if (
            mode is VerifyMode.FULL_INTENT
            and self.__provider.context.agent_state.has_sub_goals()
            and not self.__provider.context.agent_state.all_sub_goals_complete()
        ):
            raise InvariantViolation("FULL_INTENT VERIFY entered while sub-goals are still active.")

    def __prompt(
        self,
        *,
        mode: VerifyMode,
        current_sub_goal: Optional[SubGoal],
        user_guidance: list[str],
    ) -> Tuple[str, str]:
        """
        Build the verifier prompt for the explicit VERIFY mode.
        """

        if mode is VerifyMode.SUB_GOAL and current_sub_goal is not None:
            recent_trace = self.__provider.context.context_manager.get_full_context().get(
                "trace", []
            )
            logger.info(
                "Verifying sub-goal",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.subgoal.started",
                    "sub_goal.index": current_sub_goal.index,
                    "workflow.id": self.__provider.context.workflow_id,
                    "sub_goal.description": current_sub_goal.description[:80],
                },
            )
            return (
                SUBGOAL_VERIFICATION_SYSTEM,
                build_subgoal_verification_user_prompt(
                    intent=current_sub_goal.description,
                    user_guidance=user_guidance,
                    recent_trace=(recent_trace if isinstance(recent_trace, (list, tuple)) else []),
                ),
            )

        logger.info(
            "Verifying full intent",
            extra={
                "component": "graph.intent.verify",
                "event": "verify.intent.started",
                "verify.mode": mode.value,
                "workflow.id": self.__provider.context.workflow_id,
                "scope": "full_intent",
                "sub_goal.index": current_sub_goal.index if current_sub_goal else None,
            },
        )
        return (
            VERIFICATION_SYSTEM,
            build_intent_verification_user_prompt(
                intent=self.__provider.context.intent,
                user_guidance=user_guidance,
            ),
        )

    def __commit_acceptance(
        self,
        *,
        reason: str,
        current_sub_goal: SubGoal,
    ) -> IntentGraphState:
        """
        Commit the accepted active sub-goal before marking final intent success.
        """

        agent_state = self.__provider.context.agent_state
        accepted_reason = self.__accepted_reason(reason=reason)
        # PENDING_FINAL_COMMIT emits the final sub-goal lifecycle event from VERIFY, not the gate.
        has_more = agent_state.mark_current_sub_goal_complete(
            completion_signal=SubGoalCompletionSignal(
                llm_confidence=1.0,
                screen_verified=True,
                action_executed=True,
                flagged_complete=True,
                rationale_verified=bool(reason.strip()),
                evidence=self.__verification_evidence(reason=reason),
            )
        )
        agent_state.clear_verification_loop()
        agent_state.reset_completion()
        self.__provider.context.context_manager.clear_verifier_feedback()

        if has_more:
            next_sub_goal = agent_state.get_current_sub_goal()
            logger.info(
                "Sub-goal verified",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.subgoal.accepted",
                    "sub_goal.index": current_sub_goal.index,
                    "verify.mode": VerifyMode.SUB_GOAL.value,
                    "workflow.id": self.__provider.context.workflow_id,
                    "next.sub_goal.index": next_sub_goal.index if next_sub_goal else None,
                },
            )
            return cast(
                "IntentGraphState",
                {
                    IntentStateKey.PLAN: None,
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.PLANNED_STEP: None,
                    CommonStateKey.IS_COMPLETE: False,
                    IntentStateKey.SHOULD_RETRY: True,
                    CommonStateKey.COMPLETION_REASON: None,
                },
            )

        completion_reason = accepted_reason
        agent_state.mark_complete(reason=completion_reason)
        logger.info(
            "Intent completed after VERIFY acceptance",
            extra={
                "component": "graph.intent.verify",
                "event": CompletionEvent.INTENT_COMPLETED.value,
                "sub_goal.index": current_sub_goal.index,
                "workflow.id": self.__provider.context.workflow_id,
                "verify.mode": VerifyMode.PENDING_FINAL_COMMIT.value,
                "sub_goal.description": current_sub_goal.description[:80],
            },
        )
        return cast(
            "IntentGraphState",
            {
                IntentStateKey.VERIFY_MODE: None,
                IntentStateKey.SHOULD_RETRY: False,
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: completion_reason,
            },
        )

    @staticmethod
    def __accepted_reason(*, reason: str) -> str:
        """
        Return the non-empty reason stored for accepted verification.
        """

        return reason.strip() or DEFAULT_ACCEPTED_VERIFICATION_REASON

    @staticmethod
    def __verification_evidence(*, reason: str) -> str:
        """
        Return evidence text that preserves whether the verifier supplied a rationale.
        """

        if not reason.strip():
            return DEFAULT_ACCEPTED_VERIFICATION_REASON

        return f"Verified by screenshot: {reason[:MAX_VERIFICATION_EVIDENCE_CHARS]}"

    def __verification_loop_screen(self, *, capture: ScreenCapture) -> Optional[ScreenState]:
        """
        Return the screen identity used for same-screen verifier-loop accounting.
        """

        if capture.state is not None:
            return capture.state

        observer = getattr(self.__provider, "observer", None)
        if observer is not None:
            try:
                hashes = observer.resolve_capture_hashes(capture=capture, elements=[])
                return cast(
                    "ScreenState",
                    observer.build_screen_state(
                        capture=capture,
                        visual_hash=hashes.visual_hash,
                        xml_hash=hashes.xml_hash,
                        interaction_hash=hashes.interaction_hash,
                    ),
                )
            except Exception as exception:
                logger.warning(
                    "Failed to build verifier-loop screen from fresh capture",
                    extra={
                        "component": "graph.intent.verify",
                        "event": "verify.loop.screen_build_failed",
                        "error.message": str(exception),
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )

        screen = self.__provider.context.agent_state.current_screen
        if screen is None:
            return None

        if screen.activity != capture.activity:
            logger.warning(
                "Skipping verifier-loop screen match because capture activity changed",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.loop.screen_mismatch",
                    "screen.activity": screen.activity,
                    "capture.activity": capture.activity,
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            return None

        return screen

    def __fail_invariant(self, *, exception: InvariantViolation) -> IntentGraphState:
        """
        Convert corrupt VERIFY state into a structured terminal failure.
        """

        diagnostic = str(exception)
        logger.error(
            "VERIFY invariant violation",
            extra={
                "component": "graph.intent.verify",
                "event": "verify.invariant.failed",
                "workflow.id": self.__provider.context.workflow_id,
                "failure.diagnostic": diagnostic,
            },
        )
        self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.VERIFY_MODE: None,
                IntentStateKey.SHOULD_RETRY: False,
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.FAILURE_DIAGNOSTIC: diagnostic,
                CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
            },
        )
        self.__provider.persistence.persist(result=result)
        return result

    async def __emit_verification_artifact(
        self,
        *,
        reason: str,
        complete: bool,
        capture: ScreenCapture,
    ) -> None:
        """
        Hand the verifier capture + verdict to the artifact pipeline.
        """

        pipeline = self.__provider.context.artifact_pipeline
        if pipeline is None:
            return

        verdict = CompletionVerdict(
            missing=[],
            reason=reason,
            complete=complete,
            next_state=ExecutionTaskState.SUCCEEDED if complete else ExecutionTaskState.FAILED,
        )
        await pipeline.emit(
            record=ArtifactRecord(
                package_name=capture.activity,
                created=int(time.time() * 1000),
                session_id=self.__provider.context.workflow_id,
                step_number=self.__provider.context.agent_state.step_count,
                payload=VerificationPayload(capture=capture, verdict=verdict),
            ),
        )
