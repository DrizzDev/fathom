from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, cast

from fathom.constants.runtime import DEFAULT_VERIFICATION_REJECTION_LIMIT
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.prompts.templates import (
    SUBGOAL_VERIFICATION_SYSTEM,
    VERIFICATION_SYSTEM,
    build_intent_verification_user_prompt,
    build_subgoal_verification_user_prompt,
)
from fathom.schemas.artifact import ArtifactRecord, VerificationPayload
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.tasks import ExecutionTaskState
from fathom.strategies.graph.state import IntentGraphState
from fathom.utils.parsing import strip_code_fences

if TYPE_CHECKING:
    from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider

logger = logging.getLogger(__name__)


class VerifyNode:
    """
    VERIFY graph node; checks final intent satisfaction.
    """

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider

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
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        agent_state = self.__provider.context.agent_state
        current_sub_goal = agent_state.get_current_sub_goal()
        is_subgoal_verify = (
            current_sub_goal is not None
            and agent_state.has_sub_goals()
            and not agent_state.all_sub_goals_complete()
        )

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
            logger.error(
                f"Screen capture failed: {exception}",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)

            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        # 2. Construct binary validation prompt. When sub-goals are active,
        # verify the current sub-goal before advancing; verify the full intent
        # only once the sub-goal chain is complete.
        user_guidance = [
            guidance.content
            for guidance in self.__provider.context.context_manager.get_user_guidance()
        ]
        if is_subgoal_verify and current_sub_goal is not None:
            system_prompt = SUBGOAL_VERIFICATION_SYSTEM
            recent_trace = self.__provider.context.context_manager.get_full_context().get(
                "trace", []
            )
            user_prompt = build_subgoal_verification_user_prompt(
                intent=current_sub_goal.description,
                user_guidance=user_guidance,
                recent_trace=(recent_trace if isinstance(recent_trace, (list, tuple)) else []),
            )
            logger.info(
                "Verifying sub-goal %s: %s",
                current_sub_goal.index,
                current_sub_goal.description[:80],
            )
        else:
            system_prompt = VERIFICATION_SYSTEM
            user_prompt = build_intent_verification_user_prompt(
                intent=self.__provider.context.intent,
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
            logger.error(
                f"LLM verification failed: {exception}",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            is_truly_complete = False
            reason = f"Verification failed due to error: {exception}"

        duration = time.time() - start_time
        logger.info(
            f"Verification finished in {duration:.2f}s: is_complete={is_truly_complete}, reason={reason}",
            extra={
                "component": "graph.intent.verify",
                "event": "verify.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        await self.__emit_verification_artifact(
            capture=capture,
            complete=is_truly_complete,
            reason=reason,
        )

        if is_truly_complete and is_subgoal_verify and current_sub_goal is not None:
            has_more = agent_state.mark_current_sub_goal_complete(
                completion_signal=SubGoalCompletionSignal(
                    evidence=f"Verified by screenshot: {reason}",
                    flagged_complete=True,
                    rationale_verified=False,
                    action_executed=True,
                    screen_verified=True,
                    llm_confidence=1.0,
                )
            )
            agent_state.clear_verification_loop()
            agent_state.reset_completion()
            self.__provider.context.context_manager.clear_verifier_feedback()

            if has_more:
                next_sub_goal = agent_state.get_current_sub_goal()
                logger.info(
                    "Sub-goal %s verified; advancing to sub-goal %s",
                    current_sub_goal.index,
                    next_sub_goal.index if next_sub_goal is not None else None,
                )
                result = cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: False,
                        IntentStateKey.SHOULD_RETRY: True,
                        IntentStateKey.PLAN: None,
                        IntentStateKey.PLANNED_STEP: None,
                        CommonStateKey.COMPLETION_REASON: None,
                    },
                )
                self.__provider.persistence.persist(result=result)
                return result

            agent_state.mark_complete(reason="All sub-goals completed and verified sequentially")
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: (
                        "All sub-goals completed and verified sequentially"
                    ),
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        if is_truly_complete:
            self.__provider.context.agent_state.clear_verification_loop()
            self.__provider.context.agent_state.mark_complete(reason=reason)
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: reason,
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
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.MAX_STEPS.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        self.__provider.context.agent_state.reset_completion()
        loop_state = self.__provider.context.agent_state.record_verify_rejection(
            screen=self.__provider.context.agent_state.current_screen,
            activity=capture.activity,
        )
        if loop_state.consecutive_rejections >= DEFAULT_VERIFICATION_REJECTION_LIMIT:
            logger.warning(
                "Verifier rejected completion repeatedly on the same step/screen; terminating",
                extra={
                    "component": "graph.intent.verify",
                    "event": "verify.loop.terminated",
                    "workflow.id": self.__provider.context.workflow_id,
                    "consecutive.rejections": loop_state.consecutive_rejections,
                    "limit": DEFAULT_VERIFICATION_REJECTION_LIMIT,
                },
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.STUCK.value)
            result = cast(
                "IntentGraphState",
                {
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

        feedback = f"Verification failed: {reason}"
        logger.warning(
            f"{feedback}",
            extra={
                "component": "graph.intent.verify",
                "event": "verify.log",
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
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.SHOULD_RETRY: True,
                IntentStateKey.PLAN: None,
                IntentStateKey.PLANNED_STEP: None,
                CommonStateKey.COMPLETION_REASON: None,
            },
        )
        self.__provider.persistence.persist(result=result)
        return result

    async def __emit_verification_artifact(
        self,
        *,
        capture: ScreenCapture,
        complete: bool,
        reason: str,
    ) -> None:
        """
        Hand the verifier capture + verdict to the artifact pipeline.
        """

        pipeline = self.__provider.context.artifact_pipeline
        if pipeline is None:
            return

        verdict = CompletionVerdict(
            complete=complete,
            next_state=ExecutionTaskState.SUCCEEDED if complete else ExecutionTaskState.FAILED,
            reason=reason,
            missing=[],
        )
        await pipeline.emit(
            record=ArtifactRecord(
                session_id=self.__provider.context.workflow_id,
                package_name=capture.activity,
                step_number=self.__provider.context.agent_state.step_count,
                created=int(time.time() * 1000),
                payload=VerificationPayload(capture=capture, verdict=verdict),
            ),
        )
