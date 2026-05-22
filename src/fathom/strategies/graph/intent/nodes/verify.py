from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, cast

from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.prompts.templates import VERIFICATION_SYSTEM, VERIFICATION_USER_TEMPLATE
from fathom.core.recovery import (
    RecoveryTrigger,
)
from fathom.schemas.artifact import ArtifactRecord, VerificationPayload
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.state import VerificationLoopPhase
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

        all_sub_goals_done = (
            self.__provider.context.agent_state.has_sub_goals()
            and self.__provider.context.agent_state.all_sub_goals_complete()
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

                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    },
                )
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

            return cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                },
            )

        # 2. Construct binary validation prompt
        intent = self.__provider.context.intent
        system_prompt = VERIFICATION_SYSTEM

        guidance_section = ""
        user_guidance = self.__provider.context.context_manager.get_user_guidance()

        if user_guidance:
            guidance_text = "\n".join([f"- {guidance.content}" for guidance in user_guidance])
            guidance_section = f"\nUser Guidance:\n{guidance_text}\n"

        user_prompt = VERIFICATION_USER_TEMPLATE.format(
            intent=intent, guidance_section=guidance_section
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

        if all_sub_goals_done:
            logger.warning(
                f"LLM rejected completion after local completion gate passed. "
                f"Keeping workflow open. LLM reason: {reason}"
            )
            # Final intent verification owns completion. When it rejects after the
            # last sub-goal was locally marked complete, restore that sub-goal as
            # the active mission so recovery/replanning keep the same contract.
            self.__provider.context.agent_state.reopen_last_completed_sub_goal()

        self.__provider.context.agent_state.reset_completion()
        verification_loop = self.__provider.context.agent_state.record_verify_rejection(
            screen=self.__provider.context.agent_state.current_screen,
            activity=capture.activity,
        )

        if (
            verification_loop.phase is VerificationLoopPhase.RECOVERY_ATTEMPTED
            and verification_loop.consecutive_rejections
            >= self.__provider.context.recovery.verify_threshold
        ):
            diagnostic = (
                "Verification kept rejecting without recorded progress after recovery already ran: "
                f"count={verification_loop.consecutive_rejections}, "
                f"step_count={self.__provider.context.agent_state.step_count}, "
                f"activity={capture.activity!r}, reason={reason}"
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.STUCK.value)
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.FAILURE_DIAGNOSTIC: diagnostic,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.STUCK.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        if (
            verification_loop.consecutive_rejections
            >= self.__provider.context.recovery.verify_threshold
        ):
            self.__provider.context.agent_state.mark_verify_recovery_attempted()

            # Signal the rejection to the recovery coordinator; fall through
            # to the standard rejection path if no strategy commits.
            recovered = await self.__provider.recovery.try_recover(
                reason=reason,
                capture=capture,
                trigger=RecoveryTrigger.VERIFY_REJECTED,
            )
            if recovered is not None:
                return recovered

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
