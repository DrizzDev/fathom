from __future__ import annotations

import asyncio
import contextlib
import uuid
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, cast

from fathom.constants import DRAIN_TIMEOUT
from fathom.core.context.engines.gcc import GitContextEngine
from fathom.interfaces.context import ContextEngine
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.summarization import SummarizationPort
from fathom.schemas.actions import Action
from fathom.schemas.feedback import (
    CompletionFeedback,
    UserGuidance,
    VerifierFeedback,
)

logger = getLogger(__name__)


class ContextManager:
    """
    Coordinates the agent's context and memory lifecycle: human-in-the-loop guidance injection,
    state persistence through the :class:`MemoryPort`, non-blocking background summarization, and
    delegation of versioning and branching to a :class:`ContextEngine`.
    """

    def __init__(
        self,
        *,
        memory: MemoryPort,
        workflow_id: Optional[str] = None,
        engine: Optional[ContextEngine] = None,
        summarizer: Optional[SummarizationPort] = None,
    ) -> None:
        """
        Wire the manager to its persistence port and, when omitted, default the context engine to
        GCC; ``summarizer`` supplies the semantic-compression backend for background distillation.
        """

        self.__memory = memory
        self.__engine = engine or GitContextEngine()

        self.__summarizer = summarizer
        self.__workflow_id = workflow_id or uuid.uuid4().hex

        # Tier 1: Immutable Roadmap
        self.__roadmap_intent: str = "unknown"

        # User-sourced instructions. Each entry has a small ANALYZE-turn
        # TTL so a human correction survives one ignored model turn but
        # cannot become a permanent stale instruction.
        self.__user_guidance: List[UserGuidance] = []

        # System-sourced verifier rejection messages (use-once, planner clears after consuming for the next planning iteration)
        self.__verifier_feedback: List[VerifierFeedback] = []

        # System-sourced completion refute reasons (use-once, planner clears after consuming next iteration)
        self.__completion_feedback: List[CompletionFeedback] = []

        # Async Lifecycle
        self.__background_tasks: Set[asyncio.Task[None]] = set()

        # Persistence queue for non-blocking I/O. Currently disabled: GCC context is not persisted to Ledger
        # (see __persistence_worker). Fields kept so call sites and shutdown logic stay valid.
        self.__persistence_task: Optional[asyncio.Task[None]] = None
        self.__persist_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def __start_persistence_loop(self) -> None:
        """
        Starts the background worker for state persistence.
        """

        loop = asyncio.get_running_loop()
        self.__persistence_task = loop.create_task(self.__persistence_worker())

    async def __persistence_worker(self) -> None:
        """
        Background worker that drains the persistence queue.

        NOTE: GCC context persistence to Ledger is DISABLED.

        Rationale:
        - GCC context is internal system state, not user-actionable memory
        - Storing it in Ledger causes memory pollution (thousands of tokens)
        - GCC context is already available in-memory via get_full_context()
        - If persistence is needed, use separate storage mechanism (not Ledger)

        The queue draining logic is kept for potential future use with separate context storage.
        """

        while True:
            try:
                state_data = await self.__persist_queue.get()
            except asyncio.CancelledError:
                break

            try:
                # GCC context is not persisted to Ledger by design; Ledger is reserved for user-actionable
                # memory only. To add GCC persistence, serialize state_data into a separate context store.
                logger.info(
                    "[ContextManager] skipping GCC persistence to Ledger",
                    extra={
                        "component": "context",
                        "event": "persist_skipped",
                        "workflow_id": self.__workflow_id,
                        "state_keys": list(state_data.keys()),
                    },
                )
            except Exception as exception:
                logger.error(
                    "[ContextManager] background persistence failure",
                    extra={
                        "component": "context",
                        "error": str(exception),
                        "event": "persist_failed",
                        "workflow_id": self.__workflow_id,
                    },
                )
            finally:
                # Always mark the item done — whether the work succeeded, was
                # cancelled, or raised — otherwise queue.join() in shutdown hangs.
                self.__persist_queue.task_done()

    async def hydrate(self) -> None:
        """
        Restores the entire context hierarchy from the distributed store.

        NOTE: GCC context hydration from Ledger is DISABLED.

        Rationale:
        - GCC context is NOT stored in Ledger (see __persistence_worker)
        - Each session starts with fresh GCC context
        - If persistence is needed, implement separate context storage
        """

        # GCC context is not loaded from Ledger; each session starts fresh. If GCC persistence is required,
        # implement a separate context store.
        logger.info(
            f"[ContextManager] Starting fresh session | "
            f"workflow_id={self.__workflow_id} | "
            f"gcc_persistence=disabled"
        )

    async def __enqueue_persist(self) -> None:
        """
        Captures a snapshot of the current state and queues it for persistence.
        This operation is O(1) in-memory and non-blocking.

        DISABLED: persistence worker is not spawned (see __init__). Returning
        early avoids growing __persist_queue unbounded with snapshots that no
        consumer will drain. Re-enable in lockstep with __start_persistence_loop.
        """

        return

    async def commit(self, *, observation: str, thought: str, action: Action) -> None:
        """
        Record an atomic reasoning cycle.
        """

        action_data = action.model_dump() if hasattr(action, "model_dump") else {"raw": str(action)}

        await self.__engine.record(observation=observation, thought=thought, action=action_data)

        trace_len = len(self.__engine.get_context().get("trace", []))
        logger.info(f"[ContextManager] After record: trace_length={trace_len}")

        await self.__enqueue_persist()

    async def branch(self) -> None:
        """
        Trigger non-blocking semantic compression (the GCC commit logic), offloading
        summarization to a background task to keep foreground latency low.
        """

        logger.info(
            f"[ContextManager] branch() called, trace_length_before={len(self.__engine.get_context().get('trace', []))}"
        )

        # GitContextEngine moves the active log to the shadow buffer for background summarization.
        if not hasattr(self.__engine, "prepare_summarization"):
            # Engines without semantic branching skip compression entirely.
            return

        engine_with_summarization = cast("Any", self.__engine)
        segment = engine_with_summarization.prepare_summarization()
        if not segment:
            return

        logger.info(
            f"[ContextManager] After prepare_summarization: segment_length={len(segment)}, trace_length={len(self.__engine.get_context().get('trace', []))}"
        )

        await self.__enqueue_persist()

        if not self.__summarizer:
            await self.__engine.commit(summary=f"Captured {len(segment)} steps.")
            logger.info(
                f"[ContextManager] After commit (no summarizer): trace_length={len(self.__engine.get_context().get('trace', []))}"
            )
            await self.__enqueue_persist()
            return

        task = asyncio.create_task(self.__async_summarize(segment=segment))
        self.__background_tasks.add(task)
        task.add_done_callback(self.__background_tasks.discard)

    async def __async_summarize(self, *, segment: List[Dict[str, Any]]) -> None:
        """
        Background worker for semantic distillation.
        """

        try:
            logger.info(
                f"[ContextManager] __async_summarize() started: segment_length={len(segment)}"
            )

            if self.__summarizer:
                summary = await self.__summarizer.summarize_trace(trace=segment)
                logger.info("[ContextManager] Summarization complete, calling commit()")
                await self.__engine.commit(summary=summary)
            else:
                await self.__engine.commit(summary=f"Captured {len(segment)} steps.")

            logger.info(
                f"[ContextManager] After commit in __async_summarize: trace_length={len(self.__engine.get_context().get('trace', []))}"
            )
            await self.__enqueue_persist()
        except Exception as exception:
            logger.error(f"Context: Background summarization failed: {exception}")
            await self.__engine.commit(summary=f"Completed {len(segment)} steps.")
            await self.__enqueue_persist()

    def get_full_context(self) -> Dict[str, Any]:
        """
        Assembles the final context payload for the LLM.
        """

        engine_context = self.__engine.get_context()

        return {
            "intent": self.__roadmap_intent,
            "trace": engine_context.get("trace", []),
            "milestones": engine_context.get("milestones", []),
            "active_count": engine_context.get("active_count", 0),
            "guidance": [entry.render() for entry in self.__active_user_guidance()],
            "verifier_feedback": [entry.content for entry in self.__verifier_feedback],
            "completion_feedback": [entry.content for entry in self.__completion_feedback],
        }

    def set_roadmap(self, *, intent: str) -> None:
        """
        Set Tier 1 Roadmap.
        """

        self.__roadmap_intent = intent

    async def inject_user_guidance(self, *, guidance: str, step: Optional[int] = None) -> None:
        """
        Append a real-user instruction to the run-scoped user-guidance channel.
        """

        self.__user_guidance.append(UserGuidance(content=guidance, step_number=step))
        await self.__enqueue_persist()

    def get_user_guidance(self) -> List[UserGuidance]:
        """
        Return active user-guidance entries.
        """

        return self.__active_user_guidance()

    def consume_user_guidance(self) -> None:
        """
        Age active user guidance after one planner exposure.
        """

        self.__user_guidance = [
            entry.consume() if entry.active else entry for entry in self.__user_guidance
        ]

    def clear_user_guidance(self) -> None:
        """
        Drop all user-guidance entries (e.g. on explicit revoke).
        """

        self.__user_guidance.clear()

    def __active_user_guidance(self) -> List[UserGuidance]:
        """
        Return active guidance entries in injection order.
        """

        return [entry for entry in self.__user_guidance if entry.active]

    async def inject_verifier_feedback(self, *, feedback: str, step: Optional[int] = None) -> None:
        """
        Append a verifier rejection reason to the use-once verifier-feedback
        channel. The planner consumes and clears this on the next iteration.
        """

        self.__verifier_feedback.append(VerifierFeedback(content=feedback, step_number=step))
        await self.__enqueue_persist()

    def get_verifier_feedback(self) -> List[VerifierFeedback]:
        """
        Return the current verifier-feedback entries (copy).
        """

        return self.__verifier_feedback.copy()

    def clear_verifier_feedback(self) -> None:
        """
        Drop all verifier-feedback entries; called by the planner after one
        consumption cycle so the next VERIFY round produces fresh evidence.
        """

        self.__verifier_feedback.clear()

    async def inject_completion_feedback(
        self, *, feedback: str, step: Optional[int] = None
    ) -> None:
        """
        Append a vision refute reason to the use-once completion-feedback channel; the planner
        consumes and clears it on the next iteration.
        """

        self.__completion_feedback.append(CompletionFeedback(content=feedback, step_number=step))
        await self.__enqueue_persist()

    def get_completion_feedback(self) -> List[CompletionFeedback]:
        """
        Return the current completion-feedback entries (copy).
        """

        return self.__completion_feedback.copy()

    def clear_completion_feedback(self) -> None:
        """
        Drop all completion-feedback entries; called by the planner after one consumption cycle.
        """

        self.__completion_feedback.clear()

    @property
    def workflow_id(self) -> str:
        """
        Unique session ID.
        """

        return self.__workflow_id

    async def shutdown(self) -> None:
        """
        Gracefully shuts down all background tasks and the persistence worker.
        """

        pending = [task for task in self.__background_tasks if not task.done()]
        if pending:
            logger.info(
                f"[ContextManager] awaiting {len(pending)} background tasks (timeout={DRAIN_TIMEOUT}s)"
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=DRAIN_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[ContextManager] background task drain timed out, cancelling remaining"
                )
                for task in pending:
                    if not task.done():
                        task.cancel()

        if self.__persistence_task:
            if not self.__persist_queue.empty():
                try:
                    await asyncio.wait_for(self.__persist_queue.join(), timeout=DRAIN_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning("[ContextManager] persistence queue drain timed out")

            self.__persistence_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.__persistence_task

        logger.info(f"[ContextManager] workflow={self.__workflow_id} shutdown complete")
