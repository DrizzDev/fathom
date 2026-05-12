from __future__ import annotations

import asyncio
import contextlib
import uuid
from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants import DRAIN_TIMEOUT
from fathom.core.context.engines.gcc import GitContextEngine
from fathom.interfaces.context import ContextEngine
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.summarization import SummarizationPort
from fathom.schemas.actions import Action
from fathom.schemas.feedback import UserGuidance, VerifierFeedback

logger = getLogger(__name__)


class ContextManager:
    """
    Coordinator for the agent context and memory lifecycle.

    Responsibilities:
    - HITL guidance injection.
    - Distributed state persistence (via MemoryPort).
    - Background summarization management (Zero Latency).
    - Delegation of versioning/branching to a ContextEngine.
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
        Initialize the manager.

        Args:
            memory: Distributed persistence port.
            workflow_id: Unique session identifier.
            engine: Memory construction strategy (defaults to GCC).
            summarizer: Intelligence port for semantic compression.
        """

        self.__memory = memory
        self.__engine = engine or GitContextEngine()

        self.__summarizer = summarizer
        self.__workflow_id = workflow_id or uuid.uuid4().hex[:8]

        # Tier 1: Immutable Roadmap
        self.__roadmap_intent: str = "unknown"

        # User-sourced instructions (run-scoped, persists until run end)
        self.__user_guidance: List[UserGuidance] = []

        # System-sourced verifier rejection messages (use-once, planner clears after consuming for the next planning iteration)
        self.__verifier_feedback: List[VerifierFeedback] = []

        # Async Lifecycle
        self.__background_tasks: set[asyncio.Task[None]] = set()

        # Persistence Queue for Non-Blocking I/O.
        # Currently DISABLED: GCC context is not persisted to Ledger (see
        # __persistence_worker docstring). Fields kept so call sites and
        # shutdown logic stay valid; re-enable by uncommenting the spawn below.
        self.__persistence_task: Optional[asyncio.Task[None]] = None
        self.__persist_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        # self.__start_persistence_loop()

    def __start_persistence_loop(self) -> None:
        """
        Starts the background worker for state persistence.
        """

        loop = asyncio.get_running_loop()
        self.__persistence_task = loop.create_task(self.__persistence_worker())

        # self.__background_tasks.add(self.__persistence_task)
        # self.__persistence_task.add_done_callback(self.__background_tasks.discard)

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
                # GCC context is NOT persisted to Ledger by design.
                # Ledger is reserved for user-actionable memory only.
                # If you need GCC persistence, implement separate context storage
                # by replacing the no-op below with serialization + memory.set.
                logger.debug(
                    "[ContextManager] skipping GCC persistence to Ledger",
                    extra={
                        "component": "context",
                        "event": "persist_skipped",
                        "workflow_id": self.__workflow_id,
                        "state_keys": list(state_data.keys()),
                    },
                )
                # Reference implementation for when separate context storage is added:
                #     json_data = await asyncio.to_thread(json.dumps, state_data)
                #     await self.__memory.set(
                #         key=f"context:v3:{self.__workflow_id}",
                #         value=json_data,
                #     )
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

        # GCC context is NOT loaded from Ledger
        # Each session starts fresh, If GCC persistence is required, implement separate context storage

        logger.info(
            f"[ContextManager] Starting fresh session | "
            f"workflow_id={self.__workflow_id} | "
            f"gcc_persistence=disabled"
        )

        """
        try:
            if state_raw := await self.__memory.get(key=f"context:v3:{self.__workflow_id}"):
                data = await asyncio.to_thread(json.loads, state_raw)
                self.__roadmap_intent = data.get("intent", "unknown")
                self.__user_guidance = [
                    UserGuidance(**guidance) for guidance in data.get("guidance", [])
                ]
                # Delegate engine hydration
                await self.__engine.hydrate(data=data.get("engine", {}))

            logger.info(f"Context: Hydrated session {self.__workflow_id}")
        except Exception as exception:
            logger.error(f"Context: Hydration failure: {exception}")
        """

    async def __enqueue_persist(self) -> None:
        """
        Captures a snapshot of the current state and queues it for persistence.
        This operation is O(1) in-memory and non-blocking.

        DISABLED: persistence worker is not spawned (see __init__). Returning
        early avoids growing __persist_queue unbounded with snapshots that no
        consumer will drain. Re-enable in lockstep with __start_persistence_loop.
        """

        return
        # Reference snapshot composition for when persistence is re-enabled:
        #     try:
        #         state_data = {
        #             "intent": self.__roadmap_intent,
        #             "engine": self.__engine.dehydrate(),
        #             "guidance": [g.model_dump() for g in self.__user_guidance],
        #         }
        #         self.__persist_queue.put_nowait(state_data)
        #     except Exception as exception:
        #         logger.error(
        #             "[ContextManager] failed to enqueue persistence",
        #             extra={
        #                 "component": "context",
        #                 "event": "enqueue_failed",
        #                 "workflow_id": self.__workflow_id,
        #                 "error": str(exception),
        #             },
        #         )

    async def commit(self, *, observation: str, thought: str, action: Action) -> None:
        """
        Record an atomic reasoning cycle.
        """

        # Clean action dump
        action_data = action.model_dump() if hasattr(action, "model_dump") else {"raw": str(action)}

        await self.__engine.record(observation=observation, thought=thought, action=action_data)

        # Log trace length after record
        trace_len = len(self.__engine.get_context().get("trace", []))
        logger.info(f"[ContextManager] After record: trace_length={trace_len}")

        # Non-blocking persist
        await self.__enqueue_persist()

    async def branch(self) -> None:
        """
        Triggers non-blocking semantic compression (The GCC COMMIT logic).
        Offloads summarization to a background task to maintain Zero Latency (P0).
        """

        logger.info(
            f"[ContextManager] branch() called, trace_length_before={len(self.__engine.get_context().get('trace', []))}"
        )

        # 1. Prepare engine for background work
        # (GitContextEngine moves active log to shadow buffer)
        if not hasattr(self.__engine, "prepare_summarization"):
            # If engine doesn't support semantic branching, we do nothing or simple commit
            return

        engine_with_summarization = cast("Any", self.__engine)
        segment = engine_with_summarization.prepare_summarization()
        if not segment:
            return

        logger.info(
            f"[ContextManager] After prepare_summarization: segment_length={len(segment)}, trace_length={len(self.__engine.get_context().get('trace', []))}"
        )

        # 2. Persist structural change immediately (non-blocking)
        await self.__enqueue_persist()

        # 3. Offload intelligence
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
            "guidance": [entry.content for entry in self.__user_guidance],
            "verifier_feedback": [entry.content for entry in self.__verifier_feedback],
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
        Return the current user-guidance entries (copy).
        """

        return self.__user_guidance.copy()

    def clear_user_guidance(self) -> None:
        """
        Drop all user-guidance entries (e.g. on explicit revoke).
        """

        self.__user_guidance.clear()

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

        # 1. Wait for in-flight summarization tasks with bounded timeout
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

        # 2. Drain the persistence queue
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
