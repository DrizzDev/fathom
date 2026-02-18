from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fathom.core.context.engines.gcc import GitContextEngine
from fathom.interfaces.context import ContextEngine
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.summarization import SummarizationPort
from fathom.schemas.actions import Action
from fathom.schemas.context import UserGuidance

logger = logging.getLogger(__name__)


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

        # External interventions
        self.__user_guidance: List[UserGuidance] = []

        # Async Lifecycle
        self.__background_tasks: set[asyncio.Task[None]] = set()

        # Persistence Queue for Non-Blocking I/O
        self.__persistence_task: Optional[asyncio.Task[None]] = None
        self.__persist_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        self.__start_persistence_loop()

    def __start_persistence_loop(self) -> None:
        """
        Starts the background worker for state persistence.
        """

        loop = asyncio.get_running_loop()
        self.__persistence_task = loop.create_task(self.__persistence_worker())

        self.__background_tasks.add(self.__persistence_task)
        self.__persistence_task.add_done_callback(self.__background_tasks.discard)

    async def __persistence_worker(self) -> None:
        """
        Background worker that drains the persistence queue.
        Ensures main execution loop is never blocked by I/O.
        """

        while True:
            try:
                # Wait for next state snapshot
                state_data = await self.__persist_queue.get()

                # Perform CPU-bound serialization in thread
                json_data = await asyncio.to_thread(json.dumps, state_data)

                # Perform I/O
                await self.__memory.set(
                    key=f"context:v3:{self.__workflow_id}",
                    value=json_data,
                )
                self.__persist_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as exception:
                logger.error(f"Context: Background persistence failure: {exception}")

    async def hydrate(self) -> None:
        """
        Restores the entire context hierarchy from the distributed store.
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

    async def __enqueue_persist(self) -> None:
        """
        Captures a snapshot of the current state and queues it for persistence.
        This operation is O(1) in-memory and non-blocking.
        """

        try:
            # Snapshot state immediately (Deep copy/Dehydration happens here)
            # We dehydrate on the main thread to ensure consistency, but writing to DB happens in background.
            state_data = {
                "intent": self.__roadmap_intent,
                "engine": self.__engine.dehydrate(),
                "guidance": [guidance.model_dump() for guidance in self.__user_guidance],
            }
            # Push snapshot to queue
            self.__persist_queue.put_nowait(state_data)
        except Exception as exception:
            logger.error(f"Context: Failed to enqueue persistence: {exception}")

    async def commit(self, *, observation: str, thought: str, action: Action) -> None:
        """
        Record an atomic reasoning cycle.
        """

        # Clean action dump
        action_data = action.model_dump() if hasattr(action, "model_dump") else {"raw": str(action)}

        await self.__engine.record(observation=observation, thought=thought, action=action_data)
        # Non-blocking persist
        await self.__enqueue_persist()

    async def branch(self) -> None:
        """
        Triggers non-blocking semantic compression (The GCC COMMIT logic).
        Offloads summarization to a background task to maintain Zero Latency (P0).
        """

        # 1. Prepare engine for background work
        # (GitContextEngine moves active log to shadow buffer)
        if not hasattr(self.__engine, "prepare_summarization"):
            # If engine doesn't support semantic branching, we do nothing or simple commit
            return

        segment = self.__engine.prepare_summarization()
        if not segment:
            return

        # 2. Persist structural change immediately (non-blocking)
        await self.__enqueue_persist()

        # 3. Offload intelligence
        if not self.__summarizer:
            await self.__engine.commit(summary=f"Captured {len(segment)} steps.")
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
            if self.__summarizer:
                summary = await self.__summarizer.summarize_trace(trace=segment)
                await self.__engine.commit(summary=summary)
            else:
                await self.__engine.commit(summary=f"Captured {len(segment)} steps.")

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
            "guidance": [guidance.content for guidance in self.__user_guidance],
        }

    def set_roadmap(self, *, intent: str) -> None:
        """
        Set Tier 1 Roadmap.
        """

        self.__roadmap_intent = intent

    async def inject_user_guidance(self, *, guidance: str, step: Optional[int] = None) -> None:
        """
        Inject priority HITL instruction.
        """

        self.__user_guidance.append(UserGuidance(content=guidance, step_number=step))
        await self.__enqueue_persist()

    def get_user_guidance(self) -> List[UserGuidance]:
        """
        Retrieve active instructions.
        """

        return self.__user_guidance.copy()

    def clear_user_guidance(self) -> None:
        """
        Reset guidance buffer.
        """

        self.__user_guidance.clear()

    @property
    def workflow_id(self) -> str:
        """
        Unique session ID.
        """

        return self.__workflow_id

    async def shutdown(self) -> None:
        """
        Gracefully shuts down the persistence worker.
        """

        if self.__persistence_task:
            # Wait for queue to drain
            if not self.__persist_queue.empty():
                await self.__persist_queue.join()

            self.__persistence_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.__persistence_task
