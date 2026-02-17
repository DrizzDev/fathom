"""
Application-layer Context Manager.
Orchestrates memory construction by delegating to a ContextEngine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fathom.core.context.engines.gcc import GitContextEngine
from fathom.schemas.context import UserGuidance

if TYPE_CHECKING:
    from fathom.interfaces.context import ContextEngine
    from fathom.interfaces.memory import MemoryPort
    from fathom.interfaces.summarization import SummarizationPort
    from fathom.schemas.actions import Action

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Coordinator for the agent context and memory lifecycle.
    
    Responsibilities:
    - Distributed state persistence (via MemoryPort).
    - Background summarization management (Zero Latency).
    - HITL guidance injection.
    - Delegation of versioning/branching to a ContextEngine.
    """

    def __init__(
        self,
        *,
        memory: MemoryPort,
        engine: Optional[ContextEngine] = None,
        summarizer: Optional[SummarizationPort] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the manager.
        
        Args:
            memory: Distributed persistence port.
            engine: Memory construction strategy (defaults to GCC).
            summarizer: Intelligence port for semantic compression.
            workflow_id: Unique session identifier.
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
        self.__background_tasks: set[asyncio.Task] = set()

    async def hydrate(self) -> None:
        """Restores the entire context hierarchy from the distributed store."""
        try:
            state_raw = await self.__memory.get(key=f"ctx_v3:{self.__workflow_id}")
            if state_raw:
                data = json.loads(state_raw)
                self.__roadmap_intent = data.get("intent", "unknown")
                self.__user_guidance = [
                    UserGuidance(**g) for g in data.get("guidance", [])
                ]
                # Delegate engine hydration
                await self.__engine.hydrate(data=data.get("engine", {}))
                
            logger.info(f"Context: Hydrated session {self.__workflow_id}")
        except Exception as exception:
            logger.error(f"Context: Hydration failure: {exception}")

    async def __persist(self) -> None:
        """Syncs the current state across the distributed system."""
        try:
            state_data = {
                "intent": self.__roadmap_intent,
                "guidance": [g.model_dump() for g in self.__user_guidance],
                "engine": self.__engine.dehydrate()
            }
            await self.__memory.set(
                key=f"ctx_v3:{self.__workflow_id}",
                value=json.dumps(state_data)
            )
        except Exception as exception:
            logger.error(f"Context: Persistence failure: {exception}")

    async def commit(self, *, observation: str, thought: str, action: Action) -> None:
        """Record an atomic reasoning cycle."""
        # Clean action dump
        action_data = action.model_dump() if hasattr(action, "model_dump") else {"raw": str(action)}
        
        await self.__engine.record(
            observation=observation,
            thought=thought,
            action=action_data
        )
        await self.__persist()

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

        # 2. Persist structural change immediately
        await self.__persist()

        # 3. Offload intelligence
        if not self.__summarizer:
            await self.__engine.commit(summary=f"Captured {len(segment)} steps.")
            await self.__persist()
            return

        task = asyncio.create_task(self.__async_summarize(segment=segment))
        self.__background_tasks.add(task)
        task.add_done_callback(self.__background_tasks.discard)

    async def __async_summarize(self, *, segment: List[Dict[str, Any]]) -> None:
        """Background worker for semantic distillation."""
        try:
            summary = await self.__summarizer.summarize_trace(trace=segment)
            await self.__engine.commit(summary=summary)
            await self.__persist()
        except Exception as exception:
            logger.error(f"Context: Background summarization failed: {exception}")
            await self.__engine.commit(summary=f"Completed {len(segment)} steps.")
            await self.__persist()

    def get_full_context(self) -> Dict[str, Any]:
        """Assembles the final context payload for the LLM."""
        engine_context = self.__engine.get_context()
        
        return {
            "intent": self.__roadmap_intent,
            "milestones": engine_context.get("milestones", []),
            "trace": engine_context.get("trace", []),
            "guidance": [g.content for g in self.__user_guidance]
        }

    # --- Domain Commands ---

    def set_roadmap(self, *, intent: str) -> None:
        """Set Tier 1 Roadmap."""
        self.__roadmap_intent = intent

    async def inject_user_guidance(self, *, guidance: str, step: Optional[int] = None) -> None:
        """Inject priority HITL instruction."""
        self.__user_guidance.append(UserGuidance(content=guidance, step_number=step))
        await self.__persist()

    def get_user_guidance(self) -> List[UserGuidance]:
        """Retrieve active instructions."""
        return self.__user_guidance.copy()

    def clear_user_guidance(self) -> None:
        """Reset guidance buffer."""
        self.__user_guidance.clear()

    @property
    def workflow_id(self) -> str:
        """Unique session ID."""
        return self.__workflow_id
