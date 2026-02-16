"""
Context management implementing GCC-inspired three-tier versioned context.

This module provides the ContextManager which maintains:
- Roadmap: Original intent + milestones
- Milestones: Summaries of completed sub-goals
- Trace: Fine-grained OTA (Observe-Thought-Action) log
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fathom.schemas.orchestration import ExecutionContext, ExecutionRoadmap
from fathom.schemas.context import UserGuidance

if TYPE_CHECKING:
    from fathom.interfaces.memory import MemoryPort
    from fathom.schemas.actions import Action


class ContextManager:
    """
    Manages three-tier versioned context (GCC-inspired).

    Tiers:
    - roadmap: Original intent + milestones (high-level plan)
    - milestones: Summaries of completed sub-goals (compressed progress)
    - trace: Fine-grained OTA log (every Observe-Thought-Action cycle)
    """

    def __init__(self, memory: MemoryPort, workflow_id: Optional[str] = None) -> None:
        """Initialize context manager with memory port."""
        self.__memory = memory
        self.__workflow_id = workflow_id or uuid.uuid4().hex[:8]

        # Core State
        self.__context = ExecutionContext(workflow_id=self.__workflow_id)
        self.__roadmap: Optional[ExecutionRoadmap] = None

        # 3-Tier Context State
        self.__milestones: List[str] = []
        self.__trace: List[Dict[str, Any]] = []
        self.__user_guidance: List[UserGuidance] = []

    def set_roadmap(self, intent: str) -> None:
        """Set the execution roadmap based on intent."""
        self.__roadmap = ExecutionRoadmap(intent=intent)

    async def commit(self, observation: str, thought: str, action: Action) -> None:
        """
        Commit OTA (Observe-Thought-Action) cycle to trace.
        """
        entry = {
            "observation": observation,
            "thought": thought,
            "action": action.model_dump() if hasattr(action, "model_dump") else str(object=action),
        }
        self.__trace.append(entry)

    async def branch(self, milestone: str) -> None:
        """
        Create milestone and compress trace.
        """
        self.__milestones.append(milestone)
        # Clear trace for the next phase
        self.__trace = []

    async def recall(self, tier: str) -> Any:
        """
        Retrieve context from specified tier.
        """
        if tier == "roadmap":
            return self.__roadmap
        elif tier == "milestones":
            return self.__milestones
        elif tier == "trace":
            return self.__trace
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def get_full_context(self) -> Dict[str, Any]:
        """Get the full execution context for reasoning."""
        return {
            "intent": self.__roadmap.intent if self.__roadmap else "unknown",
            "roadmap": self.__roadmap,
            "milestones": self.__milestones,
            "trace": self.__trace,
            "guidance": [g.content for g in self.__user_guidance],
            # Kept for compatibility with existing runners
            "history": self.__context.get_history_summary(),
        }

    async def inject_user_guidance(self, guidance: str, step: Optional[int] = None) -> None:
        """Inject manual guidance into the context."""
        instruction = UserGuidance(content=guidance, step_number=step)
        self.__user_guidance.append(instruction)

        # Persist guidance to memory
        await self.__memory.set(
            key=f"user_guidance_{len(self.__user_guidance)}", 
            value=instruction.model_dump_json()
        )

    def get_user_guidance(self) -> List[UserGuidance]:
        """Get currently pending user guidance."""
        return self.__user_guidance.copy()

    def clear_user_guidance(self) -> None:
        """Clear all pending user guidance."""
        self.__user_guidance.clear()

    @property
    def context(self) -> ExecutionContext:
        """Get the underlying execution context."""
        return self.__context
