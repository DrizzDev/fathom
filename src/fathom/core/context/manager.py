"""Context manager for execution state."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fathom.schemas.orchestration import ExecutionContext, ExecutionRoadmap

if TYPE_CHECKING:
    from fathom.interfaces.memory import MemoryPort
    from fathom.schemas.actions import Action


class ContextManager:
    """
    Manages the current execution context and history.

    This service tracks the progress of a workflow and provides
    the necessary context for LLM reasoning and state tracking.
    """

    def __init__(self, memory: MemoryPort, workflow_id: Optional[str] = None) -> None:
        """Initialize context manager with memory port."""
        self.__memory = memory
        self.__workflow_id = workflow_id or uuid.uuid4().hex[:8]
        self.__context = ExecutionContext(workflow_id=self.__workflow_id)
        self.__roadmap: Optional[ExecutionRoadmap] = None
        self.__user_guidance: List[str] = []

    def set_roadmap(self, intent: str) -> None:
        """Set the execution roadmap based on intent."""
        self.__roadmap = ExecutionRoadmap(intent=intent)

    def get_full_context(self) -> Dict[str, Any]:
        """Get the full execution context for reasoning."""
        return {
            "intent": self.__roadmap.intent if self.__roadmap else "unknown",
            "history": self.__context.get_history_summary(),
            "guidance": self.__user_guidance,
        }

    async def commit(self, observation: str, thought: str, action: Action) -> None:
        """Commit an execution step to history."""
        # This implementation can be expanded based on need
        pass

    async def inject_user_guidance(self, guidance: str) -> None:
        """Inject manual guidance into the context."""
        self.__user_guidance.append(guidance)

    def get_user_guidance(self) -> List[str]:
        """Get currently pending user guidance."""
        return self.__user_guidance.copy()

    def clear_user_guidance(self) -> None:
        """Clear all pending user guidance."""
        self.__user_guidance.clear()

    @property
    def context(self) -> ExecutionContext:
        """Get the underlying execution context."""
        return self.__context
