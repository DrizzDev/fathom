"""
Context management implementing GCC-inspired three-tier versioned context.

This module provides the ContextManager which maintains:
- Roadmap: Original intent + milestones
- Milestones: Summaries of completed sub-goals
- Trace: Fine-grained OTA (Observe-Thought-Action) log
"""

from __future__ import annotations

from typing import Any, Dict, List

from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action


class ContextManager:
    """
    Manages three-tier versioned context (GCC-inspired).
    
    Tiers:
    - roadmap: Original intent + milestones (high-level plan)
    - milestones: Summaries of completed sub-goals (compressed progress)
    - trace: Fine-grained OTA log (every Observe-Thought-Action cycle)
    
    The context manager provides operations for:
    - commit(): Add OTA cycle to trace
    - branch(): Create milestone and compress trace
    - recall(): Retrieve context from specified tier
    """
    
    def __init__(self, memory: MemoryPort) -> None:
        """
        Initialize context manager.
        
        Args:
            memory: Memory port for persistent storage
        """
        self.__memory = memory
        self.__roadmap: str = ""
        self.__milestones: List[str] = []
        self.__trace: List[Dict[str, Any]] = []
        self.__user_guidance: List[str] = []  # Store user-injected guidance
    
    def set_roadmap(self, intent: str) -> None:
        """
        Set the roadmap (original intent).
        
        Args:
            intent: User's original intent/goal
        """
        self.__roadmap = intent
    
    async def commit(self, observation: str, thought: str, action: Action) -> None:
        """
        Commit OTA (Observe-Thought-Action) cycle to trace.
        
        Args:
            observation: What was observed (screen state, etc.)
            thought: Reasoning about what to do
            action: Action that was taken
        """
        self.__trace.append({
            "observation": observation,
            "thought": thought,
            "action": action.model_dump() if hasattr(action, "model_dump") else str(action),
        })
    
    async def branch(self, milestone: str) -> None:
        """
        Create milestone and compress trace.
        
        This operation:
        1. Adds the milestone to the milestones list
        2. Compresses the trace into a summary
        3. Clears the trace for the next phase
        
        Args:
            milestone: Description of the completed sub-goal
        """
        self.__milestones.append(milestone)
        
        # Compress trace into milestone summary
        # In a full implementation, this would use LLM to summarize
        # For now, we just clear the trace
        self.__trace = []
    
    async def recall(self, tier: str) -> Any:
        """
        Retrieve context from specified tier.
        
        Args:
            tier: Context tier to retrieve ("roadmap", "milestones", or "trace")
        
        Returns:
            Context data for the specified tier
        
        Raises:
            ValueError: If tier is unknown
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
        """
        Get complete context across all tiers.
        
        Returns:
            Dictionary with roadmap, milestones, trace, and user guidance
        """
        return {
            "roadmap": self.__roadmap,
            "milestones": self.__milestones,
            "trace": self.__trace,
            "user_guidance": self.__user_guidance,
        }
    
    def get_trace_length(self) -> int:
        """
        Get current trace length.
        
        Returns:
            Number of OTA cycles in current trace
        """
        return len(self.__trace)
    
    def get_milestone_count(self) -> int:
        """
        Get number of milestones achieved.
        
        Returns:
            Number of milestones
        """
        return len(self.__milestones)
    
    async def inject_user_guidance(self, guidance: str) -> None:
        """
        Inject user guidance into context.
        
        This is called when user provides additional context or clarification
        during HITL interaction. The guidance is added to the context and will
        be included in the next LLM reasoning cycle.
        
        Args:
            guidance: User-provided guidance or context
        """
        self.__user_guidance.append(guidance)
        
        # Also store in memory for persistence
        await self.__memory.set(
            key=f"user_guidance_{len(self.__user_guidance)}",
            value=guidance
        )
    
    def get_user_guidance(self) -> List[str]:
        """
        Get all user-injected guidance.
        
        Returns:
            List of user guidance strings
        """
        return self.__user_guidance.copy()
    
    def clear_user_guidance(self) -> None:
        """Clear user guidance after it's been used."""
        self.__user_guidance = []
