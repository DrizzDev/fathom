"""
Interface for Graph Builders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph


class GraphBuilder(ABC):
    """
    Abstract base class for building LangGraph workflows.
    Enforces dependency injection and consistent assembly patterns.
    """

    @abstractmethod
    def build(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[List[str]] = None,
    ) -> CompiledStateGraph:
        """
        Builds and compiles the graph.

        Args:
            checkpointer: Optional persistence layer for state.
            interrupt_before: List of node names to interrupt before.

        Returns:
            CompiledStateGraph: The ready-to-execute graph.
        """
        raise NotImplementedError
