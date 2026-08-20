from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph


class GraphBuilder(ABC):
    """
    Port for assembling a runnable LangGraph workflow from injected ports.
    """

    @abstractmethod
    def build(
        self,
        interrupt_before: Optional[List[str]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> CompiledStateGraph:
        """
        Wire the injected nodes into a compiled, runnable graph.

        ``interrupt_before`` lists nodes where the run pauses for human-in-the-loop control,
        and ``checkpointer`` persists graph state so an interrupted run resumes across restarts.
        """

        raise NotImplementedError
