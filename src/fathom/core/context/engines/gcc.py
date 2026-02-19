from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.interfaces.context import ContextEngine
from fathom.schemas.gcc import BranchState, CommitNode, ExecutionRecord

logger = getLogger(__name__)


class GitContextEngine(ContextEngine):
    """
    Implementation of the Git-Context-Controller (GCC) logic.
    GCC provides hierarchical context management for long-running agent workflows:

    Tier 1 (Milestones): High-level semantic summaries of completed segments
    Tier 2 (Shadow Buffer): Recent detailed execution trace (sliding window)
    Tier 3 (Active Log): Current uncommitted actions

    Context Flow:
    1. Actions are recorded to active_log (branch.log)
    2. When trace reaches threshold, prepare_summarization() moves items to shadow_buffer
    3. Background task summarizes the segment and creates a milestone
    4. commit() keeps last N items in shadow_buffer for continuity
    5. LLM sees: Milestones (semantic) + Shadow Buffer (recent details) + Active Log (current)

    This design scales to 100+ step workflows while maintaining context continuity.
    """

    def __init__(self, *, context_window: int = 7) -> None:
        """
        Initialize the Git engine with a default main branch.

        Args:
            context_window:
                Number of recent items to keep in shadow_buffer after commit.
                Balances context continuity with memory efficiency.
                Default: 7 (provides ~7 steps of recent detailed history for better continuity)
        """

        self.__current_branch: str = "main"
        self.__context_window = context_window
        self.__commit_nodes: Dict[str, CommitNode] = {}
        self.__branches: Dict[str, BranchState] = {"main": BranchState(name="main")}

        # Shadow buffer ensures O(1) consistency during background operations
        self.__shadow_buffer: List[ExecutionRecord] = []

    async def record(self, *, observation: str, thought: str, action: Dict[str, Any]) -> None:
        """
        Tier 3: Record an Observe-Thought-Action cycle.
        """

        record = ExecutionRecord(observation=observation, thought=thought, action=action)
        self.__branches[self.__current_branch].log.append(record)

    async def commit(self, *, summary: str) -> None:
        """
        Tier 2: Consolidate active log into a versioned commit.
        Creates a milestone from the summarized segment while maintaining a sliding window of recent context for continuity.
        """

        branch = self.__branches[self.__current_branch]

        new_node = CommitNode(parent_id=branch.head_id, summary=summary)

        self.__commit_nodes[new_node.commit_id] = new_node
        branch.head_id = new_node.commit_id

        logger.info(
            f"[GCC] commit() called: shadow_buffer_length_before={len(self.__shadow_buffer)}"
        )

        # Keep last N items in shadow_buffer for context continuity
        # This ensures the agent always has recent detailed history while milestones provide high-level semantic context
        if len(self.__shadow_buffer) > self.__context_window:
            self.__shadow_buffer = self.__shadow_buffer[-self.__context_window :]
            logger.info(
                f"[GCC] commit(): kept last {self.__context_window} items, shadow_buffer_length_after={len(self.__shadow_buffer)}"
            )
        else:
            logger.info(
                f"[GCC] commit(): shadow_buffer has {len(self.__shadow_buffer)} items, keeping all"
            )

    async def branch(self, *, branch_name: str) -> None:
        """
        Isolate reasoning into a new branch.
        """

        parent = self.__branches[self.__current_branch]
        new_branch = BranchState(name=branch_name, head_id=parent.head_id, log=list(parent.log))

        self.__branches[branch_name] = new_branch
        self.__current_branch = branch_name

    def get_context(self) -> Dict[str, Any]:
        """
        Construct the three-tier reasoning hierarchy.
        """

        branch = self.__branches[self.__current_branch]
        trace = [record.model_dump() for record in (self.__shadow_buffer + branch.log)]

        logger.info(
            f"[GCC] get_context(): shadow_buffer_length={len(self.__shadow_buffer)}, branch.log_length={len(branch.log)}, total_trace_length={len(trace)}"
        )

        return {
            "trace": trace,  # Merge shadow + active log to ensure no context gaps
            "milestones": self.__get_commit_chain(head_id=branch.head_id),
            "active_count": len(branch.log),
        }

    def __get_commit_chain(self, *, head_id: Optional[str]) -> List[str]:
        """
        Backtracks from HEAD to root to build the semantic milestone list.
        """

        chain: List[str] = []
        current_id = head_id

        while current_id and current_id in self.__commit_nodes:
            node = self.__commit_nodes[current_id]
            chain.insert(0, node.summary)
            current_id = node.parent_id

        return chain

    def dehydrate(self) -> Dict[str, Any]:
        """
        Serialize state for persistent storage.
        """

        return {
            "current": self.__current_branch,
            "shadow": [record.model_dump() for record in self.__shadow_buffer],
            "commits": {key: value.model_dump() for key, value in self.__commit_nodes.items()},
            "branches": {key: value.model_dump() for key, value in self.__branches.items()},
        }

    async def hydrate(self, *, data: Dict[str, Any]) -> None:
        """
        Restore state from serialized data.
        """

        self.__branches = {
            key: BranchState(**value) for key, value in data.get("branches", {}).items()
        }
        self.__commit_nodes = {
            key: CommitNode(**value) for key, value in data.get("commits", {}).items()
        }

        self.__current_branch = data.get("current", "main")
        self.__shadow_buffer = [ExecutionRecord(**record) for record in data.get("shadow", [])]

    @property
    def active_log(self) -> List[Dict[str, Any]]:
        """
        Provides a snapshot of the current uncommitted log.
        """

        branch = self.__branches[self.__current_branch]
        return [record.model_dump() for record in branch.log]

    def prepare_summarization(self) -> List[Dict[str, Any]]:
        """
        Atomically moves logs to shadow buffer and returns them for summarization.
        """

        import logging

        logger = logging.getLogger(__name__)

        branch = self.__branches[self.__current_branch]
        segment = list(branch.log)

        logger.info(
            f"[GCC] prepare_summarization(): branch.log_length={len(branch.log)}, shadow_buffer_length_before={len(self.__shadow_buffer)}"
        )

        self.__shadow_buffer.extend(segment)
        branch.log.clear()

        logger.info(
            f"[GCC] prepare_summarization(): shadow_buffer_length_after={len(self.__shadow_buffer)}, branch.log_length_after={len(branch.log)}"
        )

        return [record.model_dump() for record in segment]
