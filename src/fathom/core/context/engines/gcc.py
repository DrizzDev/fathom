from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.interfaces.context import ContextEngine
from fathom.schemas.gcc import BranchState, CommitNode, ExecutionRecord

logger = getLogger(__name__)


class GitContextEngine(ContextEngine):
    """
    Implementation of the Git-Context-Controller (GCC) logic.
    Handles versioned commits, isolated branching, and shadow-buffer consistency.
    """

    def __init__(self) -> None:
        """
        Initialize the Git engine with a default main branch.
        """

        self.__current_branch: str = "main"
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
        """

        branch = self.__branches[self.__current_branch]

        new_node = CommitNode(parent_id=branch.head_id, summary=summary)

        self.__commit_nodes[new_node.commit_id] = new_node
        branch.head_id = new_node.commit_id

        logger.info(
            f"[GCC] commit() called: shadow_buffer_length_before={len(self.__shadow_buffer)}"
        )

        # Clear shadow_buffer after creating milestone
        # The milestone now represents the summarized context
        self.__shadow_buffer.clear()

        logger.info("[GCC] commit(): cleared shadow_buffer")

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

        import logging

        logger = logging.getLogger(__name__)

        branch = self.__branches[self.__current_branch]

        trace = [record.model_dump() for record in (self.__shadow_buffer + branch.log)]

        logger.info(
            f"[GCC] get_context(): shadow_buffer_length={len(self.__shadow_buffer)}, branch.log_length={len(branch.log)}, total_trace_length={len(trace)}"
        )

        return {
            "trace": trace,  # Merge shadow + active log to ensure no context gaps
            "milestones": self.__get_commit_chain(head_id=branch.head_id),
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
