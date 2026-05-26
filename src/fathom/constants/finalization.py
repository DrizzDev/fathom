"""
Stable phase identifiers for post-terminal finalization observability.
"""

from __future__ import annotations

from enum import StrEnum


class FinalizationPhase(StrEnum):
    """
    Identifiers for each post-terminal finalization phase emitted by intent strategy and runtime.
    """

    EXECUTOR_RUN = "fathom.intent.executor"
    HISTORY_FLUSH = "fathom.finalization.history.flush"
    HISTORY_SCRIPT = "fathom.finalization.history.script"
    GRAPH_STATE_READ = "fathom.finalization.graph.state"
    CHECKPOINTER_CLOSE = "fathom.finalization.checkpointer.close"
    BACKGROUND_DRAIN = "fathom.runner.background.drain"
    MEMORY_SUMMARY = "fathom.runner.memory.summary"
    CONTEXT_SHUTDOWN = "fathom.runner.context.shutdown"
    RUNNER_CLEANUP = "fathom.runner.cleanup"
