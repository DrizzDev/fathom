"""
Finalization timeout policy for intent strategy and runtime post-terminal awaits.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HistoryFinalizationBudget(BaseModel):
    """
    Timeout budgets for history persistence phases.
    """

    model_config = ConfigDict(frozen=True)

    flush: float = Field(
        ge=0.1,
        le=120.0,
        default=10.0,
        description="Maximum wait in seconds for pending history operations to drain after graph terminal route",
    )
    script: float = Field(
        ge=0.1,
        le=300.0,
        default=60.0,
        description="Maximum wait in seconds for the script export to complete",
    )


class GraphFinalizationBudget(BaseModel):
    """
    Timeout budgets for graph state and checkpointer phases.
    """

    model_config = ConfigDict(frozen=True)

    state_read: float = Field(
        ge=0.1,
        le=60.0,
        default=5.0,
        description="Maximum wait in seconds for the post-terminal graph state read",
    )
    checkpointer_close: float = Field(
        ge=0.1,
        le=60.0,
        default=10.0,
        description="Maximum wait in seconds before abandoning the LangGraph checkpointer lifecycle close",
    )


class RuntimeFinalizationBudget(BaseModel):
    """
    Timeout budgets for runtime finalization phases.
    """

    model_config = ConfigDict(frozen=True)

    background_drain: float = Field(
        ge=0.1,
        le=60.0,
        default=5.0,
        description="Maximum wait in seconds before abandoning prewarm or background task drains",
    )
    memory_summary: float = Field(
        ge=0.1,
        le=30.0,
        default=3.0,
        description="Maximum wait in seconds for memory summary retrieval",
    )
    context_shutdown: float = Field(
        ge=0.1,
        le=60.0,
        default=10.0,
        description="Maximum wait in seconds before abandoning graph context shutdown",
    )
    cleanup: float = Field(
        ge=0.1,
        le=120.0,
        default=15.0,
        description="Maximum wait in seconds before abandoning runner cleanup",
    )


class FinalizationBudgetPolicy(BaseModel):
    """
    Aggregate timeout policy for the post-terminal finalization sequence.
    """

    model_config = ConfigDict(frozen=True)

    history: HistoryFinalizationBudget = Field(
        default_factory=HistoryFinalizationBudget,
        description="History persistence phase budgets",
    )
    graph: GraphFinalizationBudget = Field(
        default_factory=GraphFinalizationBudget,
        description="Graph state and checkpointer phase budgets",
    )
    runtime: RuntimeFinalizationBudget = Field(
        default_factory=RuntimeFinalizationBudget,
        description="Runtime cleanup and shutdown phase budgets",
    )
