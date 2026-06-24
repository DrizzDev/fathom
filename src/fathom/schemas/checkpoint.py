"""
Backend-specific configuration for the LangGraph checkpoint store.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.exploration import BFSPhase
from fathom.schemas.actions import Action
from fathom.schemas.exploration import BFSQueueEntry


class SqliteCheckpointPolicy(BaseModel):
    """
    Operational policy for the SQLite-backed checkpoint store.
    """

    model_config = ConfigDict(frozen=True)

    busy_timeout: int = Field(
        default=5_000,
        ge=0,
        le=60_000,
        description="SQLite busy_timeout in milliseconds before SQLITE_BUSY is raised under lock contention",
    )
    sweep_age: int = Field(
        default=86_400,
        ge=300,
        le=604_800,
        description="Age in seconds beyond which orphaned per-workflow checkpoint files become eligible for sweep",
    )
    sweep_min_interval: int = Field(
        default=300,
        ge=10,
        le=86_400,
        description="Minimum seconds between sweeper invocations per process to throttle filesystem scans",
    )


class PostgresCheckpointPolicy(BaseModel):
    """
    Operational policy for the Postgres-backed checkpoint store.
    """

    model_config = ConfigDict(frozen=True)

    statement_timeout: int = Field(
        default=5_000,
        ge=0,
        le=60_000,
        description="Postgres statement_timeout in milliseconds applied to each checkpoint connection",
    )
    lock_timeout: int = Field(
        default=3_000,
        ge=0,
        le=60_000,
        description="Postgres lock_timeout in milliseconds applied to each checkpoint connection",
    )
    retention_age: int = Field(
        default=86_400,
        ge=300,
        le=604_800,
        description="Age in seconds beyond which orphaned checkpoint rows become eligible for retention sweep",
    )


class SqliteCheckpointConfiguration(BaseModel):
    """
    SQLite checkpoint backend selection with policy.
    """

    model_config = ConfigDict(frozen=True)

    backend: Literal["sqlite"] = Field(
        default="sqlite",
        description="Backend discriminator for SQLite checkpoint store",
    )
    policy: SqliteCheckpointPolicy = Field(
        default_factory=SqliteCheckpointPolicy,
        description="SQLite-specific operational policy",
    )


class PostgresCheckpointConfiguration(BaseModel):
    """
    Postgres checkpoint backend selection with policy and connection material.
    """

    model_config = ConfigDict(frozen=True)

    backend: Literal["postgres"] = Field(
        default="postgres",
        description="Backend discriminator for Postgres checkpoint store",
    )
    connection_string: str = Field(
        description="Postgres connection URL used to construct the PostgresSaver pool",
    )
    policy: PostgresCheckpointPolicy = Field(
        default_factory=PostgresCheckpointPolicy,
        description="Postgres-specific operational policy",
    )


CheckpointConfiguration = Union[SqliteCheckpointConfiguration, PostgresCheckpointConfiguration]


class ExplorationCheckpoint(BaseModel):
    """
    Snapshot of the depth-first exploration path, frontier, and progress.

    Persisted at each step boundary so an interrupted run resumes the actual
    traversal (the path back to the root and the recovery queue), not merely the
    knowledge-graph frontier.
    """

    model_config = ConfigDict(frozen=True)

    phase: BFSPhase = Field(description="DFS phase the run was in.")
    root_hash: Optional[str] = Field(
        default=None, description="Visual hash of the run's root screen."
    )
    current_path: List[Tuple[str, Action]] = Field(
        default_factory=list, description="Path of (screen hash, action) from the root."
    )
    bfs_queue: List[BFSQueueEntry] = Field(
        default_factory=list, description="Pending frontier entries with replayable paths."
    )
    fully_scanned: List[str] = Field(
        default_factory=list, description="Screen hashes already fully scanned."
    )
    exhaustion_retries: Dict[str, int] = Field(
        default_factory=dict, description="Per-screen exhaustion-retry counters."
    )
