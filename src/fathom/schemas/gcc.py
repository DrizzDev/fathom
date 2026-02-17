"""
Domain schemas for Generative Context Construction (GCC).
Strictly typed to support a versioned, navigable memory hierarchy.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExecutionRecord(BaseModel):
    """
    Tier 3: Individual Observe-Thought-Action (OTA) cycle.
    """

    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = Field(default_factory=time.time)
    observation: str
    thought: str
    action: Dict[str, Any]


class CommitNode(BaseModel):
    """
    Tier 2: A versioned unit of memory.
    Encapsulates a summarized segment of the execution log.
    """

    commit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    summary: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BranchState(BaseModel):
    """
    Represents the state of a specific reasoning branch.
    """

    name: str
    head_id: Optional[str] = None  # Last commit ID in this branch
    log: List[ExecutionRecord] = Field(default_factory=list)  # Uncommitted Tier 3 records
