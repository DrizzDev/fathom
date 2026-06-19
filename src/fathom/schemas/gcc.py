from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from fathom.schemas.trace import ExecutionRecord


class CommitNode(BaseModel):
    """
    Tier 2: A versioned unit of memory.
    Encapsulates a summarized segment of the execution log.
    """

    parent_id: Optional[str] = None
    commit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])

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
