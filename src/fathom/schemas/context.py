from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TraceRecord(BaseModel):
    """
    Tier 3: Fine-grained Observe-Thought-Action (OTA) record.
    Representing the active scratchpad of the agent.
    """

    thought: str
    observation: str
    action: Dict[str, Any]

    timestamp: float = Field(default_factory=time.time)
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])


class Commit(BaseModel):
    """
    Tier 2: A coherent unit of memory squashed from trace records.
    Equivalent to a Git Commit in the GCC architecture.
    """

    summary: str
    step_range: tuple[int, int]

    parent_id: Optional[str] = None
    commit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])

    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)
