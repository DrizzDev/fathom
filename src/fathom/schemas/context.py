"""
Domain schemas for Generative Context Construction (GCC).
Strictly typed to support versioning and semantic navigation.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class UserGuidance(BaseModel):
    """
    Structured user instruction injected during execution.
    """

    content: str
    timestamp: float = Field(default_factory=time.time)
    source: str = "hitl"
    step_number: Optional[int] = None


class TraceRecord(BaseModel):
    """
    Tier 3: Fine-grained Observe-Thought-Action (OTA) record.
    Representing the active scratchpad of the agent.
    """

    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = Field(default_factory=time.time)
    observation: str
    thought: str
    action: Dict[str, Any]


class Commit(BaseModel):
    """
    Tier 2: A coherent unit of memory squashed from trace records.
    Equivalent to a Git Commit in the GCC architecture.
    """

    commit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    summary: str
    timestamp: float = Field(default_factory=time.time)
    step_range: tuple[int, int]
    metadata: Dict[str, Any] = Field(default_factory=dict)
