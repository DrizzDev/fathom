from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from pydantic import BaseModel, Field


class ExecutionRecord(BaseModel):
    """
    Single observe-thought-action turn captured on the agent's running trace.
    """

    thought: str = Field(description="Planner's rationale for the turn.")
    observation: str = Field(description="Screen observation summary at the turn.")
    action: Dict[str, Any] = Field(description="Dispatched action payload for the turn.")

    timestamp: float = Field(
        default_factory=time.time,
        description="Wall-clock timestamp of the turn.",
    )
    record_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="Short unique identifier for the turn.",
    )
