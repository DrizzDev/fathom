"""
Domain schemas for context management.
"""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field


class UserGuidance(BaseModel):
    """
    Structured user instruction injected during execution.
    """
    content: str
    timestamp: float = Field(default_factory=time.time)
    source: str = "hitl"
    step_number: Optional[int] = None
    
    def __str__(self) -> str:
        return self.content
