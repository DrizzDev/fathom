"""Schemas for LLM-backed intent decomposition."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, field_validator


class DecompositionSchema(BaseModel):
    """Strict schema for intent decomposition tool-less output."""

    sub_goals: List[str]
    confidence: Optional[float] = 0.9

    @field_validator("sub_goals")
    @classmethod
    def validate_sub_goals(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("sub_goals must not be empty")
        if len(v) > 50:
            raise ValueError("sub_goals must not exceed 50 items")
        return [goal.strip() for goal in v if goal.strip()]
