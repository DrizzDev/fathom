"""
Pydantic request models for tool call validation.

These models enforce schema compliance for all LLM tool responses,
ensuring type safety and field validation before domain object construction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExecuteUIRequest(BaseModel):
    """Request schema for execute_ui tool call."""

    action: Dict[str, Any] = Field(
        description="Action object with action_type, rationale, target_name, bbox, etc."
    )
    assistant_message: str = Field(description="Brief message explaining the action")
    screen_description: Optional[str] = Field(
        default=None, description="Description of the current screen state"
    )
    content_exhausted: Optional[bool] = Field(
        default=False, description="Signal that scrollable content is exhausted"
    )
    memory_updates: Optional[Dict[str, str]] = Field(
        default=None, description="Key-value memory updates"
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure action dict contains required fields."""
        if not isinstance(v, dict):
            raise ValueError("action must be a dictionary")
        if "action_type" not in v:
            raise ValueError("action must contain 'action_type'")
        return v

    @field_validator("memory_updates", mode="before")
    @classmethod
    def parse_memory_updates(cls, v: Any) -> Optional[Dict[str, str]]:
        """Handle memory_updates that may come as JSON string or dict."""
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if not v or v == "[]" or v == "{}":
                return None
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return None


class VerifyGoalRequest(BaseModel):
    """Request schema for verify_goal tool call."""

    goal_completed: bool = Field(description="Whether the goal has been completed")
    assistant_message: str = Field(description="Reasoning for the assessment")
    current_screen: Optional[str] = Field(default=None, description="Current screen state")
    evidence: Optional[str] = Field(default=None, description="Evidence supporting the assessment")

    model_config = ConfigDict(extra="allow")


class ValidateStateRequest(BaseModel):
    """Request schema for validate_state tool call."""

    evidence: str = Field(description="Evidence from state inspection")
    assistant_message: str = Field(description="Assessment message")
    goal_completed: Optional[bool] = Field(default=False, description="Whether goal is complete")

    model_config = ConfigDict(extra="allow")


class CompleteGoalRequest(BaseModel):
    """Request schema for complete_goal tool call."""

    assistant_message: str = Field(description="Reasoning for completion")
    evidence: Optional[str] = Field(default=None, description="Evidence that goal is complete")

    model_config = ConfigDict(extra="allow")


class StoreMemoryRequest(BaseModel):
    """Request schema for store_memory tool call."""

    category: str = Field(description="Memory category (e.g., 'prices', 'urls')")
    item: str = Field(description="Item key within the category")
    value: str = Field(description="Value to store")

    model_config = ConfigDict(extra="allow")


class RecallMemoryRequest(BaseModel):
    """Request schema for recall_memory tool call."""

    category: str = Field(description="Memory category to query")
    item: Optional[str] = Field(
        default=None,
        description="Specific item to recall; if omitted, return all items in category",
    )

    model_config = ConfigDict(extra="allow")
