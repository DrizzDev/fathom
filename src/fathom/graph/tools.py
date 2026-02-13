"""
LangChain tool wrappers for Fathom's VLM tool definitions.

These tools are **schema-only** — they exist so that the LLM can produce
structured tool-call outputs.  Actual device execution is performed by
:class:`~fathom.orchestration.executor.StepExecutor` inside the graph's
``execute`` node.

The tool functions simply return their arguments as-is; the graph routing
logic interprets them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ── Pydantic input schemas ─────────────────────────────────────────────


class BBoxInput(BaseModel):
    """Bounding box for a UI element."""

    x: int = Field(description="Top-left X coordinate (0-1000 normalized)")
    y: int = Field(description="Top-left Y coordinate (0-1000 normalized)")
    width: int = Field(description="Width of the element")
    height: int = Field(description="Height of the element")
    coord_system: str = Field(
        default="normalized",
        description="Coordinate system: 'normalized' (0-1000) or 'pixel'",
    )


class UIActionInput(BaseModel):
    """A single UI action to perform."""

    action_type: str = Field(
        description="Action type: tap, type, scroll, swipe_left, swipe_right, "
        "swipe_up, swipe_down, wait, home, back, enter"
    )
    rationale: str = Field(description="Why this action is being taken")
    is_valid: bool = Field(default=True, description="Self-validation flag")
    target_name: Optional[str] = Field(
        default=None,
        description="Descriptive name of the element",
    )
    bbox: Optional[BBoxInput] = Field(
        default=None,
        description="Bounding box for the action target",
    )
    text_to_type: Optional[str] = Field(
        default=None,
        description="Text to type (only for 'type' action)",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence level (0.0–1.0)",
    )
    validation_reason: Optional[str] = Field(
        default=None,
        description="Reasoning for validity judgment",
    )


class ExecuteUIInput(BaseModel):
    """Input for the execute_ui tool."""

    assistant_message: str = Field(
        description="Reasoning behind these actions",
    )
    actions: List[UIActionInput] = Field(
        description="Ordered list of UI actions to execute",
    )
    goal_completed: bool = Field(
        description="True if the user's goal is fully achieved after these actions",
    )
    content_exhausted: Optional[bool] = Field(
        default=None,
        description="True if scrollable content has reached its end",
    )
    memory_updates: Optional[Dict[str, str]] = Field(
        default=None,
        description="Key-value pairs to persist in memory",
    )


class ValidateStateInput(BaseModel):
    """Input for the validate_state tool."""

    assistant_message: str = Field(description="Explanation of the verification result")
    condition_to_verify: str = Field(description="The condition being verified")
    condition_met: bool = Field(description="Whether the condition is met")
    evidence: str = Field(description="Visual evidence supporting the conclusion")
    goal_completed: bool = Field(description="Whether the overall goal is achieved")


class VerifyGoalInput(BaseModel):
    """Input for the verify_goal tool."""

    assistant_message: str = Field(description="Goal completion explanation")
    goal_completed: bool = Field(description="Whether the goal is fully completed")
    current_screen: str = Field(description="The current screen description")
    evidence: str = Field(description="Visual evidence proving goal completion")


class StoreMemoryInput(BaseModel):
    """Input for the store_memory tool."""

    key: str = Field(description="Memory key")
    value: str = Field(description="Value to store")
    assistant_message: str = Field(description="Explanation of what is being saved")


class RecallMemoryInput(BaseModel):
    """Input for the recall_memory tool."""

    key: str = Field(description="Memory key to retrieve")
    assistant_message: str = Field(description="Why this information is needed")


# ── Tool functions ──────────────────────────────────────────────────────


@tool(args_schema=ExecuteUIInput)
def execute_ui(
    assistant_message: str,
    actions: List[Dict[str, Any]],
    goal_completed: bool,
    content_exhausted: Optional[bool] = None,
    memory_updates: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute a sequence of UI actions on the device to achieve a specific sub-goal or the final goal."""

    return {
        "assistant_message": assistant_message,
        "actions": actions,
        "goal_completed": goal_completed,
        "content_exhausted": content_exhausted,
        "memory_updates": memory_updates,
    }


@tool(args_schema=ValidateStateInput)
def validate_state(
    assistant_message: str,
    condition_to_verify: str,
    condition_met: bool,
    evidence: str,
    goal_completed: bool,
) -> Dict[str, Any]:
    """Verify if the screen state matches specific criteria."""

    return {
        "assistant_message": assistant_message,
        "condition_to_verify": condition_to_verify,
        "condition_met": condition_met,
        "evidence": evidence,
        "goal_completed": goal_completed,
    }


@tool(args_schema=VerifyGoalInput)
def verify_goal(
    assistant_message: str,
    goal_completed: bool,
    current_screen: str,
    evidence: str,
) -> Dict[str, Any]:
    """Verify if the user's overall goal has been fully completed."""

    return {
        "assistant_message": assistant_message,
        "goal_completed": goal_completed,
        "current_screen": current_screen,
        "evidence": evidence,
    }


@tool(args_schema=StoreMemoryInput)
def store_memory(
    key: str,
    value: str,
    assistant_message: str,
) -> Dict[str, Any]:
    """Store important information or progress in memory."""

    return {
        "key": key,
        "value": value,
        "assistant_message": assistant_message,
    }


@tool(args_schema=RecallMemoryInput)
def recall_memory(
    key: str,
    assistant_message: str,
) -> Dict[str, Any]:
    """Retrieve previously stored information from memory."""

    return {
        "key": key,
        "assistant_message": assistant_message,
    }


# ── Registry helper ─────────────────────────────────────────────────────

ALL_TOOLS = [execute_ui, validate_state, verify_goal, store_memory, recall_memory]


def get_tools_for_mode(mode: str) -> List[Any]:
    """
    Return the subset of LangChain tools appropriate for the given
    :class:`~fathom.prompts.modes.PromptMode` value.

    Mirrors the scoping logic in ``GeminiVisionTool.__scope_tools``.
    """

    base = [execute_ui, store_memory]

    if mode == "default":
        return base + [recall_memory, validate_state, verify_goal]

    if mode == "interaction":
        return base + [recall_memory]

    if mode == "verification":
        return base + [validate_state, verify_goal, recall_memory]

    # discovery — minimal
    return base
