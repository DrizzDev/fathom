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

from typing import Any, Dict, List, Literal, Optional

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
    target_type: Optional[Literal["stable", "positional", "dynamic"]] = Field(
        default=None,
        description="Optional. Script export classification: stable, positional, or dynamic. Omit if unsure.",
    )
    script_target: Optional[str] = Field(
        default=None,
        description="Optional. When target_type is positional/dynamic, exact phrase for script (e.g. 'the first search result').",
    )


class ExecuteUIInput(BaseModel):
    """Input for the execute_ui tool."""

    assistant_message: str = Field(
        description="Reasoning behind this action",
    )
    action: UIActionInput = Field(
        description="The UI action to execute",
    )
    screen_description: Optional[str] = Field(
        default=None,
        description="Goal-relevant screen state in ≤15 words",
    )
    content_exhausted: Optional[bool] = Field(
        default=None,
        description="True if scrollable content has reached its end",
    )
    memory_updates: Optional[Dict[str, str]] = Field(
        default=None,
        description="Key-value pairs to persist in memory",
    )


class CompleteGoalInput(BaseModel):
    """Input for the complete_goal tool."""

    assistant_message: str = Field(
        description="Explanation of why the goal is considered complete",
    )
    evidence: str = Field(
        description="Visual evidence from the current screen proving the goal is complete",
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

    category: str = Field(
        description="Kind of information: 'visited', 'progress', 'state', or 'data'",
    )
    item: str = Field(
        description="Identifier for the specific thing, in snake_case. "
        "Examples: 'carousel_card_1', 'checkout_step', 'product_price'",
    )
    value: str = Field(description="Value to store")
    assistant_message: str = Field(description="Explanation of what is being saved")


class RecallMemoryInput(BaseModel):
    """Input for the recall_memory tool."""

    category: str = Field(
        description="Category used when storing: 'visited', 'progress', 'state', or 'data'",
    )
    item: str = Field(
        description="Item identifier used when storing, in snake_case. "
        "Examples: 'carousel_card_1', 'checkout_step', 'product_price'",
    )
    assistant_message: str = Field(description="Why this information is needed")


# ── Tool functions ──────────────────────────────────────────────────────


@tool(args_schema=ExecuteUIInput)
def execute_ui(
    assistant_message: str,
    action: Dict[str, Any],
    screen_description: Optional[str] = None,
    content_exhausted: Optional[bool] = None,
    memory_updates: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute a UI action on the device (tap, type, scroll, swipe, etc.).

    Use this tool for ALL physical interactions with the app UI.
    Do NOT use this to signal goal completion — use complete_goal instead.
    """

    return {
        "assistant_message": assistant_message,
        "action": action,
        "screen_description": screen_description,
        "content_exhausted": content_exhausted,
        "memory_updates": memory_updates,
    }


@tool(args_schema=CompleteGoalInput)
def complete_goal(
    assistant_message: str,
    evidence: str,
) -> Dict[str, Any]:
    """Signal that the user's goal has been fully achieved.

    Call this ONLY when the current screen state proves the goal is complete.
    Do NOT call this while there are still actions to perform — use execute_ui instead.
    """

    return {
        "assistant_message": assistant_message,
        "evidence": evidence,
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
    category: str,
    item: str,
    value: str,
    assistant_message: str,
) -> Dict[str, Any]:
    """Store important information or progress in memory.

    Use category + item to form a structured key (e.g., category='visited', item='card_1').
    Do NOT use this for transient observations already visible on screen.
    """

    return {
        "category": category,
        "item": item,
        "value": value,
        "assistant_message": assistant_message,
    }


@tool(args_schema=RecallMemoryInput)
def recall_memory(
    category: str,
    item: str,
    assistant_message: str,
) -> Dict[str, Any]:
    """Retrieve previously stored information from memory.

    Use the exact same category and item that were used when storing.
    Do NOT use this when the needed information is already visible on screen.
    """

    return {
        "category": category,
        "item": item,
        "assistant_message": assistant_message,
    }


# ── Registry helper ─────────────────────────────────────────────────────

ALL_TOOLS = [execute_ui, complete_goal, validate_state, verify_goal, store_memory, recall_memory]


def get_tools_for_mode(mode: str) -> List[Any]:
    """
    Return the subset of LangChain tools appropriate for the given
    :class:`~fathom.prompts.modes.PromptMode` value.

    Mirrors the scoping logic in ``GeminiVisionTool.__scope_tools``.
    """

    base = [execute_ui, complete_goal, store_memory]

    if mode == "default":
        return base + [recall_memory, validate_state, verify_goal]

    if mode == "interaction":
        return base + [recall_memory]

    if mode == "verification":
        return base + [validate_state, verify_goal, recall_memory]

    if mode == "exploration":
        # Exploration only needs execute_ui; no goal completion signaling
        return [execute_ui]

    # discovery — minimal
    return base
