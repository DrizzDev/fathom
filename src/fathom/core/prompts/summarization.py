"""Provider-neutral trace summarization prompt policy.

Owns the system instruction, the ``create_milestone`` tool schema, and
the user-prompt builder used by any LLM provider when compressing
execution traces into structured milestones. Adapter layers (e.g.
``adapters/summarization/llm.py``) are thin shims that satisfy the
``SummarizationPort`` port and delegate here for policy content.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

__all__ = [
    "SUMMARIZATION_SYSTEM",
    "SUMMARIZATION_TOOL_DEFINITION",
    "build_summarization_user_prompt",
    "format_milestone",
]


SUMMARIZATION_SYSTEM = """You are an expert at analyzing mobile UI automation execution traces.

Your task is to create a structured milestone summary that helps an AI agent understand:
1. What was accomplished in this segment
2. Key actions that led to success
3. Any challenges or failures encountered

Focus on STATE CHANGES and OUTCOMES, not routine navigation.
Be concise but informative - the agent needs to quickly understand progress."""


SUMMARIZATION_TOOL_DEFINITION: Dict[str, Any] = {
    "function_declarations": [
        {
            "name": "create_milestone",
            "description": "Create a structured milestone summary of the execution segment",
            "parameters": {
                "type": "object",
                "properties": {
                    "accomplishment": {
                        "type": "string",
                        "description": (
                            "Main outcome or state change achieved "
                            "(e.g., 'Successfully configured custom schedule', "
                            "'Navigated to payment screen')"
                        ),
                    },
                    "key_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "2-3 most important actions that led to the accomplishment "
                            "(e.g., 'Selected Monday, Tuesday, Friday', 'Entered date range')"
                        ),
                    },
                    "challenges": {
                        "type": "string",
                        "description": (
                            "Any failures or retries encountered, or 'None' if smooth execution"
                        ),
                    },
                },
                "required": ["accomplishment", "key_actions", "challenges"],
            },
        }
    ]
}


def build_summarization_user_prompt(
    *,
    total_steps: int,
    unique_screens: int,
    sample_actions: Iterable[str],
    failures: Iterable[str],
) -> List[str]:
    """Render the provider-neutral user prompt for trace summarization.

    Returned as a list of string parts so the adapter can pass it directly
    to the LLM ``generate`` call without extra joining.
    """

    actions_str = ", ".join(str(action) for action in sample_actions)
    failures_str = ", ".join(str(failure) for failure in failures) or "None"

    return [
        "Analyze this execution segment and create a milestone summary:\n",
        f"Steps: {total_steps}",
        f"Screens visited: {unique_screens}",
        f"Actions: {actions_str}",
        f"Failures: {failures_str}",
        "\nCreate a milestone that captures what was accomplished, how, and any challenges.",
    ]


def format_milestone(
    *,
    accomplishment: str,
    key_actions: Iterable[str],
    challenges: str,
) -> str:
    """Compose the final milestone string from structured tool-call output."""

    key_actions_list = list(key_actions)
    parts: List[str] = [accomplishment]

    if key_actions_list:
        parts.append(f"via {', '.join(key_actions_list)}")

    if challenges and challenges.lower() != "none":
        parts.append(f"(faced: {challenges})")

    return ". ".join(parts)
