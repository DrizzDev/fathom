from __future__ import annotations

from typing import Final

# Operator-facing failure messages surfaced by the intent graph nodes.
GROUNDING_FAILURE_MESSAGE: Final[str] = "Failed to capture the current app screen. Please retry."
RECORDING_FAILURE_MESSAGE: Final[str] = "Failed to save execution details for the current step."
HITL_DEFAULT_PROMPT: Final[str] = "I need human assistance to proceed."
HITL_UNAVAILABLE_REPLAN_DIAGNOSTIC: Final[str] = "ASK_USER on autonomous runtime; replanning."
