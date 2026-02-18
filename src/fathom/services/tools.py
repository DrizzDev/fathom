from __future__ import annotations

from typing import Any, Dict, List


class ToolsService:
    """
    Service for managing LLM tool definitions.
    """

    def get_tool_definitions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the tool definitions for the agent.
        """

        return {
            "function_declarations": [
                {
                    "name": "execute_ui_actions",
                    "description": (
                        "Execute a UI action on the mobile UI to fulfill the user's intent. "
                        "If the target element is not currently visible, you MUST use 'scroll' or "
                        "'swipe' actions to find it. Do NOT signal goal completion here — use "
                        "verify_goal_completion instead. "
                        "Do NOT use this for state validation or screen checks — verify intent visually before acting."
                    ),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "assistant_message": {
                                "type": "STRING",
                                "description": "Short reasoning or status update from the assistant.",
                            },
                            "action": {
                                "type": "OBJECT",
                                "description": "The action to execute.",
                                "properties": {
                                    "action_type": {
                                        "type": "STRING",
                                        "description": "The type of action to perform. One of: tap, type, long_press, swipe_left, swipe_right, swipe_up, swipe_down, scroll, wait, zoom_in, zoom_out.",
                                    },
                                    "rationale": {
                                        "type": "STRING",
                                        "description": "Why this action was chosen.",
                                    },
                                    "confidence": {
                                        "type": "NUMBER",
                                        "description": "Confidence score 0.0-1.0",
                                    },
                                    "bbox": {
                                        "type": "OBJECT",
                                        "description": "Bounding box for the action. CRITICAL: All coordinates MUST be in 0-1000 normalized range.",
                                        "properties": {
                                            "x": {
                                                "type": "NUMBER",
                                                "description": "Top-left X (0-1000)",
                                            },
                                            "y": {
                                                "type": "NUMBER",
                                                "description": "Top-left Y (0-1000)",
                                            },
                                            "width": {
                                                "type": "NUMBER",
                                                "description": "Width (0-1000)",
                                            },
                                            "height": {
                                                "type": "NUMBER",
                                                "description": "Height (0-1000)",
                                            },
                                            "coord_system": {
                                                "type": "STRING",
                                                "description": "Always 'normalized'",
                                            },
                                        },
                                        "required": [
                                            "x",
                                            "y",
                                            "width",
                                            "height",
                                            "coord_system",
                                        ],
                                    },
                                    "label_id": {
                                        "type": "STRING",
                                        "description": "The numeric label ID if using XML mode.",
                                    },
                                    "text": {
                                        "type": "STRING",
                                        "description": "Text to type (only for 'type' action)",
                                    },
                                    "wait_duration": {
                                        "type": "NUMBER",
                                        "description": "Duration to wait in milliseconds (only for 'wait' action)",
                                    },
                                },
                                "required": ["action_type", "rationale"],
                            },
                        },
                        "required": ["assistant_message", "action"],
                    },
                },
                {
                    "name": "verify_goal_completion",
                    "description": (
                        "Verify if the user's overall goal has been fully completed by checking the current screen state. "
                        "Do NOT use this for intermediate state checks — only for final goal verification. "
                        "Do NOT use this when there are still UI actions to perform — use execute_ui_actions first."
                    ),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "assistant_message": {
                                "type": "STRING",
                                "description": "Explanation of the goal completion status.",
                            },
                            "goal_completed": {
                                "type": "BOOLEAN",
                                "description": "True if the overall goal is FULLY completed.",
                            },
                            "evidence": {
                                "type": "STRING",
                                "description": "Visual evidence proving completion.",
                            },
                        },
                        "required": ["assistant_message", "goal_completed", "evidence"],
                    },
                },
            ]
        }
