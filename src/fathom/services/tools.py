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
                    "description": "Execute a list of actions on the mobile UI to fulfill the user's intent. If the target element is not currently visible, you MUST use 'scroll' or 'swipe' actions to find it.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "assistant_message": {
                                "type": "STRING",
                                "description": "Short reasoning or status update from the assistant.",
                            },
                            "goal_completed": {
                                "type": "BOOLEAN",
                                "description": "True if the user's intent is fully satisfied by the current screen state.",
                            },
                            "actions": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
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
                        },
                        "required": ["assistant_message", "actions"],
                    },
                },
                {
                    "name": "verify_goal_completion",
                    "description": "Verify if the user's overall goal has been fully completed by checking the current screen state.",
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
