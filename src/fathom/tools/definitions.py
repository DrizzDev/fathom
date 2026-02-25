from __future__ import annotations

from typing import Any, Dict, List


class ToolRegistry:
    """
    Registry for tool definitions used by the Vision Language Model.
    """

    @classmethod
    def get_all_definitions(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns all tool definitions in a format compatible with Gemini API.
        """

        return {
            "function_declarations": [
                cls.__execute_ui(),
                cls.__complete_goal(),
                cls.__validate_state(),
                cls.__verify_goal(),
                cls.__store_memory(),
                cls.__recall_memory(),
            ]
        }

    @staticmethod
    def __execute_ui() -> Dict[str, Any]:
        """
        Definition for execute_ui tool.

        Handles UI interactions and explicit validate events (tap, type, scroll, swipe, validate, etc.).
        Goal completion is signaled separately via complete_goal.
        """

        return {
            "name": "execute_ui",
            "description": "Execute a physical UI action on the device.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Reasoning behind this action.",
                    },
                    "action": {
                        "type": "OBJECT",
                        "description": "The action to execute.",
                        "properties": {
                            "action_type": {
                                "type": "STRING",
                                "description": "Action type.",
                                "enum": [
                                    "tap",
                                    "type",
                                    "scroll",
                                    "swipe_left",
                                    "swipe_right",
                                    "swipe_up",
                                    "swipe_down",
                                    "wait",
                                    "validate",
                                    "home",
                                    "back",
                                    "long_press",
                                ],
                            },
                            "rationale": {
                                "type": "STRING",
                                "description": "Why this action.",
                            },
                            "target_name": {
                                "type": "STRING",
                                "description": "Generic element name (e.g., 'search bar').",
                            },
                            "bbox": {
                                "type": "OBJECT",
                                "description": "Bounding box. See COORDINATES in system prompt.",
                                "properties": {
                                    "x": {
                                        "type": "INTEGER",
                                        "description": "Top-left X.",
                                    },
                                    "y": {
                                        "type": "INTEGER",
                                        "description": "Top-left Y.",
                                    },
                                    "width": {
                                        "type": "INTEGER",
                                        "description": "Width from x.",
                                    },
                                    "height": {
                                        "type": "INTEGER",
                                        "description": "Height from y.",
                                    },
                                },
                            },
                            "text_to_type": {
                                "type": "STRING",
                                "description": "Text to type (for 'type' action only).",
                            },
                            "confidence": {
                                "type": "NUMBER",
                                "description": "Confidence (0.0-1.0).",
                            },
                            "is_valid": {
                                "type": "BOOLEAN",
                                "description": "Is this action valid for the current screen?",
                            },
                            "validation_reason": {
                                "type": "STRING",
                                "description": "Validity reasoning.",
                            },
                            "condition": {
                                "type": "STRING",
                                "description": "Condition for optional/overlay actions (e.g., 'Promotional overlay is visible').",
                            },
                            "overlay_detected": {
                                "type": "BOOLEAN",
                                "description": "True when this action is dismissing an overlay/popup.",
                            },
                            "target_type": {
                                "type": "STRING",
                                "description": "Script reference type. See TOOL ROUTING.",
                                "enum": ["stable", "positional", "dynamic"],
                            },
                            "script_target": {
                                "type": "STRING",
                                "description": "Ordinal or generic phrase for script export.",
                            },
                        },
                        "required": ["action_type", "rationale", "is_valid"],
                    },
                    "screen_description": {
                        "type": "STRING",
                        "description": "Goal-relevant screen state in ≤15 words.",
                    },
                    "content_exhausted": {
                        "type": "BOOLEAN",
                        "description": "True if scrollable content fully exhausted.",
                    },
                    "memory_updates": {
                        "type": "OBJECT",
                        "description": "Key-value pairs for persistent memory updates.",
                    },
                },
                "required": ["assistant_message", "action"],
            },
        }

    @staticmethod
    def __complete_goal() -> Dict[str, Any]:
        """
        Definition for complete_goal tool.

        Dedicated signal for goal completion — separated from execute_ui
        to reduce cognitive load and prevent premature completion.
        """

        return {
            "name": "complete_goal",
            "description": "Signal goal fully achieved. Requires visual evidence on current screen.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Why the goal is complete.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": "Visual evidence from current screen.",
                    },
                },
                "required": ["assistant_message", "evidence"],
            },
        }

    @staticmethod
    def __validate_state() -> Dict[str, Any]:
        """
        Definition for validate_state tool.
        """

        return {
            "name": "validate_state",
            "description": "Check if screen state matches specific criteria without acting.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Verification result explanation.",
                    },
                    "condition_to_verify": {
                        "type": "STRING",
                        "description": "Condition being checked.",
                    },
                    "condition_met": {
                        "type": "BOOLEAN",
                        "description": "True if condition is met.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": "Visual evidence.",
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if high-level goal fully achieved.",
                    },
                },
                "required": [
                    "assistant_message",
                    "condition_to_verify",
                    "condition_met",
                    "evidence",
                    "goal_completed",
                ],
            },
        }

    @staticmethod
    def __verify_goal() -> Dict[str, Any]:
        """
        Definition for verify_goal tool.
        """

        return {
            "name": "verify_goal",
            "description": "Detailed check if overall goal is fully completed.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Goal completion status.",
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if goal FULLY completed.",
                    },
                    "current_screen": {
                        "type": "STRING",
                        "description": "Current screen displayed.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": "Visual evidence of completion.",
                    },
                },
                "required": [
                    "assistant_message",
                    "goal_completed",
                    "current_screen",
                    "evidence",
                ],
            },
        }

    @staticmethod
    def __store_memory() -> Dict[str, Any]:
        """
        Definition for store_memory tool.
        """

        return {
            "name": "store_memory",
            "description": "Store facts needed across steps. Only for cross-step persistence.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "category": {
                        "type": "STRING",
                        "description": "visited | progress | state | data.",
                        "enum": ["visited", "progress", "state", "data"],
                    },
                    "item": {
                        "type": "STRING",
                        "description": "snake_case identifier (use same key to recall).",
                    },
                    "value": {
                        "type": "STRING",
                        "description": "Information to store.",
                    },
                    "assistant_message": {
                        "type": "STRING",
                        "description": "What is being saved.",
                    },
                },
                "required": ["category", "item", "value", "assistant_message"],
            },
        }

    @staticmethod
    def __recall_memory() -> Dict[str, Any]:
        """
        Definition for recall_memory tool.
        """

        return {
            "name": "recall_memory",
            "description": "Retrieve previously stored memory. Use exact same category+item keys.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "category": {
                        "type": "STRING",
                        "description": "Must match store_memory category.",
                        "enum": ["visited", "progress", "state", "data"],
                    },
                    "item": {
                        "type": "STRING",
                        "description": "Must match store_memory item (snake_case).",
                    },
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Why recalling.",
                    },
                },
                "required": ["category", "item", "assistant_message"],
            },
        }
