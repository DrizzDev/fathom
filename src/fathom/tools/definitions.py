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
        """

        return {
            "name": "execute_ui",
            "description": "Execute a sequence of UI actions on the device to achieve a specific sub-goal or the final goal. Use this to interact with the app.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "A message to the user explaining the reasoning behind these actions.",
                    },
                    "actions": {
                        "type": "ARRAY",
                        "description": "A list of actions to execute in sequence.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "action_type": {
                                    "type": "STRING",
                                    "description": "The type of action to perform.",
                                    "enum": [
                                        "tap",
                                        "type",
                                        "scroll",
                                        "swipe_left",
                                        "swipe_right",
                                        "swipe_up",
                                        "swipe_down",
                                        "wait",
                                        "home",
                                        "back",
                                        "enter",
                                    ],
                                },
                                "rationale": {
                                    "type": "STRING",
                                    "description": "Why this specific action is being taken.",
                                },
                                "target_name": {
                                    "type": "STRING",
                                    "description": "Descriptive name of the element (e.g., 'search bar', 'bathroom cleaning service option').",
                                },
                                "bbox": {
                                    "type": "OBJECT",
                                    "description": "Bounding box for the action (tap target). Use normalized coordinates (0-1000) whenever possible.",
                                    "properties": {
                                        "x": {"type": "INTEGER"},
                                        "y": {"type": "INTEGER"},
                                        "width": {"type": "INTEGER"},
                                        "height": {"type": "INTEGER"},
                                        "coord_system": {
                                            "type": "STRING",
                                            "enum": ["normalized", "pixel"],
                                            "description": "Coordinate system used. Defaults to normalized (0-1000).",
                                        },
                                    },
                                },
                                "text_to_type": {
                                    "type": "STRING",
                                    "description": "Text to type (only for 'type' action).",
                                },
                                "confidence": {
                                    "type": "NUMBER",
                                    "description": "Confidence level (0.0-1.0) for this action.",
                                },
                            },
                            "required": ["action_type", "rationale"],
                        },
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the user's high-level goal is fully achieved after these actions.",
                    },
                    "memory_updates": {
                        "type": "OBJECT",
                        "description": "Optional key-value pairs to update in persistent memory. Use this to track progress (e.g., 'visited_card1': 'true').",
                    },
                },
                "required": ["assistant_message", "actions", "goal_completed"],
            },
        }

    @staticmethod
    def __validate_state() -> Dict[str, Any]:
        """
        Definition for validate_state tool.
        """

        return {
            "name": "validate_state",
            "description": "Verify if the screen state matches specific criteria. Use this when the intent implies checking, validating, or verifying something.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Explanation of the verification result.",
                    },
                    "condition_to_verify": {
                        "type": "STRING",
                        "description": "The condition being verified (e.g., 'Settings screen is open').",
                    },
                    "condition_met": {
                        "type": "BOOLEAN",
                        "description": "True if the condition is met based on the visual evidence.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": "Visual evidence supporting the conclusion.",
                    },
                },
                "required": [
                    "assistant_message",
                    "condition_to_verify",
                    "condition_met",
                    "evidence",
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
                        "description": "True if the overall goal is FULLY completed based on screen state.",
                    },
                    "current_screen": {
                        "type": "STRING",
                        "description": "The actual screen currently displayed.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": "Visual evidence proving goal completion.",
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
            "description": "Store important information, progress, or state in memory to remember it later. Use this to track what you've already done (e.g., 'interacted with card 1 in carousel').",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "key": {
                        "type": "STRING",
                        "description": "The key under which to store the information.",
                    },
                    "value": {
                        "type": "STRING",
                        "description": "The information to store.",
                    },
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Explanation of what is being saved.",
                    },
                },
                "required": ["key", "value", "assistant_message"],
            },
        }

    @staticmethod
    def __recall_memory() -> Dict[str, Any]:
        """
        Definition for recall_memory tool.
        """

        return {
            "name": "recall_memory",
            "description": "Retrieve previously stored information from memory. Use this to check your progress or recall specific details.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "key": {
                        "type": "STRING",
                        "description": "The key of the information to retrieve.",
                    },
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Why this information is being retrieved.",
                    },
                },
                "required": ["key", "assistant_message"],
            },
        }
