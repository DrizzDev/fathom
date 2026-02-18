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

        Handles ONLY physical UI interactions (tap, type, scroll, swipe, etc.).
        Goal completion is signaled separately via complete_goal.
        """

        return {
            "name": "execute_ui",
            "description": (
                "Execute a UI action on the device (tap, type, scroll, swipe, etc.). "
                "Use this for ALL physical interactions with the app UI. "
                "Do NOT use this to signal goal completion — use complete_goal instead. "
                "Do NOT use this for state validation or screen checks without a UI action — use validate_state instead. "
                "Do NOT use this for verifying goal completion — use verify_goal or complete_goal instead."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "A message to the user explaining the reasoning behind this action.",
                    },
                    "action": {
                        "type": "OBJECT",
                        "description": "The action to execute.",
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
                                    "long_press",
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
                            "is_valid": {
                                "type": "BOOLEAN",
                                "description": "Self-correction: Is this action valid given the current screen state?",
                            },
                            "validation_reason": {
                                "type": "STRING",
                                "description": "Reasoning for the validity judgment.",
                            },
                            "target_type": {
                                "type": "STRING",
                                "description": (
                                    "Optional. How this target should be referenced in exported scripts: "
                                    "'stable' (fixed UI label, e.g. Login, Settings), "
                                    "'positional' (ordinal in list/carousel, e.g. the first search result), "
                                    "'dynamic' (content that may change, e.g. the promotional banner). "
                                    "Leave unset if unsure; the system will classify later."
                                ),
                                "enum": ["stable", "positional", "dynamic"],
                            },
                            "script_target": {
                                "type": "STRING",
                                "description": (
                                    "Optional. When target_type is 'positional' or 'dynamic', set this to the exact phrase "
                                    "for script export (e.g. 'the first search result', 'the second card', 'the promotional banner'). "
                                    "Use natural ordinals. Omit when target_type is 'stable' or when not classifying."
                                ),
                            },
                        },
                        "required": ["action_type", "rationale", "is_valid"],
                    },
                    "screen_description": {
                        "type": "STRING",
                        "description": "Goal-relevant screen state in ≤15 words (e.g., 'Settings page with Wi-Fi and Bluetooth toggles visible').",
                    },
                    "content_exhausted": {
                        "type": "BOOLEAN",
                        "description": "Set to true if you can visually confirm that the scrollable content (carousel, list) has reached its end and no new items will appear on further swiping.",
                    },
                    "memory_updates": {
                        "type": "OBJECT",
                        "description": "Optional key-value pairs to update in persistent memory. Use this to track progress (e.g., 'visited_card1': 'true').",
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
            "description": (
                "Signal that the user's goal has been fully achieved. "
                "Call this ONLY when the current screen state proves the goal is complete. "
                "Do NOT call this while there are still actions to perform — use execute_ui instead. "
                "Do NOT call this for intermediate progress checks — use validate_state instead. "
                "Do NOT call this unless visual evidence of completion is on the CURRENT screen."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Explanation of why the goal is considered complete.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": "Visual evidence from the current screen proving the goal is complete.",
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
            "description": (
                "Verify if the screen state matches specific criteria. "
                "Use this when the intent implies checking, validating, or verifying something. "
                "Do NOT use this when a UI action (tap, type, scroll) is needed — use execute_ui instead. "
                "Do NOT use this for final goal completion — use complete_goal to signal done, or verify_goal for a detailed completion check."
            ),
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
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the user's high-level goal is fully achieved after this validation.",
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
            "description": (
                "Verify if the user's overall goal has been fully completed by checking the current screen state. "
                "Do NOT use this for intermediate state checks (e.g., 'is the menu open?') — use validate_state instead. "
                "Do NOT use this when there are still UI actions to perform — use execute_ui instead."
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
            "description": (
                "Store important information, progress, or state in memory to remember it later. "
                "Use this to track what you've already done (e.g., category='visited', item='carousel_card_1'). "
                "Do NOT use this for transient observations already visible on screen — only store facts needed across multiple steps. "
                "Do NOT use this for data that can be re-derived from the current screenshot."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "category": {
                        "type": "STRING",
                        "description": (
                            "The kind of information being stored. "
                            "Use 'visited' for elements/screens already interacted with, "
                            "'progress' for step tracking in a flow, "
                            "'state' for captured app state facts (e.g., toggle on/off, current tab), "
                            "'data' for extracted values (e.g., price, name, count)."
                        ),
                        "enum": ["visited", "progress", "state", "data"],
                    },
                    "item": {
                        "type": "STRING",
                        "description": (
                            "Identifier for the specific thing being stored, in snake_case. "
                            "Examples: 'carousel_card_1', 'checkout_step', 'product_price', 'search_query'. "
                            "Use the SAME item name when recalling later."
                        ),
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
            "description": (
                "Retrieve previously stored information from memory. "
                "Use this to check your progress or recall specific details. "
                "You MUST use the exact same category and item values that were used when storing. "
                "Do NOT use this when the needed information is already visible on screen. "
                "Do NOT use this for keys you have not previously stored."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "category": {
                        "type": "STRING",
                        "description": (
                            "The category used when storing: "
                            "'visited', 'progress', 'state', or 'data'. "
                            "Must match the category used in the corresponding store_memory call."
                        ),
                        "enum": ["visited", "progress", "state", "data"],
                    },
                    "item": {
                        "type": "STRING",
                        "description": (
                            "The item identifier used when storing, in snake_case. "
                            "Examples: 'carousel_card_1', 'checkout_step', 'product_price'. "
                            "Must match the item used in the corresponding store_memory call."
                        ),
                    },
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Why this information is being retrieved.",
                    },
                },
                "required": ["category", "item", "assistant_message"],
            },
        }
