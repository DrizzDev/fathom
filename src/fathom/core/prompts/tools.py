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
                cls.__ask_user(),
            ]
        }

    @classmethod
    def get_export_definitions(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns tool definitions for script export composition.
        """

        return {"function_declarations": [cls.__emit_script()]}

    @staticmethod
    def __execute_ui() -> Dict[str, Any]:
        """
        Definition for execute_ui tool.
        """

        return {
            "name": "execute_ui",
            "description": "Execute a sequence of UI actions on the device to achieve a specific sub-goal or the final goal. Use this to interact with the app, including explicit validation checks via action_type='validate'.",
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
                                        "validate",
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
                                "label_id": {
                                    "type": "STRING",
                                    "description": "The ID of the element from the manifest (e.g. '4'). REQUIRED if the element is in the manifest.",
                                },
                                "bbox": {
                                    "type": "OBJECT",
                                    "description": "Bounding box for the action target. x,y are TOP-LEFT and width,height extend right/down. Use normalized values (0-1000) by default; use pixel values only with coord_system='pixel'.",
                                    "properties": {
                                        "x": {"type": "INTEGER"},
                                        "y": {"type": "INTEGER"},
                                        "width": {"type": "INTEGER"},
                                        "height": {"type": "INTEGER"},
                                        "coord_system": {
                                            "type": "STRING",
                                            "enum": ["normalized", "pixel"],
                                            "description": "Coordinate system for bbox values. Default is 'normalized' (0-1000). Set 'pixel' only when using raw pixel coordinates.",
                                        },
                                    },
                                },
                                "text_to_type": {
                                    "type": "STRING",
                                    "description": "Text to type (only for 'type' action).",
                                },
                                "wait_duration": {
                                    "type": "NUMBER",
                                    "description": "Duration to wait in seconds (e.g. 2.0, 5.0). Use this for 'wait' actions to specify how long to pause.",
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
                                "condition": {
                                    "type": "STRING",
                                    "description": "Condition required (e.g. 'Popup is visible', 'Section is collapsed', 'Error displayed')",
                                },
                                "is_conditional": {
                                    "type": "BOOLEAN",
                                    "description": "Set true when this action should be executed only under a visible guard condition.",
                                },
                                "conditional_type": {
                                    "type": "STRING",
                                    "enum": ["blocker", "transient", "error", "optional"],
                                    "description": "Optional condition class when is_conditional=true. Use blocker/transient/error/optional.",
                                },
                                "overlay_detected": {
                                    "type": "BOOLEAN",
                                    "description": "Set true when this action is specifically handling an overlay/popup blocker.",
                                },
                                "target_type": {
                                    "type": "STRING",
                                    "enum": ["stable", "positional", "dynamic"],
                                    "description": "How the target should be referenced in exported scripts: stable (fixed label), positional (ordinal in list), or dynamic (content that may change). Leave unset if unsure.",
                                },
                                "script_target": {
                                    "type": "STRING",
                                    "description": "When target_type is positional or dynamic, the exact phrase for script export (e.g. 'the first search result', 'the promotional banner'). Omit for stable.",
                                },
                            },
                            "required": ["action_type", "rationale", "is_valid"],
                        },
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the user's high-level goal is fully achieved after these actions.",
                    },
                    "content_exhausted": {
                        "type": "BOOLEAN",
                        "description": "True when scrolling/swiping reveals no more new content and the list/feed appears exhausted.",
                    },
                    "previous_screen_summary": {
                        "type": "STRING",
                        "description": "Optional short summary of the previous screen for semantic delta analysis.",
                    },
                    "current_screen_summary": {
                        "type": "STRING",
                        "description": "Optional short summary of the current screen for semantic delta analysis.",
                    },
                    "delta_observed": {
                        "type": "BOOLEAN",
                        "description": "REQUIRED: Whether a meaningful screen change was observed since the previous screenshot.",
                    },
                    "delta_reasoning": {
                        "type": "STRING",
                        "description": "Optional rationale behind the delta_observed hint.",
                    },
                    "delta_confidence": {
                        "type": "NUMBER",
                        "description": "REQUIRED: Confidence score (0.0-1.0) for delta_observed.",
                    },
                    "visible_anchors": {
                        "type": "ARRAY",
                        "description": "Optional key UI anchors currently visible (e.g., top card title, footer label).",
                        "items": {"type": "STRING"},
                    },
                    "top_anchor": {
                        "type": "STRING",
                        "description": "Optional anchor near the top of the viewport.",
                    },
                    "bottom_anchor": {
                        "type": "STRING",
                        "description": "Optional anchor near the bottom of the viewport.",
                    },
                    "memory_updates": {
                        "type": "OBJECT",
                        "description": "Optional key-value pairs to update in persistent memory. Use this to track progress (e.g., 'visited_card1': 'true').",
                    },
                },
                "required": [
                    "assistant_message",
                    "actions",
                    "goal_completed",
                    "delta_observed",
                    "delta_confidence",
                ],
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

    @staticmethod
    def __ask_user() -> Dict[str, Any]:
        """
        Definition for ask_user tool.
        """

        return {
            "name": "ask_user",
            "description": "Ask the human user for clarification, guidance, or assistance when stuck or uncertain. Use this ONLY in interactive mode.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "question": {
                        "type": "STRING",
                        "description": "The specific question or request for the user.",
                    },
                    "context": {
                        "type": "STRING",
                        "description": "Context explaining what you were trying to do and why you need help.",
                    },
                },
                "required": ["question"],
            },
        }

    @staticmethod
    def __emit_script() -> Dict[str, Any]:
        """
        Definition for script export output tool.
        """

        return {
            "name": "emit_script",
            "description": "Return structured script sections derived only from allowed step action lines. Do not paraphrase executable actions.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "conditional_blocks": {
                        "type": "ARRAY",
                        "description": "Ordered IF blocks for condition-scoped actions using action IDs.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "condition": {
                                    "type": "STRING",
                                    "description": "Condition text for IF block.",
                                },
                                "action_ids": {
                                    "type": "ARRAY",
                                    "description": "Executable action IDs under this IF block. Must be selected from provided action catalog.",
                                    "items": {"type": "STRING"},
                                },
                            },
                            "required": ["condition", "action_ids"],
                        },
                    },
                    "remaining_action_ids": {
                        "type": "ARRAY",
                        "description": "Ordered executable action IDs outside IF blocks. Must be selected from provided action catalog.",
                        "items": {"type": "STRING"},
                    },
                    "action_validations": {
                        "type": "OBJECT",
                        "description": (
                            "Optional mapping of action ID to intermediate validation line "
                            "(e.g., {'A3': 'Validate that results are visible'}). "
                            "Each value must start with 'Validate'."
                        ),
                        "additionalProperties": {"type": "STRING"},
                    },
                    "final_validation": {
                        "type": "STRING",
                        "description": "Final goal-validation line. Must start with 'Validate'.",
                    },
                },
                "required": ["remaining_action_ids", "final_validation"],
            },
        }
