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
            "description": (
                "Execute a sequence of UI actions on the device to achieve a specific sub-goal "
                "or the final goal. Use this to interact with the app, including explicit "
                "validation checks via action_type='validate'. "
                "IMPORTANT: When launching a target app (when a package_name is known), prefer "
                "signaling app completion via 'goal_completed: true' or 'sub_goal_completed: true' "
                "rather than emitting an explicit 'tap' action on the app icon. The system will "
                "normalize app launch intents automatically. "
                "CRITICAL: For every UI action you MUST provide a concrete, user-facing target "
                "phrase via 'target_name' or 'script_target' (e.g., 'Search box', "
                "'Add to cart button', 'the first search result'). NEVER use placeholders like "
                "'UI Element', 'element', 'button', 'label', 'icon', 'field', or 'text' as the "
                "only target description."
            ),
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
                                    "description": "The type of action to perform. Use swipe_* for all scrolling gestures.",
                                    "enum": [
                                        "tap",
                                        "type",
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
                                    "description": (
                                        "Descriptive, user-facing name of the element "
                                        "(e.g., 'Search box', 'Add to cart button', "
                                        "'Settings tab'). MUST NOT be a generic placeholder "
                                        "like 'element', 'UI Element', 'button', 'label', "
                                        "or 'icon'. Always choose the text a human tester "
                                        "would naturally say when referring to this element."
                                    ),
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
                                    "description": (
                                        "When target_type is 'positional' or 'dynamic', the "
                                        "exact natural-language phrase that should appear in "
                                        "exported scripts (e.g. 'the first search result', "
                                        "'the promotional banner', 'the selected cart item'). "
                                        "Treat this field as REQUIRED whenever target_type is "
                                        "'positional' or 'dynamic'. The phrase MUST be specific "
                                        "and user-facing, not a generic placeholder."
                                    ),
                                },
                                "scroll_target": {
                                    "type": "STRING",
                                    "description": "For scroll/swipe actions: the element or section being scrolled to find (e.g., 'Vitamins and supplements', 'Lab tests and packages'). Use the exact phrase from the UI when possible.",
                                },
                                "wait_subject": {
                                    "type": "STRING",
                                    "description": "For wait actions: what we're waiting for (e.g., 'app to load', 'search results to appear', 'Home page content'). Describe the expected state or element.",
                                },
                                "validation_subject": {
                                    "type": "STRING",
                                    "description": "For validate actions: what specifically is being validated (e.g., 'login status', 'banner visibility', 'item alignment'). Be specific about the validation target.",
                                },
                                "target_is_generic": {
                                    "type": "BOOLEAN",
                                    "description": "Set to true when this action taps/selects a non-specific target (e.g., 'any item', 'random category', 'first result'). Signals that target should be generalized in export.",
                                },
                                "target_element_type": {
                                    "type": "STRING",
                                    "enum": [
                                        "button",
                                        "icon",
                                        "option",
                                        "link",
                                        "field",
                                        "text",
                                        "checkbox",
                                    ],
                                    "description": "For tap/interact actions: the element type/role (button, icon, option, etc.). Helps refine target descriptions when product-specific elements are tapped.",
                                },
                                "validation_pattern": {
                                    "type": "STRING",
                                    "enum": ["blocker", "transient", "error", "generic"],
                                    "description": "For validate actions: the pattern category - blocker (permission/popup/consent), transient (loading/spinner), error (network/validation error), or generic check.",
                                },
                                "wait_pattern": {
                                    "type": "STRING",
                                    "enum": ["ad", "splash", "load", "search", "generic"],
                                    "description": "For wait actions: the wait category - ad (ad to finish), splash (app splash screen), load (content loading), search (search results), or generic.",
                                },
                            },
                            "required": ["action_type", "rationale", "is_valid"],
                        },
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the user's high-level goal is fully achieved after these actions.",
                    },
                    "goal_completion_reason": {
                        "type": "STRING",
                        "description": "Explicit reason why the goal is complete (e.g., 'Order placed successfully', 'Feature verified on screen'). Required when goal_completed=true.",
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the current sub-goal is completed after these actions. CONSTRAINT: You CANNOT skip sub-goals. All sub-goals must be executed in order. If you cannot complete the current sub-goal, ask the user for help or explain the blockage. Do NOT emit any signal that means 'skip this sub-goal'.",
                    },
                    "subgoal_completion_reason": {
                        "type": "STRING",
                        "description": "Explicit reason why the sub-goal is complete (e.g., 'Item added to cart', 'User authenticated'). Required when sub_goal_completed=true.",
                    },
                    "completion_criteria_met": {
                        "type": "ARRAY",
                        "description": "List of criteria/conditions that triggered completion (e.g., ['payment_processed', 'order_confirmed']). Use for multi-condition completions.",
                        "items": {"type": "STRING"},
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
                        "description": "Whether a meaningful screen change was observed since the previous screenshot. Strongly RECOMMENDED when the model can assess semantic deltas; omit only when unsure or when no comparison is possible.",
                    },
                    "delta_reasoning": {
                        "type": "STRING",
                        "description": "Optional rationale behind the delta_observed hint.",
                    },
                    "delta_confidence": {
                        "type": "NUMBER",
                        "description": "Confidence score (0.0-1.0) for delta_observed. Strongly RECOMMENDED whenever delta_observed is provided; omit only when the model cannot estimate confidence.",
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
                    "sub_goal_completed",
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
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True only if the overall goal is complete.",
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the current sub-goal is complete.",
                    },
                },
                "required": [
                    "assistant_message",
                    "condition_to_verify",
                    "condition_met",
                    "evidence",
                    "goal_completed",
                    "sub_goal_completed",
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
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the current sub-goal is completed based on screen state.",
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
                    "sub_goal_completed",
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
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "Must be false unless the overall goal is complete.",
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "Must be false unless the current sub-goal is complete.",
                    },
                },
                "required": ["question", "goal_completed", "sub_goal_completed"],
            },
        }

    @staticmethod
    def __emit_script() -> Dict[str, Any]:
        """
        Definition for script export output tool.
        """

        return {
            "name": "emit_script",
            "description": (
                "Return structured script sections derived only from allowed step action lines. "
                "Do not paraphrase executable actions. Rendered scripts use IF <condition> on one line "
                "and the opening { on the following line before indented block body lines."
            ),
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
                            "Optional map of catalog action_id -> intermediate validation line after that action "
                            "(e.g. list or results visible right after a search or scroll). Each value must start "
                            "with 'Validate'. Use for mid-flow checks; do not put those in final_validation."
                        ),
                        "additionalProperties": {"type": "STRING"},
                    },
                    "final_validation": {
                        "type": "STRING",
                        "description": (
                            "Single terminal UI-state line after the last catalog action. Must start with 'Validate'. "
                            "State only: what screen/page/primary content is visible or displayed. One short clause—"
                            "no tap/click/type/select/navigate/search instructions (those belong in catalog actions "
                            "or action_validations). No chained 'and then' procedures."
                        ),
                    },
                },
                "required": ["remaining_action_ids", "final_validation"],
            },
        }
