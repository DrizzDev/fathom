from __future__ import annotations

from typing import Any, Dict, List, Optional


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
    def get_export_definitions(
        cls, *, action_ids: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns tool definitions for script export composition.

        Args:
            action_ids: When provided, constrains action ID fields to only
                        these values via enum in the tool schema.
        """

        return {"function_declarations": [cls.__emit_script(action_ids=action_ids)]}

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
                "signaling sub-goal completion via 'sub_goal_completed: true' "
                "rather than emitting an explicit 'tap' action on the app icon. The system will "
                "normalize app launch intents automatically. "
                "CRITICAL: For every UI action you MUST fill 'target_name' with a concrete, "
                "user-facing element label (e.g., 'Search box', 'Add to cart button'). "
                "Additionally, set 'script_target' ONLY when 'target_type' is 'positional' or "
                "'dynamic' to provide an abstracted phrase (e.g., 'the first search result'). "
                "NEVER use placeholders like 'UI Element', 'element', 'button', 'label', 'icon', "
                "'field', or 'text' as the only target description."
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
                                # --- Critical execution fields (prioritized) ---
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
                                    ],
                                },
                                "label_id": {
                                    "type": "STRING",
                                    "description": "The ID of the element from the manifest (e.g. '4'). REQUIRED if the element is in the manifest.",
                                },
                                "bbox": {
                                    "type": "OBJECT",
                                    "description": (
                                        "Target coordinates. x,y are the CENTER of the target element. "
                                        "REQUIRED for tap, type, swipe_*, and long_press actions. "
                                        "Use normalized values (0-1000) by default; "
                                        "use pixel values only with coord_system='pixel'."
                                    ),
                                    "properties": {
                                        "x": {"type": "INTEGER"},
                                        "y": {"type": "INTEGER"},
                                        "coord_system": {
                                            "type": "STRING",
                                            "enum": ["normalized", "pixel"],
                                            "description": "Coordinate system. Default is 'normalized' (0-1000).",
                                        },
                                    },
                                },
                                "target_name": {
                                    "type": "STRING",
                                    "description": (
                                        "REQUIRED for every UI action. The single canonical, "
                                        "user-facing name of the element (e.g., 'Search box', "
                                        "'Add to cart button', 'Settings tab'). This is the "
                                        "only target field you need to fill for stable targets. "
                                        "MUST NOT be a generic placeholder like 'element', "
                                        "'UI Element', 'button', 'label', or 'icon'. Choose the "
                                        "text a human tester would naturally say when referring "
                                        "to this element."
                                    ),
                                },
                                "text_to_type": {
                                    "type": "STRING",
                                    "description": "Text to type (only for 'type' action).",
                                },
                                "wait_duration": {
                                    "type": "NUMBER",
                                    "description": "Duration to wait in seconds (e.g. 2.0, 5.0). Use this for 'wait' actions to specify how long to pause.",
                                },
                                # --- Execution signals ---
                                "confidence": {
                                    "type": "NUMBER",
                                    "description": "Confidence level (0.0-1.0) for this action.",
                                },
                                "is_valid": {
                                    "type": "BOOLEAN",
                                    "description": "Self-correction: Is this action valid given the current screen state?",
                                },
                                # --- Non-critical metadata ---
                                "rationale": {
                                    "type": "STRING",
                                    "description": "Why this specific action is being taken.",
                                },
                                "condition": {
                                    "type": "STRING",
                                    "description": (
                                        "Optional human-readable guard text (e.g. 'Cookie banner visible', "
                                        "'Loading spinner active'). When omitted, conditional_type is used "
                                        "to derive a default."
                                    ),
                                },
                                "is_conditional": {
                                    "type": "BOOLEAN",
                                    "description": (
                                        "Set true when this action should run only under a visible guard. "
                                        "Implied automatically if conditional_type or condition is set."
                                    ),
                                },
                                "conditional_type": {
                                    "type": "STRING",
                                    "enum": ["blocker", "transient", "error", "optional"],
                                    "description": (
                                        "Conditional class. Setting this implies is_conditional=true. "
                                        "blocker (overlay/popup/permission dialog blocking the UI), "
                                        "transient (spinner, skeleton shimmer, splash screen), "
                                        "error (red/orange error banner, toast, validation message), "
                                        "optional (non-blocking informational element)."
                                    ),
                                },
                                "target_type": {
                                    "type": "STRING",
                                    "enum": ["stable", "positional", "dynamic"],
                                    "description": "How the target should be referenced in exported scripts: stable (fixed label), positional (ordinal in list), or dynamic (content that may change). Leave unset if unsure.",
                                },
                                "script_target": {
                                    "type": "STRING",
                                    "description": (
                                        "OPTIONAL — set ONLY when target_type is 'positional' "
                                        "or 'dynamic'. The abstracted natural-language phrase "
                                        "for the exported test script (e.g. 'the first search "
                                        "result', 'the promotional banner', 'the selected cart "
                                        "item'). For 'stable' targets, omit this field — "
                                        "target_name alone is sufficient. The phrase MUST be "
                                        "specific and user-facing, not a generic placeholder.\n"
                                        "POSITIONAL TARGETS (CRITICAL): When tapping items in a "
                                        "list, grid, carousel, or product catalog, NEVER use the "
                                        "specific product name. Use positional references: "
                                        "'the 1st product', 'the 3rd item', 'Add button for the "
                                        "1st item'. Product names change between runs."
                                    ),
                                },
                                "scroll_target": {
                                    "type": "STRING",
                                    "description": (
                                        "REQUIRED for all scroll/swipe actions. The DESTINATION element you "
                                        "are trying to bring into view — the specific item, label, or "
                                        "section header you want to tap or read after the scroll completes "
                                        "(e.g., 'Washington state', 'Vitamins and supplements', 'Lab tests "
                                        "and packages'). NEVER name the scrollable container ('State list "
                                        "container', 'Settings page') — that is not the target. Use the "
                                        "exact label text from the UI when possible. Must not be empty for "
                                        "swipe_up, swipe_down, swipe_left, swipe_right, or scroll. "
                                        "REASON: the exporter emits 'Scroll until you see <scroll_target>' "
                                        "verbatim, so naming the scrollable container instead of the "
                                        "destination produces a script that never terminates. The "
                                        "distinction between container and destination is the whole "
                                        "point of this field."
                                    ),
                                },
                                "wait_subject": {
                                    "type": "STRING",
                                    "description": (
                                        "REQUIRED for all wait actions. What we're waiting for (e.g., 'app to "
                                        "load', 'search results to appear', 'Home page content'). Describe the "
                                        "expected state or element. Must not be empty for wait actions. "
                                        "REASON: the runtime uses wait_subject to decide retry budget and "
                                        "to write trace/history lines like 'Wait for <wait_subject>'. A "
                                        "vague subject (e.g., 'loading') gives the LLM no signal on the "
                                        "next turn about whether the wait succeeded."
                                    ),
                                },
                                "validation_subject": {
                                    "type": "STRING",
                                    "description": (
                                        "REQUIRED for validate actions. A short noun phrase (max 8 words) "
                                        "that names the SPECIFIC visible thing being checked — the button, "
                                        "label, icon, field, card, page, section, toast, or banner you "
                                        "can point to in the screenshot. "
                                        "NEVER use the filler word 'element' — it is meaningless in a "
                                        "script line. Name the actual thing: 'Submit button enabled', "
                                        "'Cart total visible', 'Home tab selected', NOT 'Submit element' "
                                        "or 'Home element visible'. "
                                        "NO full sentences, NO 'I can see', NO 'the presence of', NO "
                                        "locations like 'at the bottom'. "
                                        "GOOD: 'categories visible', 'home button selected', "
                                        "'footer text visible', 'HealthTap homepage content loaded'. "
                                        "BAD: 'HealthTap homepage content, element visible' "
                                        "(the word 'element' is forbidden); "
                                        "'I can clearly see the footer at the bottom of the screen'. "
                                        "REASON: exported test scripts read validation_subject verbatim "
                                        "into 'Validate that <subject> is visible' lines, and the trace "
                                        "history uses the same field to describe the step on the next "
                                        "turn. Vague subjects produce assertions nobody can debug and "
                                        "context lines nobody can act on."
                                    ),
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
                                    "description": (
                                        "For validate actions: blocker (modal/scrim/permission dialog blocking content), "
                                        "transient (spinner, progress bar, or shimmer placeholder still visible), "
                                        "error (red/orange text, error icon, or 'try again' message on screen), "
                                        "or generic (general state check like toggle position or text presence)."
                                    ),
                                },
                                "wait_pattern": {
                                    "type": "STRING",
                                    "enum": ["ad", "splash", "load", "search", "generic"],
                                    "description": (
                                        "For wait actions: ad (full-screen interstitial ad with countdown/skip button), "
                                        "splash (branded launch screen with app logo, no interactive elements), "
                                        "load (spinner, progress bar, skeleton/shimmer placeholders, or 'Loading...' text), "
                                        "search (waiting for search results list to populate), or generic."
                                    ),
                                },
                            },
                            # Universally-required fields only. Conditionally-required
                            # fields (bbox, export_target, scroll_target, wait_subject,
                            # validation_subject) are enforced by the Pydantic validators
                            # on ExecuteAction in fathom.schemas.tool_args, keyed off
                            # action_type. Keeping them out of the schema-level 'required'
                            # list prevents Gemini from inventing placeholder values
                            # (e.g. all-zero bbox, 'none' export_target) for actions that
                            # genuinely do not need them, such as back/home/wait/validate.
                            "required": [
                                "action_type",
                                "rationale",
                                "is_valid",
                            ],
                        },
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
                        "description": "Describe ONLY what is visible on screen (e.g., 'Swiggy home page with search bar, Instamart tab, food categories'). Do NOT describe actions being performed or navigation intent. State the screen name and key visible elements.",
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
            "description": (
                "Verify if the screen state matches specific criteria by inspecting the screenshot. "
                "Use when the intent implies checking or verifying something: "
                "e.g., a toggle is green/on, a success toast is visible, a specific tab is highlighted, "
                "or an expected heading/text is displayed on screen."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Explanation of the verification result.",
                    },
                    "validation_subject": {
                        "type": "STRING",
                        "description": (
                            "Short noun phrase naming what is being validated "
                            "(e.g., 'Settings screen open', 'cart is empty'). "
                            "Same field name as ExecuteAction.validation_subject "
                            "so the same vocabulary works across both tools. "
                            "NEVER use the filler word 'element'."
                        ),
                    },
                    "condition_met": {
                        "type": "BOOLEAN",
                        "description": "True if the condition is met based on the visual evidence.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": (
                            "Visual evidence from the screenshot supporting the conclusion "
                            "(e.g., 'Toggle is green and positioned right', 'Success banner shows Order Confirmed', "
                            "'Error text not visible anywhere on screen')."
                        ),
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the current sub-goal is complete.",
                    },
                },
                "required": [
                    "assistant_message",
                    "validation_subject",
                    "condition_met",
                    "evidence",
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
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "Must be false unless the current sub-goal is complete.",
                    },
                },
                "required": ["question", "sub_goal_completed"],
            },
        }

    @staticmethod
    def __emit_script(
        action_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Definition for script export output tool.
        """

        # When action_ids are provided, constrain the schema so Gemini can
        # only output valid catalog IDs (prevents missing/extra/duplicated IDs).
        action_id_item: Dict[str, Any] = {"type": "STRING"}
        if action_ids:
            action_id_item["enum"] = list(action_ids)

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
                                "condition_type": {
                                    "type": "STRING",
                                    "enum": ["blocker", "transient", "error", "optional"],
                                    "description": (
                                        "Classification of this condition: blocker (popup/permission/consent "
                                        "that blocks progress), transient (loading/splash/spinner that will pass), "
                                        "error (error message that may appear), or optional (nice-to-have check)."
                                    ),
                                },
                                "action_ids": {
                                    "type": "ARRAY",
                                    "description": "Executable action IDs under this IF block. Must be selected from provided action catalog.",
                                    "items": action_id_item,
                                },
                            },
                            "required": ["condition", "action_ids"],
                        },
                    },
                    "remaining_action_ids": {
                        "type": "ARRAY",
                        "description": "Ordered executable action IDs outside IF blocks. Must be selected from provided action catalog.",
                        "items": action_id_item,
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
