from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List

from fathom.constants.tools import ToolName


class ToolRegistry:
    """
    Registry for tool definitions used by the Vision Language Model.
    """

    @classmethod
    def get_all_definitions(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns all tool definitions in a format compatible with Gemini API.
        """

        return {"function_declarations": [factory() for factory in cls.__factories().values()]}

    @classmethod
    def definitions(
        cls,
        *,
        names: FrozenSet[ToolName],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return the function-declarations payload for the requested tools.
        """

        factories = cls.__factories()
        ordered = [factories[name]() for name in factories if name in names]

        return {"function_declarations": ordered}

    @classmethod
    def __factories(cls) -> Dict[ToolName, Callable[[], Dict[str, Any]]]:
        """
        Return the name → declaration-factory map in catalog order.
        """

        return {
            ToolName.ASK_USER: cls.__ask_user,
            ToolName.EXECUTE_UI: cls.__execute_ui,
            ToolName.VERIFY_GOAL: cls.__verify_goal,
            ToolName.STORE_MEMORY: cls.__store_memory,
            ToolName.RECALL_MEMORY: cls.__recall_memory,
            ToolName.VALIDATE_STATE: cls.__validate_state,
        }

    @staticmethod
    def __execute_ui() -> Dict[str, Any]:
        """
        Definition for execute_ui tool.
        """

        return {
            "name": "execute_ui",
            "description": (
                "Execute one UI action on the device to achieve a specific sub-goal or the final goal. "
                "Use this to interact with the app, including explicit validation checks via action_type='validate'. "
                "IMPORTANT: When launching a target app (when a package_name is known), prefer "
                "signaling app completion via 'goal_completed: true' or 'sub_goal_completed: true' "
                "rather than emitting an explicit 'tap' action on the app icon. The system will "
                "normalize app launch intents automatically. "
                "CRITICAL: For every UI action you MUST provide the script-owned semantic field: "
                "tap/type use export_target or script_target, scroll/swipe use scroll_target, "
                "wait uses wait_subject, validate uses validation_subject, and store uses capture. "
                "For tap/type, target_name is the exact visible execution target; export_target or "
                "script_target is the stable replay target. Choose the visible UI role and purpose "
                "from the screen, such as a dropdown, field, row, card, chip, button, icon, tab, or "
                "menu item. NEVER use placeholders like 'UI Element', 'element', 'button', 'label', "
                "'icon', 'field', or 'text' as the only target description."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "A message to the user explaining the reasoning behind these actions.",
                    },
                    "action": {
                        "type": "OBJECT",
                        "description": "The single UI action to execute for this turn.",
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
                                    "store",
                                ],
                            },
                            "label_id": {
                                "type": "STRING",
                                "description": "The ID of the element from the manifest (e.g. '4'). REQUIRED when the target or scroll container exists in the manifest.",
                            },
                            "capture": {
                                "type": "OBJECT",
                                "description": (
                                    "For action_type='store' ONLY. Store an actual value read from "
                                    "the screen or task context as a named variable. subject names "
                                    "what the user asked to capture; value is the concrete captured "
                                    "text. Never use store for agent memory or to ask the user, and "
                                    "never invent the captured value."
                                ),
                                "properties": {
                                    "name": {
                                        "type": "STRING",
                                        "description": "Variable name to store the value under.",
                                    },
                                    "subject": {
                                        "type": "STRING",
                                        "description": "What the intent asked to capture.",
                                    },
                                    "value": {
                                        "type": "STRING",
                                        "description": "Concrete value read from the screen or task context.",
                                    },
                                },
                                "required": ["name", "subject", "value"],
                            },
                            "bbox": {
                                "type": "OBJECT",
                                "description": (
                                    "Bounding box for the action target. x,y are TOP-LEFT and "
                                    "width,height extend right/down. The rectangle MUST hug the "
                                    "visible glyph or icon pixels of the specific interactive "
                                    "control only — exclude surrounding card, shadow, halo, and "
                                    "empty padding. The runtime taps the GEOMETRIC CENTER of this "
                                    "rectangle. Use normalized values (0-1000) for visually "
                                    "estimated regions; set coordinate_system='pixel' when "
                                    "copying manifest or screenshot-space bounds."
                                ),
                                "properties": {
                                    "x": {"type": "INTEGER"},
                                    "y": {"type": "INTEGER"},
                                    "width": {"type": "INTEGER"},
                                    "height": {"type": "INTEGER"},
                                    "coordinate_system": {
                                        "type": "STRING",
                                        "enum": ["normalized", "pixel"],
                                        "description": "Coordinate system for bbox values. Default is 'normalized' (0-1000). Set 'pixel' only when using raw pixel coordinates.",
                                    },
                                },
                            },
                            "target_name": {
                                "type": "STRING",
                                "description": (
                                    "EXACT visible text or glyph label of the target control as "
                                    "rendered on screen, verbatim. Do NOT append interaction-kind "
                                    "suffixes such as 'button', 'icon', 'tab', 'link', 'chip', "
                                    "'cell', or 'row' — action_type already carries the "
                                    "interaction. For unlabelled icons, describe the visible "
                                    "symbol concisely (e.g., 'Magnifying glass'). MUST NOT be a "
                                    "generic placeholder like 'element', 'UI Element', 'button', "
                                    "'label', or 'icon' alone."
                                ),
                            },
                            "text_to_type": {
                                "type": "STRING",
                                "description": "Text to type (only for 'type' action).",
                            },
                            "wait_duration": {
                                "type": "NUMBER",
                                "description": (
                                    "Optional duration to wait in seconds (e.g. 2.0, 5.0). "
                                    "For wait actions, wait_subject is the required semantic field; "
                                    "duration only tunes how long to pause."
                                ),
                            },
                            # --- Execution signals ---
                            "confidence": {
                                "type": "NUMBER",
                                "description": "Required confidence level (0.0-1.0) for this action.",
                            },
                            "is_valid": {
                                "type": "BOOLEAN",
                                "description": "Self-correction: Is this action valid given the current screen state?",
                            },
                            # --- Reasoning and conditional execution fields ---
                            "rationale": {
                                "type": "STRING",
                                "description": "Why this specific action is being taken.",
                            },
                            "validation_reason": {
                                "type": "STRING",
                                "description": "Reasoning for the validity judgment.",
                            },
                            "condition": {
                                "type": "STRING",
                                "description": (
                                    "Present-tense sentence describing the visible guard "
                                    "this action depends on. MANDATORY whenever "
                                    "is_conditional=true. Examples: 'Popup is visible', "
                                    "'Section is collapsed', 'Error banner is displayed', "
                                    "'Main menu is visible'. For a conditional wait, "
                                    "describe the awaited state (e.g., 'Search results are visible'), "
                                    "not the act of waiting."
                                ),
                            },
                            "is_conditional": {
                                "type": "BOOLEAN",
                                "description": (
                                    "Set true when this action should be executed only "
                                    "under a visible guard. When true, the 'condition' "
                                    "field is REQUIRED — describe the guard in the present tense."
                                ),
                            },
                            "conditional_type": {
                                "type": "STRING",
                                "enum": ["blocker", "transient", "error", "optional"],
                                "description": (
                                    "Condition class when is_conditional=true: "
                                    "blocker (overlay/popup/permission dialog blocking the UI), "
                                    "transient (spinner, skeleton shimmer, or splash screen that will auto-resolve), "
                                    "error (red/orange error banner, toast, or validation message), "
                                    "optional (non-blocking informational element)."
                                ),
                            },
                            "overlay_detected": {
                                "type": "BOOLEAN",
                                "description": (
                                    "Set true only when the screenshot shows an unrelated overlay blocking "
                                    "the active target (dimmed scrim, modal dialog, permission prompt, "
                                    "or banner). Do not mark a dialog, menu, action sheet, or bottom sheet "
                                    "as an overlay when it contains the control, option, value, or "
                                    "confirmation required by the active step. Dismissal actions must also "
                                    "set condition to the specific visible overlay/dialog state."
                                ),
                            },
                            "export_target": {
                                "type": "STRING",
                                "description": (
                                    "Canonical phrase for this action in exported test scripts. "
                                    "REQUIRED for tap and type actions unless script_target is provided. "
                                    "This is the stable replay target, separate from the exact visible "
                                    "target_name used for execution. Must be specific and human-readable "
                                    "by combining the control's screen role and purpose. Use the actual "
                                    "visible role, such as dropdown, field, row, card, chip, button, icon, "
                                    "tab, or menu item. For dynamic controls, name the control purpose; "
                                    "do not copy runtime values such as addresses, user data, ETA text, "
                                    "cart totals, or content-description sentences. "
                                    "NEVER use generic placeholders like 'element', 'UI Element', "
                                    "'button', 'label', 'icon', 'field', or 'text' alone."
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
                                    "When target_type is 'positional' or 'dynamic', the "
                                    "exact natural-language phrase that should appear in "
                                    "exported scripts. Use the actual visible role and purpose "
                                    "when the target is a dropdown, field, row, card, chip, button, "
                                    "icon, tab, menu item, or ordinal result. "
                                    "Treat this field as REQUIRED whenever target_type is "
                                    "'positional' or 'dynamic'. The phrase MUST be specific "
                                    "and user-facing, not a generic placeholder or runtime value."
                                ),
                            },
                            "scroll_target": {
                                "type": "STRING",
                                "description": (
                                    "REQUIRED for all scroll/swipe actions. The element or section being "
                                    "scrolled to find (e.g., 'Vitamins and supplements', 'Lab tests and "
                                    "packages'). Use the exact phrase from the UI when possible. "
                                    "When the manifest exposes the scrollable container, ground the action with that "
                                    "container's label_id and use scroll_target only for the intended content. "
                                    "Must not be empty for swipe_up, swipe_down, swipe_left, swipe_right, or scroll."
                                ),
                            },
                            "wait_subject": {
                                "type": "STRING",
                                "description": (
                                    "REQUIRED for all wait actions. What we're waiting for (e.g., 'app to "
                                    "load', 'search results to appear', 'Home page content'). Describe the "
                                    "expected state or element. Must not be empty for wait actions."
                                ),
                            },
                            "validation_subject": {
                                "type": "STRING",
                                "description": (
                                    "REQUIRED for validate actions. The state or subject being asserted "
                                    "(e.g., 'login screen', 'cart page', 'order confirmation'). Do not use "
                                    "an incidental visible field as the subject when it is only evidence "
                                    "for a broader state."
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
                        "required": ["action_type", "rationale", "is_valid", "confidence"],
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
                    "action",
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
                        "description": (
                            "Visual evidence from the screenshot supporting the conclusion "
                            "(e.g., 'Toggle is green and positioned right', 'Success banner shows Order Confirmed', "
                            "'Error text not visible anywhere on screen')."
                        ),
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
                    "goal_completion_reason": {
                        "type": "STRING",
                        "description": "Explicit reason why the goal is complete (e.g., 'Order placed successfully', 'Login screen visible as requested'). Required when goal_completed=true.",
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the current sub-goal is completed based on screen state.",
                    },
                    "subgoal_completion_reason": {
                        "type": "STRING",
                        "description": "Explicit reason why the sub-goal is complete (e.g., 'Product page rendered after tapping result'). Required when sub_goal_completed=true.",
                    },
                    "completion_criteria_met": {
                        "type": "ARRAY",
                        "description": "List of observable criteria that triggered completion (e.g., ['title_visible', 'price_visible']). Optional; use for multi-condition completions.",
                        "items": {"type": "STRING"},
                    },
                    "current_screen": {
                        "type": "STRING",
                        "description": "The actual screen currently displayed.",
                    },
                    "assertion": {
                        "type": "STRING",
                        "description": (
                            "Crisp observable condition this verification asserts on the current "
                            "screen (e.g. 'Cart contains Diet Coke x1'). Prefer a checkable "
                            "condition over a screen description."
                        ),
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
