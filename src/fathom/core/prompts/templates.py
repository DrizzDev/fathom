from __future__ import annotations

from types import MappingProxyType

# Coordinate system and confidence guidance
COORD_RULES = (
    "COORDINATE SYSTEM (CRITICAL):\n"
    "- GROUNDING: IF the target exists in the Element Manifest, you MUST include its 'label_id' (e.g., label_id='4').\n"
    "- VISION FALLBACK: If no label exists, use 'bbox' with normalized coordinates (0-1000).\n"
    "- FORMULA: norm_x = (x / width) * 1000, norm_y = (y / height) * 1000.\n"
    "- PIXEL MODE: Only use pixel coordinates if you explicitly set coord_system='pixel'. Default is normalized."
)

CONFIDENCE_RULES = "CONFIDENCE: 0.9+ clear match, 0.7-0.89 certain. Below 0.7 indicates ambiguity."

# Bbox precision rules (Immutable)
_PRECISION_RULES_RAW = {
    "text": "TEXT: Bbox wraps ONLY visible text pixels. Exclude padding/margins/icons.",
    "icon": "ICONS/BUTTONS: Snap bbox TIGHTLY to visible edges. Exclude whitespace/containers.",
    "input": "INPUTS: Wrap editable area only (borders/background). Exclude labels/icons.",
    "list": "LIST ITEMS: Wrap ONLY the specific item's text (not entire list or row).",
}
PRECISION_RULES = MappingProxyType(_PRECISION_RULES_RAW)

# Action rules (Immutable)
_ACTION_RULES_RAW = {
    "scroll": (
        "SCROLL/SWIPE: swipe_left (carousel), swipe_right, swipe_up (lists), swipe_down. "
        "Bbox wraps scrollable region only (exclude fixed headers/footers)."
    ),
    "wait": (
        "WAIT: Use if screen shows a SPINNER, LOADING TEXT, or SKELETON/SHIMMER (gray shapes). "
        "CRITICAL: Even if XML elements are present, if the visual is a Skeleton/Shimmer, you MUST WAIT. "
        "Include wait_duration (default 2.0)."
    ),
    "zoom": "ZOOM: 'zoom_in' to enlarge, 'zoom_out' to shrink. Target the relevant region.",
    "type": (
        "CRITICAL - TAP BEFORE TYPE: Always tap input first to gain focus. "
        "Then generate 'type' with text_to_type. "
        "Both target SAME bbox. text_to_type is a literal value (no prefixes)."
    ),
}
ACTION_RULES = MappingProxyType(_ACTION_RULES_RAW)

# UI handling rules (Immutable)
_UI_RULES_RAW = {
    "goal_lock": (
        "GOAL LOCK: Never change intent. Dismiss blockers (cookie consent, permission prompts, "
        "privacy popups, login walls, app update dialogs, survey popups, rating requests) FIRST, "
        "then proceed."
    ),
    "dropdown": "DROPDOWN DISMISS: Dismiss obstructions first: down arrows, X, 'Close'/'Done'.",
    "overlay": "OVERLAY: Ignore system overlays. Focus on actual app UI elements.",
    "icon_vs_page": (
        "ICON vs PAGE: Distinguish small icons vs full-page content. "
        "If menu/page is open, interact WITHIN it."
    ),
}
UI_RULES = MappingProxyType(_UI_RULES_RAW)

COMMON_RULES = f"""
{COORD_RULES}
{CONFIDENCE_RULES}

{UI_RULES["icon_vs_page"]}
{UI_RULES["dropdown"]}
{UI_RULES["overlay"]}
{UI_RULES["goal_lock"]}

BBOX PRECISION:
- {PRECISION_RULES["text"]}
- {PRECISION_RULES["icon"]}
- {PRECISION_RULES["input"]}
- {PRECISION_RULES["list"]}

ACTIONS:
- {ACTION_RULES["type"]}
- {ACTION_RULES["scroll"]}
- {ACTION_RULES["wait"]}
- {ACTION_RULES["zoom"]}

STRICT FORMAT: Return only valid tool calls using provided schema fields.
"""

TOOL_GUIDANCE = """
TOOL SELECTION & VALIDATION:
- execute_ui: PRIMARY tool for interactions (tap, type, swipe, scroll, wait, zoom).
  * Evaluate is_valid and validation_reason for EVERY action.
  * If action is risky/ambiguous, set is_valid=False and explain.
  * COMMAND NAMING: In 'target' and 'natural_language_target', use GENERIC, RELATIVE DESCRIPTIONS (e.g., 'Tap on edit CVV box', 'Tap on Submit button', 'Tap on 1st search result').
    DO NOT use IDs like 'edt_cvv' or 'button_23'. Describe WHAT it is functionally.
  * STATE TRACKING (CRITICAL): Use the 'memory_updates' field to atomically track your progress.
    Example: memory_updates={'selected_days': 'Mon,Tue', 'roadmap_step_1': 'complete'}
    ALWAYS use this to "tick off" requirements from the user's goal as you complete them.
- validate_state: Use for explicit state checks when no immediate UI action is required.
- verify_goal: Use for explicit completion checks.
- store_memory: Secondary tool. Use ONLY for saving complex text data that doesn't fit in execute_ui.
- recall_memory: Check what you've already done to avoid repeating actions.
- ask_user: Use this tool to ask the user for help or clarification when you are stuck or confused.

MEMORY STRATEGY:
- The system has NO implicit memory of what you "meant" to do. You MUST write it down.
- If you select 'Monday', you MUST write memory_updates={'monday': 'selected'}.
- If you don't write it, you WILL forget it when the screen changes.
"""

STUCK_PROMPT = "SYSTEM ALERT: You are stuck in a repetitive loop (same action/target multiple times with no progress). DO NOT try the same action again. You MUST use the 'ask_user' tool to ask the human for help immediately. This is a mandatory requirement."

# Summarization system instruction (for GCC milestone creation)
SUMMARIZATION_SYSTEM = """You are an expert at analyzing mobile UI automation execution traces.

Your task is to create a structured milestone summary that helps an AI agent understand:
1. What was accomplished in this segment
2. Key actions that led to success
3. Any challenges or failures encountered

Focus on STATE CHANGES and OUTCOMES, not routine navigation.
Be concise but informative - the agent needs to quickly understand progress."""
