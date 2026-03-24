from __future__ import annotations

# Coordinate system and confidence guidance
COORD_RULES = (
    "COORDINATES: Use NORMALIZED coords (0-1000 grid). "
    "bbox.x and bbox.y MUST be the TOP-LEFT corner of the element bounding box. "
    "width and height extend rightward and downward from that corner."
)

CONFIDENCE_RULES = "CONFIDENCE: 0.9+ clear match, 0.7-0.89 certain. Below 0.7 indicates ambiguity."

# Bbox precision rules
PRECISION_RULES = {
    "text": "TEXT: Bbox wraps ONLY visible text pixels. Exclude padding/margins/icons.",
    "icon": "ICONS/BUTTONS: Snap bbox TIGHTLY to visible edges. Exclude whitespace/containers.",
    "input": "INPUTS: Wrap editable area only (borders/background). Exclude labels/icons.",
    "list": "LIST ITEMS: Wrap ONLY the specific item's text (not entire list or row).",
}

# Action rules
ACTION_RULES = {
    "scroll": (
        "SCROLL/SWIPE: swipe_left (carousel), swipe_right, swipe_up (lists), swipe_down. "
        "Bbox wraps scrollable region only (exclude fixed headers/footers). "
        "If the screen does not change after swiping (same items visible, carousel at last dot, "
        "bounce effect), set content_exhausted=true to signal end of scrollable content. "
        "Optionally include delta_observed, delta_reasoning, and anchor fields when available."
    ),
    "wait": (
        "WAIT: ONLY for active loading (skeleton, spinner, 'Loading...' text). "
        "NOT for sparse screens with visible text/buttons. Include wait_duration_ms (default 2000ms)."
    ),
    "zoom": "",
    "type": (
        "CRITICAL - TAP BEFORE TYPE: Always tap input first to gain focus. "
        "Then generate 'type' with text_to_type. "
        "Both target SAME bbox. text_to_type is a literal value (no prefixes)."
    ),
}

# UI handling rules
UI_RULES = {
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
    "conditional": (
        "CONDITIONAL: When handling ANY optional/dynamic element (e.g. 'Close' on popup, 'Expand' on collapsed section, "
        "'Skip' on tutorial, 'Retry' on error), SET THE `condition` FIELD to the visible requirement "
        "(e.g. 'Promo popup is visible', 'Section is collapsed', 'Error message is displayed')."
    ),
}

# Backward-compatible aggregated blocks used by current builder
COMMON_RULES = f"""
{COORD_RULES}
{CONFIDENCE_RULES}

{UI_RULES["icon_vs_page"]}
{UI_RULES["dropdown"]}
{UI_RULES["overlay"]}
{UI_RULES["goal_lock"]}
{UI_RULES["conditional"]}

BBOX PRECISION:
- {PRECISION_RULES["text"]}
- {PRECISION_RULES["icon"]}
- {PRECISION_RULES["input"]}
- {PRECISION_RULES["list"]}

ACTIONS:
- {ACTION_RULES["type"]}
- {ACTION_RULES["scroll"]}
- {ACTION_RULES["wait"]}

STRICT FORMAT: Return only valid tool calls using provided schema fields.
"""

TOOL_GUIDANCE = """
TOOL ROUTING & RESPONSE FORMAT:
- execute_ui: ALL physical UI interactions (tap, type, swipe, scroll, wait).
  * For explicit checks/validation, use execute_ui with action.action_type='validate'.
  * For overlay/popup dismissal actions, set action.overlay_detected=true and action.condition to the exact visible guard
    (e.g., "Promotional overlay is visible", "Cookie consent popup is visible").
  * Keep assistant_message and validation_reason concise, grammatical, and evidence-based.
  * ALWAYS set screen_description (≤15 words) and evaluate is_valid.
  * NAMING: Use generic labels, not IDs. List items -> positional ("1st result"). Unique UI -> visible label ("Submit button").
  * TARGET_TYPE: stable (fixed UI, omit script_target), positional (list/grid item, set script_target to ordinal), dynamic (changing content, set script_target to generic desc). Wait actions -> always dynamic. Omit if uncertain.
  * Return fields: action (object), assistant_message (str), is_valid (bool). ALWAYS include evidence or short reasoning.
- complete_goal: Goal done signal. Call ONLY when current screen proves completion with visual evidence.
- validate_state: Legacy fallback only. Prefer execute_ui(action_type='validate').
- Validation: If the goal requires validation (e.g., price or presence), include a short evidence note in assistant_message or evidence.
  Use clear statements like "The first lemon item shows a $0.40 price."
- Optional no-XML delta hints: provide `delta_observed`, `delta_reasoning`, `delta_confidence`,
  `previous_screen_summary`, `current_screen_summary`, and anchors (`visible_anchors`, `top_anchor`, `bottom_anchor`)
  when they are clear. These are soft signals for loop detection.
- verify_goal: Detailed goal completion verification.
- store_memory / recall_memory: Persistent cross-step memory (use same category+item keys).

ERROR RECOVERY:
- If action fails (no UI change, button not tapped): Use 'back', scroll to refocus, or try alternative target.
- If overlay/popup blocks target: Dismiss first (X, 'Close', 'Done'), then retry.
- If action loops (same screen after 3+ attempts): Change strategy (scroll, navigate, wait for load).

FOR ALL TOOL CALLS:
- Respond ONLY with valid JSON tool calls using the provided schema.
- Do NOT output plain text, markdown, or explain your reasoning outside of fields.
- Confidence < 0.7 = mark is_valid=false and explain briefly in assistant_message.
"""

RESPONSE_DIRECTIVE = (
    "RESPONSE: Return one primary tool call (execute_ui, complete_goal, or verify_goal). "
    "You MAY include store_memory or recall_memory alongside the primary call. "
    "Execute the next best step via a tool call. Return only valid tool calls as JSON. No plain text, markdown, or explanations."
)
