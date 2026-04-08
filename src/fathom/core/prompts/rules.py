"""Provider-neutral rule text blocks used by vision prompts.

These constants are pure product policy (coordinate rules, precision
rules, UI handling rules, action rules, tool-selection guidance) and are
composed into full prompts by ``fathom.core.prompts.policy``. They are
kept as plain strings so that any provider adapter can reuse them
verbatim.
"""

from __future__ import annotations

from types import MappingProxyType

# Coordinate system and confidence guidance
COORD_RULES = (
    "COORDINATE SYSTEM (CRITICAL):\n"
    "- GROUNDING FIRST: ALWAYS prefer label_id over coordinates. If the target exists in the "
    "Element Manifest, you MUST include its 'label_id' — label-snapped coordinates are exact, "
    "while predicted coordinates are approximate.\n"
    "- TEXT-FIRST GROUNDING (MANDATORY when label_id is unavailable): When the target carries "
    "ANY visible text — a label, a caption, a menu item, a button title — your coordinates MUST "
    "be the geometric center of that text's RENDERED GLYPHS, not the center of the surrounding "
    "button shape, icon, padding, or wrapper. Rendered text is the single most reliable visual "
    "anchor: you can see exactly where each character sits in the screenshot, while button "
    "rectangles and tap regions are inferred and routinely wrong by 20-100 units.\n"
    "  HOW TO COMPUTE:\n"
    "  1. Locate the visible text label that names the target.\n"
    "  2. Find the bounding box of the literal characters (left edge of the first letter to the "
    "right edge of the last letter; top edge of the tallest letter to the bottom edge of the "
    "lowest letter).\n"
    "  3. Place x,y at the geometric center of THAT character box.\n"
    "  4. Ignore the button outline, the icon, the padding, and the row background entirely.\n"
    "  WHEN TO USE TEXT-FIRST:\n"
    "  * Button with an icon + label → center of the LABEL TEXT, not the icon and not the "
    "button surface midpoint. The label glyphs are the unambiguous anchor.\n"
    "  * Tab bar item with icon + label → center of the LABEL TEXT, not the icon.\n"
    "  * Menu item, list row, link, chip, filter pill — center of the visible text.\n"
    "  * Card with a title and an image — center of the TITLE TEXT.\n"
    "  * Bottom-sheet action — center of the action label text.\n"
    "  * Confirmation dialog button — center of 'OK' / 'Cancel' / 'Confirm' text glyphs.\n"
    "  WHEN TEXT-FIRST DOES NOT APPLY (target has no text label):\n"
    "  * Icon-only button (X close, hamburger menu, FAB) → center of the icon glyph itself, "
    "NOT the surrounding whitespace.\n"
    "  * Toggle / switch / checkbox → center of the control itself.\n"
    "  * Slider thumb / drag handle → center of the draggable element.\n"
    "  * Image thumbnail tap → center of the image.\n"
    "  Even in these cases, target the smallest visible glyph, not the inferred tap area.\n"
    "- CENTER-OF-FIELD RULE (MANDATORY when no text and no icon glyph applies): x,y MUST be the "
    "geometric center of the SPECIFIC interactive field you want to activate — NOT the center "
    "of a wrapper, card, row, or container that holds it. Always drill down to the innermost "
    "interactive element.\n"
    "  BASIC ELEMENTS:\n"
    "  * Checkbox inside a list row → center of the checkbox itself, not the row.\n"
    "  * Text input with a leading icon → center of the editable text area, not the icon.\n"
    "  * Toggle switch inside a settings row → center of the toggle track, not the row.\n"
    "  COMPLEX LAYOUTS:\n"
    "  * Star rating → center of the specific star you want to tap (e.g., 4th star).\n"
    "  * Carousel dot indicator → center of the specific dot, not the dot strip.\n"
    "  * Stepper (+/−) → center of the + or − button glyph, not the quantity display.\n"
    "  * Slider thumb → center of the draggable thumb, not the track.\n"
    "  * Bottom sheet handle → center of the drag handle bar, not the sheet content.\n"
    "  OVERLAPPING ELEMENTS:\n"
    "  * Badge on an icon → if tapping the icon, target the icon center; if tapping the badge, target the badge center.\n"
    "  * Image with overlay text/button → target the overlay element's text, not the image center.\n"
    "  * Card with multiple tappable zones — target the SPECIFIC zone's text or icon, not the card center.\n"
    "  * Search bar with clear (X) button → if clearing, target the X glyph; if typing, target the text input area.\n"
    "- DEFAULT: Use normalized coordinates (0-1000). 0 = left/top edge, 1000 = right/bottom edge.\n"
    "- SMALL ELEMENTS: For small icons/buttons, be extra precise — an error of 20+ units will "
    "miss the target. Anchor on the visible glyph and double-check your x,y against the "
    "element's pixel position in the screenshot.\n"
    "- SELF-CHECK BEFORE EMITTING: Read the literal pixel position of the visible text or "
    "glyph you intend to tap. If your x,y does not land on those exact characters/strokes, "
    "correct it before submitting.\n"
    "- PIXEL MODE: Use raw pixels ONLY when you explicitly set coord_system='pixel'.\n"
    "- COORD_SYSTEM CONSISTENCY: coord_system must match the numbers you provide."
)

CONFIDENCE_RULES = "CONFIDENCE: 0.9+ clear match, 0.7-0.89 certain. Below 0.7 indicates ambiguity."

# Bbox precision rules (Immutable)
_PRECISION_RULES_RAW = {
    "text": "TEXT: Target the CENTER of the visible text glyphs themselves — the literal characters as rendered in the screenshot. Exclude padding, margins, leading icons, button outlines, or surrounding card/container.",
    "icon": "ICONS/BUTTONS: If the button has a visible text LABEL, target the center of the label text glyphs (text-first). For icon-only buttons, target the center of the icon glyph itself, NOT the surrounding tap area or button rectangle.",
    "input": "INPUTS: Target the CENTER of the editable text area. Exclude labels, hint icons, or wrapper borders. If the field has a leading icon, target to the right of it.",
    "list": "LIST ITEMS: Target the CENTER of the specific item's tappable region (text or thumbnail), NOT the center of the entire row or list.",
    "toggle": "TOGGLES/SWITCHES: Target the CENTER of the toggle track or switch element, NOT the center of the settings row containing it.",
    "checkbox": "CHECKBOXES/RADIO: Target the CENTER of the checkbox or radio circle itself, NOT the label text or the row.",
    "tab": "TABS: Target the CENTER of the specific tab label or icon, NOT the center of the tab bar. For bottom navigation, target the icon+label pair for the specific tab.",
    "dropdown": "DROPDOWNS/SELECTS: Target the CENTER of the select field or dropdown trigger, NOT the form label or section header above it.",
    "chip": "CHIPS/TAGS: Target the CENTER of the specific chip's text/surface, NOT the chip group container. For dismissible chips, target the X icon to remove.",
    "slider": "SLIDERS: For dragging, target the CENTER of the thumb/handle. For setting a value, target the position on the track corresponding to the desired value.",
    "card": "CARDS WITH ACTIONS: If a card has multiple tappable zones (title, thumbnail, action buttons), target the specific zone you intend to interact with, NOT the card center.",
    "stepper": "STEPPERS (+/−): Target the CENTER of the specific + or − button, NOT the quantity number between them.",
    "search": "SEARCH BARS: Target the CENTER of the text input area. If clearing, target the clear (X) icon. Exclude the search icon on the left.",
    "fab": "FABs: Target the CENTER of the floating button circle itself, regardless of nearby content or screen edges.",
}
PRECISION_RULES = MappingProxyType(_PRECISION_RULES_RAW)

# Action rules (Immutable)
_ACTION_RULES_RAW = {
    "scroll": (
        "SWIPE: swipe_left (carousel), swipe_right, swipe_up (lists), swipe_down. "
        "Target the center of the scrollable region (exclude fixed headers/footers). "
        "Do NOT use 'scroll' as an action_type; always use the appropriate swipe_* variant."
    ),
    "wait": (
        "WAIT: Use if screen shows a SPINNER (circular animation), LOADING TEXT, PROGRESS BAR, "
        "or SKELETON/SHIMMER (gray rectangular placeholders where content will appear). "
        "CRITICAL: Even if XML elements are present, if the visual is a Skeleton/Shimmer, you MUST WAIT. "
        "Also wait for pull-to-refresh indicators and full-screen interstitial ads with countdown. "
        "Include wait_duration (default 2.0)."
    ),
    "validate": (
        "VALIDATE: Use when the next best step is an explicit check rather than a touch action. "
        "Visual cues: confirm toggle is on/off (green vs gray), banner/toast text is visible, "
        "expected page title or section header is displayed, error message is present or absent.\n"
        "MULTI-CHECK RULE: When the sub-goal requires validating MULTIPLE conditions "
        "(e.g., 'validate that X, Y, and Z are present'), check ALL conditions in a SINGLE "
        "validate action. List all conditions in the rationale. If all pass, set "
        "'sub_goal_completed: true' in the same tool call. Do NOT validate one element per step.\n"
        "VALIDATION_SUBJECT FORMAT (CRITICAL): validation_subject must be a SHORT noun phrase "
        "(max 8 words). State ONLY the element and its expected state. "
        "NEVER include reasoning, evidence, descriptions, locations, or full sentences. "
        "NEVER use first-person language ('I am', 'I can see', 'I will', 'I do not'). "
        "NEVER start with 'Validating', 'Checking', 'Confirming'. "
        "Write as a third-person assertion, not a narration. "
        "Put reasoning in 'rationale', NOT in validation_subject.\n"
        "GOOD: 'Instamart tab selected', 'item added to cart'\n"
        "BAD: 'I am on the Instamart section as indicated by the active tab indicator'"
    ),
    "zoom": "ZOOM: 'zoom_in' to enlarge, 'zoom_out' to shrink. Target the relevant region.",
    "type": (
        "CRITICAL - TAP BEFORE TYPE: Always tap input first to gain focus. "
        "Then generate 'type' with text_to_type. "
        "Both target SAME coordinates. text_to_type is a literal value (no prefixes)."
    ),
}
ACTION_RULES = MappingProxyType(_ACTION_RULES_RAW)

# UI handling rules (Immutable)
_UI_RULES_RAW = {
    "goal_lock": (
        "GOAL LOCK: Never change intent. Dismiss blockers FIRST, then proceed. "
        "Common blockers: cookie banners (bottom bar with Accept/Reject), permission dialogs "
        "(system-styled Allow/Deny), privacy popups, login walls, app update dialogs, "
        "rating requests (star icons with Maybe Later), survey popups."
    ),
    "dropdown": (
        "DROPDOWN DISMISS: If a dropdown, picker, or action sheet is open (floating list over dimmed content, "
        "or bottom tray with options), dismiss it first via X, 'Close', 'Done', or tapping outside."
    ),
    "overlay": (
        "OVERLAY DETECTION: Identify app overlays by visual cues: dimmed/darkened background scrim, "
        "centered card/dialog, bottom sheet rising from screen edge, or full-screen modal with X/close button. "
        "Dismiss these BEFORE interacting with underlying content. Ignore only fixed system chrome (status bar, nav bar)."
    ),
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

TARGET PRECISION:
- {PRECISION_RULES["text"]}
- {PRECISION_RULES["icon"]}
- {PRECISION_RULES["input"]}
- {PRECISION_RULES["list"]}

ACTIONS:
- {ACTION_RULES["type"]}
- {ACTION_RULES["scroll"]}
- {ACTION_RULES["wait"]}
- {ACTION_RULES["validate"]}
- {ACTION_RULES["zoom"]}

STRICT FORMAT: Return only valid tool calls using provided schema fields.
"""

TOOL_GUIDANCE = """
TOOL SELECTION & VALIDATION:
- execute_ui: PRIMARY tool for interactions (tap, type, swipe, scroll, wait, validate, zoom).
  * Delta telemetry is MANDATORY on every execute_ui call: always include both delta_observed (boolean) and delta_confidence (0.0-1.0).
  * For explicit checks/validation, prefer execute_ui with action_type='validate'.
  * For any guard-based step, set conditional_type (blocker/transient/error/optional). is_conditional is implied automatically.
  * Provide condition text when visible (e.g. 'Cookie consent banner visible', 'Loading spinner active'). If omitted, the system derives a default from conditional_type.
  * For overlay/popup dismissal, set conditional_type='blocker'. The condition text and is_conditional flag are filled in for you.
  * Evaluate is_valid and validation_reason for EVERY action.
  * If action is risky/ambiguous, set is_valid=False and explain.
  * COMMAND NAMING: In 'target' and 'natural_language_target', use GENERIC, RELATIVE DESCRIPTIONS (e.g., 'Tap on edit CVV box', 'Tap on Submit button', 'Tap on 1st search result').
    DO NOT use IDs like 'edt_cvv' or 'button_23'. Describe WHAT it is functionally.
  * STATE TRACKING (CRITICAL): Use the 'memory_updates' field to atomically track your progress.
    Example: memory_updates={'selected_days': 'Mon,Tue', 'roadmap_step_1': 'complete'}
    ALWAYS use this to "tick off" requirements from the user's goal as you complete them.
- validate_state: Legacy fallback for explicit checks when no immediate UI action is required.
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
