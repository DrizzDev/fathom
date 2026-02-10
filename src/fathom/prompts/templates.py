from __future__ import annotations

# SHARED RULES AND FRAGMENTS
COMMON_RULES = """
ICON vs MENU/PAGE: Distinguish ICONS/BUTTONS (small, nav bars) vs MENU/PAGE CONTENT (large, fills screen).
If menu/page is open, interact WITHIN it, don't tap icon again.

DROPDOWN DISMISS: If dropdowns/selectors obstruct content, dismiss first: down arrows (▼), X buttons (×), 'Close'/'Done' buttons.

TEXT ELEMENT PRECISION: For text elements (buttons, labels, search suggestions, list items, menu items) EXCEPT INPUT FIELDS, bbox must tightly wrap ONLY the visible text content. Exclude padding, margins, icons, and surrounding whitespace. Measure the actual text pixels, not the container or touch target.

ICON/BUTTON PRECISION: Snap bbox TIGHTLY to the visible edges of the icon graphic or button text. Exclude whitespace, padding, and background containers. Do not wrap the full touch target.

INPUT FIELD PRECISION: Bbox must tightly wrap editable text area (borders/background), not labels/icons.

LIST ITEMS/SEARCH SUGGESTIONS: For search suggestions, list items, or dropdown options, bbox must tightly wrap ONLY the specific item's text content (typically 200-400px width, 50-100px height). Do NOT include the entire list, multiple items, or surrounding whitespace. Focus on the FIRST/TARGET item's visible text only.

OVERLAY HANDLING: Ignore system overlays. Focus on ACTUAL app UI elements behind overlays.

GOAL LOCK: Never change user intent. If blockers appear, choose action that progresses toward SAME intent.

SWIPE/SCROLL REGIONS: Identify and tightly wrap the RELEVANT scrollable region (e.g., vertical list, horizontal carousel).
Exclude fixed headers, footers, navigation bars, and non-scrollable areas. Bbox must match the visible scrollable area exactly.
SWIPE DIRECTIONS: swipe_left (right→left), swipe_right (left→right), swipe_up (bottom→top), swipe_down (top→bottom).
Use swipe_left/swipe_right for horizontal carousels (filter chips, product rows, tabs).
Use swipe_up/swipe_down for vertical lists (product lists, search results, menus).
SCROLL: Use 'scroll' for vertical scrolling in the center of screen (defaults to center if bbox not specified).
Bbox for scroll should wrap the scrollable content area (typically full width, excludes fixed nav/headers).
For horizontal swipe regions (carousels), bbox should wrap the carousel row (typically full width, height of one row).
For vertical swipe regions (lists), bbox should wrap the list content (typically full width, excludes headers/footers).

WAIT ACTION: Use 'wait' action ONLY when screen is actively loading.
No bbox required. Include wait_duration_ms (default 2000ms, typically 1000-5000ms).
Return wait action ONLY if screen shows: skeleton placeholders (grey animated rectangles), loading spinner, progress indicator, or explicit 'Loading...' text.
Do NOT use wait for sparse screens that have readable text, buttons, or navigation elements - those are valid interactive screens.

ZOOM ACTIONS: Use 'zoom_in' to enlarge content (pinch open) and 'zoom_out' to shrink content (pinch close).
Bbox should target the region to zoom (e.g. map area, image view). If unsure, use screen center/main content area.

STRICT FORMAT: Return ONLY valid responses using the provided tools.
"""

TOOL_GUIDANCE = """
TOOL SELECTION & VALIDATION:
- execute_ui: The PRIMARY tool. Use for all interactions (tap, type, swipe, scroll).
  * CRITICAL: You MUST evaluate 'is_valid' (true/false) and 'validation_reason' for EVERY action.
  * Self-Correction: If an action seems risky or the element is ambiguous, set is_valid=False and explain why.
- validate_state: Use ONLY for explicit final verification where no further actions are needed.
"""
