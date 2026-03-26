from __future__ import annotations

# All heuristic inference functions have been removed.
# Condition inference is now handled by the VLM via the authoritative
# `condition` field on Action (required when is_conditional=True).
# Scroll target inference replaced by required `scroll_target` field.
# Wait subject inference replaced by required `wait_subject` field.
# See gemini_tools.py for schema enforcement.
