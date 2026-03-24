from __future__ import annotations

# All heuristic target resolution functions have been removed.
# Target resolution is now handled by the VLM via the authoritative
# `export_target` field on Action. See gemini_tools.py for the
# schema enforcement that rejects generic targets at parse time.
