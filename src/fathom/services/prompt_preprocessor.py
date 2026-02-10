import re
from typing import Any, Dict

class PromptPreprocessor:
    """Code-based intent analysis to reduce prompt tokens."""

    @staticmethod
    def extract_hints(intent: str, current_activity: str = "") -> Dict[str, Any]:
        """Extract deterministic info from intent string."""
        hints = {}
        # Screen name extraction
        match = re.search(r'on the ["\']?([^"\']+)["\']?\s*(screen|page)', intent, re.I)
        if match:
            hints["target_screen"] = match.group(1).strip()

        # Typing text extraction
        match = re.search(r'(?:type|enter|input)\s+["\']([^"\']+)["\']', intent, re.I)
        if match:
            hints["typing_text"] = match.group(1)

        # Navigation detection
        if hints.get("target_screen") and current_activity:
            # Simple heuristic: if target screen not in current activity name
            if hints["target_screen"].lower() not in current_activity.lower():
                hints["needs_navigation"] = True

        return hints

    @staticmethod
    def build_context_prefix(hints: Dict[str, Any]) -> str:
        """Build conditional prompt prefix from hints."""
        parts = []
        if hints.get("typing_text"):
            parts.append(f"[HINT] Text to type: \"{hints['typing_text']}\"")
        if hints.get("needs_navigation"):
            parts.append(f"[HINT] May need to navigate to: {hints['target_screen']}")
        return "\n".join(parts)
