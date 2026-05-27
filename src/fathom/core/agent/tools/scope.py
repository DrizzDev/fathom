from __future__ import annotations

from fathom.constants.tools import (
    BASE_TOOLS,
    VERIFICATION_KEYWORDS,
    VERIFICATION_TOOLS,
    ToolName,
)
from fathom.schemas.capabilities import RuntimeCapabilities
from fathom.schemas.tools import AllowedTools


class ToolScope:
    """Decides which tools the language model may invoke per turn."""

    def compute(
        self,
        *,
        intent: str,
        capabilities: RuntimeCapabilities,
    ) -> AllowedTools:
        """Return the allowed tool set for the intent and runtime."""

        names: set[ToolName] = set(BASE_TOOLS)

        if capabilities.hitl.enabled:
            names.add(ToolName.ASK_USER)

        if self.__requires_verification_tools(intent=intent):
            names.update(VERIFICATION_TOOLS)

        return AllowedTools(names=frozenset(names))

    def __requires_verification_tools(self, *, intent: str) -> bool:
        """Return whether the intent calls for verification tools."""

        lowered = intent.lower()
        return any(keyword in lowered for keyword in VERIFICATION_KEYWORDS)
