"""
Frozen test-only copy of the pre-refactor :class:`ToolScope` and its supporting
intent-keyword constants. Used exclusively by the migration parity test to lock
the new-vs-old truth table. Delete one release after the rollout completes.
"""

from __future__ import annotations

from typing import FrozenSet

from fathom.constants.tools import BASE_TOOLS, ToolName
from fathom.schemas.capabilities import RuntimeCapabilities
from fathom.schemas.tools import AllowedTools

_LEGACY_VERIFICATION_TOOLS: FrozenSet[ToolName] = frozenset(
    {ToolName.VERIFY_GOAL, ToolName.VALIDATE_STATE},
)
_LEGACY_VERIFICATION_KEYWORDS: FrozenSet[str] = frozenset(
    {"verify", "check", "confirm", "validate"},
)


class _LegacyToolScope:
    """
    Pre-refactor behavior: intent keyword match decides verification-tool exposure.
    """

    def compute(self, *, intent: str, capabilities: RuntimeCapabilities) -> AllowedTools:
        """
        Replicate the legacy compute path verbatim for parity assertions.
        """

        names: set[ToolName] = set(BASE_TOOLS)

        if capabilities.hitl.enabled:
            names.add(ToolName.ASK_USER)

        if self.__requires_verification_tools(intent=intent):
            names.update(_LEGACY_VERIFICATION_TOOLS)

        return AllowedTools(names=frozenset(names))

    @staticmethod
    def __requires_verification_tools(*, intent: str) -> bool:
        """
        Legacy intent-keyword match for verification gating.
        """

        lowered = intent.lower()
        return any(keyword in lowered for keyword in _LEGACY_VERIFICATION_KEYWORDS)
