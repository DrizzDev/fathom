from __future__ import annotations

import unittest

from fathom.constants.tools import ToolName
from fathom.core.agent.tools import ToolScope
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities


class ToolScopeTest(unittest.TestCase):
    """Pins the rules ToolScope uses to choose allowed tools per turn."""

    def test_autonomous_runtime_excludes_ask_user(self) -> None:
        """ASK_USER must not be exposed when no human is available."""

        scope = ToolScope()
        result = scope.compute(
            intent="open the app",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        self.assertNotIn(ToolName.ASK_USER, result.names)

    def test_hitl_runtime_includes_ask_user(self) -> None:
        """ASK_USER is exposed only when a human is available."""

        scope = ToolScope()
        result = scope.compute(
            intent="open the app",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

        self.assertIn(ToolName.ASK_USER, result.names)

    def test_base_tools_always_present(self) -> None:
        """EXECUTE_UI, STORE_MEMORY, RECALL_MEMORY are exposed on every turn."""

        scope = ToolScope()
        result = scope.compute(
            intent="x",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        self.assertIn(ToolName.EXECUTE_UI, result.names)
        self.assertIn(ToolName.STORE_MEMORY, result.names)
        self.assertIn(ToolName.RECALL_MEMORY, result.names)

    def test_verification_intent_adds_verification_tools(self) -> None:
        """Verification keywords in the intent add VERIFY_GOAL and VALIDATE_STATE."""

        scope = ToolScope()

        for keyword in ("verify", "check", "confirm", "validate"):
            result = scope.compute(
                intent=f"please {keyword} the order status",
                capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
            )

            self.assertIn(ToolName.VERIFY_GOAL, result.names, msg=keyword)
            self.assertIn(ToolName.VALIDATE_STATE, result.names, msg=keyword)

    def test_non_verification_intent_excludes_verification_tools(self) -> None:
        """Non-verification intents do not get VERIFY_GOAL/VALIDATE_STATE."""

        scope = ToolScope()
        result = scope.compute(
            intent="add item to cart",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        self.assertNotIn(ToolName.VERIFY_GOAL, result.names)
        self.assertNotIn(ToolName.VALIDATE_STATE, result.names)

    def test_verification_keyword_match_is_case_insensitive(self) -> None:
        """Case-insensitive match on verification keywords."""

        scope = ToolScope()
        result = scope.compute(
            intent="VERIFY login succeeded",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        self.assertIn(ToolName.VERIFY_GOAL, result.names)
