from __future__ import annotations

import unittest

from fathom.constants.tools import BASE_TOOLS, VERIFICATION_TOOLS, ToolName
from fathom.core.prompts.templates import build_tool_guidance
from fathom.schemas.tools import AllowedTools


class ToolGuidanceTest(unittest.TestCase):
    """Covers the dynamic system-prompt guidance assembled from AllowedTools."""

    @staticmethod
    def __hitl_tools() -> AllowedTools:
        """Allowed tool set for an HITL-capable runtime."""

        return AllowedTools(names=BASE_TOOLS | VERIFICATION_TOOLS | {ToolName.ASK_USER})

    @staticmethod
    def __autonomous_tools() -> AllowedTools:
        """Allowed tool set for an autonomous runtime (no ASK_USER)."""

        return AllowedTools(names=BASE_TOOLS | VERIFICATION_TOOLS)

    def test_omits_legacy_tool_names(self) -> None:
        """Guidance must not advertise tools that no longer exist."""

        guidance = build_tool_guidance(tools=self.__hitl_tools())

        self.assertNotIn("request_replan", guidance)
        self.assertNotIn("report_screen_unactionable", guidance)

    def test_contains_mandatory_progress_safety_block(self) -> None:
        """The prompt guidance must include the mandatory progress-safety rule."""

        guidance = build_tool_guidance(tools=self.__hitl_tools())

        self.assertIn("PROGRESS SAFETY (MANDATORY)", guidance)

    def test_forbids_visual_snap_fallback(self) -> None:
        """Guidance must forbid semantically unrelated visual snap fallback."""

        guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertIn(
            "snap to a visually similar but semantically unrelated label",
            guidance,
        )

    def test_permits_bbox_grounding_when_manifest_lacks_target(self) -> None:
        """Guidance must allow visual bbox grounding when manifest labels are insufficient."""

        guidance = build_tool_guidance(tools=self.__hitl_tools())

        self.assertIn("'bbox'", guidance)
        self.assertIn("'label_id'", guidance)
        self.assertIn("manifest is a hint, not a precondition", guidance)

    def test_describes_ask_user_only_when_allowed(self) -> None:
        """The ask_user tool description must appear iff ASK_USER is allowed."""

        self.assertIn("ask_user:", build_tool_guidance(tools=self.__hitl_tools()))
        self.assertNotIn("ask_user:", build_tool_guidance(tools=self.__autonomous_tools()))

    def test_fallback_rule_matches_runtime_capability(self) -> None:
        """The fallback safety rule must steer toward ask_user only when it's exposed."""

        hitl_guidance = build_tool_guidance(tools=self.__hitl_tools())
        autonomous_guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertIn("ask the user instead of guessing", hitl_guidance)
        self.assertNotIn("ask the user instead of guessing", autonomous_guidance)
        self.assertIn("deliberate recovery action", autonomous_guidance)
