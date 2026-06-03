from __future__ import annotations

import unittest

from fathom.constants.tools import BASE_TOOLS, VERIFICATION_TOOLS, ToolName
from fathom.core.prompts.templates import build_tool_guidance
from fathom.schemas.tools import AllowedTools


class ToolGuidanceTest(unittest.TestCase):
    """
    Covers the dynamic system-prompt guidance assembled from AllowedTools.
    """

    @staticmethod
    def __hitl_tools() -> AllowedTools:
        """
        Allowed tool set for an HITL-capable runtime.
        """

        return AllowedTools(names=BASE_TOOLS | VERIFICATION_TOOLS | {ToolName.ASK_USER})

    @staticmethod
    def __autonomous_tools() -> AllowedTools:
        """
        Allowed tool set for an autonomous runtime (no ASK_USER).
        """

        return AllowedTools(names=BASE_TOOLS | VERIFICATION_TOOLS)

    def test_omits_legacy_tool_names(self) -> None:
        """
        Guidance must not advertise tools that no longer exist.
        """

        guidance = build_tool_guidance(tools=self.__hitl_tools())

        self.assertNotIn("request_replan", guidance)
        self.assertNotIn("report_screen_unactionable", guidance)

    def test_contains_mandatory_progress_safety_block(self) -> None:
        """
        The prompt guidance must include the mandatory progress-safety rule.
        """

        guidance = build_tool_guidance(tools=self.__hitl_tools())

        self.assertIn("PROGRESS SAFETY (MANDATORY)", guidance)

    def test_forbids_visual_snap_fallback(self) -> None:
        """
        Guidance must forbid semantically unrelated visual snap fallback.
        """

        guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertIn(
            "snap to a visually similar but semantically unrelated label",
            guidance,
        )

    def test_marks_condition_field_mandatory_when_conditional(self) -> None:
        """
        Guidance must instruct the planner to fill condition whenever is_conditional=true.
        """

        guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertIn("'condition' field is MANDATORY", guidance)

    def test_describes_conditional_wait_in_present_tense(self) -> None:
        """
        Guidance must teach the planner to write a conditional wait as the awaited state.
        """

        guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertIn("present tense", guidance)
        self.assertIn("conditional wait", guidance)

    def test_drops_misleading_default_guard_text_fallback(self) -> None:
        """
        The previous wording suggested conditional_type could replace condition; remove it.
        """

        guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertNotIn(
            "conditional_type is used for default guard text",
            guidance,
        )

    def test_permits_bbox_grounding_when_manifest_lacks_target(self) -> None:
        """
        Guidance must allow visual bbox grounding when manifest labels are insufficient.
        """

        guidance = build_tool_guidance(tools=self.__hitl_tools())

        self.assertIn("'bbox'", guidance)
        self.assertIn("'label_id'", guidance)
        self.assertIn("manifest is a hint, not a precondition", guidance)

    def test_describes_ask_user_only_when_allowed(self) -> None:
        """
        The ask_user tool description must appear iff ASK_USER is allowed.
        """

        self.assertIn("ask_user:", build_tool_guidance(tools=self.__hitl_tools()))
        self.assertNotIn("ask_user:", build_tool_guidance(tools=self.__autonomous_tools()))

    def test_fallback_rule_matches_runtime_capability(self) -> None:
        """
        The fallback safety rule must steer toward ask_user only when it's exposed.
        """

        hitl_guidance = build_tool_guidance(tools=self.__hitl_tools())
        autonomous_guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertIn("ask the user instead of guessing", hitl_guidance)
        self.assertIn("deliberate recovery action", autonomous_guidance)
        self.assertNotIn("ask the user instead of guessing", autonomous_guidance)

    def test_bbox_precision_directive_is_present(self) -> None:
        """
        Guidance must teach the planner to hug visible glyph extent in bbox.
        """

        guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertIn("BBOX PRECISION", guidance)
        self.assertIn("hug the visible glyph", guidance)

    def test_target_name_must_not_contain_interaction_kind(self) -> None:
        """
        Guidance forbids interaction-kind suffixes in target_name.
        """

        guidance = build_tool_guidance(tools=self.__autonomous_tools())

        self.assertIn("EXACT visible text", guidance)
        self.assertIn("Do NOT append interaction-kind suffixes", guidance)
