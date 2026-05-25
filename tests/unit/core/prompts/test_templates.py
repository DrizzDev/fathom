from __future__ import annotations

from fathom.core.prompts.templates import TOOL_GUIDANCE


class TestToolGuidance:
    """
    Covers prompt guidance constants from templates.py.
    """

    def test_tool_guidance_does_not_list_request_replan(self) -> None:
        """
        Runtime re-planning is disabled, so guidance must not advertise it.
        """

        assert "request_replan" not in TOOL_GUIDANCE

    def test_tool_guidance_omits_legacy_tool_name(self) -> None:
        """
        Legacy screen-unactionable tool names must not remain in guidance.
        """

        assert "report_screen_unactionable" not in TOOL_GUIDANCE

    def test_tool_guidance_contains_mandatory_progress_safety_block(self) -> None:
        """
        The prompt guidance must include the mandatory progress-safety rule.
        """

        assert "PROGRESS SAFETY (MANDATORY)" in TOOL_GUIDANCE
        assert "ask the user instead of guessing" in TOOL_GUIDANCE

    def test_tool_guidance_forbids_visual_snap_fallback(self) -> None:
        """
        Guidance must forbid semantically unrelated visual snap fallback.
        """

        assert "snap to a visually similar but semantically unrelated label" in TOOL_GUIDANCE

    def test_tool_guidance_permits_bbox_grounding_when_manifest_lacks_target(self) -> None:
        """
        Guidance must allow visual bbox grounding when manifest labels are insufficient.
        """

        assert "'bbox'" in TOOL_GUIDANCE
        assert "'label_id'" in TOOL_GUIDANCE
        assert "manifest is a hint, not a precondition" in TOOL_GUIDANCE

    def test_tool_guidance_escape_requires_both_paths_to_fail(self) -> None:
        """
        Guidance must require manifest and visual grounding to fail before asking.
        """

        assert (
            "no matching manifest label AND no element you can visually identify" in TOOL_GUIDANCE
        )
