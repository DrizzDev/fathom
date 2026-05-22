"""
Pins the PROGRESS SAFETY rule that mandates the generic
``request_replan`` escape valve whenever the agent cannot make safe
forward progress on the active sub-goal.

The rule teaches the model the :class:`EscapeCategory` taxonomy rather
than a single symptom-specific instruction. A future prompt refactor
that drops any of the five typed categories breaks one of these tests.
"""

from __future__ import annotations

from fathom.core.prompts.gemini import GeminiPromptBuilder
from fathom.core.prompts.templates import TOOL_GUIDANCE
from fathom.schemas.escape import EscapeCategory


class TestProgressSafetyRule:
    """
    Behavioural pins for the request_replan prompt taxonomy.
    """

    def test_tool_guidance_lists_request_replan(self) -> None:
        """
        The tool selection block must enumerate ``request_replan`` so
        the model knows the tool exists.
        """

        assert "request_replan" in TOOL_GUIDANCE

    def test_tool_guidance_omits_legacy_tool_name(self) -> None:
        """
        The legacy ``report_screen_unactionable`` name must not appear —
        the atomic rename leaves no stale references for the model.
        """

        assert "report_screen_unactionable" not in TOOL_GUIDANCE

    def test_tool_guidance_contains_mandatory_progress_safety_block(self) -> None:
        """
        The PROGRESS SAFETY block must be present so the rule has a
        named home in the prompt.
        """

        assert "PROGRESS SAFETY (MANDATORY)" in TOOL_GUIDANCE
        assert "you MUST call request_replan" in TOOL_GUIDANCE

    def test_tool_guidance_teaches_every_escape_category(self) -> None:
        """
        Every :class:`EscapeCategory` value must appear in the prompt
        so the model can choose any of them. This pins the taxonomy
        contract: adding a category to the enum requires adding it to
        the prompt text in the same change.
        """

        for category in EscapeCategory:
            assert category.value in TOOL_GUIDANCE, (
                f"EscapeCategory.{category.name} ({category.value!r}) is missing from "
                "TOOL_GUIDANCE — the prompt rule must enumerate every typed category."
            )

    def test_tool_guidance_forbids_visual_snap_fallback(self) -> None:
        """
        The rule must explicitly forbid snapping to a visually similar
        but semantically unrelated label — picking the wrong manifest
        entry just because it looks like a button.
        """

        assert "snap to a visually similar but semantically unrelated label" in TOOL_GUIDANCE

    def test_tool_guidance_permits_bbox_grounding_when_manifest_lacks_target(self) -> None:
        """
        Pins Fathom's XML-optional invariant: the rule must accept a
        visually-grounded ``bbox`` as a valid alternative to ``label_id``
        when the target is rendered outside the element manifest (Canvas,
        custom overlays, video, web views). A rule that requires
        ``label_id`` unconditionally breaks vision-only operation.
        """

        assert "'bbox'" in TOOL_GUIDANCE
        assert "'label_id'" in TOOL_GUIDANCE
        assert "manifest is a hint, not a precondition" in TOOL_GUIDANCE

    def test_tool_guidance_escape_requires_both_paths_to_fail(self) -> None:
        """
        The escape mandate must require BOTH grounding paths to fail
        before request_replan is invoked. Escaping just because the
        manifest is empty would defeat vision-only operation.
        """

        assert (
            "no matching manifest label AND no element you can visually identify" in TOOL_GUIDANCE
        )

    def test_gemini_system_prompt_surfaces_progress_safety_block(self) -> None:
        """
        The end-to-end system prompt assembled by
        :class:`GeminiPromptBuilder` must surface the rule and the
        taxonomy so the production prompt cannot drift.
        """

        builder = GeminiPromptBuilder()
        system_prompt = builder.build()

        assert "request_replan" in system_prompt
        assert "PROGRESS SAFETY (MANDATORY)" in system_prompt
        for category in EscapeCategory:
            assert category.value in system_prompt

    def test_gemini_system_prompt_does_not_require_label_id_when_manifest_lacks_target(
        self,
    ) -> None:
        """
        The top-level system prompt must stay consistent with bbox fallback:
        manifest grounding is preferred, but not an unconditional requirement.
        """

        system_prompt = GeminiPromptBuilder().build()

        assert "MUST include 'label_id' from manifest for every interaction" not in system_prompt
        assert "Otherwise ground the action visually via bbox" in system_prompt
