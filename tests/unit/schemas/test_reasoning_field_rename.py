"""Tests for the reasoning-field rename (reasoning/reason -> rationale).

Both ``AnalysisResult`` and ``PlanResult`` were renamed to use the
canonical ``rationale`` field. Each model also has a
``mode='before'`` validator that accepts the old field name and
rewrites it, so checkpoints written before the rename still load
cleanly.
"""

from __future__ import annotations

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult, PlanResult


def _stub_action() -> Action:
    return Action(
        action_type=ActionType.TAP,
        rationale="r",
        target="x",
        natural_language_target="x",
    )


class TestAnalysisResultRename:
    def test_rationale_field_is_accepted(self) -> None:
        result = AnalysisResult(
            action=_stub_action(),
            rationale="fresh reasoning",
            screen_description="screen",
        )
        assert result.rationale == "fresh reasoning"
        assert not hasattr(result, "reasoning")

    def test_legacy_reasoning_field_is_renamed_on_load(self) -> None:
        """Checkpoints written before the rename have ``reasoning``; the
        validator must rewrite it to ``rationale`` transparently."""

        payload = {
            "action": _stub_action().model_dump(),
            "reasoning": "legacy reasoning",
            "screen_description": "screen",
        }
        result = AnalysisResult.model_validate(payload)
        assert result.rationale == "legacy reasoning"

    def test_rationale_wins_when_both_are_provided(self) -> None:
        """Defensive: if somehow both fields are in a dict, prefer the
        canonical one to avoid silent downgrade."""

        payload = {
            "action": _stub_action().model_dump(),
            "reasoning": "legacy",
            "rationale": "canonical",
            "screen_description": "screen",
        }
        result = AnalysisResult.model_validate(payload)
        assert result.rationale == "canonical"


class TestPlanResultRename:
    def test_rationale_field_is_accepted(self) -> None:
        plan = PlanResult(
            step=None,
            is_complete=True,
            rationale="fresh reasoning",
        )
        assert plan.rationale == "fresh reasoning"
        assert not hasattr(plan, "reason")

    def test_legacy_reason_field_is_renamed_on_load(self) -> None:
        """Checkpoints written before the rename have ``reason``; the
        validator must rewrite it to ``rationale`` transparently."""

        payload = {
            "reason": "legacy reason",
            "is_complete": True,
            "step": None,
        }
        plan = PlanResult.model_validate(payload)
        assert plan.rationale == "legacy reason"

    def test_rationale_wins_when_both_are_provided(self) -> None:
        payload = {
            "reason": "legacy",
            "rationale": "canonical",
            "is_complete": True,
            "step": None,
        }
        plan = PlanResult.model_validate(payload)
        assert plan.rationale == "canonical"
