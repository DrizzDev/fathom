"""Tests for completion signal handling in tool schemas and parsing."""

from __future__ import annotations

from pydantic import ValidationError

from fathom.schemas.gemini_tools import ExecuteUIArgs, GeminiCompletionFlags


class TestGoalCompletedDefault:
    def test_goal_completed_defaults_false(self) -> None:
        """execute_ui no longer requires goal_completed — it defaults to False."""

        flags = GeminiCompletionFlags(sub_goal_completed=False)
        assert flags.goal_completed is False

    def test_sub_goal_completed_is_required(self) -> None:
        """sub_goal_completed is still mandatory."""

        with __import__("pytest").raises(ValidationError):
            GeminiCompletionFlags()  # type: ignore[call-arg]

    def test_execute_ui_without_goal_completed(self) -> None:
        """execute_ui args should parse without goal_completed field."""

        args = ExecuteUIArgs(
            assistant_message="tapping button",
            sub_goal_completed=False,
            actions=[
                {
                    "action_type": "tap",
                    "bbox": {"x": 500, "y": 500},
                    "rationale": "tap the button",
                    "is_valid": True,
                    "export_target": "Submit button",
                }
            ],
        )
        assert args.goal_completed is False
        assert args.sub_goal_completed is False
