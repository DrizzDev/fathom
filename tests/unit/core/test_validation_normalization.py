"""Tests for cross-tool validate-action shape normalization.

After the validation field-zoo cleanup, all three paths that produce a
validate-kind Action (``execute_ui`` with action_type='validate',
``validate_state``, and ``verify_goal``) must emit the same
``Action`` shape:

* ``action_type == ActionType.VALIDATE``
* ``validation_subject`` — sanitized short noun phrase
* ``rationale`` — merged reason + evidence
* no ``validation_reason`` field (deleted; rationale is the single
  place reasoning lives)

``verify_goal`` additionally supports ``action_type == ActionType.COMPLETE``
when ``goal_completed=True`` — in that case ``validation_subject`` is
``None`` and ``target`` is the raw current_screen string (the screen
itself IS the goal state).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fathom.constants import ActionType
from fathom.core.services.parsing import ToolResponseParser
from fathom.schemas.results import GenerateResult


@pytest.fixture
def parser() -> ToolResponseParser:
    return ToolResponseParser()


def _call(name: str, args: dict) -> GenerateResult:
    return GenerateResult(content="", tool_calls=[SimpleNamespace(name=name, args=args)])


class TestValidateStateFieldRename:
    """ValidateStateArgs now uses ``validation_subject`` instead of
    the old ``condition_to_verify`` name, matching ExecuteAction."""

    def test_validate_state_reads_validation_subject(self, parser: ToolResponseParser) -> None:
        result = parser.parse(
            _call(
                "validate_state",
                {
                    "assistant_message": "Checking login status",
                    "validation_subject": "Login success banner visible",
                    "condition_met": True,
                    "evidence": "green toast at the top",
                    "sub_goal_completed": True,
                },
            )
        )
        assert result.action.action_type == ActionType.VALIDATE
        assert result.action.validation_subject == "Login success banner visible"

    def test_validate_state_merges_reason_and_evidence_into_rationale(
        self, parser: ToolResponseParser
    ) -> None:
        result = parser.parse(
            _call(
                "validate_state",
                {
                    "assistant_message": "Checking login",
                    "validation_subject": "Login success banner visible",
                    "condition_met": True,
                    "evidence": "green toast visible",
                    "sub_goal_completed": True,
                },
            )
        )
        assert "Checking login" in result.action.rationale
        assert "Evidence: green toast visible" in result.action.rationale

    def test_validate_state_sanitizes_first_person_subject(
        self, parser: ToolResponseParser
    ) -> None:
        """Legacy behavior: the sanitizer strips 'I am verifying...' prefixes."""

        result = parser.parse(
            _call(
                "validate_state",
                {
                    "assistant_message": "ok",
                    "validation_subject": "I am validating the cart page is displayed",
                    "condition_met": True,
                    "evidence": "",
                    "sub_goal_completed": True,
                },
            )
        )
        # First-person prefix stripped.
        assert "i am" not in (result.action.validation_subject or "").lower()
        assert "validating" not in (result.action.validation_subject or "").lower()


class TestVerifyGoalNormalization:
    def test_verify_goal_not_complete_emits_validate_shape(
        self, parser: ToolResponseParser
    ) -> None:
        result = parser.parse(
            _call(
                "verify_goal",
                {
                    "assistant_message": "Cart not yet visible",
                    "goal_completed": False,
                    "sub_goal_completed": False,
                    "current_screen": "Product detail page",
                    "evidence": "still on product page, no cart visible",
                },
            )
        )
        assert result.action.action_type == ActionType.VALIDATE
        assert result.action.validation_subject is not None
        assert "Product detail page" in result.action.validation_subject
        # Rationale merges reason + evidence.
        assert "Cart not yet visible" in result.action.rationale
        assert "Evidence: still on product page" in result.action.rationale

    def test_verify_goal_complete_emits_complete_shape(self, parser: ToolResponseParser) -> None:
        result = parser.parse(
            _call(
                "verify_goal",
                {
                    "assistant_message": "Order placed successfully",
                    "goal_completed": True,
                    "sub_goal_completed": True,
                    "current_screen": "Order confirmation page",
                    "evidence": "confirmation number visible",
                },
            )
        )
        assert result.action.action_type == ActionType.COMPLETE
        # COMPLETE actions leave validation_subject None — the screen IS
        # the goal state, not something to check.
        assert result.action.validation_subject is None
        assert result.action.target == "Order confirmation page"
        assert "Evidence: confirmation number visible" in result.action.rationale


class TestValidationReasonDeleted:
    """The old validation_reason field was merged into rationale. Tests
    assert the field is gone from both schema layers."""

    def test_action_has_no_validation_reason_attribute(self) -> None:
        from fathom.schemas.actions import Action

        assert "validation_reason" not in Action.model_fields

    def test_execute_action_has_no_validation_reason_attribute(self) -> None:
        from fathom.schemas.tool_args import ExecuteAction

        assert "validation_reason" not in ExecuteAction.model_fields

    def test_plan_result_has_no_validation_reasoning_attribute(self) -> None:
        from fathom.schemas.results import PlanResult

        assert "validation_reasoning" not in PlanResult.model_fields


class TestCrossToolShapeConsistency:
    """All three validate-kind paths produce an Action with the same
    populated fields: action_type, validation_subject, target, rationale."""

    def test_all_three_paths_produce_validate_kind_with_subject(
        self, parser: ToolResponseParser
    ) -> None:
        # Path A: execute_ui + action_type=validate
        execute_result = parser.parse(
            _call(
                "execute_ui",
                {
                    "sub_goal_completed": False,
                    "actions": [
                        {
                            "action_type": "validate",
                            "rationale": "checking cart",
                            "is_valid": True,
                            "validation_subject": "cart visible",
                        }
                    ],
                },
            )
        )

        # Path B: validate_state
        state_result = parser.parse(
            _call(
                "validate_state",
                {
                    "assistant_message": "ok",
                    "validation_subject": "cart visible",
                    "condition_met": True,
                    "evidence": "",
                    "sub_goal_completed": True,
                },
            )
        )

        # Path C: verify_goal (not complete)
        verify_result = parser.parse(
            _call(
                "verify_goal",
                {
                    "assistant_message": "ok",
                    "goal_completed": False,
                    "sub_goal_completed": False,
                    "current_screen": "cart visible",
                    "evidence": "",
                },
            )
        )

        for result in (execute_result, state_result, verify_result):
            assert result.action.action_type == ActionType.VALIDATE
            assert result.action.validation_subject is not None
            assert result.action.rationale != ""
