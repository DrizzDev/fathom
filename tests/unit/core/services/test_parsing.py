from __future__ import annotations

import unittest
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from fathom.constants import ActionType
from fathom.core.exceptions import ToolValidationError
from fathom.core.services.parsing import ToolResponseParser
from fathom.schemas.actions import CoordinateSystem
from fathom.schemas.results import GenerateResult


class _Call(BaseModel):
    """
    Minimal tool-call stand-in consumed by the parser.
    """

    name: str = Field(description="Tool name")
    args: Dict[str, Any] = Field(description="Tool arguments")


class ToolResponseParserExecuteUiTest(unittest.TestCase):
    """
    Covers bbox sanitation for execute_ui actions.
    """

    @staticmethod
    def __response(*, calls: List[_Call]) -> GenerateResult:
        """
        Wrap tool calls in a generate result.
        """

        return GenerateResult(content="", tool_calls=list(calls), metrics={})

    def test_coerces_malformed_normalized_bbox_to_logical(self) -> None:
        """
        Pixel-scale bbox values mislabeled as normalized must not survive parsing.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "scroll down",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "swipe_up",
                            "target_name": "Restaurant list area",
                            "scroll_target": "Restaurant list area",
                            "confidence": 0.84,
                            "bbox": {
                                "x": 0,
                                "y": 858,
                                "width": 1206,
                                "height": 1396,
                                "coordinate_system": "normalized",
                            },
                        },
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        self.assertEqual(result.action.action_type, ActionType.SWIPE_UP)
        assert result.action.bounds is not None
        self.assertEqual(result.action.bounds.system, CoordinateSystem.LOGICAL)

    def test_keeps_valid_normalized_bbox_normalized(self) -> None:
        """
        Legitimate normalized bboxes must preserve their declared system.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "scroll down",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "swipe_up",
                            "target_name": "Restaurant list area",
                            "scroll_target": "Restaurant list area",
                            "confidence": 0.84,
                            "bbox": {
                                "x": 100,
                                "y": 200,
                                "width": 800,
                                "height": 500,
                                "coordinate_system": "normalized",
                            },
                        },
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        assert result.action.bounds is not None
        self.assertEqual(result.action.bounds.system, CoordinateSystem.NORMALIZED)

    def test_missing_confidence_fails_validation(self) -> None:
        """
        Missing planner confidence must fail validation instead of being guessed.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "tap restaurant",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "tap",
                            "target_name": "Asha Tiffin",
                            "label_id": "12",
                            "bbox": {
                                "x": 100,
                                "y": 200,
                                "width": 300,
                                "height": 120,
                                "coordinate_system": "normalized",
                            },
                        },
                    },
                )
            ]
        )

        with self.assertRaises(ToolValidationError):
            parser.parse(response=response)

    def test_scroll_objective_does_not_replace_surface_target(self) -> None:
        """
        scroll_target must stay the objective, not become the swipe surface label.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "scroll down",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "swipe_up",
                            "scroll_target": "Asha Tiffin",
                            "confidence": 0.84,
                            "bbox": {
                                "x": 0,
                                "y": 858,
                                "width": 1206,
                                "height": 1396,
                                "coordinate_system": "pixel",
                            },
                        },
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        self.assertEqual(result.action.target, "main scrollable area")
        self.assertEqual(result.action.scroll_target, "Asha Tiffin")

    def test_parses_enter_action_without_wait_fallback(self) -> None:
        """
        The prompt schema exposes enter, so the parser must preserve it.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "submit search",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "enter",
                            "target_name": "Search button on keyboard",
                            "confidence": 0.84,
                        },
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        self.assertEqual(result.action.action_type, ActionType.ENTER)
        self.assertEqual(result.action.target, "Search button on keyboard")


class ToolResponseParserCompletionReasonAutofillTest(unittest.TestCase):
    """
    Pins the parser's autofill of missing ``subgoal_completion_reason`` /
    ``goal_completion_reason`` fields so the downstream completion gate
    can verify the claim without falling back to fuzzy matching.
    """

    @staticmethod
    def __response(*, args: Dict[str, Any]) -> GenerateResult:
        """
        Wrap an ``execute_ui`` tool call in a generate result.
        """

        return GenerateResult(
            content="",
            metrics={},
            tool_calls=[_Call(name="execute_ui", args=args)],
        )

    @staticmethod
    def __completion_args(
        *,
        rationale: str,
        sub_goal_completed: bool,
        goal_completed: bool = False,
        subgoal_completion_reason: Any = None,
        goal_completion_reason: Any = None,
    ) -> Dict[str, Any]:
        """
        Build a baseline ``execute_ui`` argument set that flags completion.
        """

        return {
            "assistant_message": "OK",
            "goal_completed": goal_completed,
            "sub_goal_completed": sub_goal_completed,
            "goal_completion_reason": goal_completion_reason,
            "subgoal_completion_reason": subgoal_completion_reason,
            "action": {
                "action_type": "complete",
                "target_name": "Sub-goal verification",
                "rationale": rationale,
                "is_valid": True,
                "confidence": 1.0,
            },
        }

    def test_autofills_subgoal_reason_from_rationale_when_missing(self) -> None:
        """
        Missing ``subgoal_completion_reason`` is filled from ``action.rationale``
        when the model flagged the sub-goal as complete.
        """

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=True,
                rationale="Offerwall menu has been dismissed successfully",
            )
        )

        result = parser.parse(response=response)

        self.assertTrue(result.is_sub_goal_complete)
        self.assertEqual(
            result.subgoal_completion_reason,
            "Offerwall menu has been dismissed successfully",
        )

    def test_preserves_model_supplied_subgoal_reason(self) -> None:
        """
        A non-empty ``subgoal_completion_reason`` from the model is preserved verbatim.
        """

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=True,
                rationale="Backup rationale not used",
                subgoal_completion_reason="Order placed successfully",
            )
        )

        result = parser.parse(response=response)

        self.assertEqual(result.subgoal_completion_reason, "Order placed successfully")

    def test_does_not_autofill_when_completion_flag_false(self) -> None:
        """
        Autofill only triggers when the model claims completion.
        """

        parser = ToolResponseParser()
        args = self.__completion_args(
            sub_goal_completed=False,
            rationale="Just a regular tap on the search box",
        )
        args["action"]["action_type"] = "tap"
        args["action"]["target_name"] = "Search box"
        response = self.__response(args=args)

        result = parser.parse(response=response)

        self.assertFalse(result.is_sub_goal_complete)
        self.assertIsNone(result.subgoal_completion_reason)

    def test_autofills_goal_reason_from_rationale(self) -> None:
        """
        Missing ``goal_completion_reason`` is filled from rationale when the
        model claims overall goal completion.
        """

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=True,
                goal_completed=True,
                rationale="Workflow finished as per operator instruction",
            )
        )

        result = parser.parse(response=response)

        self.assertTrue(result.is_goal_complete)
        self.assertEqual(
            result.goal_completion_reason,
            "Workflow finished as per operator instruction",
        )

    def test_leaves_reason_null_when_rationale_is_whitespace_only(self) -> None:
        """
        With both the reason field empty AND the rationale collapsing to
        whitespace, the autofill must leave the reason None — never guess.
        """

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=True,
                rationale="   \t\n  ",
            )
        )

        result = parser.parse(response=response)
        self.assertTrue(result.is_sub_goal_complete)
        self.assertIsNone(result.subgoal_completion_reason)

    def test_complete_action_with_false_sub_goal_flag_autofills_after_normalization(
        self,
    ) -> None:
        """When ``action_type=COMPLETE`` arrives with ``sub_goal_completed=False``, the normalizer must force the flag to True FIRST, then the autofill must see the True flag and populate ``subgoal_completion_reason`` from the rationale — pinning the COMPLETE-flag → autofill ordering."""

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=False,
                rationale="Offerwall confirmation visible",
            )
        )

        result = parser.parse(response=response)

        self.assertTrue(result.is_sub_goal_complete)
        self.assertEqual(
            result.subgoal_completion_reason,
            "Offerwall confirmation visible",
        )


if __name__ == "__main__":
    unittest.main()
