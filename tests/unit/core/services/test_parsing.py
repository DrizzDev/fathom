"""
Pins for execute_ui bbox normalization at the parser boundary.
"""

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
