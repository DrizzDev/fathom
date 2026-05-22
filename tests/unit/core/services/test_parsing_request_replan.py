"""
Pins for the ``request_replan`` tool parser path.

The tool is the agent's structured escape valve: when the agent cannot
make safe forward progress on the active sub-goal, it invokes
``request_replan`` with a typed :class:`EscapeCategory` and a
one-sentence detail. The parser must produce an
:class:`AnalysisResult` with ``outcome=REQUEST_REPLAN`` and a populated
:class:`EscapeReport`; invalid arguments must raise a
:class:`ToolValidationError` at the boundary so the planner never sees
a half-formed escape signal (engineering standards §17, §19).
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from fathom.constants import ActionType
from fathom.core.exceptions import ToolValidationError
from fathom.core.services.parsing import ToolResponseParser
from fathom.schemas.escape import EscapeCategory
from fathom.schemas.results import AnalysisOutcome, GenerateResult


class _Call(BaseModel):
    """
    Minimal stand-in for the adapter's tool-call object — duck-typed
    by attribute access (``name`` / ``args``) in the parser.
    """

    name: str = Field(description="Tool name")
    args: Dict[str, Any] = Field(description="Tool arguments")


class ToolResponseParserRequestReplanTest(unittest.TestCase):
    """
    Pins for the structured REQUEST_REPLAN parser branch.
    """

    @staticmethod
    def __response(*, calls: List[_Call]) -> GenerateResult:
        """
        Wrap a list of tool calls in a :class:`GenerateResult`.
        """

        return GenerateResult(content="", tool_calls=list(calls), metrics={})

    def test_target_not_available_produces_structured_outcome(self) -> None:
        """
        Calling ``request_replan`` with category ``target_not_available``
        yields an :class:`AnalysisResult` with
        ``outcome=REQUEST_REPLAN`` and a populated escape report.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="request_replan",
                    args={
                        "category": "target_not_available",
                        "detail": "no Continue button on this screen",
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        self.assertEqual(result.outcome, AnalysisOutcome.REQUEST_REPLAN)
        self.assertIsNotNone(result.escape_report)
        assert result.escape_report is not None
        self.assertEqual(result.escape_report.category, EscapeCategory.TARGET_NOT_AVAILABLE)
        self.assertEqual(result.escape_report.detail, "no Continue button on this screen")
        self.assertFalse(result.is_goal_complete)
        self.assertFalse(result.is_sub_goal_complete)
        self.assertEqual(result.action.action_type, ActionType.WAIT)

    def test_wrong_screen_category_round_trips(self) -> None:
        """
        Every taxonomy value the prompt advertises must be accepted
        end-to-end without silent coercion.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="request_replan",
                    args={
                        "category": "wrong_screen",
                        "detail": "on the CleverTap debug overlay, not the Swiggy home",
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        assert result.escape_report is not None
        self.assertEqual(result.escape_report.category, EscapeCategory.WRONG_SCREEN)

    def test_unsafe_action_category_round_trips(self) -> None:
        """
        Human-routed categories (UNSAFE_ACTION, AMBIGUOUS_TARGET) must
        be parsed the same way as replan-routed categories — routing
        decisions belong to the planner, not the parser.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="request_replan",
                    args={
                        "category": "unsafe_action",
                        "detail": "tapping Delete account would be irreversible",
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        assert result.escape_report is not None
        self.assertEqual(result.escape_report.category, EscapeCategory.UNSAFE_ACTION)
        self.assertTrue(result.escape_report.routes_to_human())

    def test_missing_category_raises_tool_validation_error(self) -> None:
        """
        A model response that omits the typed category is malformed;
        the parser must fail fast at the boundary rather than silently
        defaulting (engineering standards §19 Fail Fast).
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[_Call(name="request_replan", args={"detail": "something"})]
        )

        with self.assertRaises(ToolValidationError):
            parser.parse(response=response)

    def test_unknown_category_raises_tool_validation_error(self) -> None:
        """
        A category value outside the typed taxonomy must be rejected
        at the parser boundary so it cannot leak into Domain.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="request_replan",
                    args={"category": "made_up_value", "detail": "..."},
                )
            ]
        )

        with self.assertRaises(ToolValidationError):
            parser.parse(response=response)

    def test_empty_detail_raises_tool_validation_error(self) -> None:
        """
        An empty detail defeats the typed contract (the decomposer or
        the human gets a blank justification) and must be rejected.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="request_replan",
                    args={"category": "target_not_available", "detail": ""},
                )
            ]
        )

        with self.assertRaises(ToolValidationError):
            parser.parse(response=response)

    def test_ask_user_outcome_is_structured(self) -> None:
        """
        ``ask_user`` is a structured non-action outcome, not a synthetic
        UI action disguised as normal execution.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="ask_user",
                    args={
                        "question": "Where to?",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                    },
                )
            ]
        )

        result = parser.parse(response=response)
        self.assertEqual(result.outcome, AnalysisOutcome.ASK_USER)
