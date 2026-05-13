"""
Pins for the ``report_screen_unactionable`` tool parser path.

The tool is the agent's structured escape valve: when the active
sub-goal does not match the current screen, the agent emits this
tool call and the system routes it to the recovery coordinator
(:data:`RecoveryTrigger.REPORT_UNACTIONABLE`). The parser must
produce an :class:`AnalysisResult` with the structured outcome —
not a synthetic action — so the planner can branch deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import unittest

from fathom.constants import ActionType
from fathom.core.services.parsing import ToolResponseParser
from fathom.schemas.results import AnalysisOutcome, GenerateResult


@dataclass
class _Call:
    """
    Minimal stand-in for the adapter's tool-call object — duck-typed
    by attribute access (``name`` / ``args``) in the parser.
    """

    name: str
    args: Dict[str, Any]


class ToolResponseParserUnactionableTest(unittest.TestCase):
    """
    Pins for the structured REPORT_UNACTIONABLE parser branch.
    """

    @staticmethod
    def __response(*, calls: List[_Call]) -> GenerateResult:
        """
        Wrap a list of tool calls in a :class:`GenerateResult`.
        """

        return GenerateResult(content="", tool_calls=list(calls), metrics={})

    def test_report_unactionable_produces_structured_outcome(self) -> None:
        """
        Calling ``report_screen_unactionable`` yields an
        :class:`AnalysisResult` with
        ``outcome=REPORT_UNACTIONABLE`` and the supplied reason.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="report_screen_unactionable",
                    args={"reason": "no Continue button on this screen"},
                )
            ]
        )

        result = parser.parse(response=response)

        self.assertEqual(result.outcome, AnalysisOutcome.REPORT_UNACTIONABLE)
        self.assertEqual(result.unactionable_reason, "no Continue button on this screen")
        self.assertFalse(result.is_goal_complete)
        self.assertFalse(result.is_sub_goal_complete)
        # The action is a placeholder WAIT; the planner branches on outcome,
        # not on the action — confirm no synthetic spatial target sneaks in.
        self.assertEqual(result.action.action_type, ActionType.WAIT)

    def test_report_unactionable_defaults_reason_when_missing(self) -> None:
        """
        Missing ``reason`` does not raise — the parser supplies a
        stable default so the recovery path stays unblocked even on a
        malformed model response.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[_Call(name="report_screen_unactionable", args={})]
        )

        result = parser.parse(response=response)

        self.assertEqual(result.outcome, AnalysisOutcome.REPORT_UNACTIONABLE)
        self.assertIsNotNone(result.unactionable_reason)
        self.assertNotEqual(result.unactionable_reason, "")

    def test_default_outcome_for_other_tools_is_act(self) -> None:
        """
        Regression guard: the new ``outcome`` field defaults to
        ``ACT`` for any non-unactionable tool — existing parsers
        must not silently change semantics.
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
        self.assertEqual(result.outcome, AnalysisOutcome.ACT)
