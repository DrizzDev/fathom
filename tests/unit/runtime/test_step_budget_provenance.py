"""
The step budget that governed a run must be readable from its result.

A benchmark host pins a budget and needs to know the run honoured it. The requested
number proves nothing on its own: the runner takes ``max_steps`` from the request and
falls back to the intent configuration, so three different values can be in play and
only one of them bounded the run. Emitting the requested number as though it were the
effective one would turn an unperformed check into a passed one.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional

from fathom.runtime.command.executor import CommandExecutor
from fathom.schemas.results import IntentResult


class _CapturingStream:
    """
    A trajectory stream that keeps what was emitted instead of writing to stdout.
    """

    def __init__(self) -> None:
        self.enabled = True
        self.payload: Optional[Dict[str, Any]] = None

    def event(self, *, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        return None

    def outcome(self, *, payload: Dict[str, Any]) -> None:
        self.payload = payload


def _emit(*, result: Any) -> Dict[str, Any]:
    """
    Run the emitter and return the outcome payload.
    """

    stream = _CapturingStream()
    CommandExecutor._CommandExecutor__emit_trajectory(  # type: ignore[attr-defined]  # noqa: SLF001
        stream=stream, result=result, grounding_policy=None
    )
    assert stream.payload is not None
    return stream.payload


class TheResultCarriesTheResolvedBudgetTest(unittest.TestCase):
    """
    The field the emitter reads.
    """

    def test_a_run_records_the_budget_that_bounded_it(self) -> None:
        result = IntentResult(status="completed", success=True, steps_taken=5, max_steps=40)

        self.assertEqual(result.max_steps, 40)

    def test_an_unreached_run_records_no_budget(self) -> None:
        """
        None rather than a default. A run rejected before execution was bounded by
        nothing, and naming a number would claim a budget took effect.
        """

        result = IntentResult(status="completed", success=False, steps_taken=0)

        self.assertIsNone(result.max_steps)

    def test_a_nonsense_budget_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            IntentResult(status="completed", success=True, steps_taken=0, max_steps=0)


class TheBudgetReachesTheOutcomeTest(unittest.TestCase):
    """
    What crosses the process boundary to the host.
    """

    def test_the_effective_budget_is_reported(self) -> None:
        payload = _emit(
            result=IntentResult(status="completed", success=True, steps_taken=5, max_steps=40)
        )

        self.assertEqual(payload["effective_max_steps"], 40)

    def test_a_budget_that_differs_from_the_request_is_reported_as_it_is(self) -> None:
        """
        The whole point: the emitter must not launder the requested number back out.
        A run that fell back to the intent configuration reports what bounded it.
        """

        payload = _emit(
            result=IntentResult(status="completed", success=True, steps_taken=9, max_steps=50)
        )

        self.assertEqual(payload["effective_max_steps"], 50)

    def test_an_unknown_budget_is_reported_as_unknown(self) -> None:
        payload = _emit(result=IntentResult(status="completed", success=False, steps_taken=0))

        self.assertIsNone(payload["effective_max_steps"])

    def test_a_duck_typed_result_without_the_field_still_emits(self) -> None:
        """
        The emitter reads defensively, so an older result object must not crash it.
        """

        payload = _emit(
            result=SimpleNamespace(
                status="completed",
                success=True,
                steps_taken=3,
                completion_reason="done",
                duration=1.0,
                error=None,
                counters=None,
            )
        )

        self.assertIsNone(payload["effective_max_steps"])


if __name__ == "__main__":
    unittest.main()
