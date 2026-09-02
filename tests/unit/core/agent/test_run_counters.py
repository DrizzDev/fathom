"""
Run counters: measured, never defaulted.

These five fields were emitted as hardcoded zeros. The host's fields are tri-state
precisely so an uninstrumented harness reports UNVERIFIED, so a fabricated zero turned
three checks nobody performed into three that passed — and when the placeholders were
removed the same runs became UNVERIFIED and every cell INVALID. Neither state is usable;
the counters have to be real.

Two properties carry this and are pinned in both directions: a measured zero is a claim
("nothing happened"), an absent counter is not ("nobody looked"), and the two must never
be confused on the wire.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.core.agent.counters import measure_run
from fathom.runtime.command.executor import CommandExecutor
from fathom.runtime.plumb_stream import TrajectoryStream
from fathom.schemas.actions import Action, Bounds, CoordinateSource
from fathom.schemas.results import RunCounters
from fathom.schemas.steps import Step, StepResult


def _step(
    *,
    action_type: ActionType = ActionType.TAP,
    executed: bool = True,
    screen_changed: bool = True,
    source: CoordinateSource | None = CoordinateSource.XML,
) -> StepResult:
    """
    One recorded step, spelled out so each test's shape is visible at the call site.
    """

    bounds = None if source is None else Bounds(x=1, y=1, width=10, height=10, source=source)
    return StepResult(
        step=Step(
            action=Action(action_type=action_type, target="element", rationale="r", bounds=bounds),
            step_number=1,
            screen_hash="pre",
        ),
        success=True,
        executed=executed,
        screen_changed=screen_changed,
        pre_hash="pre",
        post_hash="post" if screen_changed else "pre",
        duration=10,
    )


class MeasuredZeroTest(TestCase):
    """
    A run that did nothing says so, and that is a measurement.
    """

    def test_a_run_with_no_steps_counts_zero(self) -> None:
        counters = measure_run(step_results=[], human_interventions=0, completion_reason=None)

        self.assertEqual(counters.total_actions, 0)
        self.assertEqual(counters.no_op_actions, 0)
        self.assertEqual(counters.uncorroborated_dispatches, 0)
        self.assertEqual(counters.human_interventions, 0)
        self.assertFalse(counters.awaiting_human_at_end)

    def test_a_clean_run_reports_zero_no_ops_and_zero_uncorroborated(self) -> None:
        """
        The shape of the airplane-mode smoke cell: three taps, all grounded, all moved
        the screen. Every counter zero except the total.
        """

        counters = measure_run(
            step_results=[_step() for _ in range(3)],
            human_interventions=0,
            completion_reason=CompletionReason.SUCCESS.value,
        )

        self.assertEqual(counters.total_actions, 3)
        self.assertEqual(counters.no_op_actions, 0)
        self.assertEqual(counters.uncorroborated_dispatches, 0)
        self.assertFalse(counters.awaiting_human_at_end)


class MeasuredNonZeroTest(TestCase):
    """
    Each counter moves for its own reason and for no other.
    """

    def test_no_ops_are_counted(self) -> None:
        counters = measure_run(
            step_results=[_step(screen_changed=True), _step(screen_changed=False)],
            human_interventions=0,
            completion_reason=None,
        )

        self.assertEqual(counters.total_actions, 2)
        self.assertEqual(counters.no_op_actions, 1)

    def test_only_blind_planner_bounds_count_as_uncorroborated(self) -> None:
        """
        ``MODEL_GROUNDED`` bounds came from the planner too, but OCR evidence inside the
        proposed region corroborated them. Counting those would erase the distinction
        the localization cascade exists to draw.
        """

        counters = measure_run(
            step_results=[
                _step(source=CoordinateSource.MODEL),
                _step(source=CoordinateSource.MODEL_GROUNDED),
                _step(source=CoordinateSource.OCR),
                _step(source=CoordinateSource.XML),
                _step(source=None),
            ],
            human_interventions=0,
            completion_reason=None,
        )

        self.assertEqual(counters.total_actions, 5)
        self.assertEqual(counters.uncorroborated_dispatches, 1)

    def test_human_answers_are_carried_through(self) -> None:
        counters = measure_run(
            step_results=[_step()], human_interventions=2, completion_reason=None
        )

        self.assertEqual(counters.human_interventions, 2)

    def test_a_run_that_ended_needing_an_answer_says_so(self) -> None:
        counters = measure_run(
            step_results=[_step()],
            human_interventions=0,
            completion_reason=CompletionReason.INTERVENTION_REQUIRED.value,
        )

        self.assertTrue(counters.awaiting_human_at_end)

    def test_any_other_ending_leaves_nothing_outstanding(self) -> None:
        for reason in (
            CompletionReason.SUCCESS.value,
            CompletionReason.STUCK.value,
            CompletionReason.MAX_STEPS.value,
            None,
        ):
            with self.subTest(reason=reason):
                counters = measure_run(
                    step_results=[_step()], human_interventions=0, completion_reason=reason
                )
                self.assertFalse(counters.awaiting_human_at_end)


class WhatCountsAsADispatchTest(TestCase):
    """
    Only steps that actually touched the device.
    """

    def test_unexecuted_steps_are_not_dispatches(self) -> None:
        counters = measure_run(
            step_results=[_step(executed=False), _step(executed=True)],
            human_interventions=0,
            completion_reason=None,
        )

        self.assertEqual(counters.total_actions, 1)

    def test_non_interactive_commands_are_not_dispatches(self) -> None:
        """
        A VALIDATE or a WAIT executes without dispatching anything, and never changes
        the screen. Counting them would inflate the total and report every one of them
        as a no-op — which is exactly the false signal these counters must not produce.
        """

        counters = measure_run(
            step_results=[
                _step(action_type=ActionType.TAP, screen_changed=True),
                _step(action_type=ActionType.VALIDATE, screen_changed=False, source=None),
                _step(action_type=ActionType.WAIT, screen_changed=False, source=None),
            ],
            human_interventions=0,
            completion_reason=None,
        )

        self.assertEqual(counters.total_actions, 1)
        self.assertEqual(counters.no_op_actions, 0)

    def test_subsets_never_exceed_the_total(self) -> None:
        """
        The host rejects a subset larger than its superset, so the invariant is checked
        here rather than discovered at the boundary.
        """

        counters = measure_run(
            step_results=[_step(screen_changed=False, source=CoordinateSource.MODEL)] * 4,
            human_interventions=0,
            completion_reason=None,
        )

        self.assertLessEqual(counters.no_op_actions, counters.total_actions)
        self.assertLessEqual(counters.uncorroborated_dispatches, counters.total_actions)


def _emitted(*, result: object) -> dict:
    """
    Run the real emitter and return the parsed PLUMB_OUTCOME payload.
    """

    stream = TrajectoryStream(run_id="run-1")
    with patch("fathom.runtime.plumb_stream.sys.stdout") as stdout:
        CommandExecutor._CommandExecutor__emit_trajectory(  # noqa: SLF001
            stream=stream, result=result
        )
    written = "".join(str(c.args[0]) for c in stdout.write.call_args_list)
    line = next(p for p in written.splitlines() if p.startswith("PLUMB_OUTCOME "))
    return json.loads(line[len("PLUMB_OUTCOME ") :])


def _result(*, counters: RunCounters | None) -> MagicMock:
    result = MagicMock()
    result.status = "completed"
    result.success = True
    result.steps_taken = 3
    result.max_steps = 40
    result.completion_reason = "Completed successfully"
    result.duration = 62.2
    result.error = None
    result.counters = counters
    return result


class CountersSurviveIntoPlumbOutcomeTest(TestCase):
    """
    The wire is where this actually matters.
    """

    def test_measured_counters_reach_the_payload(self) -> None:
        payload = _emitted(
            result=_result(
                counters=RunCounters(
                    total_actions=3,
                    no_op_actions=1,
                    uncorroborated_dispatches=2,
                    human_interventions=4,
                    awaiting_human_at_end=True,
                )
            )
        )

        self.assertEqual(payload["total_actions"], 3)
        self.assertEqual(payload["no_op_actions"], 1)
        self.assertEqual(payload["uncorroborated_dispatches"], 2)
        self.assertEqual(payload["human_interventions"], 4)
        self.assertIs(payload["awaiting_human_at_end"], True)

    def test_a_measured_zero_is_emitted_rather_than_omitted(self) -> None:
        """
        The regression that blocked the benchmark. Omitting a zero reports UNVERIFIED
        and invalidates the cell; the run really did dispatch nothing uncorroborated,
        and it is entitled to say so.
        """

        payload = _emitted(
            result=_result(
                counters=RunCounters(
                    total_actions=3,
                    no_op_actions=0,
                    uncorroborated_dispatches=0,
                    human_interventions=0,
                    awaiting_human_at_end=False,
                )
            )
        )

        for field in ("no_op_actions", "uncorroborated_dispatches", "human_interventions"):
            with self.subTest(field=field):
                self.assertIn(field, payload)
                self.assertEqual(payload[field], 0)
        self.assertIs(payload["awaiting_human_at_end"], False)

    def test_an_unmeasured_run_omits_them_entirely(self) -> None:
        """
        The other half of the contract. A run that recorded no counters must not claim
        zeros — that is the fabricated pass this work removed.
        """

        payload = _emitted(result=_result(counters=None))

        for field in (
            "total_actions",
            "no_op_actions",
            "uncorroborated_dispatches",
            "human_interventions",
            "awaiting_human_at_end",
        ):
            with self.subTest(field=field):
                self.assertNotIn(field, payload)


class HitlServiceCountsAnswersTest(TestCase):
    """
    Counted where an answer arrives, not where one is requested.
    """

    @staticmethod
    def __service():
        from fathom.core.services.hitl import HITLService
        from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities

        return HITLService(
            signal=SimpleNamespace(),
            phase=SimpleNamespace(),
            telemetry=SimpleNamespace(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

    def test_a_fresh_service_has_answered_nothing(self) -> None:
        self.assertEqual(self.__service().answers_received, 0)
