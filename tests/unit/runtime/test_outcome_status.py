"""
Regression coverage for the benchmark outcome status.

The emitter used to write ``"status": "COMPLETED"`` as a literal, so PROVIDER_EXHAUSTED,
TIMED_OUT, CANCELLED and HARNESS_ERROR could never leave the process. The host's gate
for terminal provider failure reads exactly that field, which made the gate unreachable:
a run the provider shut out was scored as a model that failed the task.

These tests pin the translation in both directions - that a real ending reaches the wire
under its own name, and that COMPLETED still means what the host thinks it means.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

from fathom.core.exceptions import (
    ProviderExhaustedError,
    VisionError,
    WorkflowCancelledError,
    WorkflowTimeoutError,
)
from fathom.runtime.command.executor import CommandExecutor
from fathom.runtime.plumb_stream import (
    OutcomeStatus,
    TrajectoryStream,
    status_for_exception,
    status_for_result,
)


class ResultStatusTest(unittest.TestCase):
    """
    A workflow that returned a result gets its own ending, not a constant.
    """

    def test_a_finished_run_is_completed_whether_or_not_it_passed(self) -> None:
        """
        The distinction the contract insists on: COMPLETED describes the attempt, and
        the host decides the task from the device. Mapping "failed" to anything else
        would let a model failure masquerade as an invalid cell.
        """

        self.assertIs(status_for_result(status="completed"), OutcomeStatus.COMPLETED)
        self.assertIs(status_for_result(status="failed"), OutcomeStatus.COMPLETED)

    def test_a_cancelled_run_reports_cancelled(self) -> None:
        """
        The regression: this ending existed in the runner and was overwritten on the way
        out.
        """

        self.assertIs(status_for_result(status="cancelled"), OutcomeStatus.CANCELLED)

    def test_status_matching_ignores_case_and_padding(self) -> None:
        """
        The status crosses a schema field as a bare string, so it is normalised rather
        than trusted to arrive exactly as the runner spelled it.
        """

        self.assertIs(status_for_result(status="  Cancelled "), OutcomeStatus.CANCELLED)

    def test_an_unnamed_status_is_a_harness_error(self) -> None:
        """
        ``WorkflowResult`` defaults status to "unknown". Reporting that as COMPLETED is
        the original defect in miniature - an answer nobody computed - so it must
        instead invalidate the cell.
        """

        self.assertIs(status_for_result(status="unknown"), OutcomeStatus.HARNESS_ERROR)
        self.assertIs(status_for_result(status=""), OutcomeStatus.HARNESS_ERROR)


class ExceptionStatusTest(unittest.TestCase):
    """
    A run that raised never reached the result emitter at all.
    """

    def test_provider_exhaustion_reports_provider_exhausted(self) -> None:
        """
        The whole point of the typed exception: the host's PROVIDER_ERROR gate reads
        this value and nothing else can set it.
        """

        self.assertIs(
            status_for_exception(exception=ProviderExhaustedError("429 after 5 attempts")),
            OutcomeStatus.PROVIDER_EXHAUSTED,
        )

    def test_provider_exhaustion_outranks_its_own_base_class(self) -> None:
        """
        ProviderExhaustedError is a VisionError so existing handlers keep working; the
        translation must not therefore collapse it back into a generic harness error.
        """

        self.assertIs(
            status_for_exception(exception=VisionError("LLM fail: something else")),
            OutcomeStatus.HARNESS_ERROR,
        )

    def test_timeouts_report_timed_out(self) -> None:
        """
        A clock that ran out is not a model that failed.
        """

        for exception in (WorkflowTimeoutError("w-1", 60.0), TimeoutError()):
            with self.subTest(exception=type(exception).__name__):
                self.assertIs(status_for_exception(exception=exception), OutcomeStatus.TIMED_OUT)

    def test_cancellation_reports_cancelled(self) -> None:
        """
        Including the BaseException-derived ones, which a bare ``except Exception``
        would never see and which the host must not read as a completed attempt.
        """

        for exception in (
            asyncio.CancelledError(),
            KeyboardInterrupt(),
            WorkflowCancelledError("w-1"),
        ):
            with self.subTest(exception=type(exception).__name__):
                self.assertIs(status_for_exception(exception=exception), OutcomeStatus.CANCELLED)

    def test_anything_else_is_a_harness_error(self) -> None:
        """
        An unclassified crash is the harness's fault until someone classifies it, which
        keeps the cell out of the model comparison rather than counting it as a loss.
        """

        self.assertIs(
            status_for_exception(exception=RuntimeError("boom")), OutcomeStatus.HARNESS_ERROR
        )


def _emitted(call) -> dict:
    """
    Run one emission and return the parsed outcome payload.
    """

    with patch("fathom.runtime.plumb_stream.sys.stdout") as stdout:
        call()
    written = "".join(str(item.args[0]) for item in stdout.write.call_args_list)
    line = next(part for part in written.splitlines() if part.startswith("PLUMB_OUTCOME "))
    return json.loads(line[len("PLUMB_OUTCOME ") :])


class EmittedPayloadTest(unittest.TestCase):
    """
    What actually crosses the process boundary.
    """

    @staticmethod
    def __result(*, status: str, success: bool) -> MagicMock:
        result = MagicMock()
        result.status = status
        result.success = success
        result.steps_taken = 7
        result.max_steps = 40
        result.completion_reason = "Stuck: No progress"
        result.duration = 58.6
        result.error = None
        result.counters = None
        return result

    def test_a_failed_run_emits_completed_not_a_literal(self) -> None:
        """
        End to end through the emitter, since the constant lived there rather than in
        the translation.
        """

        stream = TrajectoryStream(run_id="run-1")
        payload = _emitted(
            lambda: CommandExecutor._CommandExecutor__emit_trajectory(  # noqa: SLF001
                grounding_policy=None,
                stream=stream,
                result=self.__result(status="failed", success=False),
            ),
        )

        self.assertEqual(payload["status"], "COMPLETED")
        self.assertEqual(payload["exit_status"], 1)
        self.assertEqual(payload["status_source"], "PAYLOAD")

    def test_a_cancelled_run_emits_cancelled(self) -> None:
        """
        The value the old emitter could not produce.
        """

        stream = TrajectoryStream(run_id="run-1")
        payload = _emitted(
            lambda: CommandExecutor._CommandExecutor__emit_trajectory(  # noqa: SLF001
                grounding_policy=None,
                stream=stream,
                result=self.__result(status="cancelled", success=False),
            ),
        )

        self.assertEqual(payload["status"], "CANCELLED")

    def test_unmeasured_counters_are_absent_rather_than_zero(self) -> None:
        """
        The host's counters are tri-state so an uninstrumented harness reports
        UNVERIFIED. Sending 0 answered "were any dispatches uncorroborated?" with a
        number nobody counted, turning an unperformed check into a passed one.

        A run that *did* record counters emits them, zeros included — that contract is
        pinned in tests/unit/core/agent/test_run_counters.py. This is the other half:
        when nothing was measured, nothing is claimed.
        """

        stream = TrajectoryStream(run_id="run-1")
        payload = _emitted(
            lambda: CommandExecutor._CommandExecutor__emit_trajectory(  # noqa: SLF001
                grounding_policy=None,
                stream=stream,
                result=self.__result(status="completed", success=True),
            ),
        )

        for field in (
            "total_actions",
            "no_op_actions",
            "uncorroborated_dispatches",
            "human_interventions",
            "awaiting_human_at_end",
        ):
            with self.subTest(field=field):
                self.assertNotIn(field, payload)

    def test_duration_is_reported_in_seconds(self) -> None:
        """
        The runner records ``time.time() - start_time``. The emitter divided it by
        1000, so a minute-long run was published as 0.06 seconds.
        """

        stream = TrajectoryStream(run_id="run-1")
        payload = _emitted(
            lambda: CommandExecutor._CommandExecutor__emit_trajectory(  # noqa: SLF001
                grounding_policy=None,
                stream=stream,
                result=self.__result(status="completed", success=True),
            ),
        )

        self.assertAlmostEqual(payload["duration_seconds"], 58.6)

    def test_a_provider_exhausted_run_emits_an_outcome_at_all(self) -> None:
        """
        The reachability half of the defect: this path emitted nothing, so the host
        fell back to the exit status and read a terminal provider failure as a model
        that did not finish the task.
        """

        stream = TrajectoryStream(run_id="run-1")
        payload = _emitted(
            lambda: CommandExecutor._CommandExecutor__emit_failure(  # noqa: SLF001
                stream=stream,
                exception=ProviderExhaustedError("OpenAI rate limit after 5 attempts"),
                started=0.0,
            ),
        )

        self.assertEqual(payload["status"], "PROVIDER_EXHAUSTED")
        self.assertEqual(payload["status_source"], "PAYLOAD")
        self.assertEqual(payload["completion_reason"], "ProviderExhaustedError")
        self.assertNotIn("steps_taken", payload)

    def test_an_inert_stream_still_emits_nothing(self) -> None:
        """
        Ordinary interactive runs must stay silent; the failure path is new and could
        have started printing a contract line to anyone's terminal.
        """

        stream = TrajectoryStream(run_id="")
        with patch("fathom.runtime.plumb_stream.sys.stdout") as stdout:
            CommandExecutor._CommandExecutor__emit_failure(  # noqa: SLF001
                stream=stream, exception=RuntimeError("boom"), started=0.0
            )
        stdout.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
