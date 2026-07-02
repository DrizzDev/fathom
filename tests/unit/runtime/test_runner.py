from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fathom.constants.collaboration import TaskCode
from fathom.constants.state import CompletionReason
from fathom.runtime.runner import FathomRunner
from fathom.schemas.recording import Handle, ScriptOutput


class FathomRunnerGeneratedScriptTest(unittest.IsolatedAsyncioTestCase):
    """
    Runner persists final generated script content without relying on artifact files.
    """

    def __runner(self, *, recorder: AsyncMock) -> FathomRunner:
        """
        Build a runner instance with only the recorder dependency needed by this unit.
        """

        runner = object.__new__(FathomRunner)
        runner._FathomRunner__recorder = recorder  # type: ignore[attr-defined]
        return runner

    @staticmethod
    def __handle() -> Handle:
        """
        Return stable run identifiers for script persistence tests.
        """

        return Handle(
            task="7c5f7738-8c0b-4597-b9a0-988a1d22bc24",
            tenant="tenant-1",
            thread="9d243b7d-2a52-457a-b799-72d4bc420e3a",
            workflow="workflow-1",
            execution="7dcb8a47-f3e7-435b-8a0e-c596dd2fdd90",
            workspace=None,
            requester="requester",
            responder="responder",
            request="request-1",
            context="context-1",
        )

    async def test_record_generated_script_uses_content_without_file_path(self) -> None:
        """
        Regression: final scripts must be saved from generated content, not only script.txt.
        """

        recorder = AsyncMock()
        runner = self.__runner(recorder=recorder)
        record = runner._FathomRunner__record_generated_script  # type: ignore[attr-defined]
        created = datetime(2026, 6, 30, tzinfo=timezone.utc)

        await record(
            title="Search shoes",
            handle=self.__handle(),
            created=created,
            content="open browser\nsearch shoes",
            metadata={"workflow": "workflow-1"},
        )

        recorder.record_script.assert_awaited_once()
        output: ScriptOutput = recorder.record_script.await_args.kwargs["output"]

        self.assertEqual(output.task, "7c5f7738-8c0b-4597-b9a0-988a1d22bc24")
        self.assertEqual(output.title, "Search shoes")
        self.assertEqual(output.content, "open browser\nsearch shoes")
        self.assertEqual(output.created, created)
        self.assertEqual(output.metadata, {"workflow": "workflow-1"})

    async def test_record_generated_script_skips_empty_content(self) -> None:
        """
        Empty finalization output should not create an empty script row.
        """

        recorder = AsyncMock()
        runner = self.__runner(recorder=recorder)
        record = runner._FathomRunner__record_generated_script  # type: ignore[attr-defined]

        await record(
            title="Search shoes",
            handle=self.__handle(),
            created=datetime(2026, 6, 30, tzinfo=timezone.utc),
            content="   ",
            metadata={},
        )

        recorder.record_script.assert_not_awaited()

    def test_completion_reason_preserves_recorded_reason(self) -> None:
        """
        Completion reason preserves runtime data without synthetic step prose.
        """

        recorder = AsyncMock()
        runner = self.__runner(recorder=recorder)
        reason = runner._FathomRunner__completion_reason  # type: ignore[attr-defined]

        result = reason(
            status="cancelled",
            fallback="User stopped the execution.",
        )

        self.assertEqual("User stopped the execution.", result)

    def test_task_code_uses_exact_cancelled_reason(self) -> None:
        """
        Task code mapping must not treat arbitrary failure prose as cancellation.
        """

        recorder = AsyncMock()
        runner = self.__runner(recorder=recorder)
        task_code = runner._FathomRunner__task_code  # type: ignore[attr-defined]

        self.assertEqual(
            task_code(success=False, reason=CompletionReason.CANCELLED.value),
            TaskCode.USER_CANCELLED,
        )
        self.assertEqual(
            task_code(success=False, reason="Failed while trying to cancel my order"),
            TaskCode.UNKNOWN_ERROR,
        )

    def test_task_code_maps_operator_aborted_to_user_cancelled(self) -> None:
        """
        Intent strategy emits OPERATOR_ABORTED on HITL cancel; that must land as USER_CANCELLED.
        """

        recorder = AsyncMock()
        runner = self.__runner(recorder=recorder)
        task_code = runner._FathomRunner__task_code  # type: ignore[attr-defined]

        self.assertEqual(
            task_code(success=False, reason=CompletionReason.OPERATOR_ABORTED.value),
            TaskCode.USER_CANCELLED,
        )

    def test_task_code_uses_exact_max_steps_reason(self) -> None:
        """
        Task code mapping must only timeout for the canonical max-step reason.
        """

        recorder = AsyncMock()
        runner = self.__runner(recorder=recorder)
        task_code = runner._FathomRunner__task_code  # type: ignore[attr-defined]

        self.assertEqual(
            task_code(success=False, reason=CompletionReason.MAX_STEPS.value),
            TaskCode.TIMEOUT,
        )
        self.assertEqual(
            task_code(success=False, reason="Failed after max steps in app copy"),
            TaskCode.UNKNOWN_ERROR,
        )
