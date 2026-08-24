from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from fathom.adapters.interaction.noop import NoopInteraction
from fathom.constants.collaboration import TaskCode
from fathom.constants.llm import ModelProvider
from fathom.constants.manifest import ModelRole
from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.constants.state import CompletionReason
from fathom.core.config.loader import RuntimeConfigLoader
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.perception import PerceptionPort
from fathom.runtime.builder import Fathom
from fathom.runtime.runner import FathomRunner
from fathom.schemas.llm import ModelIdentity
from fathom.schemas.qualification import QualificationVerdict, Rationale
from fathom.schemas.recording import Handle, ScriptOutput
from fathom.settings.env import FathomSettings


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


class FathomRunnerExecutionIdentityTest(unittest.IsolatedAsyncioTestCase):
    """
    Runner owns the execution identity independently of the conversation layer.
    """

    def __runner(self) -> FathomRunner:
        """
        Build a runner instance without wiring conversation collaborators.
        """

        return object.__new__(FathomRunner)

    def test_reserve_execution_derives_stable_identity_from_workflow(self) -> None:
        """
        A run with no pre-reserved id derives a deterministic execution identity.
        """

        runner = self.__runner()
        reserve = runner._FathomRunner__reserve_execution  # type: ignore[attr-defined]

        first = reserve(execution_id=None, workflow_id="workflow-1")
        second = reserve(execution_id=None, workflow_id="workflow-1")

        self.assertTrue(first)
        self.assertEqual(first, second)
        self.assertNotEqual(first, reserve(execution_id=None, workflow_id="workflow-2"))

    def test_reserve_execution_honors_caller_supplied_identity(self) -> None:
        """
        A pre-reserved execution id is preserved verbatim.
        """

        runner = self.__runner()
        reserve = runner._FathomRunner__reserve_execution  # type: ignore[attr-defined]

        self.assertEqual(
            reserve(execution_id="execution-7", workflow_id="workflow-1"),
            "execution-7",
        )

    def test_reserve_execution_never_consults_the_recorder(self) -> None:
        """
        Execution identity must not depend on any recorder state or return value.
        """

        runner = self.__runner()
        reserve = runner._FathomRunner__reserve_execution  # type: ignore[attr-defined]

        self.assertFalse(hasattr(runner, "_FathomRunner__recorder"))
        self.assertTrue(reserve(execution_id=None, workflow_id="workflow-1"))


class FathomRunnerManifestEmissionTest(unittest.TestCase):
    """
    The runner resolves the model-role manifest at run start, records it for the host to read,
    and emits it as one structured event.
    """

    @staticmethod
    def __runner(*, planner: LLMPort) -> FathomRunner:
        """
        Build a runner over the minimum ports needed to resolve a manifest.
        """

        return (
            Fathom.builder()
            .with_llm(port=planner)
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_interaction(port=NoopInteraction())
            .build()
        )

    @staticmethod
    def __loader() -> RuntimeConfigLoader:
        """
        Loader bound to explicit settings so the manifest never depends on ambient env.
        """

        return RuntimeConfigLoader(settings=FathomSettings(ensemble_localizer_enabled=False))

    def test_resolution_records_the_manifest_on_the_runner(self) -> None:
        """
        After resolution the runner exposes the manifest with the effective step budget.
        """

        runner = self.__runner(planner=MagicMock(spec=LLMPort))
        resolve = runner._FathomRunner__resolve_manifest  # type: ignore[attr-defined]

        returned = resolve(steps=17, loader=self.__loader())

        self.assertIs(runner.manifest, returned)
        self.assertEqual(returned.steps, 17)

    def test_resolution_emits_one_structured_manifest_event(self) -> None:
        """
        The run-start event carries the serialized manifest under a stable event name.
        """

        runner = self.__runner(planner=MagicMock(spec=LLMPort))
        resolve = runner._FathomRunner__resolve_manifest  # type: ignore[attr-defined]

        with self.assertLogs("fathom.runtime.runner", level="INFO") as captured:
            resolve(steps=9, loader=self.__loader())

        events = [
            record
            for record in captured.records
            if getattr(record, "event", None) == "fathom.manifest.resolved"
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].manifest["steps"], 9)  # type: ignore[attr-defined]
        self.assertEqual(
            [entry["role"] for entry in events[0].manifest["roles"]],  # type: ignore[attr-defined]
            [role.value for role in ModelRole],
        )

    def test_recorded_provider_comes_from_the_declared_identity(self) -> None:
        """
        Regression: the recorded provider is what the adapter declares, not its class name.

        A class named ``StubPlanner`` would have been recorded as "stubplanner" by the previous
        class-name derivation; the declared identity is the only source now.
        """

        planner = MagicMock(spec=LLMPort)
        planner.identity = ModelIdentity(
            provider=ModelProvider.ANTHROPIC,
            model="claude-x",
        )
        runner = self.__runner(planner=planner)

        provider = runner._FathomRunner__llm_provider()  # type: ignore[attr-defined]

        self.assertEqual(provider, "anthropic")

    def test_undeclared_adapter_records_unknown_rather_than_a_guess(self) -> None:
        """
        An adapter that declares no identity is recorded as unknown.
        """

        runner = self.__runner(planner=MagicMock(spec=LLMPort))

        provider = runner._FathomRunner__llm_provider()  # type: ignore[attr-defined]

        self.assertEqual(provider, ModelProvider.UNKNOWN.value)


class FathomRunnerRejectedRunManifestTest(unittest.IsolatedAsyncioTestCase):
    """
    A run the qualifier rejects still reports which models were composed, so a host can tell
    a rejection apart from a harness misconfiguration.
    """

    async def test_rejection_result_carries_the_runs_manifest(self) -> None:
        """
        The terminal rejection result carries the manifest resolved for that run.
        """

        runner = (
            Fathom.builder()
            .with_llm(port=MagicMock(spec=LLMPort))
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_telemetry(port=AsyncMock())
            .with_interaction(port=NoopInteraction())
            .build()
        )
        resolve = runner._FathomRunner__resolve_manifest  # type: ignore[attr-defined]
        build = runner._FathomRunner__build_rejection_result  # type: ignore[attr-defined]
        manifest = resolve(
            steps=25,
            loader=RuntimeConfigLoader(settings=FathomSettings(ensemble_localizer_enabled=False)),
        )

        result = await build(
            intent="pay the invoice",
            workflow_id="workflow-1",
            start_time=0.0,
            manifest=manifest,
            verdict=QualificationVerdict(
                confidence=1.0,
                message="Cannot run that.",
                label=QualificationLabel.NOT_EXECUTABLE,
                rationale=Rationale(
                    category=RationaleCategory.EMPTY,
                    reasoning="empty",
                ),
            ),
            verdict_payload={},
        )

        assert result.manifest is not None
        self.assertIs(result.manifest, manifest)
        self.assertEqual(result.manifest.steps, 25)
        self.assertEqual(result.completion_reason, CompletionReason.NOT_EXECUTABLE.value)

    async def test_rejection_result_manifest_never_leaks_the_intent(self) -> None:
        """
        The manifest attached to a rejected run carries no trace of the user's intent text.
        """

        runner = (
            Fathom.builder()
            .with_llm(port=MagicMock(spec=LLMPort))
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_telemetry(port=AsyncMock())
            .with_interaction(port=NoopInteraction())
            .build()
        )
        resolve = runner._FathomRunner__resolve_manifest  # type: ignore[attr-defined]
        build = runner._FathomRunner__build_rejection_result  # type: ignore[attr-defined]

        result = await build(
            intent="transfer 500 to account 12345",
            workflow_id="workflow-1",
            start_time=0.0,
            manifest=resolve(
                steps=25,
                loader=RuntimeConfigLoader(
                    settings=FathomSettings(ensemble_localizer_enabled=False)
                ),
            ),
            verdict=QualificationVerdict(
                confidence=1.0,
                message=None,
                label=QualificationLabel.NOT_EXECUTABLE,
                rationale=Rationale(category=RationaleCategory.EMPTY, reasoning="empty"),
            ),
            verdict_payload={},
        )

        assert result.manifest is not None
        payload = json.dumps(result.manifest.model_dump(mode="json"))
        self.assertNotIn("12345", payload)
        self.assertNotIn("transfer", payload)
