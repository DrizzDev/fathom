from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple
from unittest.mock import AsyncMock, Mock, patch

from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.adapters.signing.noop import NoopSigner
from fathom.constants.conversation import EntryKind, Visibility
from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.constants.state import CompletionReason
from fathom.conversation.identity import InteractionIdentity
from fathom.core.services.conversation import ConversationService
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.runtime.runner import FathomRunner
from fathom.schemas.conversation import TaskTreeQuery, TimelineQuery
from fathom.schemas.qualification import QualificationVerdict, Rationale
from fathom.schemas.results import ExecutionResult
from fathom.schemas.run import Principal


class PassingQualifier(IntentQualifierPort):
    """
    Test qualifier that always allows execution.
    """

    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Return an executable verdict for runner tests.
        """

        return QualificationVerdict(
            label=QualificationLabel.EXECUTABLE,
            confidence=1.0,
            rationale=Rationale(category=RationaleCategory.UI_TASK, reasoning="test"),
        )


class SuccessfulStrategy:
    """
    Fake intent strategy that completes successfully.
    """

    completion_reason = CompletionReason.SUCCESS.value
    step_results: List[object] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        """
        Accept the same construction surface as the real strategy.
        """

    async def execute(self) -> ExecutionResult:
        """
        Return a successful execution result.
        """

        return ExecutionResult(success=True, duration=10)

    def get_progress(self) -> Dict[str, object]:
        """
        Return deterministic progress.
        """

        return {"step_count": 2}

    def get_subgoal_execution_audit(self) -> Tuple[List[str], List[str], int]:
        """
        Return empty sub-goal audit data.
        """

        return [], [], 0

    def get_metrics(self) -> None:
        """
        Return no metrics.
        """

        return None


class FailingStrategy(SuccessfulStrategy):
    """
    Fake intent strategy that fails during execution.
    """

    async def execute(self) -> ExecutionResult:
        """
        Raise a deterministic runtime failure.
        """

        raise RuntimeError("planner failed")

    def get_progress(self) -> Dict[str, object]:
        """
        Return progress available at failure time.
        """

        return {"step_count": 1}


class ContextManagerStub:
    """
    Lightweight context manager stub for runner tests.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """
        Accept the real context manager construction surface.
        """

    def set_roadmap(self, *, intent: str) -> None:
        """
        Accept roadmap updates.
        """

    async def shutdown(self) -> None:
        """
        Release no resources.
        """


class TestFathomRunnerConversationRecording(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for runtime conversation recording wiring.
    """

    def setUp(self) -> None:
        """
        Create an isolated runner with mocked runtime ports.
        """

        self.__temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__temporary_directory.cleanup)
        self.__base = Path(self.__temporary_directory.name)
        self.__interaction = SQLiteInteraction(path=self.__base / "interaction.db")
        self.__conversation = ConversationService(
            signer=NoopSigner(), interaction=self.__interaction
        )
        self.__context = patch("fathom.runtime.runner.ContextManager", ContextManagerStub)
        self.__context.start()
        self.addCleanup(self.__context.stop)
        self.__runner = self.__runner_with(interaction=self.__interaction)

    async def asyncTearDown(self) -> None:
        """
        Release runner resources after each test.
        """

        await self.__runner.cleanup()

    async def test_run_intent_records_conversation_lifecycle_and_artifacts(self) -> None:
        """
        Record run start, result, context, task, and artifacts through the recorder.
        """

        self.__write_artifact(
            category="history",
            package="com.example",
            workflow="workflow-1",
            name="script.txt",
        )
        self.__write_artifact(
            category="screenshot",
            package="com.example",
            workflow="workflow-1",
            name="capture.png",
        )

        with patch("fathom.runtime.runner.IntentStrategy", SuccessfulStrategy):
            result = await self.__runner.run_intent(
                intent="Buy milk",
                request_id="workflow-1",
                package_name="com.example",
                principal=Principal(
                    tenant="default",
                    operator="human-1",
                    agent="agent-1",
                    conversation="thread-1",
                ),
            )

        timeline = await self.__conversation.timeline(
            query=TimelineQuery(tenant="default", thread="thread-1")
        )
        audit = await self.__conversation.timeline(
            query=TimelineQuery(tenant="default", thread="thread-1", mode=Visibility.AUDIT)
        )
        tree = await self.__conversation.tasks(
            query=TaskTreeQuery(tenant="default", thread="thread-1")
        )

        self.assertTrue(result.success)
        self.assertEqual(
            {
                InteractionIdentity(workflow="workflow-1").message(
                    name="request"
                ): EntryKind.MESSAGE,
                InteractionIdentity(workflow="workflow-1").message(
                    name="result"
                ): EntryKind.MESSAGE,
            },
            {entry.id: entry.kind for entry in timeline.entries if entry.kind == EntryKind.MESSAGE},
        )
        self.assertIn(EntryKind.ARTIFACT, [entry.kind for entry in timeline.entries])
        self.assertEqual(InteractionIdentity(workflow="workflow-1").task(), tree.roots[0].id)
        self.assertIn(
            InteractionIdentity(workflow="workflow-1").context(name="start"),
            [entry.id for entry in audit.entries],
        )

    async def test_run_intent_records_failed_completion_before_reraising(self) -> None:
        """
        Record a failed terminal result when strategy execution raises.
        """

        with (
            patch("fathom.runtime.runner.IntentStrategy", FailingStrategy),
            self.assertRaises(RuntimeError),
        ):
            await self.__runner.run_intent(
                intent="Buy milk",
                request_id="workflow-1",
                package_name="com.example",
                principal=Principal(
                    tenant="default",
                    operator="human-1",
                    agent="agent-1",
                    conversation="thread-1",
                ),
            )

        timeline = await self.__conversation.timeline(
            query=TimelineQuery(tenant="default", thread="thread-1")
        )
        tree = await self.__conversation.tasks(
            query=TaskTreeQuery(tenant="default", thread="thread-1")
        )

        self.assertEqual(
            [
                InteractionIdentity(workflow="workflow-1").message(name="result"),
                InteractionIdentity(workflow="workflow-1").message(name="request"),
            ],
            [entry.id for entry in timeline.entries],
        )
        self.assertEqual("failed", tree.roots[0].state)
        self.assertEqual(CompletionReason.FAILED.value, tree.roots[0].summary)

    async def test_run_exploration_records_failure_before_reraising(self) -> None:
        """
        Mirror the run_intent failure recording on the exploration path so
        an exception during exploration leaves a terminal record on the
        thread instead of an orphaned RUNNING root task.
        """

        from fathom.runtime.runner import ExplorationStrategy as _RealExploration  # noqa: F401

        class _FailingExploration:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def execute(self):
                raise RuntimeError("exploration crashed")

            def get_progress(self):
                return {"steps": 4}

            @property
            def graph(self):
                return SimpleNamespace(nodes={})

        with (
            patch("fathom.runtime.runner.ExplorationStrategy", _FailingExploration),
            self.assertRaises(RuntimeError),
        ):
            await self.__runner.run_exploration(
                request_id="workflow-x",
                package_name="com.example",
                principal=Principal(
                    tenant="default",
                    operator="human-1",
                    agent="agent-1",
                    conversation="thread-x",
                ),
            )

        tree = await self.__conversation.tasks(
            query=TaskTreeQuery(tenant="default", thread="thread-x")
        )

        self.assertEqual("failed", tree.roots[0].state)

    def __runner_with(self, *, interaction: SQLiteInteraction) -> FathomRunner:
        """
        Build a runner with mocked ports and real interaction storage.
        """

        device = SimpleNamespace(
            configuration=None,
            get_current_package=AsyncMock(return_value="com.example"),
        )

        llm = Mock()
        llm.cleanup = AsyncMock()

        memory = Mock()
        memory.get_all_knowledge = AsyncMock(return_value={})

        telemetry = SimpleNamespace(info=AsyncMock(), warning=AsyncMock())

        path = SimpleNamespace(base_path=self.__base)

        return FathomRunner(
            llm=llm,
            device=device,
            perception=Mock(),
            memory=memory,
            signal=Mock(),
            storage=SimpleNamespace(),
            knowledge=Mock(),
            telemetry=telemetry,
            summarizer=Mock(),
            qualifier=PassingQualifier(),
            path_manager=path,
            interaction=interaction,
        )

    def __write_artifact(
        self,
        *,
        category: str,
        package: str,
        workflow: str,
        name: str,
    ) -> None:
        """
        Create one generated artifact in the runner's expected directory layout.
        """

        path = self.__base / category / "session" / package / workflow / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"generated at {datetime.now().isoformat()}", encoding="utf-8")
