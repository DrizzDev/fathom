from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, Dict, List, NoReturn, Optional, cast

from pydantic import JsonValue

from fathom.adapters.interaction.orm.postgres import PostgresInteraction
from fathom.adapters.signing.noop import NoopSigner
from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ExecutionState,
    IdempotencyState,
    JobKind,
    JobState,
    Label,
    MessageKind,
    PolicyScope,
    TaskCode,
    TaskKind,
    TaskState,
)
from fathom.constants.conversation import THREAD_TITLE_MAX_LENGTH, EntryKind, Visibility
from fathom.constants.events import FathomEvent
from fathom.constants.storage import PostgresMigrationMode
from fathom.conversation.identity import InteractionIdentity
from fathom.core.exceptions import InteractionError, ThreadConflictError, ThreadNotFoundError
from fathom.core.services.conversation import ConversationService, Ports
from fathom.core.services.recorder import ConversationRecorder
from fathom.interfaces.interaction import InteractionPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import PostgresInteractionConfiguration
from fathom.schemas.conversation import (
    ActorInput,
    ActorView,
    AddActor,
    ContextRecord,
    ConversationThreadQuery,
    EntryView,
    ExecutionReference,
    JoinMember,
    MemberView,
    MessageAppend,
    TaskFinish,
    TaskNodeView,
    TaskStart,
    TaskTreeQuery,
    ThreadCreate,
    ThreadView,
    TimelineQuery,
    TimelineView,
)
from fathom.schemas.interaction import (
    BeginRequest,
    ClaimJob,
    Execution,
    ExecutionQuery,
    FinishExecution,
    Idempotency,
    IdempotencyQuery,
    Identity,
    Job,
    JobQuery,
    MessageCursorQuery,
    Metadata,
    Policy,
    PolicyQuery,
    SavePolicy,
    ScheduleJob,
    StartExecution,
    TaskQuery,
    ThreadListQuery,
    ThreadQuery,
    Timing,
)
from fathom.schemas.recording import (
    ActionSummary,
    Analysis,
    Answer,
    Completion,
    Members,
    Metrics,
    Observation,
    Output,
    Question,
    Run,
    Step,
    StepCompletion,
    Usage,
)
from tests.unit.infrastructure.interaction.orm.support import PostgresSchema


class _StubTelemetry(TelemetryPort):
    """
    Minimal in-memory telemetry stub for recorder tests.
    """

    def __init__(self) -> None:
        self.__events: List[Dict[str, object]] = []

    async def info(self, message: str, **context: object) -> None:
        """
        Capture one info event.
        """

        self.__events.append({"level": "info", "message": message, **context})

    async def warning(self, message: str, **context: object) -> None:
        """
        Capture one warning event.
        """

        self.__events.append({"level": "warning", "message": message, **context})

    async def error(self, message: str, **context: object) -> None:
        """
        Capture one error event.
        """

        self.__events.append({"level": "error", "message": message, **context})

    async def debug(self, message: str, **context: object) -> None:
        """
        Capture one debug event.
        """

        self.__events.append({"level": "debug", "message": message, **context})

    async def exception(
        self,
        message: str,
        *,
        exception: Optional[BaseException] = None,
        **context: object,
    ) -> None:
        """
        Capture one exception event.
        """

        self.__events.append(
            {
                "level": "error",
                "message": message,
                "exception": exception,
                **context,
            }
        )

    @property
    def events(self) -> List[Dict[str, object]]:
        return list(self.__events)


class TestConversationRecorder(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for host-neutral runtime conversation recording.
    """

    async def asyncSetUp(self) -> None:
        """
        Create an isolated recorder backed by the ORM Postgres interaction adapter.
        """

        self.__schema = PostgresSchema(prefix="conversation_recorder")
        await self.__schema.__aenter__()
        interaction = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                database="postgres",
                host="localhost",
                migration_mode=PostgresMigrationMode.VALIDATE,
                password="postgres",
                pool_max_size=2,
                schema_name=self.__schema.name,
                user="postgres",
            )
        )
        await interaction.initialize()
        self.__interaction = interaction
        self.__conversation = ConversationService(
            signer=NoopSigner(),
            ports=Ports(interaction=interaction),
        )
        self.__telemetry = _StubTelemetry()
        self.__recorder = ConversationRecorder(
            conversation=self.__conversation,
            telemetry=self.__telemetry,
        )
        self.__now = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)

    async def asyncTearDown(self) -> None:
        """
        Close the ORM adapter and drop the disposable schema.
        """

        await self.__interaction.aclose()
        await self.__schema.__aexit__(
            exception_type=None,
            exception=None,
            traceback=None,
        )

    async def test_records_run_lifecycle_for_renderable_timeline(self) -> None:
        """
        Record run start and finish records that render as a client timeline.
        """

        handle = await self.__recorder.record_run_started(run=self.__run())
        assert handle is not None
        result = await self.__recorder.record_run_finished(
            completion=Completion(
                handle=handle,
                result="message-result-1",
                success=True,
                status="completed",
                reason="Order placed",
                code=TaskCode.COMPLETED,
                steps=4,
                finished=self.__now.replace(second=10),
                elapsed=10000,
            )
        )
        assert result is not None

        timeline = await self.__conversation.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", operator="human-1")
        )
        audit = await self.__conversation.timeline(
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                operator="human-1",
                mode=Visibility.AUDIT,
            )
        )

        self.assertEqual("task-run-1", handle.task)
        self.assertEqual(EntryKind.MESSAGE, result.kind)
        self.assertEqual(["message-result-1", "message-request-1"], self.__ids(timeline=timeline))
        self.assertIn("context-start-1", self.__ids(timeline=audit))
        request = await self.__interaction.get_idempotency(
            query=IdempotencyQuery(
                tenant="tenant-1",
                key=InteractionIdentity.stable(scope="request.run", parts=("execution-run-1",)),
            )
        )
        policy = await self.__interaction.get_policy(
            query=PolicyQuery(tenant="tenant-1", name="default")
        )
        jobs = await self.__interaction.get_jobs(
            query=JobQuery(tenant="tenant-1", thread="thread-1")
        )
        threads = await self.__interaction.list_threads(
            query=ThreadListQuery(tenant="tenant-1", actor="human-1")
        )
        execution = await self.__interaction.get_execution(
            query=ExecutionQuery(tenant="tenant-1", thread="thread-1", execution="execution-run-1")
        )
        messages = await self.__interaction.get_messages(
            query=MessageCursorQuery(
                tenant="tenant-1",
                thread="thread-1",
                kinds=(MessageKind.REQUEST, MessageKind.RESULT),
            )
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(IdempotencyState.COMPLETED, request.state)
        self.assertIsNotNone(policy)
        assert policy is not None
        self.assertEqual(PolicyScope.TENANT, policy.scope)
        self.assertEqual(1, len(jobs))
        self.assertEqual(JobKind.EXECUTION, jobs[0].kind)
        self.assertEqual(JobState.COMPLETED, jobs[0].state)
        self.assertIsNotNone(threads.items[0].digest)
        self.assertIsNotNone(execution)
        assert execution is not None
        self.assertEqual(ExecutionState.SUCCEEDED, execution.state)
        self.assertEqual(
            {"execution-run-1"},
            {message.execution for message in messages},
        )

    async def test_record_run_started_reserves_missing_execution_identity(self) -> None:
        """
        Recorder reserves execution-owned ids when the runtime does not provide them.
        """

        recorder = ConversationRecorder(
            conversation=self.__conversation,
            telemetry=self.__telemetry,
            identifier=lambda: "7dcb8a47-f3e7-435b-8a0e-c596dd2fdd90",
        )
        run = self.__run().model_copy(
            update={
                "context": None,
                "execution": None,
                "members": None,
                "request": None,
                "task": None,
            }
        )

        handle = await recorder.record_run_started(run=run)

        assert handle is not None
        self.assertEqual("7dcb8a47-f3e7-435b-8a0e-c596dd2fdd90", handle.execution)
        self.assertEqual(
            InteractionIdentity(execution=handle.execution).task(),
            handle.task,
        )

        execution = await self.__interaction.get_execution(
            query=ExecutionQuery(tenant="tenant-1", thread="thread-1", execution=handle.execution)
        )

        assert execution is not None
        self.assertEqual(handle.execution, execution.identity.id)
        self.assertEqual("workflow-1", execution.workflow_id)

    async def test_record_run_started_truncates_thread_title_not_intent(self) -> None:
        """
        Long run intents fit the thread-title boundary while preserving intent payloads.
        """

        intent = " ".join(("Open Instamart, change the address, add products," for _ in range(12)))
        run = self.__run().model_copy(update={"intent": intent})

        handle = await self.__recorder.record_run_started(run=run)

        assert handle is not None
        threads = await self.__interaction.list_threads(
            query=ThreadListQuery(tenant="tenant-1", actor="human-1")
        )
        execution = await self.__interaction.get_execution(
            query=ExecutionQuery(tenant="tenant-1", thread="thread-1", execution=handle.execution)
        )
        messages = await self.__interaction.get_messages(
            query=MessageCursorQuery(
                tenant="tenant-1",
                thread="thread-1",
                kinds=(MessageKind.REQUEST,),
            )
        )

        self.assertEqual(1, len(threads.items))
        title = threads.items[0].title
        self.assertIsNotNone(title)
        assert title is not None
        self.assertLessEqual(len(title), THREAD_TITLE_MAX_LENGTH)
        self.assertEqual(intent[:THREAD_TITLE_MAX_LENGTH].rstrip(), title)

        assert execution is not None
        self.assertEqual(intent, execution.intent)
        body = messages[0].content.body
        self.assertIsInstance(body, dict)
        assert isinstance(body, dict)
        self.assertEqual(intent, body["intent"])

    async def test_explicit_summary_and_detail_are_preserved_on_result_body(self) -> None:
        """
        Explicit terminal text is recorded as supplied.
        """

        handle = await self.__recorder.record_run_started(run=self.__run())
        assert handle is not None

        entry = await self.__recorder.record_run_finished(
            completion=Completion(
                steps=4,
                success=True,
                handle=handle,
                elapsed=11000,
                status="completed",
                code=TaskCode.COMPLETED,
                summary="Checkout reached.",
                result="message-result-summary-detail",
                finished=self.__now.replace(second=11),
                reason="ignored when explicit fields are present",
                detail="LLM cross-checked the final screen against the goal.",
            )
        )
        assert entry is not None

        body = self.__body(entry=entry)
        self.assertEqual("Checkout reached.", body["summary"])
        self.assertEqual(
            "LLM cross-checked the final screen against the goal.",
            body["detail"],
        )
        self.assertEqual("ignored when explicit fields are present", body["reason"])

    async def test_cancelled_run_records_cancelled_execution_and_task_state(self) -> None:
        """
        User-cancelled runs must not be stored as failed runs.
        """

        handle = await self.__recorder.record_run_started(run=self.__run())
        assert handle is not None

        await self.__recorder.record_run_finished(
            completion=Completion(
                steps=2,
                handle=handle,
                elapsed=12000,
                success=False,
                status="cancelled",
                reason="cancelled",
                code=TaskCode.USER_CANCELLED,
                result="message-result-cancelled",
                finished=self.__now.replace(second=12),
            )
        )

        execution = await self.__interaction.get_execution(
            query=ExecutionQuery(tenant="tenant-1", thread="thread-1", execution=handle.execution)
        )
        tasks = await self.__conversation.tasks(
            query=TaskTreeQuery(tenant="tenant-1", thread="thread-1", operator="human-1")
        )

        assert execution is not None
        self.assertEqual(ExecutionState.CANCELLED, execution.state)
        self.assertEqual("cancelled", tasks.roots[0].state)

    async def test_reason_with_parenthetical_clause_is_not_split(self) -> None:
        """
        A reason string is a headline, not an encoded summary/detail transport.
        """

        handle = await self.__recorder.record_run_started(run=self.__run())
        assert handle is not None

        entry = await self.__recorder.record_run_finished(
            completion=Completion(
                steps=4,
                success=True,
                handle=handle,
                elapsed=12000,
                status="completed",
                result="message-result-no-split",
                reason=(
                    "All sub-goals completed (LLM disagreed: the screenshot showed a "
                    "Swiggy McDonald's menu, not the running-shoes result page.)"
                ),
                code=TaskCode.COMPLETED,
                finished=self.__now.replace(second=12),
            )
        )
        assert entry is not None

        body = self.__body(entry=entry)

        self.assertEqual(
            "All sub-goals completed (LLM disagreed: the screenshot showed a "
            "Swiggy McDonald's menu, not the running-shoes result page.)",
            body["summary"],
        )
        self.assertIsNone(body["detail"])
        self.assertEqual(body["summary"], body["reason"])

    async def test_reason_without_parenthetical_clause_leaves_detail_null(self) -> None:
        """
        A flat reason string maps cleanly to summary with no detail produced.
        """

        handle = await self.__recorder.record_run_started(run=self.__run())
        assert handle is not None
        entry = await self.__recorder.record_run_finished(
            completion=Completion(
                handle=handle,
                result="message-result-flat",
                success=True,
                status="completed",
                reason="Order placed",
                code=TaskCode.COMPLETED,
                steps=2,
                finished=self.__now.replace(second=13),
                elapsed=2000,
            )
        )
        assert entry is not None

        body = self.__body(entry=entry)
        self.assertEqual("Order placed", body["summary"])
        self.assertIsNone(body["detail"])

    async def test_record_run_started_reuses_host_created_owner_membership(self) -> None:
        """
        Worker recording must reuse a client-created requester membership.
        """

        run = self.__run()
        await self.__conversation.create(
            request=ThreadCreate(
                id=run.thread,
                tenant=run.tenant,
                title=run.intent,
                created=run.created,
                creator=run.requester,
                member="member-owner-1",
            )
        )

        handle = await self.__recorder.record_run_started(run=run)
        timeline = await self.__conversation.timeline(
            query=TimelineQuery(
                tenant=run.tenant,
                thread=run.thread,
                operator=run.requester.id,
            )
        )

        assert handle is not None
        self.assertEqual(run.task, handle.task)
        self.assertEqual(["message-request-1"], self.__ids(timeline=timeline))
        self.assertFalse(
            [event for event in self.__telemetry.events if event.get("level") == "error"]
        )

    async def test_records_step_tree_and_subtask_aliases(self) -> None:
        """
        Record child tasks for graph steps and delegated sub-agent work.
        """

        await self.__recorder.record_run_started(run=self.__run())
        await self.__recorder.record_step_started(
            step=Step(
                id="task-step-1",
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                execution="execution-run-1",
                parent="task-run-1",
                root="task-run-1",
                actor="agent-1",
                kind=TaskKind.AGENT,
                objective="Search for item",
                created=self.__now.replace(second=1),
            )
        )
        await self.__recorder.record_subtask_started(
            step=Step(
                id="task-sub-1",
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                execution="execution-run-1",
                parent="task-step-1",
                root="task-run-1",
                actor="agent-1",
                kind=TaskKind.AGENT,
                objective="Verify item price",
                created=self.__now.replace(second=2),
            )
        )
        await self.__recorder.record_step_finished(
            completion=StepCompletion(
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                task="task-sub-1",
                state=TaskState.SUCCEEDED,
                code=TaskCode.COMPLETED,
                summary="Price verified",
                finished=self.__now.replace(second=3),
                elapsed=1000,
            )
        )

        tree = await self.__conversation.tasks(
            query=TaskTreeQuery(tenant="tenant-1", thread="thread-1", operator="human-1")
        )

        self.assertEqual("task-run-1", tree.roots[0].id)
        self.assertEqual("task-step-1", tree.roots[0].children[0].id)
        self.assertEqual("task-sub-1", tree.roots[0].children[0].children[0].id)
        self.assertEqual("delegation", tree.roots[0].children[0].children[0].kind)
        tasks = await self.__interaction.get_tasks(
            query=TaskQuery(tenant="tenant-1", thread="thread-1")
        )
        by_id = {task.identity.id: task for task in tasks}
        self.assertEqual(
            {
                "intent": "Buy milk",
                "package": "com.example",
            },
            by_id["task-run-1"].plan.plan.entries,
        )
        self.assertEqual(
            {
                "kind": "agent",
                "parent": "task-run-1",
                "root": "task-run-1",
            },
            by_id["task-step-1"].plan.plan.entries,
        )
        self.assertEqual({"state": "running"}, by_id["task-step-1"].plan.progress.entries)

    async def test_records_multiple_runs_inside_one_thread(self) -> None:
        """
        Reuse an existing thread when another run starts in the same conversation.
        """

        first = self.__run()
        second = first.model_copy(
            update={
                "task": "task-run-2",
                "execution": "execution-run-2",
                "workflow": "workflow-2",
                "intent": "Buy bread",
                "request": "message-request-2",
                "context": "context-start-2",
                "created": self.__now.replace(minute=5),
            }
        )

        await self.__recorder.record_run_started(run=first)
        await self.__recorder.record_run_started(run=second)

        timeline = await self.__conversation.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", operator="human-1")
        )
        tree = await self.__conversation.tasks(
            query=TaskTreeQuery(tenant="tenant-1", thread="thread-1", operator="human-1")
        )

        self.assertEqual(["message-request-2", "message-request-1"], self.__ids(timeline=timeline))
        self.assertEqual(["task-run-1", "task-run-2"], [task.id for task in tree.roots])

    async def test_records_hitl_question_and_answer(self) -> None:
        """
        Record human-in-the-loop question and answer messages.
        """

        await self.__recorder.record_run_started(run=self.__run())
        question = await self.__recorder.record_hitl_question(
            question=Question(
                id="message-question-1",
                tenant="tenant-1",
                thread="thread-1",
                task="task-run-1",
                actor="agent-1",
                execution="execution-run-1",
                body={"text": "Which payment method should I use?"},
                created=self.__now.replace(second=1),
            )
        )
        answer = await self.__recorder.record_hitl_answer(
            answer=Answer(
                id="message-answer-1",
                tenant="tenant-1",
                thread="thread-1",
                task="task-run-1",
                actor="human-1",
                execution="execution-run-1",
                question="message-question-1",
                body={"text": "Use wallet balance"},
                created=self.__now.replace(second=2),
            )
        )
        assert question is not None
        assert answer is not None

        timeline = await self.__conversation.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", operator="human-1")
        )

        self.assertEqual(EntryKind.MESSAGE, question.kind)
        self.assertEqual(EntryKind.MESSAGE, answer.kind)
        self.assertIn("message-question-1", self.__ids(timeline=timeline))
        self.assertIn("message-answer-1", self.__ids(timeline=timeline))

    async def test_records_step_planning_as_user_visible_progress_message(self) -> None:
        """
        Per-step planning lands as a user-visible PROGRESS message with the
        explicit step/action/target/confidence body fields populated.
        """

        await self.__recorder.record_run_started(run=self.__run())
        entry = await self.__recorder.record_llm_analysis(
            analysis=Analysis(
                id="message-progress-1",
                tenant="tenant-1",
                thread="thread-1",
                task="task-run-1",
                actor="agent-1",
                execution="execution-run-1",
                summary="Tap the checkout button to advance.",
                rationale="Checkout is the next required control.",
                observation=Observation(
                    summary="Checkout button is visible.",
                    evidence="The checkout CTA is the dominant element on the screen.",
                    screen="screen-before",
                    changed=True,
                ),
                evidence="The checkout CTA is the dominant element on the screen.",
                step=1,
                action=ActionSummary(
                    type="tap",
                    target="Checkout button",
                    rationale="Checkout is the next required control.",
                    confidence=0.92,
                ),
                metrics=Metrics(
                    total=440,
                    execution=80,
                    analysis=120,
                    grounding=240,
                    usage=Usage(prompt=100, completion=20, cached=10, total=120),
                ),
                created=self.__now.replace(second=1),
                metadata={
                    "metrics": {
                        "total": 440,
                        "execution": 80,
                        "analysis": 120,
                        "grounding": 240,
                        "usage": {
                            "total": 120,
                            "cached": 10,
                            "prompt": 100,
                            "completion": 20,
                        },
                    }
                },
            )
        )

        assert entry is not None
        self.assertEqual(Visibility.USER, entry.visibility)
        self.assertEqual(EntryKind.MESSAGE, entry.kind)
        payload = self.__payload(entry=entry)
        self.assertEqual(MessageKind.PROGRESS.value, payload["kind"])
        self.assertEqual(Audience.THREAD.value, payload["audience"])
        self.assertEqual(
            {
                "step": 1,
                "status": "completed",
                "summary": "Tap the checkout button to advance.",
                "rationale": "Checkout is the next required control.",
                "action": {
                    "type": "tap",
                    "target": "Checkout button",
                    "rationale": "Checkout is the next required control.",
                    "confidence": 0.92,
                },
                "observation": {
                    "summary": "Checkout button is visible.",
                    "evidence": "The checkout CTA is the dominant element on the screen.",
                    "screen": "screen-before",
                    "changed": True,
                },
            },
            payload["body"],
        )

        user = await self.__conversation.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", operator="human-1")
        )

        self.assertIn("message-progress-1", self.__ids(timeline=user))

    async def test_progress_body_passes_duplicated_fields_through_verbatim(self) -> None:
        """
        Recorder stores the analysis payload verbatim; duplication is the source's concern, not the recorder's.
        """

        await self.__recorder.record_run_started(run=self.__run())
        same = "I'll tap 'View Cart' to verify the item is present."
        entry = await self.__recorder.record_llm_analysis(
            analysis=Analysis(
                id="message-progress-dupe",
                tenant="tenant-1",
                thread="thread-1",
                task="task-run-1",
                actor="agent-1",
                execution="execution-run-1",
                summary=same,
                rationale="verify item in cart",
                observation=Observation(
                    summary=same,
                    evidence=same,
                    screen="abc",
                    changed=True,
                ),
                evidence=same,
                step=7,
                action=ActionSummary(
                    type="tap",
                    target="View Cart",
                    rationale="verify item in cart",
                    confidence=1.0,
                ),
                created=self.__now.replace(second=3),
            )
        )

        assert entry is not None
        body = self.__body(entry=entry)
        self.assertEqual(same, body["summary"])
        self.assertNotIn("analysis", body)

    async def test_progress_planning_without_action_keeps_optional_fields_null(self) -> None:
        """
        Terminal planning (no concrete action) still renders as PROGRESS with the action fields null.
        """

        await self.__recorder.record_run_started(run=self.__run())
        entry = await self.__recorder.record_llm_analysis(
            analysis=Analysis(
                id="message-progress-2",
                tenant="tenant-1",
                thread="thread-1",
                task="task-run-1",
                actor="agent-1",
                execution="execution-run-1",
                summary="All sub-goals reached; ready to finalize.",
                step=4,
                created=self.__now.replace(second=2),
            )
        )

        assert entry is not None
        body = self.__body(entry=entry)
        self.assertEqual(4, body["step"])
        self.assertEqual("completed", body["status"])
        self.assertIsNone(body["action"])
        self.assertIsNone(body["rationale"])
        self.assertIsNone(body["observation"])
        self.assertNotIn("analysis", body)
        self.assertEqual("All sub-goals reached; ready to finalize.", body["summary"])

    async def test_explicit_display_audit_label_overrides_to_audit_visibility(self) -> None:
        """
        Callers can still opt a planning record into audit-only by attaching DISPLAY_AUDIT.
        """

        await self.__recorder.record_run_started(run=self.__run())
        entry = await self.__recorder.record_llm_analysis(
            analysis=Analysis(
                id="message-progress-3",
                tenant="tenant-1",
                thread="thread-1",
                task="task-run-1",
                actor="agent-1",
                execution="execution-run-1",
                summary="Internal scaffolding note.",
                step=2,
                labels=(Label.DISPLAY_AUDIT,),
                created=self.__now.replace(second=3),
            )
        )
        assert entry is not None

        user = await self.__conversation.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", operator="human-1")
        )
        audit = await self.__conversation.timeline(
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                operator="human-1",
                mode=Visibility.AUDIT,
            )
        )

        self.assertEqual(Visibility.AUDIT, entry.visibility)
        self.assertNotIn("message-progress-3", self.__ids(timeline=user))
        self.assertIn("message-progress-3", self.__ids(timeline=audit))

    async def test_records_debug_artifact_with_visibility_label(self) -> None:
        """
        Debug-labelled artifacts surface in DEBUG mode and stay out of USER mode.
        """

        await self.__recorder.record_run_started(run=self.__run())
        artifact = await self.__recorder.record_artifact(
            output=Output(
                id="artifact-trace-1",
                tenant="tenant-1",
                thread="thread-1",
                task="task-run-1",
                actor="agent-1",
                kind=ArtifactKind.TRACE,
                uri="/tmp/trace.json",
                backend=ArtifactBackend.LOCAL,
                labels=(Label.DISPLAY_DEBUG,),
                created=self.__now.replace(second=4),
            )
        )

        assert artifact is not None
        self.assertEqual(Visibility.DEBUG, artifact.visibility)

        debug = await self.__conversation.timeline(
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                operator="human-1",
                mode=Visibility.DEBUG,
            )
        )

        self.assertIn("artifact-trace-1", self.__ids(timeline=debug))

    async def test_record_run_failed_rejects_successful_completion(self) -> None:
        """
        Fail fast when failed-run recording receives a successful completion.
        """

        handle = await self.__recorder.record_run_started(run=self.__run())
        assert handle is not None

        with self.assertRaises(InteractionError):
            await self.__recorder.record_run_failed(
                completion=Completion(
                    handle=handle,
                    result="message-result-1",
                    success=True,
                    status="completed",
                    reason="Done",
                    code=TaskCode.COMPLETED,
                    steps=1,
                    finished=self.__now.replace(second=1),
                    elapsed=1000,
                )
            )

    def __run(self) -> Run:
        """
        Build a deterministic run record for tests.
        """

        return Run(
            tenant="tenant-1",
            thread="thread-1",
            execution="execution-run-1",
            task="task-run-1",
            workflow="workflow-1",
            intent="Buy milk",
            package="com.example",
            requester=ActorInput(id="human-1", kind=ActorKind.HUMAN, name="Aman"),
            responder=ActorInput(id="agent-1", kind=ActorKind.AGENT, name="Fathom"),
            members=Members(requester="member-human-1", responder="member-agent-1"),
            request="message-request-1",
            context="context-start-1",
            created=self.__now,
        )

    def __ids(self, *, timeline: TimelineView) -> List[str]:
        """
        Return timeline entry identifiers for concise assertions.
        """

        return [entry.id for entry in timeline.entries]

    def __payload(self, *, entry: EntryView) -> Dict[str, JsonValue]:
        """
        Return a timeline payload object after validating its JSON shape.
        """

        payload = entry.payload
        if not isinstance(payload, dict):
            self.fail(f"Expected object payload for entry {entry.id}.")

        return payload

    def __body(self, *, entry: EntryView) -> Dict[str, JsonValue]:
        """
        Return a message body object after validating its JSON shape.
        """

        body = self.__payload(entry=entry).get("body")
        if not isinstance(body, dict):
            self.fail(f"Expected object body for entry {entry.id}.")

        return body

    async def test_step_finished_envelope_routes_workflow_and_thread(self) -> None:
        """
        Step-finished envelopes carry the typed thread and workflow ids, not parsed task slugs.
        """

        await self.__recorder.record_run_started(run=self.__run())
        await self.__recorder.record_step_started(
            step=Step(
                id="task-step-7",
                tenant="tenant-1",
                thread="thread-1",
                execution="execution-run-1",
                workflow="workflow-1",
                parent="task-run-1",
                root="task-run-1",
                actor="agent-1",
                kind=TaskKind.AGENT,
                objective="Tap login",
                created=self.__now.replace(second=1),
            )
        )
        await self.__recorder.record_step_finished(
            completion=StepCompletion(
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                task="task-step-7",
                state=TaskState.SUCCEEDED,
                code=TaskCode.COMPLETED,
                summary="ok",
                finished=self.__now.replace(second=2),
                elapsed=10,
            )
        )

        finished = [
            event
            for event in self.__telemetry.events
            if event.get("type") == "conversation.step.finished"
        ]
        self.assertEqual(1, len(finished))
        self.assertEqual("thread-1", finished[0]["conversation_id"])
        self.assertEqual("workflow-1", finished[0]["workflow_id"])
        self.assertEqual("task-step-7", finished[0]["task_id"])

    async def test_writes_emit_telemetry_envelope(self) -> None:
        """
        A successful run-start emits one structured telemetry event.
        """

        await self.__recorder.record_run_started(run=self.__run())

        starts = [
            event
            for event in self.__telemetry.events
            if event.get("type") == "conversation.run.started"
        ]
        self.assertEqual(1, len(starts))
        self.assertEqual("tenant-1", starts[0]["tenant"])
        self.assertEqual("thread-1", starts[0]["conversation_id"])
        self.assertEqual("workflow-1", starts[0]["workflow_id"])
        self.assertEqual("task-run-1", starts[0]["task_id"])
        self.assertEqual(EntryKind.EVENT.value, starts[0]["kind"])

    async def test_first_failure_suppresses_subsequent_writes(self) -> None:
        """
        After the first InteractionError, the recorder no-ops further writes
        and emits one error event to telemetry.
        """

        broken_conversation = ConversationService(
            signer=NoopSigner(),
            ports=Ports(interaction=cast("InteractionPort", _BrokenInteraction())),
        )
        recorder = ConversationRecorder(
            conversation=broken_conversation,
            telemetry=self.__telemetry,
        )

        first = await recorder.record_run_started(run=self.__run())
        self.assertIsNone(first)
        self.assertFalse(recorder.health.is_active())
        self.assertEqual(1, recorder.health.failure_count)

        second = await recorder.record_run_started(run=self.__run())
        self.assertIsNone(second)
        self.assertEqual(1, recorder.health.failure_count)

        errors = [event for event in self.__telemetry.events if event.get("level") == "error"]
        self.assertEqual(1, len(errors))
        self.assertEqual(FathomEvent.RECORDER_DISABLED, errors[0].get("type"))
        self.assertEqual("conversation.run.started", errors[0].get("operation"))
        self.assertNotIn("error", errors[0])
        self.assertNotIn("error_type", errors[0])
        self.assertNotIn("forced failure", str(errors[0]))

    async def test_concurrent_thread_create_race_falls_through_to_join(self) -> None:
        """
        Simulate the concurrent-run race: a second recorder starting a run
        in the same conversation observes "no thread" before its create
        races in, hits ThreadConflictError on create, and is expected to
        recover by joining as a member instead of crashing the run.
        """

        joined: List[str] = []
        actors_added: List[str] = []

        class _RaceConversation:
            @asynccontextmanager
            async def atomic(self) -> AsyncGenerator[None, None]:
                """
                Open a no-op grouped write boundary for the fake conversation service.
                """

                yield

            async def get(self, *, query: ConversationThreadQuery) -> ThreadView:
                raise ThreadNotFoundError(
                    thread=query.thread, message="Conversation thread does not exist."
                )

            async def internal_exists(self, *, query: ThreadQuery) -> bool:
                """
                Report the pre-race snapshot where the thread is absent.
                """

                return False

            async def create(self, *, request: ThreadCreate) -> ThreadView:
                # Simulate the racer winning between our get and our create.
                raise ThreadConflictError(
                    thread=request.id,
                    message="Thread identity already exists with different content.",
                )

            async def actor(self, *, request: AddActor) -> ActorView:
                actors_added.append(request.id)
                return ActorView(
                    id=request.id,
                    kind=ActorKind.HUMAN,
                    name=request.name or request.id,
                    created=request.created,
                )

            async def join(self, *, request: JoinMember) -> MemberView:
                joined.append(request.id)
                return MemberView(
                    id=request.id,
                    actor=request.actor,
                    role=request.role,
                    scope=request.scope,
                    joined=request.joined,
                )

            async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
                """
                Return no policy so the recorder creates the default row.
                """

                return None

            async def save_policy(self, *, request: SavePolicy) -> None:
                """
                Accept the default policy write.
                """

            async def begin_request(self, *, request: BeginRequest) -> Idempotency:
                """
                Return the started idempotency record.
                """

                return Idempotency(
                    tenant=request.tenant,
                    key=request.key,
                    hash=request.hash,
                    state=IdempotencyState.STARTED,
                    created_at=request.created,
                    expires_at=request.expires,
                    metadata=request.metadata,
                )

            async def schedule_job(self, *, request: ScheduleJob) -> Job:
                """
                Return the pending run job.
                """

                return Job(
                    identity=request.identity,
                    thread=request.thread,
                    task=request.task,
                    kind=request.kind,
                    state=JobState.PENDING,
                    attempts=0,
                    available_at=request.available,
                    payload=request.payload,
                    timing=Timing(
                        created_at=request.created,
                        updated_at=request.created,
                    ),
                    metadata=request.metadata,
                )

            async def claim_job(self, *, request: ClaimJob) -> Job:
                """
                Return the claimed run job.
                """

                return Job(
                    identity=Identity(
                        id=request.job or "job-workflow-1",
                        tenant=request.tenant,
                    ),
                    thread="thread-1",
                    task="task-run-1",
                    kind=JobKind.EXECUTION,
                    state=JobState.CLAIMED,
                    attempts=1,
                    owner=request.owner,
                    locked_at=request.claimed,
                    available_at=request.claimed,
                    timing=Timing(
                        created_at=request.claimed,
                        updated_at=request.claimed,
                    ),
                )

            async def start_execution(self, *, request: StartExecution) -> Execution:
                """
                Return the started execution row.
                """

                return Execution(
                    state=request.state,
                    thread=request.thread,
                    intent=request.intent,
                    outcome=Metadata(),
                    metadata=request.metadata,
                    identity=request.identity,
                    timing=Timing(
                        created_at=request.started,
                        updated_at=request.started,
                        started_at=request.started,
                    ),
                )

            async def finish_execution(self, *, request: FinishExecution) -> Execution:
                """
                Race test never finishes the run.
                """

                raise NotImplementedError

            async def start(self, *, request: TaskStart) -> TaskNodeView:
                return TaskNodeView(
                    id=request.id,
                    parent=None,
                    root=request.id,
                    execution=ExecutionReference(id=request.execution),
                    kind=request.kind.value,
                    state="running",
                    objective=request.objective,
                    assignee=request.assignee,
                    created=request.created,
                )

            async def finish(self, *, request: TaskFinish) -> TaskNodeView:
                raise NotImplementedError

            async def append(self, *, request: MessageAppend) -> EntryView:
                return EntryView(
                    id=request.id,
                    kind=EntryKind.MESSAGE,
                    visibility=Visibility.USER,
                    sequence=1,
                    created=request.created,
                    actor=request.author,
                    task=request.task,
                    payload={},
                )

            async def attach(self, *, request: object) -> EntryView:
                raise NotImplementedError

            async def record(self, *, request: ContextRecord) -> EntryView:
                return EntryView(
                    id=request.id,
                    kind=EntryKind.CONTEXT,
                    visibility=Visibility.AUDIT,
                    sequence=None,
                    created=request.created,
                    actor=request.consumer,
                    task=request.task,
                    payload={},
                )

        recorder = ConversationRecorder(
            conversation=_RaceConversation(),  # type: ignore[arg-type]
            telemetry=self.__telemetry,
        )
        handle = await recorder.record_run_started(run=self.__run())

        # Recorder must not crash on the racer; the run keeps going and the
        # requester is still added + joined as a member of the existing thread.
        self.assertIsNotNone(handle, self.__telemetry.events)
        self.assertTrue(recorder.health.is_active())
        self.assertEqual(2, len(joined))  # requester + responder

    async def test_unexpected_storage_failure_disables_recorder(self) -> None:
        """
        Non-InteractionError exceptions from the interaction port (DB lock,
        filesystem error) must also disable the recorder; the run keeps
        running, the recorder no-ops further writes, and one error event
        is emitted without leaking the internal exception.
        """

        broken_conversation = ConversationService(
            signer=NoopSigner(),
            ports=Ports(interaction=cast("InteractionPort", _RawErrorInteraction())),
        )
        recorder = ConversationRecorder(
            conversation=broken_conversation,
            telemetry=self.__telemetry,
        )

        first = await recorder.record_run_started(run=self.__run())
        self.assertIsNone(first)
        self.assertFalse(recorder.health.is_active())

        errors = [event for event in self.__telemetry.events if event.get("level") == "error"]
        self.assertEqual(1, len(errors))
        self.assertEqual(FathomEvent.RECORDER_DISABLED, errors[0].get("type"))
        self.assertEqual("conversation.run.started", errors[0].get("operation"))
        self.assertNotIn("error", errors[0])
        self.assertNotIn("error_type", errors[0])
        self.assertNotIn("raw storage failure", str(errors[0]))


class _BrokenInteraction:
    """
    Interaction stub that fails every method with InteractionError.
    """

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Raise the same typed storage failure at transaction entry.
        """

        raise InteractionError("forced failure on atomic")
        yield

    def __getattr__(self, name: str) -> Callable[..., object]:
        async def __fail(*_args: object, **_kwargs: object) -> NoReturn:
            raise InteractionError(f"forced failure on {name}")

        return __fail


class _RawErrorInteraction:
    """
    Interaction stub that raises raw, non-InteractionError exceptions to
    simulate filesystem / DB-lock errors leaking from the storage layer.
    """

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Raise the same raw storage failure at transaction entry.
        """

        raise RuntimeError("raw storage failure on atomic")
        yield

    def __getattr__(self, name: str) -> Callable[..., object]:
        async def __fail(*_args: object, **_kwargs: object) -> NoReturn:
            raise RuntimeError(f"raw storage failure on {name}")

        return __fail
