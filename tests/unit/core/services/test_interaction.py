from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.constants.collaboration import (
    Audience,
    ContextPurpose,
    EventKind,
    JobKind,
    JobState,
    Label,
    MessageKind,
)
from fathom.conversation.identity import InteractionIdentity
from fathom.core.services.interaction import InteractionProjector, InteractionService
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action
from fathom.schemas.interaction import (
    ClaimJob,
    Content,
    ContextQuery,
    EventQuery,
    Identity,
    JobQuery,
    MessageQuery,
    Projection,
    RecordMessage,
    RunFinish,
    RunStart,
)
from fathom.schemas.screens import ScreenState


class TestInteractionService(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the interaction application service.
    """

    def setUp(self) -> None:
        """
        Create an isolated service backed by SQLite.
        """

        self.__temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__temporary_directory.cleanup)
        path = Path(self.__temporary_directory.name) / "interaction.db"
        self.__interaction = SQLiteInteraction(path=path)
        self.__service = InteractionService(interaction=self.__interaction)
        self.__now = datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc)

    async def test_records_run_lifecycle(self) -> None:
        """
        Record a full run lifecycle with messages, context, task, and job.
        """

        handle = await self.__service.start_run(
            request=RunStart(
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                intent="Buy milk",
                package="com.example",
                operator="operator-1",
                agent="agent-1",
                started=self.__now,
            )
        )

        await self.__service.finish_run(
            request=RunFinish(
                handle=handle,
                success=True,
                status="completed",
                reason="Done",
                steps=3,
                finished=self.__now,
                elapsed=1000,
            )
        )

        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1")
        )
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )
        contexts = await self.__interaction.get_contexts(
            query=ContextQuery(
                tenant="tenant-1",
                thread="thread-1",
                task=handle.task,
                purpose=ContextPurpose.EXECUTION,
            )
        )
        jobs = await self.__interaction.get_jobs(
            query=JobQuery(tenant="tenant-1", thread="thread-1", kind=JobKind.MEMORY)
        )

        self.assertEqual(["request", "result"], [message.kind.value for message in messages])
        self.assertEqual([1, 2], [message.sequence for message in messages])
        self.assertEqual((handle.request,), contexts[0].references.messages)
        self.assertEqual(JobState.PENDING, jobs[0].state)
        self.assertIn(EventKind.TASK_SUCCEEDED, [event.kind for event in events])

    async def test_start_run_is_idempotent(self) -> None:
        """
        Replay the same run start without duplicating records.
        """

        request = RunStart(
            tenant="tenant-1",
            thread="thread-1",
            workflow="workflow-1",
            intent="Buy milk",
            operator="operator-1",
            agent="agent-1",
            started=self.__now,
        )

        first = await self.__service.start_run(request=request)
        second = await self.__service.start_run(request=request)
        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual(first, second)
        self.assertEqual(1, len(messages))

    async def test_start_run_reuses_thread_memberships_across_workflows(self) -> None:
        """
        Keep one stable membership per actor role when a thread receives multiple runs.
        """

        await self.__service.start_run(
            request=RunStart(
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                intent="Buy milk",
                operator="operator-1",
                agent="agent-1",
                started=self.__now,
            )
        )
        await self.__service.start_run(
            request=RunStart(
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-2",
                intent="Buy bread",
                operator="operator-1",
                agent="agent-1",
                started=self.__now.replace(minute=5),
            )
        )
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual(
            2,
            len([event for event in events if event.kind == EventKind.ACTOR_JOINED]),
        )

    async def test_finish_run_is_idempotent_after_memory_job_claim(self) -> None:
        """
        Replay run finish even after the scheduled memory job has been claimed.
        """

        handle = await self.__service.start_run(
            request=RunStart(
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                intent="Buy milk",
                operator="operator-1",
                agent="agent-1",
                started=self.__now,
            )
        )
        request = RunFinish(
            handle=handle,
            success=True,
            status="completed",
            reason="Done",
            steps=3,
            finished=self.__now,
            elapsed=1000,
        )

        await self.__service.finish_run(request=request)
        await self.__interaction.claim_job(
            request=ClaimJob(
                tenant="tenant-1",
                owner="worker-1",
                claimed=self.__now,
                kind=JobKind.MEMORY,
            )
        )
        await self.__service.finish_run(request=request)
        jobs = await self.__interaction.get_jobs(
            query=JobQuery(tenant="tenant-1", thread="thread-1", kind=JobKind.MEMORY)
        )

        self.assertEqual(1, len(jobs))

    async def test_projector_projects_memory_jobs(self) -> None:
        """
        Claim pending memory jobs and write projected messages into memory.
        """

        memory = MemoryStub()
        projector = InteractionProjector(interaction=self.__interaction, memory=memory)
        handle = await self.__service.start_run(
            request=RunStart(
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                intent="Buy milk",
                operator="operator-1",
                agent="agent-1",
                started=self.__now,
            )
        )
        await self.__service.finish_run(
            request=RunFinish(
                handle=handle,
                success=True,
                status="completed",
                reason="Done",
                steps=3,
                finished=self.__now,
                elapsed=1000,
            )
        )

        count = await projector.project(
            request=Projection(tenant="tenant-1", owner="worker-1", claimed=self.__now)
        )
        jobs = await self.__interaction.get_jobs(
            query=JobQuery(tenant="tenant-1", thread="thread-1", kind=JobKind.MEMORY)
        )
        projected = json.loads(
            memory.values[
                f"interaction.thread-1.{InteractionIdentity(workflow='workflow-1').task()}"
            ]
        )

        self.assertEqual(1, count)
        self.assertEqual(JobState.COMPLETED, jobs[0].state)
        self.assertEqual("thread-1", projected["thread"])
        self.assertEqual(2, len(projected["messages"]))

    async def test_projector_skips_unsanitized_private_messages(self) -> None:
        """
        Do not copy unsanitized private message bodies into memory.
        """

        memory = MemoryStub()
        projector = InteractionProjector(interaction=self.__interaction, memory=memory)
        handle = await self.__service.start_run(
            request=RunStart(
                tenant="tenant-1",
                thread="thread-1",
                workflow="workflow-1",
                intent="Buy milk",
                operator="operator-1",
                agent="agent-1",
                started=self.__now,
            )
        )
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=Identity(id="message-private", tenant="tenant-1"),
                thread="thread-1",
                task=handle.task,
                author="operator-1",
                sequence=None,
                kind=MessageKind.ANSWER,
                audience=Audience.TASK,
                content=Content(
                    body={"text": "OTP is 482901"},
                    labels=(Label.PRIVACY_OTP,),
                ),
                created=self.__now,
            )
        )
        await self.__service.finish_run(
            request=RunFinish(
                handle=handle,
                success=True,
                status="completed",
                reason="Done",
                steps=3,
                finished=self.__now,
                elapsed=1000,
            )
        )

        await projector.project(
            request=Projection(tenant="tenant-1", owner="worker-1", claimed=self.__now)
        )
        value = memory.values[
            f"interaction.thread-1.{InteractionIdentity(workflow='workflow-1').task()}"
        ]

        self.assertNotIn("482901", value)


class MemoryStub(MemoryPort):
    """
    In-memory memory port test double.
    """

    def __init__(self) -> None:
        """
        Initialize captured memory values.
        """

        self.values: Dict[str, str] = {}

    async def set(self, *, key: str, value: str) -> None:
        """
        Store one memory value.
        """

        self.values[key] = value

    async def get(self, *, key: str) -> Optional[str]:
        """
        Retrieve one memory value.
        """

        return self.values.get(key)

    async def get_all(self) -> Dict[str, str]:
        """
        Retrieve all memory values.
        """

        return dict(self.values)

    async def store_observation(self, *, screen: ScreenState, description: Optional[str]) -> None:
        """
        Ignore observation writes for projection tests.
        """

        _ = screen
        _ = description

    async def store_experience(self, *, visual_hash: str, action: Action, success: bool) -> None:
        """
        Ignore experience writes for projection tests.
        """

        _ = visual_hash
        _ = action
        _ = success

    async def retrieve_knowledge(self, *, visual_hash: str) -> Dict[str, Any]:
        """
        Return no knowledge for projection tests.
        """

        _ = visual_hash
        return {}

    async def get_all_knowledge(self) -> Dict[str, Any]:
        """
        Return no knowledge summary for projection tests.
        """

        return {}
