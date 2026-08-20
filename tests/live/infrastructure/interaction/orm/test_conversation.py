from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional, Type
from uuid import uuid4

import asyncpg
import pytest

from fathom.adapters.interaction.orm.postgres import PostgresInteraction
from fathom.adapters.signing.noop import NoopSigner
from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ContextPurpose,
    ExecutionState,
    IdempotencyState,
    JobCode,
    JobKind,
    JobState,
    MembershipRole,
    MessageKind,
    PolicyScope,
    ScriptStatus,
    TaskCode,
    TaskKind,
    TaskState,
    ThreadState,
)
from fathom.constants.conversation import EntryKind, RunState, Visibility
from fathom.constants.signing import SigningStatus
from fathom.constants.storage import PostgresMigrationMode
from fathom.core.exceptions import ThreadNotFoundError
from fathom.core.services.conversation import ConversationService, Ports
from fathom.interfaces.interaction import InteractionPort
from fathom.schemas import conversation as ConversationSchemas
from fathom.schemas import interaction as InteractionSchemas
from fathom.schemas.configuration import PostgresInteractionConfiguration

if TYPE_CHECKING:
    from types import TracebackType


pytestmark = pytest.mark.asyncio


class LocalPostgresConversationHarness:
    """
    Owns one disposable local Postgres schema and service instance.
    """

    def __init__(self) -> None:
        """
        Build an isolated schema name and adapter configuration.
        """

        self.__schema = f"live_conversation_{uuid4().hex}"
        self.__cleanup: Optional[asyncpg.Connection] = None

        self.store = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                pool_min_size=1,
                pool_max_size=2,
                schema_name=self.__schema,
                dsn="postgresql://localhost/postgres",
                migration_mode=PostgresMigrationMode.APPLY,
            )
        )
        self.service = ConversationService(
            signer=NoopSigner(),
            ports=self.__ports(interaction=self.store),
        )

    async def __aenter__(self) -> LocalPostgresConversationHarness:
        """
        Initialize the real Postgres adapter or skip when Postgres is unavailable.
        """

        try:
            self.__cleanup = await asyncpg.connect(database="postgres")
            await self.store.initialize()
        except (OSError, asyncpg.PostgresError) as exception:
            await self.__close()
            pytest.skip(f"Local Postgres unavailable: {exception}")

        return self

    async def __aexit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """
        Close adapter resources and drop the disposable schema.
        """

        _ = exception_type, exception, traceback
        await self.__close()

    async def __close(self) -> None:
        """
        Release the adapter pool and remove the test schema.
        """

        await self.store.aclose()

        if self.__cleanup is None:
            return

        await self.__cleanup.execute(f'DROP SCHEMA IF EXISTS "{self.__schema}" CASCADE')
        await self.__cleanup.close()

        self.__cleanup = None

    def __ports(self, *, interaction: InteractionPort) -> Ports:
        """
        Bind every narrow conversation port to the same Postgres interaction adapter.
        """

        return Ports(interaction=interaction)


class ExistingPostgresConversationHarness:
    """
    Opens the existing migrated local Fathom Postgres schema without dropping it.
    """

    __DEFAULT_SCHEMA = "fathom"
    __DEFAULT_DSN = "postgresql://localhost/fathom"

    def __init__(self) -> None:
        """
        Build a validate-only adapter using the same env names as runtime configuration.
        """

        self.store = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                dsn=self.__dsn(),
                pool_min_size=1,
                pool_max_size=2,
                schema_name=self.__schema(),
                migration_mode=PostgresMigrationMode.VALIDATE,
            )
        )
        self.service = ConversationService(
            signer=NoopSigner(),
            ports=self.__ports(interaction=self.store),
        )

    async def __aenter__(self) -> ExistingPostgresConversationHarness:
        """
        Initialize the adapter against the existing schema or skip when unavailable.
        """

        try:
            await self.store.initialize()
        except (OSError, asyncpg.PostgresError) as exception:
            await self.store.aclose()
            pytest.skip(f"Existing local Fathom Postgres unavailable: {exception}")

        return self

    async def __aexit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """
        Close adapter resources without dropping the existing schema.
        """

        _ = exception_type, exception, traceback
        await self.store.aclose()

    def __dsn(self) -> str:
        """
        Resolve the Postgres DSN used by the existing local Fathom database.
        """

        return (
            os.environ.get("DRIZZ_FATHOM_POSTGRES_DSN")
            or os.environ.get("FATHOM_INTERACTION_POSTGRES_DSN")
            or self.__DEFAULT_DSN
        )

    def __schema(self) -> str:
        """
        Resolve the Postgres schema used by the existing local Fathom database.
        """

        return (
            os.environ.get("DRIZZ_FATHOM_POSTGRES_SCHEMA")
            or os.environ.get("FATHOM_INTERACTION_POSTGRES_SCHEMA")
            or self.__DEFAULT_SCHEMA
        )

    def __ports(self, *, interaction: InteractionPort) -> Ports:
        """
        Bind every narrow conversation port to the same Postgres interaction adapter.
        """

        return Ports(interaction=interaction)


class TestLiveConversationLayerPostgres:
    """
    Exercise the conversation service and Postgres-backed interaction store together.
    """

    async def test_full_conversation_layer_flow_against_local_postgres(self) -> None:
        """
        Verify core API, service, and repository paths against a migrated local Postgres schema.
        """

        async with LocalPostgresConversationHarness() as harness:
            now = datetime(2026, 6, 20, 10, 0, tzinfo=timezone.utc)

            tenant = "tenant-live"
            ids = LiveConversationIds.build()

            await self.__write_conversation(
                ids=ids,
                now=now,
                tenant=tenant,
                service=harness.service,
            )
            await self.__verify_service_reads(
                ids=ids,
                tenant=tenant,
                service=harness.service,
            )
            await self.__verify_port_reads(
                ids=ids,
                tenant=tenant,
                store=harness.store,
            )
            await self.__verify_lifecycle_visibility(
                ids=ids,
                now=now,
                tenant=tenant,
                store=harness.store,
                service=harness.service,
            )

    async def test_full_conversation_layer_flow_against_existing_fathom_postgres(
        self,
    ) -> None:
        """
        Verify the flow against the already-migrated local Fathom database.
        """

        async with ExistingPostgresConversationHarness() as harness:
            now = datetime.now(tz=timezone.utc)

            tenant = "tenant-live-existing"
            ids = LiveConversationIds.build()

            await self.__write_conversation(
                ids=ids,
                now=now,
                tenant=tenant,
                service=harness.service,
            )
            await self.__verify_service_reads(
                ids=ids,
                tenant=tenant,
                service=harness.service,
            )
            await self.__verify_port_reads(
                ids=ids,
                tenant=tenant,
                store=harness.store,
            )
            await self.__verify_lifecycle_visibility(
                ids=ids,
                now=now,
                tenant=tenant,
                store=harness.store,
                service=harness.service,
            )

    async def __write_conversation(
        self,
        *,
        tenant: str,
        now: datetime,
        ids: LiveConversationIds,
        service: ConversationService,
    ) -> None:
        """
        Persist every conversation aggregate through typed service calls.
        """

        created = await service.create(
            request=ConversationSchemas.ThreadCreate(
                id=ids.thread,
                tenant=tenant,
                title=None,
                creator=ConversationSchemas.ActorInput(
                    id=ids.operator,
                    kind=ActorKind.HUMAN,
                    name="Live Operator",
                ),
                member=ids.owner_membership,
                created=now,
            )
        )
        assert created.id == ids.thread
        assert created.title is None

        titled = await service.title(
            tenant=tenant,
            title=ids.title,
            thread=ids.thread,
            operator=ids.operator,
            updated=now + timedelta(seconds=1),
        )
        assert titled.title == ids.title
        assert titled.metadata.title is not None
        assert titled.metadata.title.source == "intent"
        assert titled.metadata.title.refreshed == now + timedelta(seconds=1)

        agent = await service.actor(
            request=ConversationSchemas.AddActor(
                id=ids.agent,
                tenant=tenant,
                name="Live Agent",
                kind=ActorKind.AGENT,
                created=now + timedelta(seconds=2),
            )
        )
        assert agent.id == ids.agent

        member = await service.join(
            request=ConversationSchemas.JoinMember(
                tenant=tenant,
                actor=ids.agent,
                thread=ids.thread,
                id=ids.agent_membership,
                role=MembershipRole.DELEGATE,
                joined=now + timedelta(seconds=3),
            )
        )
        assert member.actor == ids.agent

        execution = await service.start_execution(
            request=InteractionSchemas.StartExecution(
                thread=ids.thread,
                actor=ids.operator,
                started_at=now + timedelta(seconds=4),
                intent="Exercise the live conversation layer",
                identity=InteractionSchemas.Identity(id=ids.execution, tenant=tenant),
            )
        )
        assert execution.identity.id == ids.execution
        assert execution.state == ExecutionState.RUNNING

        task = await service.start(
            request=ConversationSchemas.TaskStart(
                id=ids.task,
                tenant=tenant,
                thread=ids.thread,
                assignee=ids.agent,
                creator=ids.operator,
                kind=TaskKind.FATHOM,
                execution=ids.execution,
                created=now + timedelta(seconds=5),
                objective="Exercise the live conversation layer",
            )
        )
        assert task.id == ids.task
        assert task.state == TaskState.RUNNING

        message = await service.append(
            request=ConversationSchemas.MessageAppend(
                labels=(),
                task=ids.task,
                tenant=tenant,
                id=ids.message,
                thread=ids.thread,
                author=ids.operator,
                execution=ids.execution,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                created=now + timedelta(seconds=6),
                body={
                    "package": "com.example.live",
                    "intent": "Run the live Postgres path.",
                },
            )
        )
        assert message.kind == EntryKind.MESSAGE

        progress = await service.append(
            request=ConversationSchemas.MessageAppend(
                labels=(),
                task=ids.task,
                tenant=tenant,
                thread=ids.thread,
                author=ids.agent,
                id=ids.progress_message,
                execution=ids.execution,
                kind=MessageKind.PROGRESS,
                audience=Audience.THREAD,
                created=now + timedelta(seconds=7),
                body={
                    "step": 1,
                    "status": "completed",
                    "action": {
                        "type": "tap",
                        "target": "Search",
                        "confidence": 0.91,
                        "rationale": "Search is the next visible control.",
                    },
                    "analysis": "Tap Search to continue the flow.",
                    "rationale": "Search is the next visible control.",
                    "observation": {
                        "changed": True,
                        "screen": "screen-live-1",
                        "summary": "Search results are visible.",
                        "evidence": "The search field and results list are visible.",
                    },
                    "summary": "Tapped Search",
                },
            )
        )
        assert progress.kind == EntryKind.MESSAGE

        artifact = await service.attach(
            request=ConversationSchemas.ArtifactAttach(
                size=128,
                task=ids.task,
                tenant=tenant,
                id=ids.artifact,
                mime="image/png",
                thread=ids.thread,
                producer=ids.agent,
                execution=ids.execution,
                kind=ArtifactKind.SCREENSHOT,
                backend=ArtifactBackend.OBJECT,
                created=now + timedelta(seconds=8),
                uri="s3://drizz-live/screenshots/one.png",
            )
        )
        assert artifact.kind == EntryKind.ARTIFACT

        script = await service.save(
            request=ConversationSchemas.ScriptSave(
                id=ids.script,
                tenant=tenant,
                task=ids.task,
                actor=ids.agent,
                thread=ids.thread,
                title="Live script",
                content="tap('Search')",
                status=ScriptStatus.ACTIVE,
                summary="Generated by live test",
                created=now + timedelta(seconds=9),
            )
        )

        assert script.artifact is None
        assert script.identity.id == ids.script

        context = await service.record(
            request=ConversationSchemas.ContextRecord(
                task=ids.task,
                tenant=tenant,
                id=ids.context,
                thread=ids.thread,
                consumer=ids.agent,
                builder="live-test@1",
                execution=ids.execution,
                messages=(ids.message,),
                artifacts=(ids.artifact,),
                purpose=ContextPurpose.EXECUTION,
                created=now + timedelta(seconds=10),
            )
        )
        assert context.kind == EntryKind.CONTEXT

        policy = await service.save_policy(
            request=InteractionSchemas.SavePolicy(
                name=ids.policy_name,
                scope=PolicyScope.TENANT,
                created_at=now + timedelta(seconds=11),
                identity=InteractionSchemas.Identity(id=ids.policy, tenant=tenant),
                governance=InteractionSchemas.Governance(
                    retention=InteractionSchemas.Metadata(entries={"messages": "30d"})
                ),
            )
        )
        assert policy is None

        idempotency = await service.begin_request(
            request=InteractionSchemas.BeginRequest(
                tenant=tenant,
                hash="sha256:live",
                key=ids.idempotency_key,
                expires_at=now + timedelta(hours=1),
                created_at=now + timedelta(seconds=12),
            )
        )
        assert idempotency.state == IdempotencyState.STARTED

        finished_idempotency = await service.finish_request(
            request=InteractionSchemas.FinishRequest(
                tenant=tenant,
                key=ids.idempotency_key,
                response={"thread": ids.thread},
                state=IdempotencyState.COMPLETED,
                finished=now + timedelta(seconds=13),
            )
        )
        assert finished_idempotency.state == IdempotencyState.COMPLETED

        job = await service.schedule_job(
            request=InteractionSchemas.ScheduleJob(
                task=ids.task,
                thread=ids.thread,
                kind=JobKind.EXECUTION,
                execution=ids.execution,
                created_at=now + timedelta(seconds=14),
                available_at=now + timedelta(seconds=14),
                payload=InteractionSchemas.Metadata(entries={"phase": "live"}),
                identity=InteractionSchemas.Identity(id=ids.job, tenant=tenant),
            )
        )
        assert job.identity.id == ids.job

        claimed = await service.claim_job(
            request=InteractionSchemas.ClaimJob(
                job=ids.job,
                tenant=tenant,
                owner="live-worker",
                claimed=now + timedelta(seconds=15),
            )
        )
        assert claimed is not None
        assert claimed.state == JobState.CLAIMED

        finished_job = await service.finish_job(
            request=InteractionSchemas.FinishJob(
                job=ids.job,
                tenant=tenant,
                owner="live-worker",
                state=JobState.COMPLETED,
                outcome=InteractionSchemas.Outcome(
                    code=JobCode.COMPLETED,
                    detail="Live job completed.",
                ),
                finished=now + timedelta(seconds=16),
            )
        )
        assert finished_job.state == JobState.COMPLETED

        result = await service.append(
            request=ConversationSchemas.MessageAppend(
                labels=(),
                task=ids.task,
                tenant=tenant,
                thread=ids.thread,
                author=ids.agent,
                id=ids.result_message,
                execution=ids.execution,
                kind=MessageKind.RESULT,
                audience=Audience.THREAD,
                created=now + timedelta(seconds=17),
                body={
                    "status": "success",
                    "summary": "Live summary complete",
                    "detail": "The service layer returned the expected summary rows.",
                },
            )
        )
        assert result.kind == EntryKind.MESSAGE

        finished_task = await service.finish(
            request=ConversationSchemas.TaskFinish(
                tenant=tenant,
                task=ids.task,
                elapsed=15000,
                code=TaskCode.COMPLETED,
                state=TaskState.SUCCEEDED,
                summary="Live task completed.",
                ended=now + timedelta(seconds=18),
                detail="Postgres-backed flow completed.",
            )
        )
        assert finished_task.state == TaskState.SUCCEEDED

        cleanup = await service.cleanup(
            request=InteractionSchemas.CleanupRequest(
                tenant=tenant,
                events_before=now - timedelta(days=1),
                idempotency_before=now - timedelta(days=1),
                soft_deleted_before=now - timedelta(days=1),
                terminal_jobs_before=now - timedelta(days=1),
            )
        )
        assert all(value == 0 for value in cleanup.model_dump().values())

    async def __verify_service_reads(
        self,
        *,
        tenant: str,
        ids: LiveConversationIds,
        service: ConversationService,
    ) -> None:
        """
        Verify client-facing service queries return the written data.
        """

        thread = await service.get(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
            )
        )
        assert thread.id == ids.thread
        assert thread.title == ids.title

        conversations = await service.list(
            query=ConversationSchemas.ConversationListQuery(
                tenant=tenant,
                title=ids.title,
                operator=ids.operator,
            )
        )
        assert ids.thread in {item.id for item in conversations.items}

        messages = await service.messages(
            query=ConversationSchemas.MessageListQuery(
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
                kinds=(MessageKind.REQUEST,),
            )
        )
        assert [item.id for item in messages.items] == [ids.message]

        artifacts = await service.artifacts(
            query=ConversationSchemas.ArtifactListQuery(
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
                kinds=(ArtifactKind.SCREENSHOT,),
            )
        )
        assert [item.id for item in artifacts.items] == [ids.artifact]
        assert artifacts.items[0].signing_status == SigningStatus.NOT_REQUIRED

        run_script = await service.script(
            query=ConversationSchemas.RunScriptQuery(
                task=ids.task,
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
            )
        )
        assert run_script is not None
        assert run_script.id == ids.script
        assert run_script.content == "tap('Search')"

        scripts = await service.list_scripts(
            query=ConversationSchemas.ScriptsQuery(
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
            )
        )
        assert [item.id for item in scripts.items] == [ids.script]

        summary_messages = await service.summary_messages(
            query=InteractionSchemas.SummaryMessagesQuery(
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
                kinds=(MessageKind.REQUEST,),
            )
        )
        assert [item.id for item in summary_messages] == [ids.message]

        summary_scripts = await service.summary_scripts(
            query=InteractionSchemas.SummaryScriptsQuery(
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
            )
        )
        assert [item.id for item in summary_scripts] == [ids.script]

        summary = await service.summary(
            query=ConversationSchemas.SummaryQuery(
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
            )
        )

        assert summary.counts.runs == 1
        assert summary.counts.scripts == 1
        assert summary.counts.messages == 3
        assert summary.counts.artifacts == 1

        assert len(summary.runs) == 1
        assert summary.thread.id == ids.thread
        assert summary.runs[0].task == ids.task
        assert summary.runs[0].state == RunState.SUCCEEDED
        assert summary.runs[0].execution.id == ids.execution

        assert summary.runs[0].intent.packages is not None
        assert summary.runs[0].intent.packages.target == "com.example.live"
        assert summary.runs[0].intent.text == "Run the live Postgres path."

        assert summary.runs[0].outcome.status == "success"
        assert summary.runs[0].outcome.summary == "Live summary complete"

        assert len(summary.runs[0].milestones) == 1
        assert summary.runs[0].milestones[0].analysis is None
        assert summary.runs[0].milestones[0].action is not None
        assert summary.runs[0].milestones[0].action.type == "tap"
        assert summary.runs[0].milestones[0].status == "completed"
        assert summary.runs[0].milestones[0].observation is not None
        assert summary.runs[0].milestones[0].action.target == "Search"
        assert summary.runs[0].milestones[0].summary == "Tapped Search"
        assert summary.runs[0].milestones[0].observation.summary == "Search results are visible."
        assert (
            summary.runs[0].milestones[0].action.rationale == "Search is the next visible control."
        )

        assert summary.runs[0].script is not None
        assert summary.runs[0].script.id == ids.script

        assert summary.overview.status == RunState.SUCCEEDED
        assert summary.overview.activity == summary.runs[0].updated

        timeline = await service.timeline(
            query=ConversationSchemas.TimelineQuery(
                limit=10,
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
                mode=Visibility.AUDIT,
                kinds=(EntryKind.MESSAGE, EntryKind.ARTIFACT, EntryKind.CONTEXT),
            )
        )
        assert {entry.kind for entry in timeline.entries} == {
            EntryKind.CONTEXT,
            EntryKind.MESSAGE,
            EntryKind.ARTIFACT,
        }

        message_timeline = await service.timeline(
            query=ConversationSchemas.TimelineQuery(
                limit=10,
                tenant=tenant,
                thread=ids.thread,
                mode=Visibility.USER,
                operator=ids.operator,
                kinds=(EntryKind.MESSAGE,),
            )
        )
        assert message_timeline.total == 3
        assert [entry.id for entry in message_timeline.entries] == [
            ids.result_message,
            ids.progress_message,
            ids.message,
        ]
        assert [self.__payload_kind(entry=entry) for entry in message_timeline.entries] == [
            MessageKind.RESULT.value,
            MessageKind.PROGRESS.value,
            MessageKind.REQUEST.value,
        ]

        tasks = await service.tasks(
            query=ConversationSchemas.TaskTreeQuery(
                tenant=tenant,
                thread=ids.thread,
                operator=ids.operator,
            )
        )
        assert [task.id for task in tasks.roots] == [ids.task]

        state = await service.state(
            tenant=tenant,
            task=ids.task,
            thread=ids.thread,
            operator=ids.operator,
        )
        assert state == TaskState.SUCCEEDED.value

        policy = await service.get_policy(
            query=InteractionSchemas.PolicyQuery(tenant=tenant, name=ids.policy_name)
        )
        assert policy is not None
        assert policy.identity.id == ids.policy

        idempotency = await service.get_idempotency(
            query=InteractionSchemas.IdempotencyQuery(tenant=tenant, key=ids.idempotency_key)
        )
        assert idempotency is not None
        assert idempotency.state == IdempotencyState.COMPLETED

    async def __verify_port_reads(
        self,
        *,
        tenant: str,
        store: InteractionPort,
        ids: LiveConversationIds,
    ) -> None:
        """
        Verify repository-backed adapter reads that are not directly exposed as views.
        """

        events = await store.get_events(
            query=InteractionSchemas.EventQuery(tenant=tenant, thread=ids.thread)
        )
        assert {event.kind.value for event in events} >= {
            "task.opened",
            "actor.joined",
            "context.built",
            "job.scheduled",
            "job.completed",
            "thread.created",
            "task.succeeded",
            "artifact.linked",
            "message.recorded",
        }

        versions = await store.get_script_versions(
            query=InteractionSchemas.ScriptVersionQuery(tenant=tenant, script=ids.script)
        )
        assert len(versions) == 1
        assert versions[0].content == "tap('Search')"

        jobs = await store.get_jobs(
            query=InteractionSchemas.JobQuery(
                tenant=tenant,
                thread=ids.thread,
                state=JobState.COMPLETED,
            )
        )
        assert jobs[0].execution == ids.execution
        assert [job.identity.id for job in jobs] == [ids.job]

        contexts = await store.get_contexts(
            query=InteractionSchemas.ContextQuery(
                tenant=tenant,
                thread=ids.thread,
            )
        )

        assert contexts[0].execution == ids.execution
        assert [context.identity.id for context in contexts] == [ids.context]

        context_page = await store.list_contexts(
            query=InteractionSchemas.ContextCursorQuery(
                tenant=tenant,
                thread=ids.thread,
            )
        )
        assert context_page.items[0].execution == ids.execution
        assert [context.identity.id for context in context_page.items] == [ids.context]

    async def __verify_lifecycle_visibility(
        self,
        *,
        tenant: str,
        now: datetime,
        store: InteractionPort,
        ids: LiveConversationIds,
        service: ConversationService,
    ) -> None:
        """
        Verify archive, unarchive, and delete hide or expose rows correctly.
        """

        archived = await service.archive(
            request=ConversationSchemas.ConversationTransition(
                tenant=tenant,
                thread=ids.thread,
                actor=ids.operator,
                updated=now + timedelta(minutes=1),
            )
        )
        assert archived.id == ids.thread
        assert archived.state.value == "archived"

        default_page = await service.list(
            query=ConversationSchemas.ConversationListQuery(
                tenant=tenant,
                operator=ids.operator,
            )
        )
        assert ids.thread not in {item.id for item in default_page.items}

        archived_page = await service.list(
            query=ConversationSchemas.ConversationListQuery(
                tenant=tenant,
                operator=ids.operator,
                state=ThreadState.ARCHIVED,
            )
        )
        assert [item.id for item in archived_page.items] == [ids.thread]

        restored = await service.unarchive(
            request=ConversationSchemas.ConversationTransition(
                tenant=tenant,
                thread=ids.thread,
                actor=ids.operator,
                include_archived=True,
                updated=now + timedelta(minutes=2),
            )
        )
        assert restored.state.value == "active"

        deleted = await service.delete(
            request=ConversationSchemas.ConversationTransition(
                tenant=tenant,
                thread=ids.thread,
                actor=ids.operator,
                updated=now + timedelta(minutes=3),
            )
        )
        assert deleted.state.value == "deleted"

        with pytest.raises(ThreadNotFoundError):
            await service.get(
                query=ConversationSchemas.ConversationThreadQuery(
                    tenant=tenant,
                    thread=ids.thread,
                    operator=ids.operator,
                )
            )

        hidden_messages = await store.list_messages(
            query=InteractionSchemas.MessageCursorQuery(
                tenant=tenant,
                thread=ids.thread,
            )
        )
        assert hidden_messages.items == ()

        hidden_artifacts = await store.list_artifacts(
            query=InteractionSchemas.ArtifactCursorQuery(
                tenant=tenant,
                thread=ids.thread,
            )
        )
        assert hidden_artifacts.items == ()

        hidden_scripts = await store.list_scripts(
            query=InteractionSchemas.ScriptListQuery(
                tenant=tenant,
                thread=ids.thread,
            )
        )
        assert hidden_scripts.items == ()

        hidden_execution = await store.get_execution(
            query=InteractionSchemas.ExecutionQuery(
                tenant=tenant,
                thread=ids.thread,
                execution=ids.execution,
            )
        )
        assert hidden_execution is None

        hidden_contexts = await store.list_contexts(
            query=InteractionSchemas.ContextCursorQuery(
                tenant=tenant,
                thread=ids.thread,
            )
        )
        assert hidden_contexts.items == ()

        hidden_jobs = await store.get_jobs(
            query=InteractionSchemas.JobQuery(
                tenant=tenant,
                thread=ids.thread,
            )
        )
        assert hidden_jobs == []

        hidden_events = await store.list_events(
            query=InteractionSchemas.EventCursorQuery(
                tenant=tenant,
                thread=ids.thread,
            )
        )
        assert hidden_events.items == ()

    def __payload_kind(self, *, entry: ConversationSchemas.EntryView) -> str:
        """
        Return the message kind from a timeline entry payload.
        """

        if not isinstance(entry.payload, dict):
            raise AssertionError("Timeline entry payload is not an object.")

        kind = entry.payload.get("kind")
        if not isinstance(kind, str):
            raise AssertionError("Timeline entry payload kind is missing.")

        return kind


class LiveConversationIds:
    """
    Test fixture identifiers used to build real conversation schemas.
    """

    def __init__(
        self,
        *,
        job: str,
        task: str,
        agent: str,
        title: str,
        thread: str,
        script: str,
        policy: str,
        message: str,
        context: str,
        artifact: str,
        operator: str,
        execution: str,
        policy_name: str,
        result_message: str,
        idempotency_key: str,
        owner_membership: str,
        agent_membership: str,
        progress_message: str,
    ) -> None:
        """
        Store deterministic identifiers for one live test scenario.
        """

        self.job = job
        self.task = task
        self.agent = agent
        self.title = title
        self.thread = thread
        self.script = script
        self.policy = policy
        self.message = message
        self.context = context
        self.artifact = artifact
        self.operator = operator
        self.execution = execution
        self.policy_name = policy_name
        self.result_message = result_message
        self.idempotency_key = idempotency_key
        self.owner_membership = owner_membership
        self.agent_membership = agent_membership
        self.progress_message = progress_message

    @classmethod
    def build(cls) -> LiveConversationIds:
        """
        Build opaque UUID identifiers for every stored aggregate.
        """

        return cls(
            job=str(uuid4()),
            task=str(uuid4()),
            agent=str(uuid4()),
            thread=str(uuid4()),
            script=str(uuid4()),
            policy=str(uuid4()),
            message=str(uuid4()),
            operator=str(uuid4()),
            context=str(uuid4()),
            artifact=str(uuid4()),
            execution=str(uuid4()),
            result_message=str(uuid4()),
            idempotency_key=str(uuid4()),
            owner_membership=str(uuid4()),
            agent_membership=str(uuid4()),
            progress_message=str(uuid4()),
            policy_name=f"live-default-{uuid4()}",
            title=f"Live Postgres Conversation {uuid4()}",
        )
