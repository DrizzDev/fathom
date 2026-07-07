from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Type
from uuid import uuid4

import asyncpg
import pytest

from fathom.adapters.interaction.orm.postgres import PostgresInteraction
from fathom.adapters.signing.noop import NoopSigner
from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    ContextPurpose,
    ScriptFormat,
    ScriptStatus,
    ScriptVersionSource,
    TaskCode,
    TaskKind,
    TaskState,
)
from fathom.constants.storage import PostgresMigrationMode
from fathom.conversation.identity import InteractionIdentity
from fathom.core.exceptions import InteractionError
from fathom.core.services.conversation import ConversationService, Ports
from fathom.core.services.recorder import ConversationRecorder
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import PostgresInteractionConfiguration
from fathom.schemas.conversation import ActorInput
from fathom.schemas.recording import (
    Analysis,
    Answer,
    Completion,
    ContextSnapshot,
    Handle,
    Output,
    Question,
    Run,
    ScriptOutput,
    Step,
    StepCompletion,
)

pytestmark = pytest.mark.asyncio


class SilentTelemetry(TelemetryPort):
    """
    Telemetry sink that discards every event for deterministic tests.
    """

    async def debug(self, message: str, **context: Any) -> None:
        """
        Discard a debug event.
        """

    async def info(self, message: str, **context: Any) -> None:
        """
        Discard an info event.
        """

    async def warning(self, message: str, **context: Any) -> None:
        """
        Discard a warning event.
        """

    async def error(self, message: str, **context: Any) -> None:
        """
        Discard an error event.
        """

    async def exception(
        self, message: str, *, exception: Optional[BaseException] = None, **context: Any
    ) -> None:
        """
        Discard an exception event.
        """


class BrokenExecutionService(ConversationService):
    """
    Conversation service whose execution write always fails at the boundary.
    """

    async def start_execution(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fail every execution write to simulate a broken conversation layer.
        """

        raise InteractionError("Simulated conversation-layer failure.")


class FullyBrokenService(ConversationService):
    """
    Conversation service whose every durable operation fails at the boundary.
    """

    def atomic(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fail the transactional boundary used by run lifecycle checkpoints.
        """

        raise InteractionError("Simulated conversation-layer failure.")

    async def start(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fail task-start checkpoints.
        """

        raise InteractionError("Simulated conversation-layer failure.")

    async def finish(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fail task-finish checkpoints.
        """

        raise InteractionError("Simulated conversation-layer failure.")

    async def append(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fail message-append checkpoints.
        """

        raise InteractionError("Simulated conversation-layer failure.")

    async def attach(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fail artifact-attach checkpoints.
        """

        raise InteractionError("Simulated conversation-layer failure.")

    async def save(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fail script-save checkpoints.
        """

        raise InteractionError("Simulated conversation-layer failure.")

    async def record(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fail context-record checkpoints.
        """

        raise InteractionError("Simulated conversation-layer failure.")


class RecorderResilienceHarness:
    """
    Owns one disposable migrated schema and a recorder wired to it.
    """

    def __init__(self, *, service_type: Type[ConversationService] = ConversationService) -> None:
        """
        Build an isolated schema, adapter, and recorder configuration.
        """

        self.__schema = f"live_recorder_{uuid4().hex}"
        self.__admin: Optional[asyncpg.Connection] = None

        self.store = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                pool_min_size=1,
                pool_max_size=8,
                schema_name=self.__schema,
                dsn="postgresql://localhost/postgres",
                migration_mode=PostgresMigrationMode.APPLY,
            )
        )
        self.service = service_type(signer=NoopSigner(), ports=Ports(interaction=self.store))

    async def __aenter__(self) -> RecorderResilienceHarness:
        """
        Initialize the adapter and admin connection or skip when Postgres is unavailable.
        """

        try:
            self.__admin = await asyncpg.connect(database="postgres")
            await self.store.initialize()
        except (OSError, asyncpg.PostgresError) as exception:
            await self.__close()
            pytest.skip(f"Local Postgres unavailable: {exception}")

        return self

    async def __aexit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        """
        Release adapter resources and drop the disposable schema.
        """

        _ = exception_type, exception, traceback
        await self.__close()

    async def __close(self) -> None:
        """
        Close the adapter pool and remove the test schema.
        """

        await self.store.aclose()

        if self.__admin is None:
            return

        await self.__admin.execute(f'DROP SCHEMA IF EXISTS "{self.__schema}" CASCADE')
        await self.__admin.close()
        self.__admin = None

    def recorder(self) -> ConversationRecorder:
        """
        Build a fresh recorder, mirroring the per-run reset the runtime performs.
        """

        return ConversationRecorder(telemetry=SilentTelemetry(), conversation=self.service)

    async def actor_tenants(self, *, actor: str) -> List[str]:
        """
        Return the tenants that own one actor id, ordered for stable comparison.
        """

        assert self.__admin is not None
        await self.__admin.execute(f'SET search_path TO "{self.__schema}"')
        rows = await self.__admin.fetch(
            "SELECT tenant_id FROM actors WHERE id = $1 ORDER BY tenant_id", actor
        )
        return [row["tenant_id"] for row in rows]

    async def executions_for(self, *, workflow: str) -> List[str]:
        """
        Return the execution ids recorded for one workflow id.
        """

        assert self.__admin is not None
        await self.__admin.execute(f'SET search_path TO "{self.__schema}"')
        rows = await self.__admin.fetch(
            "SELECT id FROM executions WHERE workflow_id = $1 ORDER BY id", workflow
        )
        return [row["id"] for row in rows]

    async def primary_key(self, *, table: str) -> str:
        """
        Return the primary-key definition for one table in the test schema.
        """

        assert self.__admin is not None
        definition = await self.__admin.fetchval(
            "SELECT pg_get_constraintdef(constraint_record.oid) "
            "FROM pg_constraint constraint_record "
            "JOIN pg_class table_record ON table_record.oid = constraint_record.conrelid "
            "JOIN pg_namespace namespace_record "
            "  ON namespace_record.oid = table_record.relnamespace "
            "WHERE namespace_record.nspname = $1 "
            "  AND table_record.relname = $2 "
            "  AND constraint_record.contype = 'p'",
            self.__schema,
            table,
        )
        return definition or ""


class RunFactory:
    """
    Builds recorder run payloads for multi-tenant resilience tests.
    """

    @staticmethod
    def build(
        *,
        tenant: str,
        operator: str,
        workflow: Optional[str] = None,
        thread: Optional[str] = None,
        created: Optional[datetime] = None,
    ) -> Run:
        """
        Build one run payload for a tenant operator, mirroring the runtime shape.
        """

        return Run(
            tenant=tenant,
            workspace=None,
            thread=thread or str(uuid4()),
            created=created or datetime(2026, 7, 1, tzinfo=timezone.utc),
            workflow=(identifier := workflow or str(uuid4())),
            execution=InteractionIdentity.stable(scope="execution", parts=(identifier,)),
            intent="open the app and sign in",
            package=None,
            metadata={"starting_package": "com.sec.android.app.launcher"},
            requester=ActorInput(id=operator, name=operator, kind=ActorKind.HUMAN),
            responder=ActorInput(
                id="agent:fathom",
                name="agent:fathom",
                kind=ActorKind.AGENT,
                model="gemini-3-flash-preview",
                provider="gemini",
            ),
        )


class TestRecorderMultiTenantResilience:
    """
    Prove the recorder serves any tenant concurrently without cross-tenant collisions.
    """

    __TENANTS: Tuple[Tuple[str, str], ...] = (
        ("1", "alice@drizz.dev"),
        ("343", "avinash@salaryse.com"),
        ("500", "bob@othercorp.com"),
        ("900", "carol@newco.com"),
        ("1201", "dan@scaleup.io"),
        ("7788", "erin@fintech.co"),
    )

    async def __record(self, *, harness: RecorderResilienceHarness, tenant: str, operator: str):
        """
        Record one run through a fresh recorder, as the runtime does per run.
        """

        recorder = harness.recorder()
        return await recorder.record_run_started(
            run=RunFactory.build(tenant=tenant, operator=operator)
        )

    async def test_first_runs_for_many_tenants_succeed_concurrently(self) -> None:
        """
        Simultaneous first runs across tenants each reserve a distinct execution identity.
        """

        async with RecorderResilienceHarness() as harness:
            handles = await asyncio.gather(
                *(
                    self.__record(harness=harness, tenant=tenant, operator=operator)
                    for tenant, operator in self.__TENANTS
                )
            )

            assert all(isinstance(handle, Handle) for handle in handles)
            executions = [handle.execution for handle in handles]
            assert len(set(executions)) == len(self.__TENANTS)

            owners = await harness.actor_tenants(actor="agent:fathom")
            assert owners == sorted(tenant for tenant, _ in self.__TENANTS)

    async def test_repeated_concurrent_bursts_stay_collision_free(self) -> None:
        """
        Repeated bursts of concurrent runs never raise on the shared agent identity.
        """

        async with RecorderResilienceHarness() as harness:
            for _ in range(3):
                handles = await asyncio.gather(
                    *(
                        self.__record(harness=harness, tenant=tenant, operator=operator)
                        for tenant, operator in self.__TENANTS
                    )
                )
                assert all(isinstance(handle, Handle) for handle in handles)

            owners = await harness.actor_tenants(actor="agent:fathom")
            assert owners == sorted(tenant for tenant, _ in self.__TENANTS)

    async def test_actor_and_policy_keys_are_tenant_scoped(self) -> None:
        """
        The migrated schema keys actors and policies by tenant, not by shared id.
        """

        async with RecorderResilienceHarness() as harness:
            actors_key = await harness.primary_key(table="actors")
            policies_key = await harness.primary_key(table="policies")

            assert actors_key == "PRIMARY KEY (tenant_id, id)"
            assert policies_key == "PRIMARY KEY (tenant_id, id)"


class TestRecorderRetryReplay:
    """
    Prove Temporal-style retries replay onto the same identity without duplicates.
    """

    async def test_same_workflow_replays_onto_one_execution(self) -> None:
        """
        An identical replay re-derives one execution id and never violates a unique key.
        """

        async with RecorderResilienceHarness() as harness:
            workflow = str(uuid4())
            thread = str(uuid4())
            started = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)

            def attempt() -> Run:
                return RunFactory.build(
                    tenant="343",
                    thread=thread,
                    created=started,
                    workflow=workflow,
                    operator="avinash@salaryse.com",
                )

            # Two independent runs with identical content, as a deterministic retry
            # replays: identity is re-derived from workflow_id on each attempt.
            first = await harness.recorder().record_run_started(run=attempt())
            second = await harness.recorder().record_run_started(run=attempt())

            assert isinstance(first, Handle)
            assert isinstance(second, Handle)
            assert first.execution == second.execution

            # No unique-constraint violation: exactly one execution row for the workflow.
            assert await harness.executions_for(workflow=workflow) == [first.execution]
            assert await harness.actor_tenants(actor="agent:fathom") == ["343"]


class TestRecorderBestEffort:
    """
    Prove a broken conversation layer never propagates out of the recorder.
    """

    async def test_recording_failure_is_swallowed(self) -> None:
        """
        A failing execution write disables recording and returns no handle, never raising.
        """

        async with RecorderResilienceHarness(service_type=BrokenExecutionService) as harness:
            recorder = harness.recorder()

            handle = await recorder.record_run_started(
                run=RunFactory.build(tenant="343", operator="avinash@salaryse.com")
            )

            assert handle is None
            assert recorder.health.is_active() is False


class Checkpoint:
    """
    Names one recorder checkpoint and invokes it against a recorder.
    """

    __TENANT = "343"
    __THREAD = "conversation-343"
    __WORKFLOW = "workflow-343"
    __EXECUTION = "execution-343"
    __TASK = "task-343"
    __ACTOR = "agent:fathom"
    __NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)

    @classmethod
    def __scope(cls) -> dict:
        """
        Return the shared conversation-scope fields for a checkpoint payload.
        """

        return {"tenant": cls.__TENANT, "thread": cls.__THREAD, "workflow": cls.__WORKFLOW}

    @classmethod
    def __handle(cls) -> Handle:
        """
        Return a run handle for run-completion checkpoints.
        """

        return Handle(
            task=cls.__TASK,
            request="message-343",
            context="context-343",
            tenant=cls.__TENANT,
            thread=cls.__THREAD,
            workspace=None,
            workflow=cls.__WORKFLOW,
            execution=cls.__EXECUTION,
            requester="avinash@salaryse.com",
            responder=cls.__ACTOR,
        )

    @classmethod
    def all(cls) -> Tuple[Tuple[str, Any], ...]:
        """
        Return every recorder checkpoint paired with a call against the given recorder.
        """

        scope = cls.__scope()
        handle = cls.__handle()
        completion = Completion(
            handle=handle,
            steps=1,
            result="message-343",
            status="completed",
            success=True,
            reason="done",
            code=TaskCode.COMPLETED,
            finished=cls.__NOW,
            elapsed=10,
        )
        failure = completion.model_copy(update={"success": False, "status": "failed"})
        step = Step(
            **scope,
            id=cls.__TASK,
            execution=cls.__EXECUTION,
            kind=TaskKind.AGENT,
            objective="open the app",
            created=cls.__NOW,
        )
        step_done = StepCompletion(
            **scope,
            task=cls.__TASK,
            state=TaskState.SUCCEEDED,
            code=TaskCode.COMPLETED,
            finished=cls.__NOW,
            elapsed=10,
        )
        return (
            ("record_run_started", lambda r: r.record_run_started(run=cls.__run())),
            ("record_run_finished", lambda r: r.record_run_finished(completion=completion)),
            ("record_run_failed", lambda r: r.record_run_failed(completion=failure)),
            ("record_step_started", lambda r: r.record_step_started(step=step)),
            ("record_step_finished", lambda r: r.record_step_finished(completion=step_done)),
            ("record_subtask_started", lambda r: r.record_subtask_started(step=step)),
            ("record_subtask_finished", lambda r: r.record_subtask_finished(completion=step_done)),
            ("record_llm_analysis", lambda r: r.record_llm_analysis(analysis=cls.__analysis())),
            ("record_hitl_question", lambda r: r.record_hitl_question(question=cls.__question())),
            ("record_hitl_answer", lambda r: r.record_hitl_answer(answer=cls.__answer())),
            ("record_artifact", lambda r: r.record_artifact(output=cls.__output())),
            ("record_script", lambda r: r.record_script(output=cls.__script())),
            ("record_context", lambda r: r.record_context(snapshot=cls.__context())),
        )

    @classmethod
    def __run(cls) -> Run:
        """
        Return a run payload for the run-started checkpoint.
        """

        return RunFactory.build(
            tenant=cls.__TENANT,
            thread=cls.__THREAD,
            workflow=cls.__WORKFLOW,
            operator="avinash@salaryse.com",
        )

    @classmethod
    def __analysis(cls) -> Analysis:
        """
        Return an analysis payload for the analysis checkpoint.
        """

        return Analysis(
            **cls.__scope(),
            id="message-analysis-343",
            actor=cls.__ACTOR,
            execution=cls.__EXECUTION,
            summary="tap the login button",
            step=1,
            created=cls.__NOW,
        )

    @classmethod
    def __question(cls) -> Question:
        """
        Return a question payload for the human-in-the-loop question checkpoint.
        """

        return Question(
            **cls.__scope(),
            id="message-question-343",
            actor=cls.__ACTOR,
            execution=cls.__EXECUTION,
            body={"text": "which account?"},
            created=cls.__NOW,
        )

    @classmethod
    def __answer(cls) -> Answer:
        """
        Return an answer payload for the human-in-the-loop answer checkpoint.
        """

        return Answer(
            **cls.__scope(),
            id="message-answer-343",
            actor="avinash@salaryse.com",
            execution=cls.__EXECUTION,
            body={"text": "the primary one"},
            question="message-question-343",
            created=cls.__NOW,
        )

    @classmethod
    def __output(cls) -> Output:
        """
        Return an artifact payload for the artifact checkpoint.
        """

        return Output(
            **cls.__scope(),
            id="artifact-343",
            execution=cls.__EXECUTION,
            kind=ArtifactKind.TRACE,
            uri="/tmp/trace-343.json",
            backend=ArtifactBackend.LOCAL,
            created=cls.__NOW,
        )

    @classmethod
    def __script(cls) -> ScriptOutput:
        """
        Return a script payload for the script checkpoint.
        """

        return ScriptOutput(
            **cls.__scope(),
            id="script-343",
            execution=cls.__EXECUTION,
            content="open app\ntap login",
            format=ScriptFormat.TEXT_PLAIN,
            status=ScriptStatus.ACTIVE,
            source=ScriptVersionSource.GENERATED,
            created=cls.__NOW,
        )

    @classmethod
    def __context(cls) -> ContextSnapshot:
        """
        Return a context payload for the context checkpoint.
        """

        return ContextSnapshot(
            **cls.__scope(),
            id="context-snapshot-343",
            actor=cls.__ACTOR,
            execution=cls.__EXECUTION,
            purpose=ContextPurpose.EXECUTION,
            created=cls.__NOW,
        )


class TestRecorderCheckpointBestEffort:
    """
    Prove a conversation failure at every checkpoint is contained, never propagated.
    """

    async def test_every_checkpoint_failure_is_swallowed(self) -> None:
        """
        Each recorder checkpoint returns no result and disables recording without raising.
        """

        async with RecorderResilienceHarness(service_type=FullyBrokenService) as harness:
            for name, invoke in Checkpoint.all():
                recorder = harness.recorder()

                result = await invoke(recorder)

                assert result is None, f"{name} propagated a value instead of degrading"
                assert recorder.health.is_active() is False, f"{name} did not disable recording"
