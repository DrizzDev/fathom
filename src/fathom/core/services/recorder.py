from __future__ import annotations

import json
from datetime import timedelta
from hashlib import sha256
from logging import getLogger
from typing import Awaitable, Callable, Dict, Optional, TypeVar

from pydantic import JsonValue

from fathom.constants.collaboration import (
    Audience,
    ContextPurpose,
    IdempotencyState,
    JobCode,
    JobKind,
    JobState,
    MembershipRole,
    MessageKind,
    PolicyScope,
    TaskKind,
    TaskState,
)
from fathom.constants.conversation import (
    RECORDER_BUILDER,
    REQUEST_EXPIRY_DAYS,
    EntryKind,
)
from fathom.constants.events import FathomEvent
from fathom.conversation.identity import InteractionIdentity
from fathom.core.exceptions import InteractionError, ThreadConflictError, ThreadNotFoundError
from fathom.core.services.conversation import ConversationService
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.conversation import (
    AddActor,
    ArtifactAttach,
    ContextRecord,
    ConversationThreadQuery,
    EntryView,
    JoinMember,
    MessageAppend,
    ScriptSave,
    TaskFinish,
    TaskNodeView,
    TaskStart,
    ThreadCreate,
)
from fathom.schemas.interaction import (
    BeginRequest,
    ClaimJob,
    FinishJob,
    FinishRequest,
    Governance,
    Identity,
    Metadata,
    Outcome,
    PolicyQuery,
    SavePolicy,
    ScheduleJob,
    Script,
)
from fathom.schemas.recording import (
    Analysis,
    Answer,
    Completion,
    Handle,
    Output,
    Question,
    Run,
    ScriptOutput,
    Step,
    StepCompletion,
    TelemetryEnvelope,
)

T = TypeVar("T")


class RecorderHealth:
    """
    Tracks recorder activity so the first durable-write failure suppresses
    subsequent writes for the lifetime of the run.

    The recorder is best-effort by design: a broken interaction port should
    never crash a Fathom run. After the first InteractionError, the recorder
    no-ops and emits one error event so the host can surface the outage.
    """

    def __init__(self) -> None:
        """
        Start in active state with no recorded failures.
        """

        self.__active = True
        self.__failure_count = 0

    def is_active(self) -> bool:
        """
        Return True while the recorder is accepting durable writes.
        """

        return self.__active

    def record_failure(self) -> bool:
        """
        Mark a failure.
        Return True for the first failure (caller emits one error event); False for subsequent failures (silent suppression).
        """

        first = self.__active

        self.__active = False
        self.__failure_count += 1

        return first

    def reset(self) -> None:
        """
        Reset health for a new run; called by callers that own the lifetime.
        """

        self.__active = True
        self.__failure_count = 0

    @property
    def failure_count(self) -> int:
        """
        Total failures observed since the last reset.
        """

        return self.__failure_count


class ConversationRecorder:
    """
    Host-neutral recorder that translates runtime facts into conversation
    use cases. Every durable write is paired with a structured telemetry event so live clients can render activity without polling.
    """

    def __init__(
        self,
        *,
        telemetry: TelemetryPort,
        conversation: ConversationService,
    ) -> None:
        """
        Initialize the recorder with a conversation service and telemetry port.
        """

        self.__logger = getLogger(".".join((__name__, self.__class__.__name__)))

        self.__telemetry = telemetry
        self.__conversation = conversation

        self.__health = RecorderHealth()

    @property
    def health(self) -> RecorderHealth:
        """
        Expose recorder health for inspection by the runtime/tests.
        """

        return self.__health

    async def record_run_started(self, *, run: Run) -> Optional[Handle]:
        """
        Record the durable conversation records for a starting run.
        """

        return await self.__guard(
            failure_event="conversation.run.started",
            do=lambda: self.__do_record_run_started(run=run),
            envelope_for=lambda _handle: self.__envelope_run_started(run=run),
        )

    async def record_run_finished(self, *, completion: Completion) -> Optional[EntryView]:
        """
        Record the terminal message and task state for a finished run.
        """

        return await self.__guard(
            failure_event="conversation.run.finished",
            do=lambda: self.__do_record_run_finished(completion=completion),
            envelope_for=lambda _entry: self.__envelope_run_finished(completion=completion),
        )

    async def record_run_failed(self, *, completion: Completion) -> Optional[EntryView]:
        """
        Record the terminal message and task state for a failed run.
        """

        if completion.success:
            raise InteractionError("Failed run recording requires an unsuccessful completion.")

        return await self.__guard(
            failure_event="conversation.run.failed",
            do=lambda: self.__do_record_run_finished(completion=completion),
            envelope_for=lambda _entry: self.__envelope_run_failed(completion=completion),
        )

    async def record_step_started(self, *, step: Step) -> Optional[TaskNodeView]:
        """
        Record a graph step, agent task, tool task, or sub-agent task start.
        """

        return await self.__guard(
            failure_event="conversation.step.started",
            do=lambda: self.__do_record_step_started(step=step),
            envelope_for=lambda _node: self.__envelope_step_started(step=step),
        )

    async def record_step_finished(self, *, completion: StepCompletion) -> Optional[TaskNodeView]:
        """
        Record a graph step, agent task, tool task, or sub-agent task completion.
        """

        return await self.__guard(
            failure_event="conversation.step.finished",
            do=lambda: self.__do_record_step_finished(completion=completion),
            envelope_for=lambda _node: self.__envelope_step_finished(completion=completion),
        )

    async def record_subtask_started(self, *, step: Step) -> Optional[TaskNodeView]:
        """
        Record a sub-agent or delegated task start.
        """

        delegated = step.model_copy(update={"kind": TaskKind.DELEGATION})

        return await self.__guard(
            failure_event="conversation.subtask.started",
            do=lambda: self.__do_record_step_started(step=delegated),
            envelope_for=lambda _node: self.__envelope_subtask_started(step=delegated),
        )

    async def record_subtask_finished(
        self, *, completion: StepCompletion
    ) -> Optional[TaskNodeView]:
        """
        Record a sub-agent or delegated task completion.
        """

        return await self.__guard(
            failure_event="conversation.subtask.finished",
            do=lambda: self.__do_record_step_finished(completion=completion),
            envelope_for=lambda _node: self.__envelope_subtask_finished(completion=completion),
        )

    async def record_llm_analysis(self, *, analysis: Analysis) -> Optional[EntryView]:
        """
        Record an auditable model analysis summary without raw prompt payloads.
        """

        return await self.__guard(
            failure_event="conversation.analysis.recorded",
            do=lambda: self.__do_record_llm_analysis(analysis=analysis),
            envelope_for=lambda _entry: self.__envelope_llm_analysis(analysis=analysis),
        )

    async def record_hitl_question(self, *, question: Question) -> Optional[EntryView]:
        """
        Record a human-in-the-loop question as a conversation message.
        """

        return await self.__guard(
            failure_event="conversation.hitl.question",
            do=lambda: self.__do_record_hitl_question(question=question),
            envelope_for=lambda _entry: self.__envelope_hitl_question(question=question),
        )

    async def record_hitl_answer(self, *, answer: Answer) -> Optional[EntryView]:
        """
        Record a human-in-the-loop answer as a conversation message.
        """

        return await self.__guard(
            failure_event="conversation.hitl.answer",
            do=lambda: self.__do_record_hitl_answer(answer=answer),
            envelope_for=lambda _entry: self.__envelope_hitl_answer(answer=answer),
        )

    async def record_artifact(self, *, output: Output) -> Optional[EntryView]:
        """
        Record an artifact reference produced by runtime execution.
        """

        return await self.__guard(
            failure_event="conversation.artifact.linked",
            do=lambda: self.__do_record_artifact(output=output),
            envelope_for=lambda _entry: self.__envelope_artifact(output=output),
        )

    async def record_script(self, *, output: ScriptOutput) -> Optional[Script]:
        """
        Record reusable script content produced by runtime execution.
        """

        return await self.__guard(
            failure_event="conversation.script.saved",
            do=lambda: self.__do_record_script(output=output),
            envelope_for=lambda script: self.__envelope_script(output=output, script=script),
        )

    async def __guard(
        self,
        *,
        failure_event: str,
        do: Callable[[], Awaitable[T]],
        envelope_for: Callable[[T], TelemetryEnvelope],
    ) -> Optional[T]:
        """
        Run a recorder operation, suppressing further writes after the first
        durable-write failure and emitting one structured telemetry event per successful write.

        The recorder is best-effort: a broken interaction port (locked DB, filesystem error, integrity violation, network blip on a remote adapter)
        must never crash a Fathom run. We catch every Exception from the interaction port. InteractionError preserves the typed message;
        any other exception is wrapped into a generic disabled notice so the host can alert without losing the run.
        """

        if not self.__health.is_active():
            return None

        try:
            result = await do()
        except InteractionError as exception:
            await self.__handle_failure(exception=exception, operation=failure_event)
            return None

        except Exception as exception:
            await self.__handle_unexpected_failure(exception=exception, operation=failure_event)
            return None

        try:
            envelope = envelope_for(result)
            await self.__telemetry.info(
                "Conversation recorder write succeeded",
                **envelope.as_kwargs(),
            )
        except Exception:
            self.__logger.exception(
                "Conversation recorder success telemetry failed",
                extra={"operation": failure_event},
            )

        return result

    async def __handle_failure(self, *, exception: InteractionError, operation: str) -> None:
        """
        Record one InteractionError failure with a full traceback and emit a single typed error event on first failure for the live UI.
        """

        self.__logger.exception(
            "Conversation recorder write failed (interaction error)",
            extra={"operation": operation, "error_type": type(exception).__name__},
            exc_info=exception,
        )

        first = self.__health.record_failure()
        if not first:
            return

        try:
            await self.__telemetry.error(
                "Conversation recorder disabled after failure",
                type=FathomEvent.RECORDER_DISABLED,
                operation=operation,
                error=str(exception),
                error_type=type(exception).__name__,
            )
        except Exception:
            self.__logger.exception(
                "Conversation recorder failure-notice telemetry failed",
                extra={"operation": operation},
            )

    async def __handle_unexpected_failure(self, *, exception: Exception, operation: str) -> None:
        """
        Record one unexpected failure with a full traceback and emit a single typed error event on first failure for the live UI.
        """

        self.__logger.exception(
            "Conversation recorder write failed (unexpected exception)",
            extra={"operation": operation, "error_type": type(exception).__name__},
            exc_info=exception,
        )

        first = self.__health.record_failure()
        if not first:
            return

        try:
            await self.__telemetry.error(
                "Conversation recorder disabled after unexpected storage failure",
                type=FathomEvent.RECORDER_DISABLED,
                operation=operation,
                error=str(exception),
                error_type=type(exception).__name__,
            )
        except Exception:
            self.__logger.exception(
                "Conversation recorder failure-notice telemetry failed",
                extra={"operation": operation},
            )

    async def __do_record_run_started(self, *, run: Run) -> Handle:
        """
        Persist the durable records for a starting run.
        """

        async with self.__conversation.atomic():
            await self.__ensure_requester(run=run)
            await self.__ensure_responder(run=run)

            await self.__record_default_policy(run=run)
            await self.__record_request_started(run=run)

            await self.__conversation.start(
                request=TaskStart(
                    id=run.task,
                    tenant=run.tenant,
                    thread=run.thread,
                    kind=TaskKind.FATHOM,
                    created=run.created,
                    objective=run.intent,
                    reference=run.package,
                    workspace=run.workspace,
                    creator=run.requester.id,
                    assignee=run.responder.id,
                    metadata={
                        **run.metadata,
                        "workflow": run.workflow,
                    },
                )
            )
            await self.__conversation.append(
                request=MessageAppend(
                    id=run.request,
                    task=run.task,
                    tenant=run.tenant,
                    thread=run.thread,
                    workspace=run.workspace,
                    author=run.requester.id,
                    kind=MessageKind.REQUEST,
                    audience=Audience.THREAD,
                    body={
                        "intent": run.intent,
                        "package": run.package,
                        "workflow": run.workflow,
                        "starting_package": run.metadata.get("starting_package"),
                    },
                    created=run.created,
                    metadata=run.metadata,
                )
            )
            await self.__conversation.record(
                request=ContextRecord(
                    id=run.context,
                    task=run.task,
                    tenant=run.tenant,
                    thread=run.thread,
                    created=run.created,
                    metadata=run.metadata,
                    messages=(run.request,),
                    workspace=run.workspace,
                    builder=RECORDER_BUILDER,
                    consumer=run.responder.id,
                    model=run.responder.model,
                    provider=run.responder.provider,
                    purpose=ContextPurpose.EXECUTION,
                )
            )
            await self.__record_job_started(run=run)

        return Handle(
            task=run.task,
            tenant=run.tenant,
            thread=run.thread,
            request=run.request,
            context=run.context,
            workflow=run.workflow,
            workspace=run.workspace,
            requester=run.requester.id,
            responder=run.responder.id,
        )

    async def __record_default_policy(self, *, run: Run) -> None:
        """
        Ensure a tenant/workspace policy row exists for this entrypoint.
        """

        existing = await self.__conversation.get_policy(
            query=PolicyQuery(name="default", tenant=run.tenant, workspace=run.workspace)
        )
        if existing is not None:
            return

        await self.__conversation.save_policy(
            request=SavePolicy(
                identity=Identity(
                    tenant=run.tenant,
                    workspace=run.workspace,
                    id=self.__policy_id(workspace=run.workspace),
                ),
                name="default",
                created_at=run.created,
                governance=Governance(),
                scope=PolicyScope.WORKSPACE if run.workspace else PolicyScope.TENANT,
                metadata=Metadata(
                    entries={
                        "entrypoint": "fathom.run",
                        "managed_by": RECORDER_BUILDER,
                    }
                ),
            )
        )

    async def __record_request_started(self, *, run: Run) -> None:
        """
        Persist a run idempotency/audit request row.
        """

        await self.__conversation.begin_request(
            request=BeginRequest(
                tenant=run.tenant,
                created_at=run.created,
                hash=self.__request_hash(run=run),
                key=self.__request_key(workflow=run.workflow),
                expires_at=run.created + timedelta(days=REQUEST_EXPIRY_DAYS),
                metadata=Metadata(
                    entries={
                        "thread": run.thread,
                        "workflow": run.workflow,
                        "entrypoint": "fathom.run",
                    }
                ),
            )
        )

    async def __record_job_started(self, *, run: Run) -> None:
        """
        Persist and claim a run-scoped job for CLI/direct execution parity.
        """

        job = await self.__conversation.schedule_job(
            request=ScheduleJob(
                identity=Identity(
                    tenant=run.tenant,
                    workspace=run.workspace,
                    id=self.__job_id(workflow=run.workflow),
                ),
                task=run.task,
                thread=run.thread,
                kind=JobKind.EXECUTION,
                available_at=run.created,
                payload=Metadata(
                    entries={
                        "intent": run.intent,
                        "package": run.package,
                        "workflow": run.workflow,
                    }
                ),
                created_at=run.created,
                metadata=Metadata(entries={"entrypoint": "fathom.run"}),
            )
        )
        if job.state != JobState.PENDING:
            return

        await self.__conversation.claim_job(
            request=ClaimJob(
                tenant=run.tenant,
                claimed=run.created,
                owner=run.responder.id,
                job=self.__job_id(workflow=run.workflow),
            )
        )

    async def __ensure_requester(self, *, run: Run) -> None:
        """
        Create or join the requesting actor for the run thread.

        Two concurrent runs starting in the same conversation can race on
        thread creation: both observe "no thread", both call create_thread,
        and the second collides on the threads PRIMARY KEY. We catch only
        the typed ThreadConflictError (one specific case from create_thread)
        and fall through to the existing-thread path. Any other failure
        from get_thread or create_thread propagates so the recorder's failure-suppression layer can disable cleanly.
        """

        thread_existed = await self.__thread_exists(run=run)

        if not thread_existed:
            try:
                await self.__conversation.create(
                    request=ThreadCreate(
                        id=run.thread,
                        tenant=run.tenant,
                        title=run.intent,
                        created=run.created,
                        metadata=run.metadata,
                        creator=run.requester,
                        workspace=run.workspace,
                        member=run.members.requester,
                        role=MembershipRole.REQUESTER,
                    )
                )
                return
            except ThreadConflictError as exception:
                self.__logger.exception(f"Got {exception}", stack_info=True)
                # A concurrent racer created the thread between our check and our create.
                # Fall through to the existing-thread path and join as a member.

        await self.__conversation.actor(
            request=AddActor(
                tenant=run.tenant,
                id=run.requester.id,
                created=run.created,
                kind=run.requester.kind,
                name=run.requester.name,
                model=run.requester.model,
                provider=run.requester.provider,
                workspace=run.requester.workspace or run.workspace,
            )
        )
        await self.__conversation.join(
            request=JoinMember(
                tenant=run.tenant,
                joined=run.created,
                thread=run.thread,
                actor=run.requester.id,
                workspace=run.workspace,
                id=run.members.requester,
                role=MembershipRole.REQUESTER,
            )
        )

    async def __thread_exists(self, *, run: Run) -> bool:
        """
        Probe whether the run's thread already exists. Returns False on the
        typed ThreadNotFoundError; any other failure propagates so the recorder's failure-suppression layer can disable cleanly.
        """

        try:
            await self.__conversation.get(
                query=ConversationThreadQuery(tenant=run.tenant, thread=run.thread)
            )
        except ThreadNotFoundError:
            return False

        return True

    async def __ensure_responder(self, *, run: Run) -> None:
        """
        Create and join the responding actor for the run thread.
        """

        await self.__conversation.actor(
            request=AddActor(
                tenant=run.tenant,
                id=run.responder.id,
                created=run.created,
                kind=run.responder.kind,
                name=run.responder.name,
                model=run.responder.model,
                provider=run.responder.provider,
                workspace=run.responder.workspace or run.workspace,
            )
        )
        await self.__conversation.join(
            request=JoinMember(
                tenant=run.tenant,
                thread=run.thread,
                joined=run.created,
                actor=run.responder.id,
                workspace=run.workspace,
                id=run.members.responder,
                role=MembershipRole.RESPONDER,
            )
        )

    async def __do_record_run_finished(self, *, completion: Completion) -> EntryView:
        """
        Persist the terminal records for a finished or failed run.
        """

        async with self.__conversation.atomic():
            entry = await self.__conversation.append(
                request=MessageAppend(
                    id=completion.result,
                    kind=MessageKind.RESULT,
                    audience=Audience.THREAD,
                    task=completion.handle.task,
                    tenant=completion.handle.tenant,
                    thread=completion.handle.thread,
                    author=completion.handle.responder,
                    workspace=completion.handle.workspace,
                    body=self.__result_body(completion=completion),
                    created=completion.finished,
                    metadata=completion.metadata,
                )
            )
            await self.__conversation.finish(
                request=TaskFinish(
                    code=completion.code,
                    detail=completion.reason,
                    summary=completion.reason,
                    ended=completion.finished,
                    elapsed=completion.elapsed,
                    task=completion.handle.task,
                    tenant=completion.handle.tenant,
                    state=self.__run_state(completion=completion),
                )
            )
            await self.__record_job_finished(completion=completion)
            await self.__record_request_finished(completion=completion)

        return entry

    async def __record_job_finished(self, *, completion: Completion) -> None:
        """
        Finish the durable run job created for this workflow.
        """

        await self.__conversation.finish_job(
            request=FinishJob(
                finished=completion.finished,
                tenant=completion.handle.tenant,
                owner=completion.handle.responder,
                job=self.__job_id(workflow=completion.handle.workflow),
                state=JobState.COMPLETED if completion.success else JobState.FAILED,
                outcome=Outcome(
                    code=JobCode.COMPLETED if completion.success else JobCode.UNKNOWN_ERROR,
                    detail=completion.reason,
                ),
            )
        )

    async def __record_request_finished(self, *, completion: Completion) -> None:
        """
        Finish the run idempotency/audit row.
        """

        await self.__conversation.finish_request(
            request=FinishRequest(
                tenant=completion.handle.tenant,
                key=self.__request_key(workflow=completion.handle.workflow),
                state=(
                    IdempotencyState.COMPLETED if completion.success else IdempotencyState.FAILED
                ),
                response={
                    "result": completion.result,
                    "status": completion.status,
                    "success": completion.success,
                    "task": completion.handle.task,
                    "workflow": completion.handle.workflow,
                },
                finished=completion.finished,
            )
        )

    async def __do_record_step_started(self, *, step: Step) -> TaskNodeView:
        """
        Persist the durable records for a step or sub-task start.
        """

        return await self.__conversation.start(
            request=TaskStart(
                id=step.id,
                root=step.root,
                kind=step.kind,
                parent=step.parent,
                origin=step.origin,
                tenant=step.tenant,
                thread=step.thread,
                creator=step.actor,
                assignee=step.actor,
                created=step.created,
                metadata=step.metadata,
                workspace=step.workspace,
                objective=step.objective,
                reference=step.reference,
            )
        )

    async def __do_record_step_finished(self, *, completion: StepCompletion) -> TaskNodeView:
        """
        Persist the terminal records for a step or sub-task finish.
        """

        return await self.__conversation.finish(
            request=TaskFinish(
                task=completion.task,
                code=completion.code,
                state=completion.state,
                tenant=completion.tenant,
                detail=completion.reason,
                ended=completion.finished,
                summary=completion.summary,
                elapsed=completion.elapsed,
            )
        )

    async def __do_record_llm_analysis(self, *, analysis: Analysis) -> EntryView:
        """
        Persist a per-step planning record as a user-visible progress message.
        """

        return await self.__conversation.append(
            request=MessageAppend(
                id=analysis.id,
                task=analysis.task,
                author=analysis.actor,
                thread=analysis.thread,
                tenant=analysis.tenant,
                labels=analysis.labels,
                created=analysis.created,
                audience=Audience.THREAD,
                kind=MessageKind.PROGRESS,
                metadata=analysis.metadata,
                workspace=analysis.workspace,
                body=self.__progress_body(analysis=analysis),
            )
        )

    @staticmethod
    def __progress_body(*, analysis: Analysis) -> Dict[str, JsonValue]:
        """
        Build the JSON body for a per-step planning message.
        """

        return {
            "step": analysis.step,
            "action": analysis.action,
            "target": analysis.target,
            "summary": analysis.summary,
            "evidence": analysis.evidence,
            "confidence": analysis.confidence,
        }

    @classmethod
    def __result_body(cls, *, completion: Completion) -> Dict[str, JsonValue]:
        """
        Build the JSON body for a terminal run-result message.
        """

        summary = completion.summary or completion.reason

        return {
            "reason": summary,
            "summary": summary,
            "error": completion.error,
            "steps": completion.steps,
            "detail": completion.detail,
            "status": completion.status,
            "success": completion.success,
        }

    async def __do_record_hitl_question(self, *, question: Question) -> EntryView:
        """
        Persist a HITL question as a conversation message.
        """

        return await self.__conversation.append(
            request=MessageAppend(
                id=question.id,
                task=question.task,
                body=question.body,
                author=question.actor,
                tenant=question.tenant,
                thread=question.thread,
                audience=Audience.THREAD,
                kind=MessageKind.QUESTION,
                created=question.created,
                metadata=question.metadata,
                workspace=question.workspace,
            )
        )

    async def __do_record_hitl_answer(self, *, answer: Answer) -> EntryView:
        """
        Persist a HITL answer as a conversation message replying to the question.
        """

        return await self.__conversation.append(
            request=MessageAppend(
                id=answer.id,
                task=answer.task,
                body=answer.body,
                author=answer.actor,
                tenant=answer.tenant,
                thread=answer.thread,
                reply=answer.question,
                created=answer.created,
                kind=MessageKind.ANSWER,
                audience=Audience.THREAD,
                metadata=answer.metadata,
                workspace=answer.workspace,
            )
        )

    async def __do_record_artifact(self, *, output: Output) -> EntryView:
        """
        Persist an artifact reference and return its renderable entry.
        """

        return await self.__conversation.attach(
            request=ArtifactAttach(
                id=output.id,
                uri=output.uri,
                task=output.task,
                mime=output.mime,
                kind=output.kind,
                size=output.size,
                tenant=output.tenant,
                thread=output.thread,
                labels=output.labels,
                producer=output.actor,
                backend=output.backend,
                created=output.created,
                metadata=output.metadata,
                retention=output.retention,
                workspace=output.workspace,
            )
        )

    async def __do_record_script(self, *, output: ScriptOutput) -> Script:
        """
        Persist reusable script content and version audit metadata.
        """

        return await self.__conversation.save(
            request=ScriptSave(
                id=output.id,
                task=output.task,
                actor=output.actor,
                title=output.title,
                thread=output.thread,
                tenant=output.tenant,
                source=output.source,
                format=output.format,
                status=output.status,
                content=output.content,
                summary=output.summary,
                created=output.created,
                artifact=output.artifact,
                metadata=output.metadata,
                workspace=output.workspace,
            )
        )

    def __run_state(self, *, completion: Completion) -> TaskState:
        """
        Resolve the terminal task state for a run completion.
        """

        return TaskState.SUCCEEDED if completion.success else TaskState.FAILED

    def __policy_id(self, *, workspace: Optional[str]) -> str:
        """
        Return the deterministic policy id for a run entrypoint.
        """

        return InteractionIdentity.stable(scope="policy.default", parts=(workspace or "tenant",))

    def __request_key(self, *, workflow: str) -> str:
        """
        Return the deterministic idempotency key for a run workflow.
        """

        return InteractionIdentity.stable(scope="request.run", parts=(workflow,))

    def __job_id(self, *, workflow: str) -> str:
        """
        Return the deterministic durable job id for a run workflow.
        """

        return InteractionIdentity.stable(scope="job.execution", parts=(workflow,))

    def __request_hash(self, *, run: Run) -> str:
        """
        Hash the stable run request payload stored in the idempotency row.
        """

        payload = json.dumps(
            {
                "task": run.task,
                "thread": run.thread,
                "intent": run.intent,
                "package": run.package,
                "request": run.request,
                "workflow": run.workflow,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()

    def __envelope_run_started(self, *, run: Run) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for a run-started event.
        """

        return TelemetryEnvelope(
            task_id=run.task,
            tenant=run.tenant,
            kind=EntryKind.EVENT,
            workflow_id=run.workflow,
            conversation_id=run.thread,
            type="conversation.run.started",
            payload={"intent": run.intent, "package": run.package},
        )

    def __envelope_run_finished(self, *, completion: Completion) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for a successful run-finished event.
        """

        return TelemetryEnvelope(
            kind=EntryKind.MESSAGE,
            task_id=completion.handle.task,
            tenant=completion.handle.tenant,
            type="conversation.run.finished",
            workflow_id=completion.handle.workflow,
            conversation_id=completion.handle.thread,
            payload={
                "steps": completion.steps,
                "status": completion.status,
                "reason": completion.reason,
                "success": completion.success,
            },
        )

    def __envelope_run_failed(self, *, completion: Completion) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for a failed run-finished event.
        """

        return TelemetryEnvelope(
            kind=EntryKind.MESSAGE,
            task_id=completion.handle.task,
            type="conversation.run.failed",
            tenant=completion.handle.tenant,
            workflow_id=completion.handle.workflow,
            conversation_id=completion.handle.thread,
            payload={
                "error": completion.error,
                "steps": completion.steps,
                "status": completion.status,
                "reason": completion.reason,
            },
        )

    def __envelope_script(self, *, output: ScriptOutput, script: Script) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for a saved script.
        """

        _ = script

        return TelemetryEnvelope(
            task_id=output.task,
            kind=EntryKind.EVENT,
            tenant=output.tenant,
            workflow_id=output.workflow,
            conversation_id=output.thread,
            type="conversation.script.saved",
            payload={
                "script": output.id,
                "format": output.format,
                "artifact": output.artifact,
                "source": output.source.value,
            },
        )

    def __envelope_step_started(self, *, step: Step) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for a step-started event.
        """

        return TelemetryEnvelope(
            task_id=step.id,
            tenant=step.tenant,
            kind=EntryKind.EVENT,
            workflow_id=step.workflow,
            conversation_id=step.thread,
            type="conversation.step.started",
            payload={"objective": step.objective, "kind": step.kind.value},
        )

    def __envelope_step_finished(self, *, completion: StepCompletion) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for a step-finished event.
        """

        return TelemetryEnvelope(
            kind=EntryKind.EVENT,
            task_id=completion.task,
            tenant=completion.tenant,
            workflow_id=completion.workflow,
            conversation_id=completion.thread,
            type="conversation.step.finished",
            payload={"state": completion.state.value, "code": completion.code.value},
        )

    def __envelope_subtask_started(self, *, step: Step) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for a subtask-started event.
        """

        envelope = self.__envelope_step_started(step=step)
        return envelope.model_copy(update={"type": "conversation.subtask.started"})

    def __envelope_subtask_finished(self, *, completion: StepCompletion) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for a subtask-finished event.
        """

        envelope = self.__envelope_step_finished(completion=completion)
        return envelope.model_copy(update={"type": "conversation.subtask.finished"})

    def __envelope_llm_analysis(self, *, analysis: Analysis) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for an llm-analysis event.
        """

        return TelemetryEnvelope(
            task_id=analysis.task,
            kind=EntryKind.MESSAGE,
            tenant=analysis.tenant,
            workflow_id=analysis.workflow,
            conversation_id=analysis.thread,
            type="conversation.analysis.recorded",
            payload={"summary": analysis.summary},
        )

    def __envelope_hitl_question(self, *, question: Question) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for an HITL question event.
        """

        return TelemetryEnvelope(
            task_id=question.task,
            kind=EntryKind.MESSAGE,
            tenant=question.tenant,
            workflow_id=question.workflow,
            conversation_id=question.thread,
            type="conversation.hitl.question",
            payload=dict(question.body) if isinstance(question.body, dict) else {},
        )

    def __envelope_hitl_answer(self, *, answer: Answer) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for an HITL answer event.
        """

        return TelemetryEnvelope(
            task_id=answer.task,
            tenant=answer.tenant,
            kind=EntryKind.MESSAGE,
            workflow_id=answer.workflow,
            conversation_id=answer.thread,
            type="conversation.hitl.answer",
            payload=dict(answer.body) if isinstance(answer.body, dict) else {},
        )

    def __envelope_artifact(self, *, output: Output) -> TelemetryEnvelope:
        """
        Build the telemetry envelope for an artifact-linked event.
        """

        return TelemetryEnvelope(
            task_id=output.task,
            tenant=output.tenant,
            kind=EntryKind.ARTIFACT,
            workflow_id=output.workflow,
            conversation_id=output.thread,
            type="conversation.artifact.linked",
            payload={
                "uri": output.uri,
                "mime": output.mime,
                "kind": output.kind.value,
            },
        )
