from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue

from fathom.constants.collaboration import (
    INTERACTION_BUILDER,
    ActorKind,
    Audience,
    ContextPurpose,
    JobKind,
    MembershipRole,
    MessageKind,
    TaskCode,
    TaskKind,
    TaskState,
)
from fathom.constants.state import CompletionReason
from fathom.conversation.classifier import PrivacyClassifier
from fathom.conversation.identity import InteractionIdentity
from fathom.core.services.projector import MemoryProjectorHandler
from fathom.interfaces.interaction import InteractionPort
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.interaction import (
    Assignment,
    BuildContext,
    ClaimJob,
    Content,
    ContextQuery,
    CreateActor,
    CreateThread,
    FinishJob,
    FinishTask,
    Identity,
    JoinThread,
    Metadata,
    OpenTask,
    Plan,
    Projection,
    RecordMessage,
    References,
    RunFinish,
    RunHandle,
    RunStart,
    ScheduleJob,
    Terminal,
    ThreadQuery,
)

if TYPE_CHECKING:
    from datetime import datetime


class InteractionService:
    """
    Application service that records Fathom runs through the interaction port.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Initialize the service with an interaction persistence port.
        """

        self.__interaction = interaction
        self.__classifier = PrivacyClassifier()

    async def start_run(self, *, request: RunStart) -> RunHandle:
        """
        Create the durable interaction records for a starting run.
        """

        task = self.__task(workflow=request.workflow)
        message = self.__message(workflow=request.workflow, name="request")
        thread = await self.__interaction.get_thread(
            query=ThreadQuery(tenant=request.tenant, thread=request.thread)
        )

        await self.__start_run_records(
            task=task,
            request=request,
            message=message,
            thread_exists=thread is not None,
        )

        return RunHandle(
            task=task,
            request=message,
            agent=request.agent,
            thread=request.thread,
            tenant=request.tenant,
            operator=request.operator,
            workflow=request.workflow,
            workspace=request.workspace,
        )

    async def __start_run_records(
        self,
        *,
        task: str,
        message: str,
        request: RunStart,
        thread_exists: bool,
    ) -> None:
        """
        Persist all starting run records as one atomic interaction write.
        """

        async with self.__interaction.atomic():
            await self.__interaction.create_actor(
                request=CreateActor(
                    identity=Identity(
                        id=request.operator,
                        tenant=request.tenant,
                        workspace=request.workspace,
                    ),
                    kind=ActorKind.HUMAN,
                    name=request.operator,
                    created_at=request.started,
                )
            )
            await self.__interaction.create_actor(
                request=CreateActor(
                    identity=Identity(
                        id=request.agent,
                        tenant=request.tenant,
                        workspace=request.workspace,
                    ),
                    name=request.agent,
                    kind=ActorKind.AGENT,
                    created_at=request.started,
                )
            )
            if not thread_exists:
                await self.__interaction.create_thread(
                    request=CreateThread(
                        identity=Identity(
                            id=request.thread,
                            tenant=request.tenant,
                            workspace=request.workspace,
                        ),
                        title=request.intent,
                        creator=request.operator,
                        metadata=request.metadata,
                        created_at=request.started,
                    )
                )
            await self.__join_actor(
                tenant=request.tenant,
                thread=request.thread,
                actor=request.operator,
                joined=request.started,
                workspace=request.workspace,
                role=MembershipRole.REQUESTER,
            )
            await self.__join_actor(
                actor=request.agent,
                tenant=request.tenant,
                thread=request.thread,
                joined=request.started,
                workspace=request.workspace,
                role=MembershipRole.RESPONDER,
            )
            await self.__interaction.open_task(
                request=OpenTask(
                    identity=Identity(
                        id=task,
                        tenant=request.tenant,
                        workspace=request.workspace,
                    ),
                    thread=request.thread,
                    kind=TaskKind.FATHOM,
                    state=TaskState.RUNNING,
                    plan=Plan(
                        objective=request.intent,
                        reference=request.package,
                        plan=Metadata(entries={"workflow": request.workflow}),
                    ),
                    metadata=request.metadata,
                    created_at=request.started,
                    assignment=Assignment(creator=request.operator, assignee=request.agent),
                )
            )
            await self.__interaction.record_message(
                request=RecordMessage(
                    identity=Identity(
                        id=message,
                        tenant=request.tenant,
                        workspace=request.workspace,
                    ),
                    task=task,
                    sequence=None,
                    thread=request.thread,
                    author=request.operator,
                    kind=MessageKind.REQUEST,
                    audience=Audience.THREAD,
                    content=self.__classified_content(
                        body={
                            "intent": request.intent,
                            "package": request.package,
                            "workflow": request.workflow,
                        }
                    ),
                    created_at=request.started,
                )
            )
            await self.__record_context(
                task=task,
                message=message,
                agent=request.agent,
                thread=request.thread,
                tenant=request.tenant,
                created=request.started,
                workflow=request.workflow,
                workspace=request.workspace,
            )

    async def finish_run(self, *, request: RunFinish) -> None:
        """
        Record the terminal interaction records for a finished run.
        """

        code = self.__task_code(request=request)
        state = self.__task_state(request=request)

        async with self.__interaction.atomic():
            await self.__interaction.record_message(
                request=RecordMessage(
                    identity=Identity(
                        tenant=request.handle.tenant,
                        workspace=request.handle.workspace,
                        id=self.__message(workflow=request.handle.workflow, name="result"),
                    ),
                    sequence=None,
                    kind=MessageKind.RESULT,
                    audience=Audience.THREAD,
                    task=request.handle.task,
                    author=request.handle.agent,
                    thread=request.handle.thread,
                    content=self.__classified_content(
                        body={
                            "error": request.error,
                            "steps": request.steps,
                            "status": request.status,
                            "reason": request.reason,
                            "success": request.success,
                        }
                    ),
                    metadata=request.metadata,
                    created_at=request.finished,
                )
            )
            await self.__interaction.finish_task(
                request=FinishTask(
                    state=state,
                    summary=request.reason,
                    elapsed=request.elapsed,
                    task=request.handle.task,
                    ended_at=request.finished,
                    tenant=request.handle.tenant,
                    terminal=Terminal(code=code, detail=request.reason),
                )
            )
            await self.__interaction.schedule_job(
                request=ScheduleJob(
                    identity=Identity(
                        tenant=request.handle.tenant,
                        workspace=request.handle.workspace,
                        id=self.__job(workflow=request.handle.workflow, name="memory"),
                    ),
                    kind=JobKind.MEMORY,
                    task=request.handle.task,
                    thread=request.handle.thread,
                    available_at=request.finished,
                    payload=Metadata(
                        entries={
                            "task": request.handle.task,
                            "thread": request.handle.thread,
                            "workflow": request.handle.workflow,
                        }
                    ),
                    created_at=request.finished,
                )
            )

    def __classified_content(self, *, body: JsonValue) -> Content:
        """
        Build message content with application-level privacy labels applied.
        """

        return Content(body=body, labels=self.__classifier.classify(body=body))

    async def __join_actor(
        self,
        *,
        actor: str,
        thread: str,
        tenant: str,
        joined: datetime,
        role: MembershipRole,
        workspace: Optional[str],
    ) -> None:
        """
        Join one actor to the run thread.
        """

        await self.__interaction.join_thread(
            request=JoinThread(
                role=role,
                actor=actor,
                thread=thread,
                joined_at=joined,
                identity=Identity(
                    tenant=tenant,
                    workspace=workspace,
                    id=InteractionIdentity.stable(
                        scope="membership", parts=(thread, role.value, actor)
                    ),
                ),
            )
        )

    async def __record_context(
        self,
        *,
        task: str,
        agent: str,
        thread: str,
        tenant: str,
        message: str,
        workflow: str,
        created: datetime,
        workspace: Optional[str],
    ) -> None:
        """
        Record the initial reference-based context recipe for the run.
        """

        contexts = await self.__interaction.get_contexts(
            query=ContextQuery(
                task=task,
                tenant=tenant,
                thread=thread,
                purpose=ContextPurpose.EXECUTION,
            )
        )
        context = self.__context(workflow=workflow, name="start")
        if any(record.identity.id == context for record in contexts):
            return

        await self.__interaction.build_context(
            request=BuildContext(
                identity=Identity(
                    id=context,
                    tenant=tenant,
                    workspace=workspace,
                ),
                task=task,
                thread=thread,
                consumer=agent,
                created_at=created,
                builder=INTERACTION_BUILDER,
                purpose=ContextPurpose.EXECUTION,
                references=References(messages=(message,)),
            )
        )

    def __task_state(self, *, request: RunFinish) -> TaskState:
        """
        Resolve the terminal task state for a run outcome.
        """

        if request.success:
            return TaskState.SUCCEEDED

        if request.reason == CompletionReason.CANCELLED.value:
            return TaskState.CANCELLED

        return TaskState.FAILED

    def __task_code(self, *, request: RunFinish) -> TaskCode:
        """
        Resolve the terminal task code for a run outcome.
        """

        if request.success:
            return TaskCode.COMPLETED

        if request.reason == CompletionReason.CANCELLED.value:
            return TaskCode.USER_CANCELLED

        return TaskCode.UNKNOWN_ERROR

    def __task(self, *, workflow: str) -> str:
        """
        Build the root task identifier for a workflow.
        """

        return InteractionIdentity.stable(scope="task.root", parts=(workflow,))

    def __message(self, *, workflow: str, name: str) -> str:
        """
        Build a message identifier for a workflow.
        """

        return InteractionIdentity.stable(scope="message", parts=(workflow, name))

    def __context(self, *, workflow: str, name: str) -> str:
        """
        Build a context identifier for a workflow.
        """

        return InteractionIdentity.stable(scope="context", parts=(workflow, name))

    def __job(self, *, workflow: str, name: str) -> str:
        """
        Build a job identifier for a workflow.
        """

        return InteractionIdentity.stable(scope="job", parts=(workflow, name))


class InteractionProjector:
    """
    Backward-compatible projection shim.

    New hosts should wire JobSchedulerPort + MemoryProjectorHandler directly.
    """

    def __init__(self, *, interaction: InteractionPort, memory: MemoryPort) -> None:
        """
        Initialize the projector with interaction and memory ports.
        """

        self.__interaction = interaction
        self.__handler = MemoryProjectorHandler(interaction=interaction, memory=memory)

    async def project(self, *, request: Projection) -> int:
        """
        Claim and project pending memory jobs.
        """

        completed = 0

        for _ in range(request.limit):
            job = await self.__interaction.claim_job(
                request=ClaimJob(
                    kind=JobKind.MEMORY,
                    owner=request.owner,
                    tenant=request.tenant,
                    claimed=request.claimed,
                )
            )
            if job is None:
                return completed

            result = await self.__handler.handle(job=job)

            await self.__interaction.finish_job(
                request=FinishJob(
                    state=result.state,
                    owner=request.owner,
                    job=job.identity.id,
                    outcome=result.outcome,
                    finished=request.claimed,
                    tenant=job.identity.tenant,
                )
            )
            completed += 1

        return completed
