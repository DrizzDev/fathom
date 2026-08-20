from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional, Tuple

from pydantic import JsonValue

from fathom.constants.collaboration import ArtifactBackend, Label, ThreadState
from fathom.constants.conversation import (
    ARTIFACT_LIST_MAX_LIMIT,
    MESSAGE_LIST_MAX_LIMIT,
    SCRIPT_CONTENT_ENCODING,
    SHA256_HEX_LENGTH,
    SUMMARY_MESSAGE_LIMIT,
    SUMMARY_SCRIPT_LIMIT,
    TIMELINE_MAX_LIMIT,
    EntryKind,
    Visibility,
)
from fathom.constants.signing import SigningStatus
from fathom.conversation.classifier import PrivacyClassifier
from fathom.conversation.cursor import CompositeTimelineCursor
from fathom.conversation.sanitizer import ContentSanitizer
from fathom.conversation.timeline import TimelineComposer
from fathom.core.exceptions import ConversationSummaryLimitExceeded, InteractionError
from fathom.core.services.conversation.access import ConversationAccessGuard
from fathom.core.services.conversation.metadata import ThreadMetadataProjector
from fathom.core.services.conversation.ports import ConversationPorts
from fathom.core.services.conversation.summary import SummaryLoader
from fathom.core.services.conversation.title import TitleComposer
from fathom.interfaces.signing import SigningPort
from fathom.schemas import conversation as ConversationSchemas
from fathom.schemas import interaction as InteractionSchemas
from fathom.schemas.signing import SigningRequest

if TYPE_CHECKING:
    from datetime import datetime


class ConversationService:
    """
    Host-neutral application service for client-facing conversation workflows.
    """

    def __init__(
        self,
        *,
        signer: SigningPort,
        ports: ConversationPorts,
        timeline: Optional[TimelineComposer] = None,
        title: Optional[TitleComposer] = None,
    ) -> None:
        """
        Initialize the service with explicit durable ledger ports and a signer.

        Hosts that deploy without object storage construct a NoopSigner at the composition root
        and pass it here; the signer is a required port with no adapter default.
        """

        self.__ports = ports
        self.__signer = signer
        self.__access = ConversationAccessGuard(threads=ports.threads, members=ports.members)

        self.__sanitizer = ContentSanitizer()
        self.__classifier = PrivacyClassifier()

        self.__summary = SummaryLoader()
        self.__metadata = ThreadMetadataProjector()
        self.__timeline = timeline or TimelineComposer()
        self.__title = title or TitleComposer()

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Open one interaction transaction for grouped conversation writes.
        """

        async with self.__ports.lifecycle.atomic():
            yield

    async def actor(
        self, *, request: ConversationSchemas.AddActor
    ) -> ConversationSchemas.ActorView:
        """
        Register an actor and return its client-facing summary.
        """

        actor = await self.__ports.actors.create(
            request=InteractionSchemas.CreateActor(
                identity=InteractionSchemas.Identity(
                    id=request.id,
                    tenant=request.tenant,
                    workspace=request.workspace,
                ),
                kind=request.kind,
                name=request.name,
                external=request.external,
                created_at=request.created,
                metadata=InteractionSchemas.Metadata(entries=request.metadata),
                runtime=InteractionSchemas.Runtime(provider=request.provider, model=request.model),
            )
        )

        return self.__actor_view(actor=actor)

    async def join(
        self, *, request: ConversationSchemas.JoinMember
    ) -> ConversationSchemas.MemberView:
        """
        Join an actor to a conversation thread.
        """

        membership = await self.__ports.members.join(
            request=InteractionSchemas.JoinThread(
                identity=InteractionSchemas.Identity(
                    id=request.id,
                    tenant=request.tenant,
                    workspace=request.workspace,
                ),
                role=request.role,
                actor=request.actor,
                scope=request.scope,
                thread=request.thread,
                joined_at=request.joined,
                metadata=InteractionSchemas.Metadata(entries=request.metadata),
            )
        )

        return self.__member_view(membership=membership)

    async def create(
        self, *, request: ConversationSchemas.ThreadCreate
    ) -> ConversationSchemas.ThreadView:
        """
        Create a conversation thread and optional creator membership.
        """

        creator = request.creator
        creator_membership = self.__creator_membership(request=request)

        async with self.__ports.lifecycle.atomic():
            if creator is not None:
                await self.__ports.actors.create(
                    request=InteractionSchemas.CreateActor(
                        identity=InteractionSchemas.Identity(
                            id=creator.id,
                            tenant=request.tenant,
                            workspace=creator.workspace or request.workspace,
                        ),
                        kind=creator.kind,
                        name=creator.name,
                        created_at=request.created,
                        runtime=InteractionSchemas.Runtime(
                            model=creator.model,
                            provider=creator.provider,
                        ),
                    )
                )

            title = (
                self.__title.normalize(value=request.title)
                if request.title is not None
                else self.__title.initial(
                    context=ConversationSchemas.TitleContext(intent="", package=None)
                )
            )

            thread = await self.__ports.threads.create(
                request=InteractionSchemas.CreateThread(
                    identity=InteractionSchemas.Identity(
                        id=request.id,
                        tenant=request.tenant,
                        workspace=request.workspace,
                    ),
                    title=title,
                    created_at=request.created,
                    creator=creator.id if creator is not None else None,
                    metadata=InteractionSchemas.Metadata(entries=request.metadata),
                )
            )

            if creator is not None and creator_membership is not None:
                await self.__ports.members.join(
                    request=InteractionSchemas.JoinThread(
                        identity=InteractionSchemas.Identity(
                            id=creator_membership,
                            tenant=request.tenant,
                            workspace=request.workspace,
                        ),
                        actor=creator.id,
                        thread=request.id,
                        role=request.role,
                        joined_at=request.created,
                    )
                )

        return self.__thread_view(thread=thread)

    def __creator_membership(self, *, request: ConversationSchemas.ThreadCreate) -> Optional[str]:
        """
        Return the creator membership id required for creator-backed threads.
        """

        if request.creator is None:
            return None

        if request.member is None:
            raise InteractionError("Thread creator requires a stable membership identifier.")

        return request.member

    async def append(
        self, *, request: ConversationSchemas.MessageAppend
    ) -> ConversationSchemas.EntryView:
        """
        Append a message and return its renderable timeline entry.
        """

        message = await self.__ports.messages.record(
            request=InteractionSchemas.RecordMessage(
                identity=InteractionSchemas.Identity(
                    id=request.id,
                    tenant=request.tenant,
                    workspace=request.workspace,
                ),
                kind=request.kind,
                task=request.task,
                reply=request.reply,
                thread=request.thread,
                author=request.author,
                sequence=request.sequence,
                audience=request.audience,
                execution=request.execution,
                content=self.__classified_content(
                    body=request.body,
                    at=request.created,
                    labels=request.labels,
                ),
                created_at=request.created,
                metadata=InteractionSchemas.Metadata(entries=request.metadata),
            )
        )

        return self.__timeline.message_entry(message=message)

    def __classified_content(
        self,
        *,
        at: "datetime",
        body: JsonValue,
        labels: Tuple[Label, ...],
    ) -> InteractionSchemas.Content:
        """
        Build message content with classifier labels and sanitizer stamping.

        Every message runs through the sanitizer so ``sanitized_at`` / ``sanitizer`` are always
        populated; the default ``noop@1`` profile returns the body verbatim while recording the run.
        """

        classified = self.__classifier.classify(body=body, existing=labels)
        sanitized = self.__sanitizer.sanitize(body=body, labels=classified, at=at)

        return InteractionSchemas.Content(
            labels=classified,
            body=sanitized.body,
            sanitizer=sanitized.sanitizer,
            sanitized_at=sanitized.sanitized,
        )

    async def start(
        self, *, request: ConversationSchemas.TaskStart
    ) -> ConversationSchemas.TaskNodeView:
        """
        Start a conversation task and return its client-facing node.
        """

        task = await self.__ports.tasks.open(
            request=InteractionSchemas.OpenTask(
                identity=InteractionSchemas.Identity(
                    id=request.id,
                    tenant=request.tenant,
                    workspace=request.workspace,
                ),
                thread=request.thread,
                execution=request.execution,
                assignment=InteractionSchemas.Assignment(
                    creator=request.creator, assignee=request.assignee
                ),
                lineage=InteractionSchemas.Lineage(
                    root=request.root,
                    parent=request.parent,
                    origin=request.origin,
                ),
                kind=request.kind,
                state=request.state,
                plan=InteractionSchemas.Plan(
                    objective=request.objective,
                    reference=request.reference,
                    plan=InteractionSchemas.Metadata(entries=request.plan),
                    progress=InteractionSchemas.Metadata(entries=request.progress),
                ),
                created_at=request.created,
                metadata=InteractionSchemas.Metadata(entries=request.metadata),
            )
        )

        return self.__task_node(task=task, children={})

    async def start_execution(
        self, *, request: InteractionSchemas.StartExecution
    ) -> InteractionSchemas.Execution:
        """
        Start a conversation execution and return its stored row.
        """

        return await self.__ports.executions.start(request=request)

    async def finish_execution(
        self, *, request: InteractionSchemas.FinishExecution
    ) -> InteractionSchemas.Execution:
        """
        Finish a conversation execution and return its stored row.
        """

        return await self.__ports.executions.finish(request=request)

    async def finish(
        self, *, request: ConversationSchemas.TaskFinish
    ) -> ConversationSchemas.TaskNodeView:
        """
        Finish a conversation task and return its client-facing node.
        """

        task = await self.__ports.tasks.finish(
            request=InteractionSchemas.FinishTask(
                task=request.task,
                state=request.state,
                tenant=request.tenant,
                ended_at=request.ended,
                elapsed=request.elapsed,
                summary=request.summary,
                terminal=InteractionSchemas.Terminal(code=request.code, detail=request.detail),
            )
        )

        return self.__task_node(task=task, children={})

    async def attach(
        self, *, request: ConversationSchemas.ArtifactAttach
    ) -> ConversationSchemas.EntryView:
        """
        Attach an artifact reference and return its renderable timeline entry.
        """

        artifact = await self.__ports.artifacts.link(
            request=InteractionSchemas.LinkArtifact(
                identity=InteractionSchemas.Identity(
                    id=request.id,
                    tenant=request.tenant,
                    workspace=request.workspace,
                ),
                uri=request.uri,
                task=request.task,
                execution=request.execution,
                kind=request.kind,
                mime=request.mime,
                size=request.size,
                thread=request.thread,
                labels=request.labels,
                backend=request.backend,
                producer=request.producer,
                created_at=request.created,
                retention=request.retention,
                metadata=InteractionSchemas.Metadata(entries=request.metadata),
            )
        )

        return self.__timeline.artifact_entry(artifact=artifact)

    async def save(self, *, request: ConversationSchemas.ScriptSave) -> InteractionSchemas.Script:
        """
        Save a reusable script and append its immutable version audit row.
        """

        return await self.__ports.scripts.save(
            request=InteractionSchemas.SaveScript(
                identity=InteractionSchemas.Identity(
                    id=request.id,
                    tenant=request.tenant,
                    workspace=request.workspace,
                ),
                task=request.task,
                actor=request.actor,
                title=request.title,
                thread=request.thread,
                format=request.format,
                status=request.status,
                source=request.source,
                content=request.content,
                summary=request.summary,
                artifact=request.artifact,
                created_at=request.created,
                execution=request.execution,
                metadata=InteractionSchemas.Metadata(entries=request.metadata),
            )
        )

    async def record(
        self, *, request: ConversationSchemas.ContextRecord
    ) -> ConversationSchemas.EntryView:
        """
        Record a reference-based context recipe and return its audit entry.
        """

        context = await self.__ports.contexts.build(
            request=InteractionSchemas.BuildContext(
                identity=InteractionSchemas.Identity(
                    id=request.id,
                    tenant=request.tenant,
                    workspace=request.workspace,
                ),
                task=request.task,
                thread=request.thread,
                purpose=request.purpose,
                builder=request.builder,
                consumer=request.consumer,
                execution=request.execution,
                references=InteractionSchemas.References(
                    events=request.events,
                    messages=request.messages,
                    artifacts=request.artifacts,
                ),
                hash=request.hash,
                model=request.model,
                provider=request.provider,
                created_at=request.created,
                expires_at=request.expires,
                metadata=InteractionSchemas.Metadata(entries=request.metadata),
            )
        )

        return self.__timeline.context_entry(context=context)

    async def get(
        self, *, query: ConversationSchemas.ConversationThreadQuery
    ) -> ConversationSchemas.ThreadView:
        """
        Load one conversation thread as a client-facing view.
        """

        thread = await self.__thread(query=query)
        return self.__thread_view(thread=thread)

    async def internal_exists(self, *, query: InteractionSchemas.ThreadQuery) -> bool:
        """
        Check whether an internal thread exists without returning client data.
        """

        thread = await self.__ports.threads.get(query=query)
        return thread is not None

    async def archive(
        self, *, request: ConversationSchemas.ConversationTransition
    ) -> ConversationSchemas.ThreadView:
        """
        Archive one conversation thread.
        """

        return await self.__transition(
            request=request,
            state=ThreadState.ARCHIVED,
        )

    async def unarchive(
        self, *, request: ConversationSchemas.ConversationTransition
    ) -> ConversationSchemas.ThreadView:
        """
        Restore one archived conversation thread to active state.
        """

        return await self.__transition(
            state=ThreadState.ACTIVE,
            request=request.model_copy(update={"include_archived": True}),
        )

    async def delete(
        self, *, request: ConversationSchemas.ConversationTransition
    ) -> ConversationSchemas.ThreadView:
        """
        Soft-delete one conversation and its thread-owned records.
        """

        return await self.__transition(
            state=ThreadState.DELETED,
            request=request.model_copy(update={"include_archived": True}),
        )

    async def __transition(
        self,
        *,
        state: ThreadState,
        request: ConversationSchemas.ConversationTransition,
    ) -> ConversationSchemas.ThreadView:
        """
        Apply a lifecycle transition and return the resulting conversation.
        """

        await self.__access.require(
            tenant=request.tenant,
            thread=request.thread,
            operator=request.actor,
            include_archived=request.include_archived,
        )
        thread = await self.__ports.threads.transition(
            request=InteractionSchemas.ThreadTransition(
                state=state,
                actor=request.actor,
                tenant=request.tenant,
                thread=request.thread,
                updated_at=request.updated,
            )
        )
        return self.__thread_view(thread=thread)

    async def cleanup(
        self, *, request: InteractionSchemas.CleanupRequest
    ) -> InteractionSchemas.CleanupResult:
        """
        Forward a host-issued cleanup sweep to the interaction port.
        """

        return await self.__ports.cleanup.run(request=request)

    async def save_policy(self, *, request: InteractionSchemas.SavePolicy) -> None:
        """
        Persist a governance policy used by a conversation entrypoint.
        """

        await self.__ports.policies.save(request=request)

    async def get_policy(
        self, *, query: InteractionSchemas.PolicyQuery
    ) -> Optional[InteractionSchemas.Policy]:
        """
        Load a governance policy by tenant, workspace, and name.
        """

        return await self.__ports.policies.get(query=query)

    async def begin_request(
        self, *, request: InteractionSchemas.BeginRequest
    ) -> InteractionSchemas.Idempotency:
        """
        Record the start of an idempotent run request.
        """

        return await self.__ports.requests.begin(request=request)

    async def finish_request(
        self, *, request: InteractionSchemas.FinishRequest
    ) -> InteractionSchemas.Idempotency:
        """
        Record the terminal state of an idempotent run request.
        """

        return await self.__ports.requests.finish(request=request)

    async def get_idempotency(
        self, *, query: InteractionSchemas.IdempotencyQuery
    ) -> Optional[InteractionSchemas.Idempotency]:
        """
        Load one tenant-scoped idempotency record.
        """

        return await self.__ports.requests.get(query=query)

    async def schedule_job(
        self, *, request: InteractionSchemas.ScheduleJob
    ) -> InteractionSchemas.Job:
        """
        Persist a run-scoped durable job record.
        """

        return await self.__ports.jobs.schedule(request=request)

    async def claim_job(
        self, *, request: InteractionSchemas.ClaimJob
    ) -> Optional[InteractionSchemas.Job]:
        """
        Claim a run-scoped durable job record.
        """

        return await self.__ports.jobs.claim(request=request)

    async def finish_job(self, *, request: InteractionSchemas.FinishJob) -> InteractionSchemas.Job:
        """
        Finish a run-scoped durable job record.
        """

        return await self.__ports.jobs.finish(request=request)

    async def title(
        self,
        *,
        title: str,
        thread: str,
        tenant: str,
        operator: str,
        updated: datetime,
        source: str = "intent",
    ) -> ConversationSchemas.ThreadView:
        """
        Set or replace a thread title after access validation.
        """

        await self.__access.require(tenant=tenant, thread=thread, operator=operator)

        result = await self.__ports.threads.title(
            request=InteractionSchemas.SetThreadTitle(
                tenant=tenant,
                thread=thread,
                updated_at=updated,
                title=self.__title.normalize(value=title),
                metadata=InteractionSchemas.Metadata(
                    entries={
                        "source": source,
                        "refreshed_at": updated.isoformat(),
                    }
                ),
            )
        )
        return self.__thread_view(thread=result)

    async def list(
        self, *, query: ConversationSchemas.ConversationListQuery
    ) -> ConversationSchemas.ConversationPage:
        """
        Load a tenant-scoped page of conversations.
        """

        page = await self.__ports.threads.list(
            query=InteractionSchemas.ThreadListQuery(
                state=query.state,
                title=query.title,
                limit=query.limit,
                cursor=query.cursor,
                tenant=query.tenant,
                actor=query.operator,
                workspace=query.workspace,
                updated_since=query.since,
                updated_until=query.until,
                include_archived=self.__should_include_archived(state=query.state),
            )
        )

        return ConversationSchemas.ConversationPage(
            next=page.next,
            total=page.total,
            items=tuple(self.__thread_view(thread=thread) for thread in page.items),
        )

    async def messages(
        self, *, query: ConversationSchemas.MessageListQuery
    ) -> ConversationSchemas.MessagePage:
        """
        Load a cursor-paginated page of messages for one conversation.
        """

        await self.__thread(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=query.tenant,
                thread=query.thread,
                operator=query.operator,
            )
        )
        page = await self.__ports.messages.list(
            query=InteractionSchemas.MessageCursorQuery(
                task=query.task,
                order=query.order,
                kinds=query.kinds,
                since=query.since,
                until=query.until,
                limit=query.limit,
                author=query.actor,
                cursor=query.cursor,
                tenant=query.tenant,
                thread=query.thread,
            )
        )

        return ConversationSchemas.MessagePage(
            next=page.next,
            total=page.total,
            items=tuple(self.__message_view(message=message) for message in page.items),
        )

    async def artifacts(
        self, *, query: ConversationSchemas.ArtifactListQuery
    ) -> ConversationSchemas.ArtifactPage:
        """
        Load a cursor-paginated page of artifacts for one conversation.
        """

        await self.__thread(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=query.tenant,
                thread=query.thread,
                operator=query.operator,
            )
        )
        page = await self.__ports.artifacts.list(
            query=InteractionSchemas.ArtifactCursorQuery(
                task=query.task,
                order=query.order,
                kinds=query.kinds,
                since=query.since,
                until=query.until,
                limit=query.limit,
                cursor=query.cursor,
                tenant=query.tenant,
                thread=query.thread,
                producer=query.producer,
            )
        )

        signed_items: List[ConversationSchemas.ArtifactView] = []

        for artifact in page.items:
            signed_items.append(await self.__signed_artifact_view(artifact=artifact))

        return ConversationSchemas.ArtifactPage(
            next=page.next,
            total=page.total,
            items=tuple(signed_items),
        )

    async def script(
        self, *, query: ConversationSchemas.RunScriptQuery
    ) -> Optional[ConversationSchemas.ScriptView]:
        """
        Load the generated script for one run task.
        """

        await self.__thread(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=query.tenant,
                thread=query.thread,
                operator=query.operator,
            )
        )
        task = await self.__ports.tasks.get(
            query=InteractionSchemas.TaskOneQuery(
                tenant=query.tenant, thread=query.thread, task=query.task
            )
        )
        if task is None:
            return None

        page = await self.__ports.scripts.list(
            query=InteractionSchemas.ScriptListQuery(
                limit=1,
                count=False,
                task=query.task,
                tenant=query.tenant,
                thread=query.thread,
                order=InteractionSchemas.SortOrder.DESC,
            )
        )
        if not page.items:
            return None

        script = page.items[0]
        checksum = await self.__latest_script_checksum(script=script)
        return self.__script_view(script=script, checksum=checksum)

    async def list_scripts(
        self, *, query: ConversationSchemas.ScriptsQuery
    ) -> ConversationSchemas.ScriptPage:
        """
        Load a cursor-paginated page of scripts for one conversation thread.
        """

        await self.__thread(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=query.tenant,
                thread=query.thread,
                operator=query.operator,
            )
        )
        page = await self.__ports.scripts.list(
            query=InteractionSchemas.ScriptListQuery(
                task=query.task,
                count=query.count,
                since=query.since,
                until=query.until,
                limit=query.limit,
                cursor=query.cursor,
                tenant=query.tenant,
                thread=query.thread,
                order=InteractionSchemas.SortOrder.DESC,
            )
        )
        return ConversationSchemas.ScriptPage(
            next=page.next,
            total=page.total,
            items=tuple(self.__script_view(script=script, checksum=None) for script in page.items),
        )

    async def summary_messages(
        self, *, query: InteractionSchemas.SummaryMessagesQuery
    ) -> Tuple[ConversationSchemas.MessageView, ...]:
        """
        Read every message of the selected kinds in one thread up to the summary cap.
        """

        await self.__thread(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=query.tenant,
                thread=query.thread,
                operator=query.operator,
            )
        )

        page = await self.__ports.messages.list(
            query=InteractionSchemas.MessageCursorQuery(
                kinds=query.kinds,
                count_total=False,
                tenant=query.tenant,
                thread=query.thread,
                limit=SUMMARY_MESSAGE_LIMIT,
                order=InteractionSchemas.SortOrder.ASC,
            )
        )
        if page.next is not None:
            raise ConversationSummaryLimitExceeded(
                thread=query.thread,
                limit=SUMMARY_MESSAGE_LIMIT,
                kind=",".join(kind.value for kind in query.kinds) or "message",
            )

        return tuple(self.__message_view(message=message) for message in page.items)

    async def summary_scripts(
        self, *, query: InteractionSchemas.SummaryScriptsQuery
    ) -> Tuple[ConversationSchemas.ScriptView, ...]:
        """
        Read every script in one thread up to the summary cap.
        """

        await self.__thread(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=query.tenant,
                thread=query.thread,
                operator=query.operator,
            )
        )
        page = await self.__ports.scripts.list(
            query=InteractionSchemas.ScriptListQuery(
                count=False,
                tenant=query.tenant,
                thread=query.thread,
                limit=SUMMARY_SCRIPT_LIMIT,
                order=InteractionSchemas.SortOrder.ASC,
            )
        )
        if page.next is not None:
            raise ConversationSummaryLimitExceeded(
                kind="script",
                thread=query.thread,
                limit=SUMMARY_SCRIPT_LIMIT,
            )

        return tuple(self.__script_view(script=script, checksum=None) for script in page.items)

    async def summary(
        self, *, query: ConversationSchemas.SummaryQuery
    ) -> ConversationSchemas.SummaryView:
        """
        Project a conversation overview from the durable conversation records.
        """

        loaded = await self.__summary.load(source=self, query=query)
        runtime = await self.__runtime(tenant=query.tenant, thread=query.thread)

        return loaded.model_copy(update={"runtime": runtime})

    async def timeline(
        self, *, query: ConversationSchemas.TimelineQuery
    ) -> ConversationSchemas.TimelineView:
        """
        Build a renderable timeline from durable ledger records.

        The composer performs the consume/emit walk and derives the per-kind
        cursor positions; this method only fetches the per-kind candidate
        batches and wires `has_more` from each underlying page.
        """

        thread = await self.__thread(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=query.tenant,
                thread=query.thread,
                operator=query.operator,
            )
        )

        incoming = self.__decode_timeline_cursor(value=query.cursor)

        events: List[InteractionSchemas.Event] = []
        contexts: List[InteractionSchemas.Context] = []
        messages: List[InteractionSchemas.Message] = []
        artifacts: List[InteractionSchemas.Artifact] = []

        total = 0
        first_page = query.cursor is None
        candidate_limit = min(TIMELINE_MAX_LIMIT, max(query.limit * 4, query.limit + 16))
        has_more = {"messages": False, "events": False, "artifacts": False, "contexts": False}

        if self.__include_timeline_kind(query=query, kind=EntryKind.MESSAGE):
            message_page = await self.__ports.messages.list(
                query=InteractionSchemas.MessageCursorQuery(
                    task=query.task,
                    order=query.order,
                    until=query.until,
                    since=query.since,
                    author=query.actor,
                    tenant=query.tenant,
                    thread=query.thread,
                    count_total=first_page,
                    cursor=incoming.messages,
                    limit=min(candidate_limit, MESSAGE_LIST_MAX_LIMIT),
                )
            )
            messages = list(message_page.items)
            has_more["messages"] = message_page.next is not None
            total += message_page.total if first_page else len(message_page.items)

        if self.__include_timeline_kind(query=query, kind=EntryKind.EVENT):
            event_page = await self.__ports.events.list(
                query=InteractionSchemas.EventCursorQuery(
                    task=query.task,
                    order=query.order,
                    actor=query.actor,
                    since=query.since,
                    until=query.until,
                    tenant=query.tenant,
                    thread=query.thread,
                    cursor=incoming.events,
                    count_total=first_page,
                    limit=min(candidate_limit, TIMELINE_MAX_LIMIT),
                )
            )
            events = list(event_page.items)
            has_more["events"] = event_page.next is not None
            total += event_page.total if first_page else len(event_page.items)

        if self.__include_timeline_kind(query=query, kind=EntryKind.ARTIFACT):
            artifact_page = await self.__ports.artifacts.list(
                query=InteractionSchemas.ArtifactCursorQuery(
                    task=query.task,
                    order=query.order,
                    since=query.since,
                    until=query.until,
                    tenant=query.tenant,
                    thread=query.thread,
                    producer=query.actor,
                    count_total=first_page,
                    cursor=incoming.artifacts,
                    limit=min(candidate_limit, ARTIFACT_LIST_MAX_LIMIT),
                )
            )
            artifacts = list(artifact_page.items)
            has_more["artifacts"] = artifact_page.next is not None
            total += artifact_page.total if first_page else len(artifact_page.items)

        if self.__include_timeline_kind(query=query, kind=EntryKind.CONTEXT):
            context_page = await self.__ports.contexts.list(
                query=InteractionSchemas.ContextCursorQuery(
                    task=query.task,
                    order=query.order,
                    since=query.since,
                    until=query.until,
                    tenant=query.tenant,
                    thread=query.thread,
                    consumer=query.actor,
                    count_total=first_page,
                    cursor=incoming.contexts,
                    limit=min(candidate_limit, TIMELINE_MAX_LIMIT),
                )
            )
            contexts = list(context_page.items)
            has_more["contexts"] = context_page.next is not None
            total += context_page.total if first_page else len(context_page.items)

        runtime = await self.__runtime(tenant=query.tenant, thread=query.thread)

        view = self.__timeline.build(
            query=query,
            total=total,
            inbound=incoming,
            has_more=has_more,
            events=tuple(events),
            messages=tuple(messages),
            contexts=tuple(contexts),
            artifacts=tuple(artifacts),
            thread=self.__thread_view(thread=thread),
        )

        signed_entries: List[ConversationSchemas.EntryView] = []

        for entry in view.entries:
            signed_entries.append(await self.__signed_timeline_entry(entry=entry))

        return view.model_copy(update={"runtime": runtime, "entries": tuple(signed_entries)})

    async def __signed_timeline_entry(
        self,
        *,
        entry: ConversationSchemas.EntryView,
    ) -> ConversationSchemas.EntryView:
        """
        Sign URIs embedded inside artifact-kind timeline entry payloads and annotate the payload
        with typed status fields; non-artifact entries pass through untouched.
        """

        if entry.kind != EntryKind.ARTIFACT or not isinstance(entry.payload, dict):
            return entry

        payload = entry.payload

        uri = payload.get("uri")
        backend = payload.get("backend")

        if not isinstance(uri, str) or not isinstance(backend, str):
            return entry

        outcome = await self.__signer.sign(
            request=SigningRequest(uri=uri, backend=ArtifactBackend(backend)),
        )
        is_signed = outcome.status == SigningStatus.SIGNED

        updated_payload: Dict[str, JsonValue] = {
            **payload,
            "uri": outcome.uri,
            "signed": is_signed,
            "signing_status": outcome.status.value,
        }
        return entry.model_copy(update={"payload": updated_payload})

    def __decode_timeline_cursor(self, *, value: Optional[str]) -> CompositeTimelineCursor:
        """
        Decode the inbound timeline cursor into per-kind sub-cursors.
        """

        if value is None:
            return CompositeTimelineCursor()

        return CompositeTimelineCursor.decode(value=value)

    async def tasks(
        self, *, query: ConversationSchemas.TaskTreeQuery
    ) -> ConversationSchemas.TaskTreeView:
        """
        Build a renderable task tree for one conversation thread.

        When `query.task` is supplied, the result is the subtree rooted at that task (the named task plus its descendants).
        Otherwise every root-level task in the thread is returned.
        """

        thread = await self.__thread(
            query=ConversationSchemas.ConversationThreadQuery(
                tenant=query.tenant,
                thread=query.thread,
                operator=query.operator,
            )
        )
        task_query = InteractionSchemas.TaskQuery(tenant=query.tenant, thread=query.thread)
        loaded = await self.__load_tree_tasks(
            query=task_query, subtree=query.task, limit=query.limit
        )

        return ConversationSchemas.TaskTreeView(
            total=len(loaded),
            thread=self.__thread_view(thread=thread),
            roots=self.__task_roots(tasks=loaded, scoped_root=query.task),
            runtime=await self.__runtime(tenant=query.tenant, thread=query.thread),
        )

    async def __load_tree_tasks(
        self,
        *,
        query: InteractionSchemas.TaskQuery,
        subtree: Optional[str],
        limit: int,
    ) -> Tuple[InteractionSchemas.Task, ...]:
        """
        Load the task rows the tree endpoint renders, pushing the root-count cap and subtree scope down into SQL.
        """

        if subtree is not None:
            return tuple(await self.__ports.tasks.subtree(query=query, root=subtree))

        roots = await self.__ports.tasks.top_roots(query=query, limit=limit)
        if not roots:
            return ()

        root_ids = [task.identity.id for task in roots]
        descendants = await self.__ports.tasks.descendants(query=query, roots=root_ids)

        return tuple(roots) + tuple(descendants)

    async def state(self, *, operator: str, tenant: str, thread: str, task: str) -> Optional[str]:
        """
        Return one task's state without loading the whole conversation tree.
        """

        await self.__access.require(tenant=tenant, thread=thread, operator=operator)

        stored = await self.__ports.tasks.get(
            query=InteractionSchemas.TaskOneQuery(tenant=tenant, thread=thread, task=task)
        )

        if stored is None:
            return None

        return stored.state.value

    def __scoped_tasks(
        self, *, tasks: Tuple[InteractionSchemas.Task, ...], root: Optional[str]
    ) -> Tuple[InteractionSchemas.Task, ...]:
        """
        When a subtree root is named, prune to that root plus its transitive descendants.
        Otherwise pass every task through unchanged.
        """

        if root is None:
            return tasks

        ids = {root}

        # Walk children breadth-first; tasks are already loaded so no extra IO.
        changed = True

        while changed:
            changed = False
            for task in tasks:
                if task.identity.id in ids:
                    continue

                if task.lineage.parent in ids:
                    ids.add(task.identity.id)
                    changed = True

        return tuple(task for task in tasks if task.identity.id in ids)

    async def __thread(
        self, *, query: ConversationSchemas.ConversationThreadQuery
    ) -> InteractionSchemas.Thread:
        """
        Load a required thread from the ledger.
        """

        return await self.__access.require(
            tenant=query.tenant,
            thread=query.thread,
            operator=query.operator,
        )

    def __should_include_archived(self, *, state: Optional[ThreadState]) -> bool:
        """
        Include archived rows only when the caller explicitly requests them.
        """

        return state is ThreadState.ARCHIVED or state is ThreadState.DELETED

    def __include_timeline_kind(
        self, *, query: ConversationSchemas.TimelineQuery, kind: EntryKind
    ) -> bool:
        """
        Decide whether a backing record kind needs to be loaded for the timeline.
        """

        if query.kinds and kind not in query.kinds:
            return False

        if kind == EntryKind.EVENT:
            return query.mode in (Visibility.DEBUG, Visibility.AUDIT)

        if kind == EntryKind.CONTEXT:
            return query.mode == Visibility.AUDIT

        return True

    def __task_roots(
        self,
        *,
        tasks: Tuple[InteractionSchemas.Task, ...],
        scoped_root: Optional[str] = None,
    ) -> Tuple[ConversationSchemas.TaskNodeView, ...]:
        """
        Convert flat ledger tasks into deterministic root task nodes.

        When `scoped_root` names a subtree root, only that task is treated as
        the root of the rendered tree even if it has a parent in the wider
        thread. Otherwise tasks whose parent is missing from the loaded set
        are surfaced as roots (the original behavior).
        """

        task_ids = {task.identity.id for task in tasks}
        children: Dict[Optional[str], List[InteractionSchemas.Task]] = {}

        for task in tasks:
            if scoped_root is not None and task.identity.id == scoped_root:
                children.setdefault(None, []).append(task)
                continue

            parent = task.lineage.parent if task.lineage.parent in task_ids else None
            children.setdefault(parent, []).append(task)

        return tuple(
            self.__task_node(task=task, children=children)
            for task in sorted(children.get(None, []), key=self.__task_order)
        )

    def __task_node(
        self,
        *,
        task: InteractionSchemas.Task,
        children: Dict[Optional[str], List[InteractionSchemas.Task]],
    ) -> ConversationSchemas.TaskNodeView:
        """
        Convert a ledger task and its descendants into a task tree node.
        """

        return ConversationSchemas.TaskNodeView(
            kind=task.kind,
            state=task.state,
            id=task.identity.id,
            summary=task.summary,
            root=task.lineage.root,
            ended=task.timing.ended,
            parent=task.lineage.parent,
            created=task.timing.created,
            started=task.timing.started,
            objective=task.plan.objective,
            assignee=task.assignment.assignee,
            workflow=self.__workflow_reference(task=task),
            execution=ConversationSchemas.ExecutionReference(id=task.execution),
            children=tuple(
                self.__task_node(task=child, children=children)
                for child in sorted(children.get(task.identity.id, []), key=self.__task_order)
            ),
        )

    @staticmethod
    def __workflow_reference(
        *, task: InteractionSchemas.Task
    ) -> Optional[ConversationSchemas.WorkflowReference]:
        """
        Extract the runtime workflow reference from task metadata when present.
        """

        workflow = task.metadata.entries.get("workflow")

        if isinstance(workflow, str) and workflow:
            return ConversationSchemas.WorkflowReference(id=workflow)

        return None

    def __task_order(self, task: InteractionSchemas.Task) -> Tuple[str, str]:
        """
        Return deterministic sort keys for task tree rendering.
        """

        return (task.timing.created.isoformat(), task.identity.id)

    def __thread_view(self, *, thread: InteractionSchemas.Thread) -> ConversationSchemas.ThreadView:
        """
        Convert a durable thread into a client-facing view.
        """

        return ConversationSchemas.ThreadView(
            state=thread.state,
            title=thread.title,
            id=thread.identity.id,
            created=thread.timing.created,
            updated=thread.timing.updated,
            metadata=self.__metadata.view(thread=thread),
            digest=self.__public_digest(digest=thread.digest),
        )

    async def __runtime(
        self, *, tenant: str, thread: str
    ) -> Optional[ConversationSchemas.RuntimeReference]:
        """
        Build the runtime pointer for a thread from the most recent non-archived task.
        """

        recent = await self.__ports.tasks.recent(
            query=InteractionSchemas.TaskQuery(tenant=tenant, thread=thread)
        )
        if recent is None:
            return None

        return ConversationSchemas.RuntimeReference(
            workflow=self.__workflow_reference(task=recent),
            execution=ConversationSchemas.ExecutionReference(id=recent.execution),
        )

    def __public_digest(self, *, digest: Optional[str]) -> Optional[str]:
        """
        Return only human-readable conversation digests for public APIs.
        """

        if digest is None:
            return None

        if len(digest) != SHA256_HEX_LENGTH:
            return digest

        try:
            int(digest, 16)
        except ValueError:
            return digest

        return None

    def __message_view(
        self, *, message: InteractionSchemas.Message
    ) -> ConversationSchemas.MessageView:
        """
        Convert a durable message into a client-facing message row.
        """

        return ConversationSchemas.MessageView(
            task=message.task,
            kind=message.kind,
            reply=message.reply,
            author=message.author,
            id=message.identity.id,
            created=message.created,
            sequence=message.sequence,
            body=message.content.body,
            audience=message.audience,
            labels=message.content.labels,
        )

    async def __signed_artifact_view(
        self,
        *,
        artifact: InteractionSchemas.Artifact,
    ) -> ConversationSchemas.ArtifactView:
        """
        Build an artifact view with the URI signed by the configured port.
        """

        outcome = await self.__signer.sign(
            request=SigningRequest(uri=artifact.uri, backend=artifact.backend),
        )

        return ConversationSchemas.ArtifactView(
            uri=outcome.uri,
            mime=artifact.mime,
            size=artifact.size,
            task=artifact.task,
            kind=artifact.kind,
            labels=artifact.labels,
            id=artifact.identity.id,
            created=artifact.created,
            producer=artifact.producer,
            signing_status=outcome.status,
        )

    def __actor_view(self, *, actor: InteractionSchemas.Actor) -> ConversationSchemas.ActorView:
        """
        Convert a durable actor into a client-facing view.
        """

        return ConversationSchemas.ActorView(
            name=actor.name,
            kind=actor.kind,
            id=actor.identity.id,
            created=actor.timing.created,
        )

    def __member_view(
        self, *, membership: InteractionSchemas.Membership
    ) -> ConversationSchemas.MemberView:
        """
        Convert a durable membership into a client-facing view.
        """

        return ConversationSchemas.MemberView(
            role=membership.role,
            actor=membership.actor,
            scope=membership.scope,
            joined=membership.joined,
            id=membership.identity.id,
        )

    def __script_view(
        self,
        *,
        checksum: Optional[str],
        script: InteractionSchemas.Script,
    ) -> ConversationSchemas.ScriptView:
        """
        Convert a durable script into a client-facing view with an optional checksum.
        """

        return ConversationSchemas.ScriptView(
            task=script.task,
            checksum=checksum,
            title=script.title,
            format=script.format,
            status=script.status,
            id=script.identity.id,
            content=script.content,
            revision=script.revision,
            created_by=script.created_by,
            updated_by=script.updated_by,
            created=script.timing.created,
            updated=script.timing.updated,
            size=len(script.content.encode(SCRIPT_CONTENT_ENCODING)),
        )

    async def __latest_script_checksum(self, *, script: InteractionSchemas.Script) -> Optional[str]:
        """
        Fetch the checksum of the latest immutable version for the given script.
        """

        versions = await self.__ports.scripts.versions(
            query=InteractionSchemas.ScriptVersionQuery(
                version=script.revision,
                script=script.identity.id,
                tenant=script.identity.tenant,
            )
        )
        if not versions:
            return None

        return versions[0].checksum
