from __future__ import annotations

from typing import List, Optional, Protocol

from fathom.interfaces.interaction import InteractionPort
from fathom.schemas.interaction import (
    Actor,
    Artifact,
    ArtifactCursorQuery,
    ArtifactPage,
    ArtifactQuery,
    BeginRequest,
    BuildContext,
    ClaimJob,
    CleanupRequest,
    CleanupResult,
    Context,
    ContextCursorQuery,
    ContextPage,
    ContextQuery,
    CreateActor,
    CreateThread,
    Event,
    EventCursorQuery,
    EventPage,
    EventQuery,
    Execution,
    ExecutionQuery,
    FinishExecution,
    FinishJob,
    FinishRequest,
    FinishTask,
    Idempotency,
    IdempotencyQuery,
    Job,
    JoinThread,
    LinkArtifact,
    Membership,
    MembershipQuery,
    Message,
    MessageCursorQuery,
    MessagePage,
    OpenTask,
    Policy,
    PolicyQuery,
    RecordMessage,
    SavePolicy,
    SaveScript,
    ScheduleJob,
    Script,
    ScriptListQuery,
    ScriptPage,
    ScriptQuery,
    ScriptVersion,
    ScriptVersionQuery,
    SetThreadTitle,
    StartExecution,
    Task,
    TaskOneQuery,
    TaskQuery,
    Thread,
    ThreadListQuery,
    ThreadPage,
    ThreadQuery,
    ThreadTransition,
)


class ThreadStore:
    """
    Conversation thread operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def create(self, *, request: CreateThread) -> Thread:
        """
        Create one conversation thread.
        """

        return await self.__interaction.create_thread(request=request)

    async def get(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Load one conversation thread.
        """

        return await self.__interaction.get_thread(query=query)

    async def title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set the title for one conversation thread.
        """

        return await self.__interaction.set_thread_title(request=request)

    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Transition one conversation thread lifecycle state.
        """

        return await self.__interaction.transition(request=request)

    async def list(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        List conversation threads.
        """

        return await self.__interaction.list_threads(query=query)


class ActorStore:
    """
    Conversation actor operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def create(self, *, request: CreateActor) -> Actor:
        """
        Create one actor.
        """

        return await self.__interaction.create_actor(request=request)


class MemberStore:
    """
    Conversation membership operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def join(self, *, request: JoinThread) -> Membership:
        """
        Join one actor to a thread.
        """

        return await self.__interaction.join_thread(request=request)

    async def find(self, *, query: MembershipQuery) -> Optional[Membership]:
        """
        Load one active actor membership.
        """

        return await self.__interaction.find_membership(query=query)


class TaskStore:
    """
    Conversation task operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def open(self, *, request: OpenTask) -> Task:
        """
        Open one task.
        """

        return await self.__interaction.open_task(request=request)

    async def finish(self, *, request: FinishTask) -> Task:
        """
        Finish one task.
        """

        return await self.__interaction.finish_task(request=request)

    async def get(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Load one task.
        """

        return await self.__interaction.get_task(query=query)

    async def list(self, *, query: TaskQuery) -> List[Task]:
        """
        List tasks.
        """

        return await self.__interaction.get_tasks(query=query)

    async def recent(self, *, query: TaskQuery) -> Optional[Task]:
        """
        Return the most recent non-archived task in the thread, if any.
        """

        return await self.__interaction.recent_task(query=query)

    async def top_roots(self, *, query: TaskQuery, limit: int) -> List[Task]:
        """
        Return the top-N root tasks in the thread using SQL LIMIT.
        """

        return await self.__interaction.top_root_tasks(query=query, limit=limit)

    async def descendants(self, *, query: TaskQuery, roots: List[str]) -> List[Task]:
        """
        Return every descendant of the supplied root tasks in one query.
        """

        return await self.__interaction.task_descendants(query=query, roots=roots)

    async def subtree(self, *, query: TaskQuery, root: str) -> List[Task]:
        """
        Return one subtree rooted at the supplied task in one query.
        """

        return await self.__interaction.task_subtree(query=query, root=root)


class ExecutionStore:
    """
    Conversation execution operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def start(self, *, request: StartExecution) -> Execution:
        """
        Start one execution.
        """

        return await self.__interaction.start_execution(request=request)

    async def finish(self, *, request: FinishExecution) -> Execution:
        """
        Finish one execution.
        """

        return await self.__interaction.finish_execution(request=request)

    async def get(self, *, query: ExecutionQuery) -> Optional[Execution]:
        """
        Load one execution.
        """

        return await self.__interaction.get_execution(query=query)


class MessageStore:
    """
    Conversation message operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def record(self, *, request: RecordMessage) -> Message:
        """
        Record one message.
        """

        return await self.__interaction.record_message(request=request)

    async def list(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        List messages.
        """

        return await self.__interaction.list_messages(query=query)


class EventStore:
    """
    Conversation event operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def list(self, *, query: EventCursorQuery) -> EventPage:
        """
        List events.
        """

        return await self.__interaction.list_events(query=query)

    async def get(self, *, query: EventQuery) -> List[Event]:
        """
        Load matching events.
        """

        return await self.__interaction.get_events(query=query)


class ArtifactStore:
    """
    Conversation artifact operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def link(self, *, request: LinkArtifact) -> Artifact:
        """
        Link one artifact.
        """

        return await self.__interaction.link_artifact(request=request)

    async def list(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        List artifacts.
        """

        return await self.__interaction.list_artifacts(query=query)

    async def get(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load matching artifacts.
        """

        return await self.__interaction.get_artifacts(query=query)


class ScriptStore:
    """
    Conversation script operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def save(self, *, request: SaveScript) -> Script:
        """
        Save one script.
        """

        return await self.__interaction.save_script(request=request)

    async def list(self, *, query: ScriptListQuery) -> ScriptPage:
        """
        List scripts.
        """

        return await self.__interaction.list_scripts(query=query)

    async def get(self, *, query: ScriptQuery) -> List[Script]:
        """
        Load matching scripts.
        """

        return await self.__interaction.get_scripts(query=query)

    async def versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Load script versions.
        """

        return await self.__interaction.get_script_versions(query=query)


class PolicyStore:
    """
    Conversation policy operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def save(self, *, request: SavePolicy) -> Policy:
        """
        Save one policy.
        """

        return await self.__interaction.save_policy(request=request)

    async def get(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Load one policy.
        """

        return await self.__interaction.get_policy(query=query)


class JobStore:
    """
    Conversation job operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def schedule(self, *, request: ScheduleJob) -> Job:
        """
        Schedule one job.
        """

        return await self.__interaction.schedule_job(request=request)

    async def claim(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Claim one job.
        """

        return await self.__interaction.claim_job(request=request)

    async def finish(self, *, request: FinishJob) -> Job:
        """
        Finish one job.
        """

        return await self.__interaction.finish_job(request=request)


class RequestStore:
    """
    Conversation idempotency operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def begin(self, *, request: BeginRequest) -> Idempotency:
        """
        Begin one idempotent request.
        """

        return await self.__interaction.begin_request(request=request)

    async def finish(self, *, request: FinishRequest) -> Idempotency:
        """
        Finish one idempotent request.
        """

        return await self.__interaction.finish_request(request=request)

    async def get(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Load one idempotency record.
        """

        return await self.__interaction.get_idempotency(query=query)


class ContextStore:
    """
    Conversation context operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def build(self, *, request: BuildContext) -> Context:
        """
        Build one context record.
        """

        return await self.__interaction.build_context(request=request)

    async def list(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        List context records.
        """

        return await self.__interaction.list_contexts(query=query)

    async def get(self, *, query: ContextQuery) -> List[Context]:
        """
        Load matching context records.
        """

        return await self.__interaction.get_contexts(query=query)


class CleanupStore:
    """
    Conversation cleanup operations exposed with aggregate-local verbs.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Store the backing interaction adapter.
        """

        self.__interaction = interaction

    async def run(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run one cleanup request.
        """

        return await self.__interaction.cleanup(request=request)


class ConversationPorts(Protocol):
    """
    Port bundle contract consumed by the conversation application service.
    """

    actors: ActorStore
    members: MemberStore
    lifecycle: InteractionPort

    tasks: TaskStore
    threads: ThreadStore
    messages: MessageStore
    executions: ExecutionStore

    events: EventStore
    artifacts: ArtifactStore

    jobs: JobStore
    scripts: ScriptStore
    cleanup: CleanupStore

    policies: PolicyStore
    requests: RequestStore
    contexts: ContextStore


class Ports:
    """
    Adapts the composite interaction adapter into aggregate-local conversation ports.
    """

    def __init__(self, *, interaction: InteractionPort) -> None:
        """
        Build aggregate-local stores from one interaction adapter.
        """

        self.lifecycle = interaction

        self.actors = ActorStore(interaction=interaction)
        self.members = MemberStore(interaction=interaction)

        self.tasks = TaskStore(interaction=interaction)
        self.threads = ThreadStore(interaction=interaction)
        self.executions = ExecutionStore(interaction=interaction)

        self.events = EventStore(interaction=interaction)
        self.messages = MessageStore(interaction=interaction)
        self.artifacts = ArtifactStore(interaction=interaction)

        self.scripts = ScriptStore(interaction=interaction)
        self.policies = PolicyStore(interaction=interaction)

        self.jobs = JobStore(interaction=interaction)
        self.cleanup = CleanupStore(interaction=interaction)

        self.requests = RequestStore(interaction=interaction)
        self.contexts = ContextStore(interaction=interaction)
