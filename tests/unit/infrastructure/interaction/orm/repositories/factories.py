from __future__ import annotations

from tests.unit.infrastructure.interaction.orm.support import InteractionRuntimeRegistry

from fathom.infrastructure.interaction.orm.raw import InteractionSqlFiles, RawSql
from fathom.infrastructure.interaction.orm.repositories import (
    ActorRepository,
    ArtifactRepository,
    ContextRepository,
    ExecutionRepository,
    JobRepository,
    MembershipRepository,
    MessageRepository,
    PolicyRepository,
    ReferenceGuard,
    RequestRepository,
    ScriptRepository,
    TaskRepository,
    ThreadRepository,
)
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    LifecycleRecorder,
    SequenceAllocator,
    UuidIdentifierSource,
)
from fathom.interaction.digest import EventDigest
from fathom.interaction.lifecycle import Lifecycle


class InteractionRepositoryFactory:
    """
    Builds persistence repositories with explicit collaborators matching production.
    """

    def __init__(self) -> None:
        """
        Initialize shared repository collaborators for one test module.
        """

        self.__state_machine = Lifecycle()
        self.__identifiers = UuidIdentifierSource()
        self.__transaction = InteractionRuntimeRegistry.require()

        self.__sql_files = InteractionSqlFiles.bundled()
        self.__raw = RawSql(root=self.__sql_files.root)

        self.__references = ReferenceGuard()
        self.__sequences = SequenceAllocator(
            raw=self.__raw,
            identifier_source=self.__identifiers,
        )
        self.__recorder = LifecycleRecorder(
            raw=self.__raw,
            event_digest=EventDigest(),
            sequence_allocator=self.__sequences,
            identifier_source=self.__identifiers,
        )

    def actors(self) -> ActorRepository:
        """
        Build an actor repository.
        """

        return ActorRepository(transaction=self.__transaction)

    def artifacts(self) -> ArtifactRepository:
        """
        Build an artifact repository.
        """

        return ArtifactRepository(
            lifecycle=self.__recorder,
            references=self.__references,
            transaction=self.__transaction,
        )

    def contexts(self) -> ContextRepository:
        """
        Build a context repository.
        """

        return ContextRepository(
            lifecycle=self.__recorder,
            references=self.__references,
            transaction=self.__transaction,
        )

    def executions(self) -> ExecutionRepository:
        """
        Build an execution repository.
        """

        return ExecutionRepository(transaction=self.__transaction)

    def jobs(self) -> JobRepository:
        """
        Build a job repository.
        """

        return JobRepository(
            raw=self.__raw,
            lifecycle=self.__recorder,
            references=self.__references,
            transaction=self.__transaction,
            validator=self.__state_machine,
        )

    def memberships(self) -> MembershipRepository:
        """
        Build a membership repository.
        """

        return MembershipRepository(lifecycle=self.__recorder, transaction=self.__transaction)

    def messages(self) -> MessageRepository:
        """
        Build a message repository.
        """

        return MessageRepository(
            recorder=self.__recorder,
            sequences=self.__sequences,
            references=self.__references,
            transaction=self.__transaction,
            lifecycle=self.__state_machine,
        )

    def policies(self) -> PolicyRepository:
        """
        Build a policy repository.
        """

        return PolicyRepository(
            lifecycle=self.__state_machine,
            transaction=self.__transaction,
        )

    def requests(self) -> RequestRepository:
        """
        Build a request repository.
        """

        return RequestRepository(
            transaction=self.__transaction,
            lifecycle=self.__state_machine,
            identifier_source=self.__identifiers,
        )

    def scripts(self) -> ScriptRepository:
        """
        Build a script repository.
        """

        return ScriptRepository(
            references=self.__references,
            transaction=self.__transaction,
            identifier_source=self.__identifiers,
        )

    def tasks(self) -> TaskRepository:
        """
        Build a task repository.
        """

        return TaskRepository(
            recorder=self.__recorder,
            lifecycle=self.__state_machine,
            transaction=self.__transaction,
        )

    def threads(self) -> ThreadRepository:
        """
        Build a thread repository.
        """

        return ThreadRepository(lifecycle=self.__recorder, transaction=self.__transaction)
