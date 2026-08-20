from __future__ import annotations

import json
from typing import AsyncIterator, Tuple

from fathom.constants.collaboration import (
    PRIVATE_LABELS,
    JobCode,
    JobKind,
    JobState,
    Label,
)
from fathom.interfaces.interaction import MessagePort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.scheduler import JobHandlerPort
from fathom.schemas.interaction import Job, Message, MessageCursorQuery, Outcome
from fathom.schemas.scheduler import JobHandlerResult


class MemoryProjectorHandler(JobHandlerPort):
    """
    Job handler that projects safe conversation messages into memory.
    """

    def __init__(self, *, memory: MemoryPort, interaction: MessagePort) -> None:
        """
        Initialize the handler with durable interaction and memory ports.
        """

        self.__memory = memory
        self.__interaction = interaction

    async def handle(self, *, job: Job) -> JobHandlerResult:
        """
        Project one claimed memory job.
        """

        if job.kind != JobKind.MEMORY:
            return JobHandlerResult(
                state=JobState.ABANDONED,
                outcome=Outcome(
                    code=JobCode.PERMANENT_ERROR,
                    detail=f"Unsupported job kind '{job.kind.value}'.",
                ),
            )
        if job.task is None:
            return JobHandlerResult(
                state=JobState.ABANDONED,
                outcome=Outcome(
                    code=JobCode.PERMANENT_ERROR,
                    detail="Memory projection job has no task.",
                ),
            )

        projected_list = [
            message
            async for message in self.__paginated_messages(job=job)
            if self.__projectable(message=message)
        ]

        projected = tuple(projected_list)

        await self.__memory.set(
            key=self.__key(thread=job.thread, task=job.task),
            value=self.__value(thread=job.thread, task=job.task, messages=projected),
        )

        return JobHandlerResult(
            state=JobState.COMPLETED,
            outcome=Outcome(code=JobCode.COMPLETED),
        )

    async def __paginated_messages(self, *, job: Job) -> AsyncIterator[Message]:
        """
        Walk every page of task messages so long tasks are projected fully.

        Loops until the underlying cursor is exhausted; reading only the first page would silently
        truncate long task histories.
        """

        cursor: str | None = None

        while True:
            page = await self.__interaction.list_messages(
                query=MessageCursorQuery(
                    task=job.task,
                    cursor=cursor,
                    thread=job.thread,
                    tenant=job.identity.tenant,
                )
            )

            for message in page.items:
                yield message

            if page.next is None:
                return

            cursor = page.next

    def __key(self, *, thread: str, task: str) -> str:
        """
        Build a stable memory key for one interaction task.
        """

        return f"interaction.{thread}.{task}"

    def __projectable(self, *, message: Message) -> bool:
        """
        Decide whether a message is safe to project into memory.
        """

        labels = frozenset(message.content.labels)
        if Label.MEMORY_SKIP in labels:
            return False

        return not (labels & PRIVATE_LABELS and message.content.sanitized is None)

    def __value(self, *, thread: str, task: str, messages: Tuple[Message, ...]) -> str:
        """
        Serialize projected interaction messages for memory storage.
        """

        return json.dumps(
            {
                "thread": thread,
                "task": task,
                "messages": [
                    {
                        "kind": message.kind.value,
                        "body": message.content.body,
                    }
                    for message in messages
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
