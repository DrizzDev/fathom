from __future__ import annotations

from typing import Tuple, Type

from tortoise.models import Model

from fathom.infrastructure.interaction.orm.models.actor import ActorRecord
from fathom.infrastructure.interaction.orm.models.artifact import ArtifactRecord
from fathom.infrastructure.interaction.orm.models.context import ContextRecord
from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord
from fathom.infrastructure.interaction.orm.models.event import EventRecord
from fathom.infrastructure.interaction.orm.models.execution import ExecutionRecord
from fathom.infrastructure.interaction.orm.models.job import JobRecord
from fathom.infrastructure.interaction.orm.models.member import MembershipRecord
from fathom.infrastructure.interaction.orm.models.message import MessageRecord
from fathom.infrastructure.interaction.orm.models.policy import PolicyRecord
from fathom.infrastructure.interaction.orm.models.request import RequestRecord
from fathom.infrastructure.interaction.orm.models.script import (
    ScriptRecord,
    ScriptVersionRecord,
)
from fathom.infrastructure.interaction.orm.models.sequence import SequenceRecord
from fathom.infrastructure.interaction.orm.models.task import TaskRecord

__all__ = [
    "Catalog",
    "JobRecord",
    "TaskRecord",
    "ActorRecord",
    "EventRecord",
    "ScriptRecord",
    "PolicyRecord",
    "RequestRecord",
    "ContextRecord",
    "MessageRecord",
    "SequenceRecord",
    "ArtifactRecord",
    "ExecutionRecord",
    "MembershipRecord",
    "ConversationRecord",
    "ScriptVersionRecord",
]


class Catalog:
    """
    Lists persistent records that belong to the interaction store.
    """

    @classmethod
    def module(cls) -> str:
        """
        Return the importable Tortoise model module path.
        """

        return cls.__module__

    def all(self) -> Tuple[Type[Model], ...]:
        """
        Return all persistent record classes in registration order.
        """

        return (
            JobRecord,
            TaskRecord,
            EventRecord,
            ActorRecord,
            PolicyRecord,
            ScriptRecord,
            ContextRecord,
            MessageRecord,
            RequestRecord,
            ArtifactRecord,
            SequenceRecord,
            ExecutionRecord,
            MembershipRecord,
            ConversationRecord,
            ScriptVersionRecord,
        )
