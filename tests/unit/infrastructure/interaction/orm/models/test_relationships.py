from __future__ import annotations

import unittest
from typing import Protocol, Tuple, Type, cast

from tortoise.models import Model

from fathom.infrastructure.interaction.orm.models import (
    ArtifactRecord,
    ContextRecord,
    EventRecord,
    ExecutionRecord,
    JobRecord,
    MembershipRecord,
    MessageRecord,
    ScriptRecord,
    ScriptVersionRecord,
    SequenceRecord,
    TaskRecord,
)


class RelationMetadata(Protocol):
    """
    Tortoise relation metadata fields asserted by these tests.
    """

    db_constraint: bool
    source_field: str


class TestOrmRelationshipMetadata(unittest.TestCase):
    """
    Verify ORM relations mirror columns without replacing repository raw-id access.
    """

    def test_conversation_owned_records_map_existing_conversation_column(self) -> None:
        """
        Conversation-owned records expose a relation backed by `conversation_id`.
        """

        for record in self.__conversation_owned_records():
            self.__assert_source(
                record=record,
                field="conversation",
                source="conversation_id",
            )

    def test_execution_owned_records_map_existing_execution_column(self) -> None:
        """
        Execution-owned records expose a relation backed by `execution_id`.
        """

        for record in self.__execution_owned_records():
            self.__assert_source(
                record=record,
                field="execution",
                source="execution_id",
            )

    def test_script_versions_map_existing_script_column(self) -> None:
        """
        Script versions expose a relation backed by `script_id`.
        """

        self.__assert_source(
            source="script_id",
            field="script",
            record=ScriptVersionRecord,
        )

    def __assert_source(
        self,
        *,
        field: str,
        source: str,
        record: Type[Model],
    ) -> None:
        """
        Assert one Tortoise relation maps to the expected existing database column.
        """

        relation = cast("RelationMetadata", record._meta.fields_map[field])

        self.assertFalse(relation.db_constraint)
        self.assertEqual(relation.source_field, source)

    def __conversation_owned_records(self) -> Tuple[Type[Model], ...]:
        """
        Return records that carry the existing `conversation_id` column.
        """

        return (
            JobRecord,
            TaskRecord,
            EventRecord,
            ScriptRecord,
            ContextRecord,
            MessageRecord,
            SequenceRecord,
            ArtifactRecord,
            ExecutionRecord,
            MembershipRecord,
        )

    def __execution_owned_records(self) -> Tuple[Type[Model], ...]:
        """
        Return records that carry the existing `execution_id` column.
        """

        return (
            JobRecord,
            TaskRecord,
            EventRecord,
            ScriptRecord,
            ContextRecord,
            MessageRecord,
            ArtifactRecord,
        )
