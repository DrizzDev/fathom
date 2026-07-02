from __future__ import annotations

from typing import Dict, Tuple, Type

from tortoise import fields
from tortoise.models import Model

from fathom.infrastructure.interaction.orm import models as interaction_models


class TestInteractionStoreModels:
    """
    Verify the persistence metadata contract for the interaction store.
    """

    def test_exports_every_conversation_table_model(self) -> None:
        expected_tables = {
            "actors",
            "conversations",
            "executions",
            "memberships",
            "tasks",
            "messages",
            "events",
            "artifacts",
            "scripts",
            "script_versions",
            "policies",
            "contexts",
            "requests",
            "jobs",
            "sequences",
        }

        actual_tables = {model._meta.db_table for model in interaction_models.Catalog().all()}

        assert actual_tables == expected_tables

    def test_every_persisted_model_uses_plain_uuid_primary_key(self) -> None:
        for model in interaction_models.Catalog().all():
            pk_field = model._meta.pk

            assert isinstance(model, type)
            assert isinstance(pk_field, fields.CharField)
            assert pk_field.model_field_name == "id"
            assert pk_field.max_length == 36

    def test_every_persisted_model_is_tenant_scoped(self) -> None:
        for model in interaction_models.Catalog().all():
            assert "tenant_id" in model._meta.fields_map

    def test_natural_uniqueness_constraints_survive_single_pk_modeling(self) -> None:
        constraints_by_model: Dict[Type[Model], Tuple[Tuple[str, ...], ...]] = {
            interaction_models.MessageRecord: (("tenant_id", "conversation_id", "sequence"),),
            interaction_models.EventRecord: (("tenant_id", "conversation_id", "sequence"),),
            interaction_models.ScriptVersionRecord: (("tenant_id", "script_id", "version"),),
            interaction_models.RequestRecord: (("tenant_id", "key"),),
            interaction_models.SequenceRecord: (("tenant_id", "conversation_id", "scope"),),
        }

        for model, expected in constraints_by_model.items():
            assert model._meta.unique_together == expected

    def test_active_membership_uniqueness_is_schema_owned(self) -> None:
        """
        Keep active membership uniqueness in the partial database index.
        """

        assert interaction_models.MembershipRecord._meta.unique_together == ()

    def test_soft_deleted_public_rows_have_deleted_at_field(self) -> None:
        public_models = (
            interaction_models.ConversationRecord,
            interaction_models.TaskRecord,
            interaction_models.MessageRecord,
            interaction_models.ArtifactRecord,
            interaction_models.ScriptRecord,
        )

        for model in public_models:
            assert "deleted_at" in model._meta.fields_map

    def test_conversation_bound_models_use_conversation_id_column(self) -> None:
        conversation_models = (
            interaction_models.MembershipRecord,
            interaction_models.TaskRecord,
            interaction_models.MessageRecord,
            interaction_models.EventRecord,
            interaction_models.ArtifactRecord,
            interaction_models.ScriptRecord,
            interaction_models.ContextRecord,
            interaction_models.JobRecord,
            interaction_models.SequenceRecord,
        )

        for model in conversation_models:
            field = model._meta.fields_map["conversation"]
            assert field.source_field == "conversation_id"

    def test_only_approved_relations_use_source_field_aliases(self) -> None:
        """
        Verify only safe relationship metadata maps onto existing id columns.
        """

        approved = {
            "script": "script_id",
            "conversation": "conversation_id",
            "execution": "execution_id",
        }

        for model in interaction_models.Catalog().all():
            for name, field in model._meta.fields_map.items():
                assert field.source_field in (None, name, approved.get(name))

    def test_mutable_models_expose_standard_audit_fields(self) -> None:
        """
        Verify mutable tables expose full lifecycle audit columns.
        """

        expected = {
            "created_at",
            "created_by",
            "updated_at",
            "updated_by",
            "deleted_at",
            "deleted_by",
        }
        for model in interaction_models.Catalog().all():
            assert expected <= set(model._meta.fields_map)
