from __future__ import annotations

import unittest
from unittest.mock import patch

from fathom.constants.llm import InferencePriorityMode
from fathom.constants.qualification import DEFAULT_QUALIFIER_MODEL
from fathom.constants.run import TargetKind
from fathom.constants.storage import InteractionBackend, PostgresMigrationMode
from fathom.core.exceptions import StorageConfigurationError
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.schemas.configuration import (
    DeviceConfiguration,
    InteractionStorageConfiguration,
    PostgresInteractionConfiguration,
    QualifierConfiguration,
)
from fathom.schemas.run import (
    IntentObjectiveConfiguration,
    IntentRunRequest,
    Principal,
    ResourceConfiguration,
    RunRequest,
    TargetConfiguration,
)
from fathom.settings.env import FathomSettings


class RunAssemblyBuilderQualifierLLMConfigurationTest(unittest.TestCase):
    """
    Verify qualifier LLM configuration resolution.
    """

    def __principal(self) -> Principal:
        """
        Return a valid principal for runtime request construction.
        """

        return Principal(
            tenant="tenant",
            operator="operator",
            agent="agent:fathom",
            conversation="conversation",
        )

    def test_qualifier_llm_inherits_credentials_from_bound_settings(self) -> None:
        """
        Credentials must come from the settings bound to this assembly builder.
        """

        bound_settings = FathomSettings(
            gemini_api_key="test-api-key",
            vertex_location="bound-location",
            vertex_project_id="bound-project",
            google_application_credentials="/fake/credentials.json",
        )

        assembly = RunAssemblyBuilder(settings=bound_settings)
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration()
        )

        self.assertEqual(configuration.api_key, "test-api-key")
        self.assertEqual(configuration.location, "bound-location")
        self.assertEqual(configuration.project_id, "bound-project")
        self.assertEqual(configuration.credentials, "/fake/credentials.json")

    def test_qualifier_knobs_flow_into_llm_configuration(self) -> None:
        """
        Qualifier inference knobs must reach the LLM configuration.
        """

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration.evolve(thinking_level="minimal"),
        )
        self.assertEqual(configuration.thinking_level, "minimal")

    def test_qualifier_model_defaults_to_constant(self) -> None:
        """
        The qualifier model must not silently inherit the planner model.
        """

        assembly = RunAssemblyBuilder(
            settings=FathomSettings(
                gemini_api_key="x",
                gemini_model="gemini-3.5-flash",
            )
        )
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration()
        )
        self.assertEqual(configuration.model, DEFAULT_QUALIFIER_MODEL)

    def test_qualifier_model_can_be_overridden_via_evolve(self) -> None:
        """
        Callers can override only the qualifier model via evolve().
        """

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration.evolve(model="gemini-2.5-flash"),
        )
        self.assertEqual(configuration.model, "gemini-2.5-flash")

    def test_qualifier_timeout_and_retries_flow_into_llm_configuration(self) -> None:
        """
        Qualifier timeout and retry budget must reach the LLM configuration.
        """

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration.evolve(timeout=3.0, max_retries=4),
        )
        self.assertEqual(configuration.timeout, 3.0)
        self.assertEqual(configuration.max_retries, 4)

    def test_planner_priority_policy_flows_from_settings(self) -> None:
        """
        Planner LLM configuration must inherit priority settings from the bound settings object.
        """

        assembly = RunAssemblyBuilder(
            settings=FathomSettings(
                gemini_api_key="x",
                capacity_enabled=True,
                capacity_mode=InferencePriorityMode.ADAPTIVE.value,
                capacity_window=4,
                capacity_failures=2,
                capacity_slows=3,
                capacity_latency=6.5,
                capacity_recovery=2,
            ),
        )
        configuration = assembly.build_planner_model_configuration(
            request=RunRequest(
                objective=IntentObjectiveConfiguration(intent="Open settings"),
                principal=self.__principal(),
                resources={"targets": [TargetConfiguration()]},
            ),
        )

        self.assertTrue(configuration.priority.enabled)
        self.assertEqual(configuration.priority.mode, InferencePriorityMode.ADAPTIVE)
        self.assertEqual(configuration.priority.adaptive.window, 4)
        self.assertEqual(configuration.priority.adaptive.threshold.failures, 2)
        self.assertEqual(configuration.priority.adaptive.threshold.slows, 3)
        self.assertEqual(configuration.priority.adaptive.threshold.latency, 6.5)
        self.assertEqual(configuration.priority.adaptive.threshold.recovery, 2)

    def test_request_priority_configuration_overrides_settings(self) -> None:
        """
        Run requests may explicitly override planner priority policy without changing environment settings.
        """

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        request = RunRequest(
            objective=IntentObjectiveConfiguration(intent="Open settings"),
            principal=self.__principal(),
            resources={
                "targets": [TargetConfiguration()],
                "language_model_configuration": {
                    "planner_configuration": {
                        "priority": {
                            "enabled": False,
                            "mode": InferencePriorityMode.ALWAYS.value,
                        },
                    },
                },
            },
        )

        configuration = assembly.build_planner_model_configuration(request=request)

        self.assertFalse(configuration.priority.enabled)

    def test_qualifier_priority_policy_flows_from_settings(self) -> None:
        """
        Dedicated qualifier LLM configuration must use the same priority policy as planner.
        """

        assembly = RunAssemblyBuilder(
            settings=FathomSettings(
                gemini_api_key="x",
                capacity_mode=InferencePriorityMode.ADAPTIVE.value,
                capacity_recovery=3,
            ),
        )
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration(),
        )

        self.assertEqual(configuration.priority.mode, InferencePriorityMode.ADAPTIVE)
        self.assertEqual(configuration.priority.adaptive.threshold.recovery, 3)


class InteractionStorageResolutionTest(unittest.TestCase):
    """
    Verify CLI-default and host-supplied interaction storage resolution.
    """

    def __principal(self) -> Principal:
        """
        Return a deterministic Principal for tests.
        """

        return Principal(
            tenant="t",
            operator="u",
            agent="agent:fathom",
            conversation="c",
        )

    def __build_request(
        self,
        *,
        interaction_storage: InteractionStorageConfiguration | None = None,
    ) -> IntentRunRequest:
        """
        Build a minimal valid IntentRunRequest for assembly tests.
        """

        return IntentRunRequest(
            principal=self.__principal(),
            objective=IntentObjectiveConfiguration(intent="x"),
            resources=ResourceConfiguration(
                targets=[
                    TargetConfiguration(
                        kind=TargetKind.DEVICE,
                        device_configuration=DeviceConfiguration(),
                    )
                ],
                interaction_storage=interaction_storage,
            ),
        )

    def test_host_supplied_interaction_storage_wins(self) -> None:
        """
        Host-supplied configuration is returned verbatim.
        """

        host_configuration = InteractionStorageConfiguration(
            backend=InteractionBackend.POSTGRES,
            postgres=PostgresInteractionConfiguration(
                host="example",
                user="fathom",
                password="secret",
                database="fathom",
            ),
        )
        request = self.__build_request(interaction_storage=host_configuration)
        builder = RunAssemblyBuilder(settings=FathomSettings())

        result = builder.build_interaction_storage_configuration(request=request)

        self.assertIs(result, host_configuration)

    def test_cli_default_postgres_requires_dsn(self) -> None:
        """
        Postgres backend without a DSN raises a typed StorageConfigurationError.
        """

        request = self.__build_request()
        settings = FathomSettings(
            interaction_backend="postgres",
            interaction_postgres_host=None,
            interaction_postgres_user=None,
            interaction_postgres_password=None,
        )
        builder = RunAssemblyBuilder(settings=settings)

        with self.assertRaises(StorageConfigurationError) as context:
            builder.build_interaction_storage_configuration(request=request)
        self.assertEqual(InteractionBackend.POSTGRES.value, context.exception.backend)

    def test_cli_default_postgres_uses_worker_environment_settings(self) -> None:
        """
        Worker-side Postgres fallback must include schema and pool settings
        from Fathom env rather than silently defaulting to a different schema.
        """

        request = self.__build_request()
        settings = FathomSettings(
            interaction_backend="postgres",
            interaction_postgres_host="localhost",
            interaction_postgres_user="fathom",
            interaction_postgres_password="secret",
            interaction_postgres_database="fathom",
            interaction_postgres_schema="conversation",
            interaction_postgres_pool_min_size=1,
            interaction_postgres_pool_max_size=4,
            interaction_postgres_statement_timeout=2500,
            interaction_postgres_migration_mode=PostgresMigrationMode.VALIDATE,
        )
        builder = RunAssemblyBuilder(settings=settings)

        result = builder.build_interaction_storage_configuration(request=request)

        self.assertEqual(InteractionBackend.POSTGRES, result.backend)
        assert result.postgres is not None
        self.assertEqual("conversation", result.postgres.schema_name)
        self.assertEqual(1, result.postgres.pool_min_size)
        self.assertEqual(4, result.postgres.pool_max_size)
        self.assertEqual(2500, result.postgres.statement_timeout)
        self.assertEqual(PostgresMigrationMode.VALIDATE, result.postgres.migration_mode)

    def test_cli_default_postgres_infers_drizz_worker_environment(self) -> None:
        """
        DRIZZ_FATHOM_POSTGRES_* is the canonical worker env set.
        """

        request = self.__build_request()
        with patch.dict(
            "os.environ",
            {
                "DRIZZ_FATHOM_POSTGRES_HOST": "localhost",
                "DRIZZ_FATHOM_POSTGRES_PORT": "5433",
                "DRIZZ_FATHOM_POSTGRES_USER": "fathom",
                "DRIZZ_FATHOM_POSTGRES_PASSWORD": "secret",
                "DRIZZ_FATHOM_POSTGRES_DATABASE": "fathom",
                "DRIZZ_FATHOM_POSTGRES_SCHEMA": "fathom",
                "DRIZZ_FATHOM_POSTGRES_POOL_MIN_SIZE": "2",
                "DRIZZ_FATHOM_POSTGRES_POOL_MAX_SIZE": "8",
                "DRIZZ_FATHOM_POSTGRES_STATEMENT_TIMEOUT": "3000",
                "DRIZZ_FATHOM_POSTGRES_MIGRATION_MODE": "validate",
            },
            clear=True,
        ):
            settings = FathomSettings(_env_file=None)
        builder = RunAssemblyBuilder(settings=settings)

        result = builder.build_interaction_storage_configuration(request=request)

        self.assertEqual(InteractionBackend.POSTGRES, result.backend)
        assert result.postgres is not None
        self.assertEqual("localhost", result.postgres.host)
        self.assertEqual(5433, result.postgres.port)
        self.assertEqual("fathom", result.postgres.database)
        self.assertEqual("fathom", result.postgres.schema_name)
        self.assertEqual(2, result.postgres.pool_min_size)
        self.assertEqual(8, result.postgres.pool_max_size)
        self.assertEqual(3000, result.postgres.statement_timeout)
        self.assertEqual(PostgresMigrationMode.VALIDATE, result.postgres.migration_mode)

    def test_cli_default_postgres_uses_dsn_when_settings_provide_one(self) -> None:
        """
        When DRIZZ_FATHOM_POSTGRES_DSN is set the CLI assembly builds a
        DSN-mode PostgresInteractionConfiguration; host/user/password are not
        required and are absent from the resulting config.
        """

        request = self.__build_request()
        settings = FathomSettings(
            interaction_backend="postgres",
            interaction_postgres_dsn="postgresql://fathom:s%2Bcret@db.local:5432/fathom?sslmode=require",
            interaction_postgres_schema="conversation",
            interaction_postgres_pool_min_size=1,
            interaction_postgres_pool_max_size=4,
            interaction_postgres_statement_timeout=2500,
        )
        builder = RunAssemblyBuilder(settings=settings)

        result = builder.build_interaction_storage_configuration(request=request)

        self.assertEqual(InteractionBackend.POSTGRES, result.backend)
        assert result.postgres is not None
        self.assertEqual(
            "postgresql://fathom:s%2Bcret@db.local:5432/fathom?sslmode=require",
            result.postgres.dsn,
        )
        self.assertIsNone(result.postgres.host)
        self.assertIsNone(result.postgres.user)
        self.assertIsNone(result.postgres.password)
        self.assertEqual("conversation", result.postgres.schema_name)
        self.assertEqual(1, result.postgres.pool_min_size)
        self.assertEqual(4, result.postgres.pool_max_size)
        self.assertEqual(2500, result.postgres.statement_timeout)

    def test_cli_default_postgres_dsn_env_alone_selects_postgres_backend(self) -> None:
        """
        A DSN in the environment is sufficient signal for the backend resolver
        to pick Postgres; discrete host/user/password are no longer required.
        """

        request = self.__build_request()
        with patch.dict(
            "os.environ",
            {
                "DRIZZ_FATHOM_POSTGRES_DSN": "postgresql://fathom:secret@db.local:5432/fathom",
            },
            clear=True,
        ):
            settings = FathomSettings(_env_file=None)
        builder = RunAssemblyBuilder(settings=settings)

        result = builder.build_interaction_storage_configuration(request=request)

        self.assertEqual(InteractionBackend.POSTGRES, result.backend)
        assert result.postgres is not None
        self.assertEqual(
            "postgresql://fathom:secret@db.local:5432/fathom",
            result.postgres.dsn,
        )

    def test_invalid_backend_setting_raises_typed_error(self) -> None:
        """
        Unknown FATHOM_INTERACTION_BACKEND values fail with a typed error.
        """

        request = self.__build_request()
        settings = FathomSettings(interaction_backend="cassandra")
        builder = RunAssemblyBuilder(settings=settings)

        with self.assertRaises(StorageConfigurationError):
            builder.build_interaction_storage_configuration(request=request)
