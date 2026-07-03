from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fathom.constants.llm import InferencePriorityMode
from fathom.constants.run import TargetKind
from fathom.constants.storage import InteractionBackend
from fathom.core.exceptions import ConfigurationError, StorageConfigurationError
from fathom.schemas.base.common import ThresholdConfiguration
from fathom.schemas.configuration import (
    AdaptivePriorityConfiguration,
    DeviceConfiguration,
    InteractionStorageConfiguration,
    LLMConfiguration,
    NoopInteractionConfiguration,
    PostgresInteractionConfiguration,
    PriorityInferenceConfiguration,
    QualifierConfiguration,
    StorageConfiguration,
    TelemetryConfiguration,
)
from fathom.schemas.run import RunRequest, TargetConfiguration
from fathom.settings.env import FathomSettings

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager


class RunAssemblyBuilder:
    """
    Build runtime adapter configurations from a canonical run request.
    """

    def __init__(self, *, settings: FathomSettings) -> None:
        """
        Initialize the builder with environment-backed defaults.
        """

        self.__settings = settings

    def build_primary_target(self, *, request: RunRequest) -> TargetConfiguration:
        """
        Resolve the single target supported by the current runtime.
        """

        targets = request.resources.targets

        if len(targets) != 1:
            raise ConfigurationError("Current runtime supports exactly one target per run request")

        target = targets[0]
        if target.kind != TargetKind.DEVICE:
            raise ConfigurationError(
                f"Target kind {target.kind.value} is not supported by the current runtime"
            )

        return target

    def build_device_configuration(self, *, request: RunRequest) -> DeviceConfiguration:
        """
        Resolve the primary device configuration from the run request.
        """

        target = self.build_primary_target(request=request)
        return target.device_configuration

    def build_planner_model_configuration(self, *, request: RunRequest) -> LLMConfiguration:
        """
        Resolve planner model configuration using request values with environment defaults.
        """

        credentials = (
            self.__settings.google_credentials_dict
            or self.__settings.google_application_credentials
        )
        request_values = (
            request.resources.language_model_configuration.planner_configuration.model_dump(
                exclude_none=True,
                exclude_unset=True,
            )
        )
        default_values = {
            "credentials": credentials,
            "model": self.__settings.gemini_model,
            "api_key": self.__settings.gemini_api_key,
            "location": self.__settings.vertex_location,
            "priority": self.__priority_configuration(),
            "project_id": self.__settings.vertex_project_id,
            "use_cache": getattr(self.__settings, "use_cache", True),
        }
        default_values.update(request_values)

        return LLMConfiguration.model_validate(default_values)

    def build_qualifier_model_configuration(
        self, *, configuration: QualifierConfiguration
    ) -> LLMConfiguration:
        """
        Resolve qualifier model configuration from settings and the qualifier knobs.
        """

        credentials = (
            self.__settings.google_credentials_dict
            or self.__settings.google_application_credentials
        )
        return LLMConfiguration.model_validate(
            {
                "credentials": credentials,
                "model": configuration.inference.model,
                "api_key": self.__settings.gemini_api_key,
                "timeout": configuration.inference.timeout,
                "location": self.__settings.vertex_location,
                "use_cache": configuration.inference.use_cache,
                "project_id": self.__settings.vertex_project_id,
                "temperature": configuration.inference.temperature,
                "max_retries": configuration.inference.max_retries,
                "thinking_level": configuration.inference.thinking_level,
                "priority": self.__priority_configuration(),
            }
        )

    def __priority_configuration(self) -> PriorityInferenceConfiguration:
        """
        Resolve provider-neutral elevated-capacity inference policy from settings.
        """

        return PriorityInferenceConfiguration(
            enabled=self.__settings.capacity_enabled,
            mode=InferencePriorityMode(self.__settings.capacity_mode),
            adaptive=AdaptivePriorityConfiguration(
                window=self.__settings.capacity_window,
                threshold=ThresholdConfiguration(
                    slows=self.__settings.capacity_slows,
                    latency=self.__settings.capacity_latency,
                    failures=self.__settings.capacity_failures,
                    recovery=self.__settings.capacity_recovery,
                ),
            ),
        )

    def build_storage_configuration(self, *, request: RunRequest) -> StorageConfiguration:
        """
        Resolve storage configuration using request values with environment defaults.
        """

        credentials = (
            self.__settings.google_credentials_dict
            or self.__settings.google_application_credentials
        )
        request_values = request.resources.storage_configuration.model_dump(exclude_none=True)

        default_values = {
            "credentials": credentials,
            "project_id": self.__settings.vertex_project_id,
        }
        default_values.update(request_values)

        return StorageConfiguration.model_validate(default_values)

    def build_interaction_storage_configuration(
        self,
        *,
        request: RunRequest,
        path_manager: Optional[SharedPathManager] = None,
    ) -> InteractionStorageConfiguration:
        """
        Resolve interaction storage configuration for the run.

        Host-supplied values on the wire win. When absent (CLI runs),
        fall back to FathomSettings and build the configured durable backend.
        """

        if request.resources.interaction_storage is not None:
            return request.resources.interaction_storage

        backend = self.__resolve_backend()

        if backend == InteractionBackend.POSTGRES:
            return self.__cli_postgres_configuration()

        return InteractionStorageConfiguration(
            backend=InteractionBackend.NOOP,
            noop=NoopInteractionConfiguration(),
        )

    def __resolve_backend(self) -> InteractionBackend:
        """
        Resolve the CLI-default interaction backend from settings.
        """

        backend = self.__settings.resolved_interaction_backend

        try:
            return InteractionBackend(backend)
        except ValueError as exception:
            raise StorageConfigurationError(
                backend=backend,
                message=(
                    f"DRIZZ_FATHOM_INTERACTION_BACKEND must be one of "
                    f"{[backend.value for backend in InteractionBackend]}; got '{backend}'"
                ),
            ) from exception

    def __cli_postgres_configuration(self) -> InteractionStorageConfiguration:
        """
        Build a CLI-default Postgres interaction storage configuration.

        Picks `DRIZZ_FATHOM_POSTGRES_DSN` when set; otherwise requires the discrete host/user/password triple.
        The pool, schema and tunable fields are mode-independent and applied in both branches.
        """

        dsn = self.__settings.interaction_postgres_dsn

        if dsn:
            return InteractionStorageConfiguration(
                backend=InteractionBackend.POSTGRES,
                postgres=PostgresInteractionConfiguration(
                    dsn=dsn,
                    ssl=self.__settings.interaction_postgres_ssl,
                    schema_name=self.__settings.interaction_postgres_schema,
                    pool_min_size=self.__settings.interaction_postgres_pool_min_size,
                    pool_max_size=self.__settings.interaction_postgres_pool_max_size,
                    migration_mode=self.__settings.interaction_postgres_migration_mode,
                    statement_timeout=self.__settings.interaction_postgres_statement_timeout,
                ),
            )

        host = self.__settings.interaction_postgres_host
        user = self.__settings.interaction_postgres_user
        password = self.__settings.interaction_postgres_password

        if not host or not user or not password:
            raise StorageConfigurationError(
                backend=InteractionBackend.POSTGRES.value,
                message=(
                    "DRIZZ_FATHOM_POSTGRES_DSN, or all of "
                    "DRIZZ_FATHOM_POSTGRES_HOST, "
                    "DRIZZ_FATHOM_POSTGRES_USER, and "
                    "DRIZZ_FATHOM_POSTGRES_PASSWORD, are required when the interaction backend resolves to postgres"
                ),
            )
        return InteractionStorageConfiguration(
            backend=InteractionBackend.POSTGRES,
            postgres=PostgresInteractionConfiguration(
                host=host,
                user=user,
                password=password,
                ssl=self.__settings.interaction_postgres_ssl,
                port=self.__settings.interaction_postgres_port,
                database=self.__settings.interaction_postgres_database,
                schema_name=self.__settings.interaction_postgres_schema,
                pool_min_size=self.__settings.interaction_postgres_pool_min_size,
                pool_max_size=self.__settings.interaction_postgres_pool_max_size,
                migration_mode=self.__settings.interaction_postgres_migration_mode,
                statement_timeout=self.__settings.interaction_postgres_statement_timeout,
            ),
        )

    def build_telemetry_configuration(
        self,
        *,
        request: RunRequest,
        workflow_id: Optional[str] = None,
    ) -> TelemetryConfiguration:
        """
        Resolve runtime telemetry configuration from the run request.
        """

        stream_connection_string = (
            request.telemetry.stream_connection_string or self.__settings.redis_url
        )
        session_id = request.runtime.session_id
        execution_id = request.runtime.execution_id or workflow_id or session_id

        if stream_connection_string:
            return TelemetryConfiguration(
                type="REDIS",
                identity=execution_id,
                session_id=session_id,
                connection_string=stream_connection_string,
                topic="enricher:commands:v1:logs:{session_id}",
            )

        return TelemetryConfiguration(type="STRUCTLOG")
