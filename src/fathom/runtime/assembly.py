from __future__ import annotations

from typing import Optional

from fathom.constants.run import TargetKind
from fathom.core.exceptions import ConfigurationError
from fathom.schemas.configuration import (
    DeviceConfiguration,
    LLMConfiguration,
    QualifierConfiguration,
    StorageConfiguration,
    TelemetryConfiguration,
)
from fathom.schemas.run import RunRequest, TargetConfiguration
from fathom.settings.env import FathomSettings


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
                exclude_none=True
            )
        )
        default_values = {
            "credentials": credentials,
            "model": self.__settings.gemini_model,
            "api_key": self.__settings.gemini_api_key,
            "location": self.__settings.vertex_location,
            "project_id": self.__settings.vertex_project_id,
            "use_cache": getattr(self.__settings, "use_cache", True),
        }
        default_values.update(request_values)
        return LLMConfiguration.model_validate(default_values)

    def build_qualifier_model_configuration(
        self, *, configuration: QualifierConfiguration
    ) -> LLMConfiguration:
        """
        Resolve qualifier model configuration from settings and the qualifier configuration.
        """

        credentials = (
            self.__settings.google_credentials_dict
            or self.__settings.google_application_credentials
        )
        return LLMConfiguration.model_validate(
            {
                "credentials": credentials,
                "use_cache": configuration.use_cache,
                "model": self.__settings.gemini_model,
                "temperature": configuration.temperature,
                "api_key": self.__settings.gemini_api_key,
                "location": self.__settings.vertex_location,
                "thinking_level": configuration.thinking_level,
                "project_id": self.__settings.vertex_project_id,
            }
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
