from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from temporalio import activity

from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.signal.temporal import TemporalSignalAdapter
from fathom.base.paths import SharedPathManager
from fathom.interfaces.signal import SignalPort
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.runtime.builder import Fathom
from fathom.runtime.factories import (
    DeviceFactory,
    LLMFactory,
    PerceptionFactory,
    StorageFactory,
    TelemetryFactory,
)
from fathom.schemas.run import ExplorationRunRequest, IntentRunRequest, RunRequest
from fathom.settings.env import FathomSettings

if TYPE_CHECKING:
    from fathom.runtime.runner import FathomRunner

logger = getLogger(__name__)


class FathomActivities:
    """
    Temporal activities implementation for Fathom agent workflows.
    """

    def __init__(self, settings: FathomSettings | None = None) -> None:
        """
        Initialize activities with runtime settings.
        """

        self.__settings = settings or FathomSettings()
        self.__assembly = RunAssemblyBuilder(settings=self.__settings)

    def __validate_intent_request(self, *, request: dict[str, object]) -> IntentRunRequest:
        """
        Validate an intent run payload.
        """

        return IntentRunRequest.model_validate(request)

    def __validate_exploration_request(
        self,
        *,
        request: dict[str, object],
    ) -> ExplorationRunRequest:
        """
        Validate an exploration run payload.
        """

        return ExplorationRunRequest.model_validate(request)

    def __create_signal_adapter(self, *, workflow_id: str, request: RunRequest) -> SignalPort:
        """
        Create the signal adapter for the current workflow host.
        """

        if not request.runtime.interactive:
            return NoopSignal()

        temporal_host = self.__settings.temporal_host
        if not temporal_host:
            raise ValueError("temporal_host is required for interactive mode but is not configured")

        return TemporalSignalAdapter(
            workflow_id=workflow_id,
            target_host=str(temporal_host),
            api_key=self.__settings.temporal_api_key,
            namespace=activity.info().workflow_namespace,
        )

    def __build_runner(self, *, workflow_id: str, request: RunRequest) -> FathomRunner:
        """
        Build the Fathom runner for the validated run request.
        """

        device_configuration = self.__assembly.build_device_configuration(request=request)
        planner_configuration = self.__assembly.build_planner_model_configuration(request=request)
        storage_configuration = self.__assembly.build_storage_configuration(request=request)
        telemetry_configuration = self.__assembly.build_telemetry_configuration(
            request=request,
            workflow_id=workflow_id,
        )

        signal_adapter = self.__create_signal_adapter(
            request=request,
            workflow_id=workflow_id,
        )
        path_manager = SharedPathManager(settings=self.__settings)

        device_adapter = DeviceFactory().create(configuration=device_configuration)
        perception_adapter = PerceptionFactory().create(
            configuration=device_configuration,
            device=device_adapter,
            use_xml=request.objective.use_xml,
        )

        llm_adapter = LLMFactory().create(configuration=planner_configuration)
        telemetry_adapter = TelemetryFactory().create(configuration=telemetry_configuration)
        storage_adapter = StorageFactory().create(
            path_manager=path_manager,
            configuration=storage_configuration,
        )

        return (
            Fathom.builder(path_manager=path_manager)
            .with_llm(port=llm_adapter)
            .with_device(port=device_adapter)
            .with_signal(port=signal_adapter)
            .with_storage(port=storage_adapter)
            .with_telemetry(port=telemetry_adapter)
            .with_perception(port=perception_adapter)
            .with_realignment(policy=request.interaction.realignment)
            .with_intent_config(configuration=request.interaction.intent_configuration)
            .with_execution_config(configuration=request.interaction.execution_configuration)
            .with_exploration_config(configuration=request.interaction.exploration_configuration)
            .build()
        )

    @activity.defn(name="EXECUTE_INTENT")  # type: ignore[untyped-decorator]
    async def execute_intent(
        self,
        workflow_id: str,
        request: dict[str, object],
    ) -> dict[str, object]:
        """
        Execute an intent run.
        """

        activity.logger.info("Starting Fathom intent execution for workflow %s", workflow_id)
        validated_request = self.__validate_intent_request(request=request)
        runner = self.__build_runner(workflow_id=workflow_id, request=validated_request)

        try:
            activity.heartbeat("Starting execution")

            package_name = (
                validated_request.objective.package_name
                or await runner.device.get_current_package()
            )

            result = await runner.run_intent(
                request_id=workflow_id,
                package_name=package_name,
                intent=validated_request.objective.intent,
                use_xml=validated_request.objective.use_xml,
                max_steps=validated_request.objective.max_steps,
                context_scope=validated_request.memory.context_scope,
                realignment=validated_request.interaction.realignment,
                conversation_id=validated_request.memory.conversation_id,
            )
            activity.heartbeat(f"Completed: {result.steps_taken} steps")
            return {
                "error": result.error,
                "success": result.success,
                "steps": result.steps_taken,
                "duration": result.duration,
                "metrics": result.metrics if result.metrics else None,
            }
        except Exception as exception:
            activity.logger.exception("Fathom execution failed: %s", exception)
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
            }
        finally:
            await runner.cleanup()

    @activity.defn(name="EXECUTE_EXPLORATION")  # type: ignore[untyped-decorator]
    async def execute_exploration(
        self,
        workflow_id: str,
        request: dict[str, object],
    ) -> dict[str, object]:
        """
        Execute an exploration run.
        """

        activity.logger.info("Starting Fathom exploration for workflow %s", workflow_id)
        validated_request = self.__validate_exploration_request(request=request)
        runner = self.__build_runner(workflow_id=workflow_id, request=validated_request)

        try:
            activity.heartbeat("Starting exploration")
            package_name = (
                validated_request.objective.package_name
                or await runner.device.get_current_package()
            )
            result = await runner.run_exploration(
                request_id=workflow_id,
                package_name=package_name,
                max_steps=validated_request.objective.max_steps,
            )
            activity.heartbeat(f"Completed: {result.steps_executed} steps")
            return {
                "metrics": None,
                "error": result.error,
                "success": result.success,
                "duration": result.duration,
                "steps": result.steps_executed,
            }
        except Exception as exception:
            activity.logger.exception("Exploration failed: %s", exception)
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
            }
        finally:
            await runner.cleanup()
