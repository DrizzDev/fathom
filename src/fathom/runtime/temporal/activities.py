from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional, cast

from temporalio import activity

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.adapters.signal.temporal import TemporalSignalAdapter
from fathom.interfaces.signal import SignalPort
from fathom.runtime.builder import Fathom
from fathom.runtime.factories import DeviceFactory, TelemetryFactory
from fathom.schemas.configuration import (
    DeviceConfiguration,
    ExecutionConfiguration,
    ExplorationConfiguration,
    IntentConfiguration,
    LLMConfiguration,
    TelemetryConfiguration,
)

logger = getLogger(__name__)


class FathomActivities:
    """
    Temporal activities implementation for Fathom agent tasks.
    Encapsulates execution logic for intent and exploration workflows.
    """

    @activity.defn(name="EXECUTE_INTENT")  # type: ignore[untyped-decorator]
    async def execute_intent(
        self,
        workflow_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute an intent-based workflow.

        Args:
            request: Activity input parameters.
            workflow_id: ID of the parent workflow.

        Returns:
            Execution results.
        """

        activity.logger.info(f"Starting Fathom intent execution for workflow {workflow_id}")

        configuration = self.__build_configurations(workflow_id=workflow_id, request=request)

        runner = self.__build_runner(
            workflow_id=workflow_id,
            llm_configuration=configuration["llm"],
            device_configuration=configuration["device"],
            intent_configuration=configuration["intent"],
            interactive=request.get("interactive", True),
            execution_configuration=configuration["engine"],
            telemetry_configuration=configuration["telemetry"],
            exploration_configuration=configuration["exploration"],
            realignment=cast("Optional[Dict[str, Any]]", request.get("realignment")),
        )

        try:
            activity.heartbeat("Starting execution")

            # Fetch package name for accurate tracing/storage
            try:
                package_name = await runner.device.get_current_package()
            except Exception:
                package_name = "unknown_app"

            result = await runner.run_intent(
                request_id=workflow_id,
                intent=request["intent"],
                package_name=package_name,
                use_xml=request.get("use_xml", False),
                max_steps=request.get("max_steps", 100),
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
            activity.logger.exception(f"Fathom execution failed: {exception}")
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
            }

        finally:
            # Telemetry cleanup is handled by the adapter itself or GC in this simple case
            await runner.cleanup()

    @activity.defn(name="EXECUTE_EXPLORATION")  # type: ignore[untyped-decorator]
    async def execute_exploration(
        self,
        workflow_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute an autonomous exploration workflow.

        Args:
            request: Activity input parameters.
            workflow_id: ID of the parent workflow.

        Returns:
            Exploration results.
        """

        activity.logger.info(f"Starting Fathom exploration for workflow {workflow_id}")

        configuration = self.__build_configurations(workflow_id=workflow_id, request=request)

        runner = self.__build_runner(
            workflow_id=workflow_id,
            llm_configuration=configuration["llm"],
            device_configuration=configuration["device"],
            intent_configuration=configuration["intent"],
            interactive=request.get("interactive", True),
            execution_configuration=configuration["engine"],
            telemetry_configuration=configuration["telemetry"],
            exploration_configuration=configuration["exploration"],
            realignment=cast("Optional[Dict[str, Any]]", request.get("realignment")),
        )

        try:
            activity.heartbeat("Starting exploration")

            result = await runner.run_exploration(
                max_steps=request.get("max_steps", 100), request_id=workflow_id
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
            activity.logger.exception(f"Exploration failed: {exception}")
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
            }

        finally:
            await runner.cleanup()

    def __build_configurations(self, workflow_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs configuration objects from request dictionary.
        """

        # 1. LLM Configuration Merging (Legacy + New)
        llm_request_configuration = request.get("llm_config", {})
        planner_configuration = request.get("planner_configuration", {})

        # Populate parameters from legacy planner first
        llm_parameters: Dict[str, Any] = {
            "location": planner_configuration.get("location"),
            "project_id": planner_configuration.get("project_id"),
            "credentials": planner_configuration.get("credentials"),
            "use_cache": planner_configuration.get("use_cache", True),
            "model": planner_configuration.get("model", "gemini-3-flash-preview"),
        }

        llm_parameters.update(llm_request_configuration)
        llm_parameters = {key: value for key, value in llm_parameters.items() if value is not None}

        llm_configuration = LLMConfiguration(**llm_parameters)

        # 2. Device Configuration
        session_id = request.get("session_id", "default_session")
        identity = request.get("identity") or workflow_id
        execution_id = request.get("execution_id") or workflow_id

        if enricher_url := request.get("enricher_url"):
            device_configuration = DeviceConfiguration(
                type="REMOTE",
                session_id=session_id,
                execution_id=execution_id,
                provider_url=enricher_url,
                authentication_token=request.get("auth_token"),
            )
        else:
            device_configuration = DeviceConfiguration(type="LOCAL", serial_number=session_id)

        if redis_url := request.get("redis_url"):
            telemetry_configuration = TelemetryConfiguration(
                type="REDIS",
                identity=identity,
                session_id=session_id,
                connection_string=redis_url,
                topic="enricher:commands:v1:logs:{session_id}",
            )
        else:
            telemetry_configuration = TelemetryConfiguration(type="STRUCTLOG")

        return {
            "llm": llm_configuration,
            "device": device_configuration,
            "telemetry": telemetry_configuration,
            "intent": IntentConfiguration(**(request.get("intent_config", {}))),
            "engine": ExecutionConfiguration(**(request.get("execution_config", {}))),
            "exploration": ExplorationConfiguration(**(request.get("exploration_config", {}))),
        }

    def __build_runner(
        self,
        workflow_id: str,
        llm_configuration: LLMConfiguration,
        device_configuration: DeviceConfiguration,
        intent_configuration: IntentConfiguration,
        telemetry_configuration: TelemetryConfiguration,
        execution_configuration: ExecutionConfiguration,
        exploration_configuration: ExplorationConfiguration,
        *,
        interactive: bool = True,
        realignment: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Instantiates the Fathom runner with configured adapters.
        """

        from fathom.adapters.signal.noop import NoopSignal
        from fathom.schemas.orchestration import RealignmentPolicy

        if interactive:
            signal_adapter: SignalPort = TemporalSignalAdapter(
                workflow_id=workflow_id, namespace=activity.info().namespace
            )
        else:
            signal_adapter: SignalPort = NoopSignal()

        device_adapter = DeviceFactory.create(configuration=device_configuration)
        telemetry_adapter = TelemetryFactory.create(configuration=telemetry_configuration)

        builder = (
            Fathom.builder()
            .with_device(port=device_adapter)
            .with_signal(port=signal_adapter)
            .with_telemetry(port=telemetry_adapter)
            .with_intent_config(configuration=intent_configuration)
            .with_llm(port=GeminiLLM(configuration=llm_configuration))
            .with_execution_config(configuration=execution_configuration)
            .with_exploration_config(configuration=exploration_configuration)
        )

        if realignment:
            builder.with_realignment(policy=RealignmentPolicy(**realignment))

        return builder.build()
