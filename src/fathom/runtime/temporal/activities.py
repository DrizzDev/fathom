from __future__ import annotations

import asyncio
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

from temporalio import activity

from fathom.base.paths import SharedPathManager
from fathom.constants import FathomEvent
from fathom.infrastructure.temporal.state import SignalStateRegistry
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.telemetry import TelemetryLevel
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.runtime.builder import Fathom
from fathom.runtime.factories import (
    DeviceFactory,
    LLMFactory,
    PerceptionFactory,
    SignalFactory,
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

    def __init__(self, settings: Optional[FathomSettings] = None) -> None:
        """
        Initialize activities with runtime settings.
        """

        self.__settings = settings or FathomSettings()
        self.__assembly = RunAssemblyBuilder(settings=self.__settings)

    def __validate_intent_request(self, *, request: Dict[str, Any]) -> IntentRunRequest:
        """
        Validate an intent run payload.
        """

        return IntentRunRequest.model_validate(request)

    def __validate_exploration_request(self, *, request: Dict[str, Any]) -> ExplorationRunRequest:
        """
        Validate an exploration run payload.
        """

        return ExplorationRunRequest.model_validate(request)

    def __create_signal_adapter(self, *, workflow_id: str, request: RunRequest) -> SignalPort:
        """
        Create the signal adapter for the current workflow host via factory.
        """

        adapter = SignalFactory().create(
            workflow_id=workflow_id,
            signal_type=request.runtime.signal_type,
            interactive=request.runtime.interactive,
        )

        logger.info(
            f"[activity] workflow={workflow_id} event=signal_adapter "
            f"adapter={type(adapter).__name__} interactive={request.runtime.interactive} "
            f"signal_type={request.runtime.signal_type}"
        )

        return adapter

    def __build_runner(self, *, workflow_id: str, request: RunRequest) -> "FathomRunner":
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
            device=device_adapter,
            use_xml=request.objective.use_xml,
            configuration=device_configuration,
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

    async def __cleanup_runner(self, *, runner: "FathomRunner") -> None:
        """
        Cleanup runner resources.
        """

        await runner.cleanup()

    async def __cancel_runner(
        self,
        *,
        message: str,
        level: TelemetryLevel,
        runner: "FathomRunner",
        event_type: FathomEvent,
    ) -> None:
        """
        Cancel the runner and emit a client-facing message.
        """

        await runner.notify(level=level, message=message, event_type=event_type)
        runner.cancel()

    @activity.defn(name="EXECUTE_INTENT")  # type: ignore[untyped-decorator]
    async def execute_intent(self, workflow_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an intent run.
        """

        activity.logger.info(
            f"[activity] workflow={workflow_id} activity=EXECUTE_INTENT phase=starting"
        )

        validated_request = self.__validate_intent_request(request=request)

        activity.logger.info(
            f"[activity] workflow={workflow_id} activity=EXECUTE_INTENT "
            f'intent="{validated_request.objective.intent}" '
            f"max_steps={validated_request.objective.max_steps} "
            f"interactive={validated_request.runtime.interactive}"
        )

        runner = self.__build_runner(workflow_id=workflow_id, request=validated_request)

        try:
            activity.heartbeat("Starting execution")

            package_name = (
                validated_request.objective.package_name
                or await runner.device.get_current_package()
            )

            activity.logger.info(
                f"[activity] workflow={workflow_id} activity=EXECUTE_INTENT phase=executing package={package_name}"
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

            activity.logger.info(
                f"[activity] workflow={workflow_id} activity=EXECUTE_INTENT phase=completed "
                f"success={result.success} steps={result.steps_taken} duration_ms={result.duration}"
            )

            activity.heartbeat(f"Completed: {result.steps_taken} steps")
            return {
                "error": result.error,
                "success": result.success,
                "steps": result.steps_taken,
                "duration": result.duration,
                "metrics": result.metrics if result.metrics else None,
            }
        except asyncio.CancelledError:
            activity.logger.warning(
                f"[activity] workflow={workflow_id} activity=EXECUTE_INTENT phase=cancelled"
            )
            await self.__cancel_runner(
                runner=runner,
                level="warning",
                message=(
                    "Workflow execution was cancelled. Cleaning up resources and closing "
                    "active connections."
                ),
                event_type=FathomEvent.WORKFLOW_CANCELLED,
            )
            raise
        except Exception as exception:
            activity.logger.exception(
                f'[activity] workflow={workflow_id} activity=EXECUTE_INTENT phase=failed error="{exception}"'
            )
            runner.cancel()
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
            }
        finally:
            activity.logger.info(
                f"[activity] workflow={workflow_id} activity=EXECUTE_INTENT phase=cleanup"
            )
            await self.__cleanup_runner(runner=runner)
            SignalStateRegistry.shared().release(workflow_id=workflow_id)

    @activity.defn(name="EXECUTE_EXPLORATION")  # type: ignore[untyped-decorator]
    async def execute_exploration(
        self, workflow_id: str, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an exploration run.
        """

        activity.logger.info(
            f"[activity] workflow={workflow_id} activity=EXECUTE_EXPLORATION phase=starting"
        )

        validated_request = self.__validate_exploration_request(request=request)

        activity.logger.info(
            f"[activity] workflow={workflow_id} activity=EXECUTE_EXPLORATION "
            f"max_steps={validated_request.objective.max_steps} "
            f"interactive={validated_request.runtime.interactive}"
        )

        runner = self.__build_runner(workflow_id=workflow_id, request=validated_request)

        try:
            activity.heartbeat("Starting exploration")

            package_name = (
                validated_request.objective.package_name
                or await runner.device.get_current_package()
            )

            activity.logger.info(
                f"[activity] workflow={workflow_id} activity=EXECUTE_EXPLORATION phase=executing package={package_name}"
            )

            result = await runner.run_exploration(
                request_id=workflow_id,
                package_name=package_name,
                max_steps=validated_request.objective.max_steps,
            )

            activity.logger.info(
                f"[activity] workflow={workflow_id} activity=EXECUTE_EXPLORATION phase=completed "
                f"success={result.success} steps={result.steps_executed} duration_ms={result.duration}"
            )

            activity.heartbeat(f"Completed: {result.steps_executed} steps")
            return {
                "metrics": None,
                "error": result.error,
                "success": result.success,
                "duration": result.duration,
                "steps": result.steps_executed,
            }
        except asyncio.CancelledError:
            activity.logger.warning(
                f"[activity] workflow={workflow_id} activity=EXECUTE_EXPLORATION phase=cancelled"
            )
            await self.__cancel_runner(
                runner=runner,
                level="warning",
                message=(
                    "Workflow execution was cancelled. Cleaning up resources and closing "
                    "active connections."
                ),
                event_type=FathomEvent.WORKFLOW_CANCELLED,
            )
            raise
        except Exception as exception:
            activity.logger.exception(
                f'[activity] workflow={workflow_id} activity=EXECUTE_EXPLORATION phase=failed error="{exception}"'
            )
            runner.cancel()
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
            }
        finally:
            activity.logger.info(
                f"[activity] workflow={workflow_id} activity=EXECUTE_EXPLORATION phase=cleanup"
            )
            await self.__cleanup_runner(runner=runner)
            SignalStateRegistry.shared().release(workflow_id=workflow_id)
