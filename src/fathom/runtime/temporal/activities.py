from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from temporalio import activity

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.adapters.signal.temporal import TemporalSignalAdapter
from fathom.runtime.builder import Fathom
from fathom.runtime.factories import DeviceFactory
from fathom.schemas.configuration import DeviceConfiguration, LLMConfiguration

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

        configuration = self.__build_configurations(request=request)

        runner = self.__build_runner(
            workflow_id=workflow_id,
            llm_configuration=configuration["llm"],
            device_configuration=configuration["device"],
        )

        try:
            activity.heartbeat("Starting execution")

            result = await runner.run_intent(
                request_id=workflow_id,
                intent=request["intent"],
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

        configuration = self.__build_configurations(request=request)

        runner = self.__build_runner(
            workflow_id=workflow_id,
            llm_configuration=configuration["llm"],
            device_configuration=configuration["device"],
        )

        try:
            activity.heartbeat("Starting exploration")

            result = await runner.run_exploration(
                max_steps=request.get("max_steps", 50), request_id=workflow_id
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

    def __build_configurations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs configuration objects from request dictionary.
        """

        planner_configuration = request.get("planner_configuration", {})

        llm_configuration = LLMConfiguration(
            location=planner_configuration.get("location"),
            project_id=planner_configuration.get("project_id"),
            credentials=planner_configuration.get("credentials"),
            model=planner_configuration.get("model", "gemini-2.0-flash-exp"),
        )

        enricher_url = request.get("enricher_url")
        session_id = request.get("session_id", "default_session")

        if enricher_url:
            device_configuration = DeviceConfiguration(
                type="REMOTE",
                session_id=session_id,
                provider_url=enricher_url,
                authentication_token=request.get("auth_token"),
            )
        else:
            device_configuration = DeviceConfiguration(type="LOCAL", serial_number=session_id)

        return {"device": device_configuration, "llm": llm_configuration}

    def __build_runner(
        self,
        workflow_id: str,
        llm_configuration: LLMConfiguration,
        device_configuration: DeviceConfiguration,
    ) -> Any:
        """
        Instantiates the Fathom runner with configured adapters.
        """

        signal_adapter = TemporalSignalAdapter(workflow_id=workflow_id)
        device_adapter = DeviceFactory.create(configuration=device_configuration)

        return (
            Fathom.builder()
            .with_device(port=device_adapter)
            .with_signal(port=signal_adapter)
            .with_llm(port=GeminiLLM(configuration=llm_configuration))
            .build()
        )
