from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Any, Optional

from fathom.exceptions import FathomError
from fathom.infrastructure.llm import GeminiLLMClient
from fathom.infrastructure.memory import SQLiteMemoryProvider
from fathom.infrastructure.storage import GCSImageStorage, LocalImageStorage
from fathom.schemas.configuration import ADBConfig, GeminiConfig
from fathom.schemas.results import IntentResult
from fathom.services.prompts import PromptsService
from fathom.settings.env import FathomSettings
from fathom.tools.capture.adb import ADBCaptureTool
from fathom.tools.device.adb import ADBDeviceTool
from fathom.tools.vision.gemini import GeminiVisionTool
from fathom.workflows.intent import IntentWorkflow

logger = getLogger(name=__name__)


class FathomRunner:
    """
    Main entry point for executing Fathom workflows.
    Orchestrates the wiring of infrastructure, tools, and strategies.
    """

    def __init__(self, settings: FathomSettings) -> None:
        self.__settings = settings

        self.__memory_provider: Optional[SQLiteMemoryProvider] = None
        self.__vision_orchestrator: Optional[GeminiVisionTool] = None

        self.__prompts_service = PromptsService()
        self.__current_workflow: Optional[IntentWorkflow] = None

    async def run_intent(
        self,
        intent: str,
        max_steps: int = 20,
        use_xml: bool = False,
        device_serial: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> IntentResult:
        """
        Run an intent-based workflow.
        """

        _ = max_steps

        # 1. Device Wiring
        serial = device_serial or self.__settings.android_serial
        device = ADBDeviceTool(configuration=ADBConfig(device_serial=serial))

        if not await device.wait_for_device(timeout=5.0):
            raise FathomError(f"Device {serial or '(default)'} offline.")

        # 2. Vision Infrastructure Wiring
        self.__memory_provider = SQLiteMemoryProvider()

        # Select prompt version dynamically based on model and XML requirement
        model_name = self.__settings.gemini_model
        actual_version = prompt_version or self.__prompts_service.select_version(
            model_name=model_name, use_xml=use_xml
        )

        self.__vision_orchestrator = self.__build_vision_orchestrator(version=actual_version)

        # 3. Workflow Wiring
        self.__current_workflow = IntentWorkflow(
            device=device,
            intent=intent,
            capture=ADBCaptureTool(),
            memory=self.__memory_provider,
            vision=self.__vision_orchestrator,
            workflow_id=f"intent_{asyncio.get_event_loop().time()}",
        )

        # 4. Execution
        try:
            return await self.__current_workflow.execute()
        finally:
            await self.cleanup()

    def __build_vision_orchestrator(self, version: str) -> GeminiVisionTool:
        """
        Builds the Gemini-based vision orchestrator.
        """

        llm_config = GeminiConfig(
            model=self.__settings.gemini_model,
            api_key=self.__settings.gemini_api_key,
        )

        client = GeminiLLMClient(configuration=llm_config)

        return GeminiVisionTool(
            model=client,
            version=version,
            memory=self.__memory_provider,  # type: ignore[arg-type]
            prompts=self.__prompts_service,
            local_storage=LocalImageStorage(),
            cloud_storage=GCSImageStorage(
                configuration=llm_config,
                credentials=self.__settings.google_application_credentials,
            ),
        )

    def cancel(self) -> None:
        """
        Cancels the currently running workflow.
        """

        if self.__current_workflow:
            self.__current_workflow.cancel()

    async def run_exploration(self, **kwargs: object) -> Any:
        """
        Placeholder for exploration workflow.
        """

        _ = kwargs

        logger.warning(msg="Exploration workflow is not yet implemented.")
        from unittest.mock import MagicMock

        return MagicMock(
            total_actions=0,
            unique_screens=0,
            total_transitions=0,
            coverage_percentage=0.0,
        )

    async def cleanup(self) -> None:
        """
        Shut down all background resources.
        """

        if self.__vision_orchestrator:
            await self.__vision_orchestrator.cleanup()
