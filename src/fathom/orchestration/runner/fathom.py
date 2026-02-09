from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Any, Optional

from fathom.exceptions import FathomError
from fathom.infrastructure.llm import GeminiLLMClient
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.infrastructure.storage.local import LocalImageStorage
from fathom.infrastructure.memory.ledger import Ledger
from fathom.interfaces import ILedger, IMemoryProvider
from fathom.schemas.configuration import ADBConfig, GeminiConfig
from fathom.schemas.results import ExplorationResult, IntentResult
from fathom.services.prompts import PromptsService
from fathom.settings.env import FathomSettings
from fathom.tools.capture.adb import ADBCaptureTool
from fathom.tools.device.adb import ADBDeviceTool
from fathom.tools.vision.gemini import GeminiVisionTool
from fathom.workflows.intent import IntentWorkflow
from fathom.workflows.base import BaseWorkflow


logger = getLogger(name=__name__)


class FathomRunner:
    """
    Main entry point for executing Fathom workflows.
    Orchestrates the wiring of infrastructure, tools, and strategies.
    """

    def __init__(self, settings: FathomSettings) -> None:
        self.__settings = settings

        self.__memory_provider: Optional[IMemoryProvider] = None
        self.__ledger: Optional[ILedger] = None
        self.__vision_orchestrator: Optional[GeminiVisionTool] = None

        self.__prompts_service = PromptsService()
        self.__current_workflow: Optional[BaseWorkflow[Any]] = None

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

        # 1. Device Wiring
        serial = device_serial or self.__settings.android_serial
        device = ADBDeviceTool(configuration=ADBConfig(device_serial=serial))

        if not await device.wait_for_device(timeout=5.0):
            raise FathomError(f"Device {serial or '(default)'} offline.")

        # 2. Vision Infrastructure Wiring
        self.__memory_provider = SQLiteMemoryProvider()
        self.__ledger = Ledger()

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

    async def run_exploration(
        self, 
        max_steps: int = 50, 
        device_serial: Optional[str] = None
    ) -> ExplorationResult:
        """
        Run an application exploration workflow.
        """
        from fathom.workflows.exploration import ExplorationWorkflow
        from fathom.workflows.base import WorkflowConfig

        # 1. Device Wiring
        serial = device_serial or self.__settings.android_serial
        device = ADBDeviceTool(configuration=ADBConfig(device_serial=serial))

        if not await device.wait_for_device(timeout=5.0):
            raise FathomError(f"Device {serial or '(default)'} offline.")

        # 2. Infrastructure Wiring
        self.__memory_provider = SQLiteMemoryProvider()
        self.__ledger = Ledger()
        self.__vision_orchestrator = self.__build_vision_orchestrator(
            version=self.__prompts_service.select_version(
                model_name=self.__settings.gemini_model, 
                use_xml=False
            )
        )

        # 3. Workflow Wiring
        workflow = ExplorationWorkflow(
            workflow_id=f"explore_{asyncio.get_event_loop().time()}",
            device=device,
            capture=ADBCaptureTool(),
            vision=self.__vision_orchestrator,
            config=WorkflowConfig(max_steps=max_steps),
        )
        self.__current_workflow = workflow

        # 4. Execution
        try:
            return await workflow.execute()
        finally:
            await self.cleanup()

    def __build_vision_orchestrator(self, version: str) -> GeminiVisionTool:
        """
        Builds the Gemini-based vision orchestrator.
        """

        llm_configuration = GeminiConfig(
            model=self.__settings.gemini_model,
            api_key=self.__settings.gemini_api_key,
        )

        client = GeminiLLMClient(configuration=llm_configuration)

        return GeminiVisionTool(
            model=client,
            version=version,
            memory=self.__memory_provider,  # type: ignore[arg-type]
            ledger=self.__ledger,  # type: ignore[arg-type]
            prompts=self.__prompts_service,
            local_storage=LocalImageStorage(),
            cloud_storage=GCSImageStorage(
                configuration=llm_configuration,
                credentials=self.__settings.google_application_credentials,
            ),
        )

    def cancel(self) -> None:
        """
        Cancels the currently running workflow.
        """

        if self.__current_workflow:
            self.__current_workflow.cancel()

    async def cleanup(self) -> None:
        """
        Shut down all background resources.
        """

        if self.__vision_orchestrator:
            await self.__vision_orchestrator.cleanup()