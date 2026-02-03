from __future__ import annotations

from datetime import datetime
from logging import getLogger
from typing import Any, Optional

from fathom.exceptions import FathomError
from fathom.infrastructure.llm import GeminiLLMClient
from fathom.infrastructure.memory import SQLiteMemoryProvider
from fathom.infrastructure.storage import GCSImageStorage, LocalImageStorage
from fathom.schemas.configuration import ADBConfig, GeminiConfig, WorkflowConfig
from fathom.schemas.results import ExplorationResult, IntentResult
from fathom.services.prompts import PromptsService
from fathom.settings.env import FathomSettings
from fathom.tools.capture import ADBCaptureConfig, ADBCaptureTool
from fathom.tools.device import ADBDeviceTool
from fathom.tools.vision import GeminiVisionTool
from fathom.workflows import IntentWorkflow
from fathom.workflows.base import BaseWorkflow
from fathom.workflows.exploration import ExplorationWorkflow

logger = getLogger(__name__)


class FathomRunner:
    """
    Composition Root for Fathom.
    Wires up infrastructure, providers, and workflows.
    """

    def __init__(self, settings: FathomSettings) -> None:
        self.settings = settings
        self.__active_workflow: Optional[BaseWorkflow[Any]] = None
        self.__vision_orchestrator: Optional[GeminiVisionTool] = None
        self.__memory_provider: Optional[SQLiteMemoryProvider] = None

    async def run_intent(
        self,
        intent: str,
        max_steps: int = 20,
        use_xml: bool = False,
        prompt_version: str = "v2_analytical",
        device_serial: Optional[str] = None,
    ) -> IntentResult:
        """
        Run an intent-based workflow.
        """
        # 1. Device Wiring
        serial = device_serial or self.settings.android_serial
        device = ADBDeviceTool(ADBConfig(device_serial=serial))
        if not await device.wait_for_device(timeout=5.0):
            raise FathomError(f"Device {serial or '(default)'} offline.")

        # 2. Vision Infrastructure Wiring
        self.__memory_provider = SQLiteMemoryProvider()
        self.__vision_orchestrator = self.__build_vision_orchestrator(prompt_version)

        # 3. Workflow Assembly
        capture = ADBCaptureTool(config=ADBCaptureConfig(device_serial=serial))
        workflow_id = f"intent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.__active_workflow = IntentWorkflow(
            intent=intent,
            vision=self.__vision_orchestrator,
            device=device,
            capture=capture,
            memory=self.__memory_provider,
            workflow_id=workflow_id,
            config=WorkflowConfig(max_steps=max_steps, use_xml_bounding_boxes=use_xml),
        )

        logger.info("Starting workflow execution", extra={"intent": intent, "id": workflow_id})
        try:
            result = await self.__active_workflow.execute()
            # Capture what the agent learned/remembered
            if self.__memory_provider:
                result.memory_summary = await self.__memory_provider.get_all_knowledge()
            return result
        finally:
            await self.cleanup()

    async def run_exploration(
        self,
        max_steps: int = 50,
        device_serial: Optional[str] = None,
    ) -> ExplorationResult:
        """
        Run an exploration workflow to map the app.
        """
        serial = device_serial or self.settings.android_serial
        device = ADBDeviceTool(ADBConfig(device_serial=serial))
        if not await device.wait_for_device(timeout=5.0):
            raise FathomError(f"Device {serial or '(default)'} offline.")

        self.__memory_provider = SQLiteMemoryProvider()
        self.__vision_orchestrator = self.__build_vision_orchestrator("v2_analytical")

        capture = ADBCaptureTool(config=ADBCaptureConfig(device_serial=serial))
        workflow_id = f"exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.__active_workflow = ExplorationWorkflow(
            vision=self.__vision_orchestrator,
            device=device,
            capture=capture,
            workflow_id=workflow_id,
            config=WorkflowConfig(max_steps=max_steps),
        )

        try:
            return await self.__active_workflow.execute()
        finally:
            await self.cleanup()

    def __build_vision_orchestrator(self, version: str) -> GeminiVisionTool:
        """
        Wired up the vision orchestrator with concrete providers.
        """
        configuration = GeminiConfig(
            model=self.settings.gemini_model,
            api_key=self.settings.gemini_api_key,
            project_id=self.settings.vertex_project_id,
            credentials_path=self.settings.google_application_credentials,
        )

        llm_client = GeminiLLMClient(configuration)
        prompts = PromptsService()
        cloud_storage = GCSImageStorage(configuration, getattr(llm_client, "credentials", None))
        local_storage = LocalImageStorage()

        # Re-use existing memory provider if initialized
        memory = self.__memory_provider or SQLiteMemoryProvider()

        return GeminiVisionTool(
            model=llm_client,
            memory=memory,
            prompts=prompts,
            cloud_storage=cloud_storage,
            local_storage=local_storage,
            version=version,
        )

    async def cleanup(self) -> None:
        """
        Performs graceful shutdown.
        """
        if self.__active_workflow:
            self.__active_workflow.cancel()
        if self.__vision_orchestrator:
            await self.__vision_orchestrator.cleanup()

    def cancel(self) -> None:
        """
        Immediate cancellation request.
        """
        if self.__active_workflow:
            self.__active_workflow.cancel()
