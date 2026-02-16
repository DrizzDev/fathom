"""
LEGACY CODE - DEPRECATED

This module contains the old FathomRunner implementation using direct tool wiring.
It is preserved for backward compatibility via the 'fathom-old' command.

NEW CODE: Use the hexagonal architecture instead:
- Runner: src/fathom/runtime/runner.py
- Builder: src/fathom/runtime/builder.py
- CLI: src/fathom/cli_new.py (via 'fathom' command)

This code will be removed in a future major version.
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from logging import getLogger
from typing import Any, Optional

from fathom.exceptions import FathomError
from fathom.infrastructure.llm import GeminiLLMClient
from fathom.infrastructure.memory.ledger import Ledger
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.infrastructure.storage.local import LocalImageStorage
from fathom.interfaces import ILedger, IMemoryProvider
from fathom.prompts.factory import PromptFactory
from fathom.schemas.configuration import ADBCaptureConfig, ADBConfig, GeminiConfig, WorkflowConfig
from fathom.schemas.results import ExplorationResult, IntentResult
from fathom.settings.env import FathomSettings
from fathom.tools.capture.adb import ADBCaptureTool
from fathom.tools.device.adb import ADBDeviceTool
from fathom.tools.vision.gemini import GeminiVisionTool
from fathom.workflows.base import BaseWorkflow
from fathom.workflows.exploration import ExplorationWorkflow
from fathom.workflows.intent import IntentWorkflow

logger = getLogger(name=__name__)


class FathomRunner:
    """
    DEPRECATED: Old FathomRunner using direct tool wiring.

    Use the new hexagonal architecture instead:
    - from fathom.runtime.runner import FathomRunner (new)
    - from fathom.runtime.builder import FathomBuilder

    This class is preserved for backward compatibility and will be removed
    in a future major version.
    Main entry point for executing Fathom workflows.
    Orchestrates the wiring of infrastructure, tools, and strategies.
    """

    def __init__(self, settings: FathomSettings) -> None:
        warnings.warn(
            "FathomRunner from orchestration.runner is deprecated. "
            "Use fathom.runtime.runner.FathomRunner with hexagonal architecture instead. "
            "This class will be removed in a future major version.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.__settings = settings

        self.__memory_provider: Optional[IMemoryProvider] = None
        self.__ledger: Optional[ILedger] = None
        self.__vision_orchestrator: Optional[GeminiVisionTool] = None

        self.__current_workflow: Optional[BaseWorkflow[Any]] = None

    async def run_intent(
        self,
        intent: str,
        max_steps: int = 20,
        use_xml: bool = False,
        request_id: Optional[str] = None,
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

        # Start fetching package name in background
        package_task = asyncio.create_task(device.get_current_package())

        # 2. Vision Infrastructure Wiring
        self.__ledger = Ledger()
        self.__memory_provider = SQLiteMemoryProvider()

        # Determine prompt version
        actual_version = prompt_version or PromptFactory.resolve_version(
            model_name=self.__settings.gemini_model, use_xml=use_xml
        )

        workflow_id = request_id or uuid.uuid4().hex[:8]

        package_name = await package_task

        self.__vision_orchestrator = self.__build_vision_orchestrator(
            version=actual_version, session_id=workflow_id, package_name=package_name
        )

        # 3. Workflow Wiring
        workflow_configuration = WorkflowConfig(
            max_steps=max_steps,
            package_name=package_name,
            use_xml_bounding_boxes=use_xml,
        )

        self.__current_workflow = IntentWorkflow(
            device=device,
            intent=intent,
            workflow_id=workflow_id,
            memory=self.__memory_provider,
            vision=self.__vision_orchestrator,
            configuration=workflow_configuration,
            capture=ADBCaptureTool(config=ADBCaptureConfig(device_serial=serial)),
        )

        # 4. Execution
        try:
            return await self.__current_workflow.execute()
        finally:
            await self.cleanup()

    async def run_exploration(
        self,
        max_steps: int = 50,
        request_id: Optional[str] = None,
        device_serial: Optional[str] = None,
    ) -> ExplorationResult:
        """
        Run an application exploration workflow.
        """

        # 1. Device Wiring
        serial = device_serial or self.__settings.android_serial
        device = ADBDeviceTool(configuration=ADBConfig(device_serial=serial))

        if not await device.wait_for_device(timeout=5.0):
            raise FathomError(f"Device {serial or '(default)'} offline.")

        # Start fetching package name in background
        package_task = asyncio.create_task(device.get_current_package())

        # 2. Infrastructure Wiring
        self.__ledger = Ledger()
        self.__memory_provider = SQLiteMemoryProvider()

        actual_version = PromptFactory.resolve_version(
            model_name=self.__settings.gemini_model, use_xml=False
        )

        workflow_id = request_id or uuid.uuid4().hex[:8]

        package_name = await package_task

        self.__vision_orchestrator = self.__build_vision_orchestrator(
            version=actual_version, session_id=workflow_id, package_name=package_name
        )

        # 3. Workflow Wiring
        workflow = ExplorationWorkflow(
            device=device,
            workflow_id=workflow_id,
            vision=self.__vision_orchestrator,
            capture=ADBCaptureTool(config=ADBCaptureConfig(device_serial=serial)),
            configuration=WorkflowConfig(max_steps=max_steps, package_name=package_name),
        )
        self.__current_workflow = workflow

        # 4. Execution
        try:
            return await workflow.execute()
        finally:
            await self.cleanup()

    def __build_vision_orchestrator(
        self, version: str, package_name: str, session_id: str = "default"
    ) -> GeminiVisionTool:
        """
        Builds the Gemini-based vision orchestrator.
        """

        llm_configuration = GeminiConfig(
            model=self.__settings.gemini_model,
            api_key=self.__settings.gemini_api_key,
            location=self.__settings.vertex_location,
            project_id=self.__settings.vertex_project_id,
            credentials_path=self.__settings.google_application_credentials,
        )

        client = GeminiLLMClient(configuration=llm_configuration)
        cloud_storage = GCSImageStorage(
            configuration=llm_configuration, credentials=client.credentials
        )

        if not self.__ledger:
            raise FathomError(message="Ledger not initialized")

        if not self.__memory_provider:
            raise FathomError(message="Memory provider not initialized")

        return GeminiVisionTool(
            model=client,
            memory=self.__memory_provider,
            ledger=self.__ledger,
            local_storage=LocalImageStorage(),
            gcs_storage=cloud_storage,
            version=version,
            session_id=session_id,
            package_name=package_name,
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
