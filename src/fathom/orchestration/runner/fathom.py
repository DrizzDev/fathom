from __future__ import annotations

import asyncio
import uuid
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

from fathom.exceptions import FathomError
from fathom.infrastructure.llm import GeminiLLMClient
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.infrastructure.memory.ledger import Ledger
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.infrastructure.storage.local import LocalImageStorage
from fathom.interfaces import ILedger, IMemoryProvider, IVisionProvider
from fathom.prompts.factory import PromptFactory
from fathom.schemas.configuration import ADBCaptureConfig, ADBConfig, GeminiConfig, WorkflowConfig
from fathom.schemas.results import ExplorationResult, IntentResult
from fathom.services.export import GraphExportService
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
    Main entry point for executing Fathom workflows.
    Orchestrates the wiring of infrastructure, tools, and strategies.

    When ``settings.use_langgraph`` is ``True``, uses the LangChain model
    adapter and passes the flag through to :class:`IntentWorkflow` so it
    executes via the LangGraph StateGraph.
    """

    def __init__(self, settings: FathomSettings) -> None:
        self.__settings = settings

        self.__memory_provider: Optional[IMemoryProvider] = None
        self.__ledger: Optional[ILedger] = None
        self.__knowledge_graph: Optional[KnowledgeGraph] = None
        self.__vision_orchestrator: Optional[GeminiVisionTool] = None

        self.__current_workflow: Optional[BaseWorkflow[Any]] = None

    async def run_intent(
        self,
        intent: str,
        max_steps: int = 100,
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

        # Resolve package name first — it scopes the per-app knowledge graph
        package_name = await device.get_current_package()

        # 2. Vision Infrastructure Wiring
        self.__ledger = Ledger()
        knowledge_db = f"assets/memory/{package_name}/knowledge.db"

        # Only use the per-app knowledge graph if exploration has already
        # created one. Intent runs must never create a knowledge graph.
        self.__knowledge_graph = None
        if Path(knowledge_db).exists():
            self.__memory_provider = SQLiteMemoryProvider(database_path=knowledge_db)
            self.__knowledge_graph = KnowledgeGraph(database_path=knowledge_db)
            await self.__knowledge_graph.load()
        else:
            self.__memory_provider = SQLiteMemoryProvider()

        # Determine prompt version
        actual_version = prompt_version or PromptFactory.resolve_version(
            model_name=self.__settings.gemini_model, use_xml=use_xml
        )

        workflow_id = request_id or uuid.uuid4().hex[:8]

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
            use_langgraph=self.__settings.use_langgraph,
        )

        # 4. Execution
        try:
            result = await self.__current_workflow.execute()

            # Re-resolve knowledge graph from the memory provider's
            # current DB path.  If the foreground app changed during the
            # run the provider will now point at the correct per-app DB.
            await self.__attach_knowledge_graph(result)

            return result
        finally:
            await self.cleanup()

    async def run_exploration(
        self,
        max_steps: int = 50,
        request_id: Optional[str] = None,
        device_serial: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> ExplorationResult:
        """
        Run an application exploration workflow.

        Args:
            package_name: Target application package to explore. When provided,
                the agent will ensure the app is in the foreground before
                starting and will enforce package scope throughout the run.
                If ``None``, auto-detects from the current foreground app.
        """

        # 1. Device Wiring
        serial = device_serial or self.__settings.android_serial
        device = ADBDeviceTool(configuration=ADBConfig(device_serial=serial))

        if not await device.wait_for_device(timeout=5.0):
            raise FathomError(f"Device {serial or '(default)'} offline.")

        # Resolve target package
        if package_name:
            # Ensure the target app is in the foreground
            current = await device.get_current_package()
            if current != package_name:
                logger.info(
                    "Target package %s not in foreground (got %s), launching...",
                    package_name,
                    current,
                )
                launch = await device.launch_app(package_name=package_name)
                if not launch.success:
                    raise FathomError(
                        f"Failed to launch target package {package_name}: {launch.error}"
                    )
                await asyncio.sleep(2.0)  # Wait for app to fully launch
        else:
            package_name = await device.get_current_package()

        # 2. Infrastructure Wiring
        self.__ledger = Ledger()
        knowledge_db = f"assets/memory/{package_name}/knowledge.db"
        self.__memory_provider = SQLiteMemoryProvider(database_path=knowledge_db)

        # Initialize persistent knowledge graph (per-app) and load prior knowledge
        self.__knowledge_graph = KnowledgeGraph(database_path=knowledge_db)
        await self.__knowledge_graph.load()

        actual_version = PromptFactory.resolve_version(
            model_name=self.__settings.gemini_model, use_xml=False
        )

        workflow_id = request_id or uuid.uuid4().hex[:8]

        self.__vision_orchestrator = self.__build_vision_orchestrator(
            version=actual_version, session_id=workflow_id, package_name=package_name
        )

        # 3. Workflow Wiring
        workflow = ExplorationWorkflow(
            device=device,
            workflow_id=workflow_id,
            vision=self.__vision_orchestrator,
            memory=self.__memory_provider,
            capture=ADBCaptureTool(config=ADBCaptureConfig(device_serial=serial)),
            configuration=WorkflowConfig(max_steps=max_steps, package_name=package_name),
            knowledge_graph=self.__knowledge_graph,
            use_langgraph=self.__settings.use_langgraph,
            target_package=package_name,
        )
        self.__current_workflow = workflow

        # 4. Execution
        try:
            result = await workflow.execute()

            # Attach accumulated knowledge graph to result
            if self.__knowledge_graph:
                result.knowledge_graph = self.__knowledge_graph.export_json()

            # Auto-export knowledge graph artifacts alongside the DB
            self.__export_knowledge_graph(
                graph_data=result.knowledge_graph,
                package_name=package_name,
            )

            return result
        finally:
            await self.cleanup()

    def __build_vision_orchestrator(
        self, version: str, package_name: str, session_id: str = "default"
    ) -> GeminiVisionTool:
        """
        Builds the Gemini-based vision orchestrator.

        When ``settings.use_langgraph`` is enabled, uses
        :class:`LangChainLLMClient` instead of the direct Gemini SDK client.
        """

        llm_configuration = GeminiConfig(
            model=self.__settings.gemini_model,
            api_key=self.__settings.gemini_api_key,
            location=self.__settings.vertex_location,
            project_id=self.__settings.vertex_project_id,
            credentials_path=self.__settings.google_application_credentials,
        )

        client: IVisionProvider
        if self.__settings.use_langgraph:
            from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient

            client = LangChainLLMClient(configuration=llm_configuration)
        else:
            client = GeminiLLMClient(configuration=llm_configuration)

        if not self.__ledger:
            raise FathomError(message="Ledger not initialized")

        if not self.__memory_provider:
            raise FathomError(message="Memory provider not initialized")

        return GeminiVisionTool(
            model=client,
            memory=self.__memory_provider,
            ledger=self.__ledger,
            local_storage=LocalImageStorage(),
            version=version,
            session_id=session_id,
            package_name=package_name,
        )

    def __export_knowledge_graph(
        self,
        graph_data: Dict[str, Any],
        package_name: str,
    ) -> None:
        """Export knowledge graph artifacts to the per-app memory directory.

        Writes JSON, DOT, Mermaid, and PNG files alongside the
        ``knowledge.db`` so they are always up-to-date after each run.
        Failures are logged but never propagated — exports are best-effort.
        """

        if not graph_data or not graph_data.get("nodes"):
            return

        output_dir = f"assets/memory/{package_name}"
        try:
            written = GraphExportService.save(
                graph_data,
                output_dir=output_dir,
                prefix="knowledge_graph",
            )
            if written:
                logger.info(
                    "Knowledge graph exported to %s: %s",
                    output_dir,
                    ", ".join(written.keys()),
                )
        except Exception:
            logger.warning("Failed to export knowledge graph artifacts", exc_info=True)

    async def __attach_knowledge_graph(self, result: IntentResult) -> None:
        """Attach a knowledge graph snapshot to the intent result.

        After execution the memory provider may point at a different
        per-app database than the one loaded at startup (because the
        foreground app changed mid-run).  This method re-resolves the
        correct ``KnowledgeGraph`` from the provider's current path so
        the result always reflects the final app's knowledge.
        """

        if not isinstance(self.__memory_provider, SQLiteMemoryProvider):
            # Non-SQLite provider — just attach existing graph if any
            if self.__knowledge_graph:
                result.knowledge_graph = self.__knowledge_graph.export_json()
            return

        final_db = self.__memory_provider.path
        if not final_db.exists():
            # No knowledge DB for the final app — attach existing graph if any
            if self.__knowledge_graph:
                result.knowledge_graph = self.__knowledge_graph.export_json()
            return

        # If the initial graph was loaded from the same DB, reuse it
        if self.__knowledge_graph and Path(self.__knowledge_graph.provider.path) == final_db:
            result.knowledge_graph = self.__knowledge_graph.export_json()
            return

        # Otherwise, load the knowledge graph from the final DB
        try:
            kg = KnowledgeGraph(database_path=str(final_db))
            await kg.load()
            self.__knowledge_graph = kg
            result.knowledge_graph = kg.export_json()
        except Exception:
            logger.warning("Failed to load knowledge graph from %s", final_db, exc_info=True)
            if self.__knowledge_graph:
                result.knowledge_graph = self.__knowledge_graph.export_json()

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
