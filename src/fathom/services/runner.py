from logging import getLogger
from typing import Optional

from fathom.auth.credentials import CredentialsManager
from fathom.exceptions import FathomError
from fathom.schemas.configuration import ADBConfig, GeminiConfig, WorkflowConfig
from fathom.schemas.results import IntentResult
from fathom.settings.env import FathomSettings
from fathom.tools.capture import ADBCaptureConfig, ADBCaptureTool
from fathom.tools.device import ADBDeviceTool
from fathom.tools.vision import GeminiVisionTool
from fathom.workflows import IntentWorkflow

logger = getLogger(__name__)


class FathomRunner:
    """
    Orchestrates Fathom workflow execution.
    """

    def __init__(self, settings: FathomSettings) -> None:
        """
        Initialize runner with settings.
        """

        self.settings = settings
        self.__ensure_auth()

    def __ensure_auth(self) -> None:
        """
        Ensure authentication credentials are available.
        """

        if self.settings.gemini_api_key:
            return

        credentials = CredentialsManager.load_google_credentials()

        if credentials:
            self.settings.vertex_project_id = credentials.project_id
            return

        if self.settings.vertex_project_id:
            return

        raise FathomError(
            "No authentication found. Please provide GEMINI_API_KEY or "
            "GOOGLE_APPLICATION_CREDENTIALS (pointing to valid JSON)."
        )

    async def run_intent(
        self,
        intent: str,
        max_steps: int = 20,
        device_serial: Optional[str] = None,
    ) -> IntentResult:
        """
        Run an intent-based workflow.

        Args:
            intent: The goal to achieve.
            device_serial: Optional device serial overrides settings.
            max_steps: Maximum steps allowed.

        Returns:
            IntentResult containing success status and step history.
        """

        serial = device_serial or self.settings.android_serial
        logger.info("Initializing tools", extra={"serial": serial})

        # Initialize Tools
        device_tool = ADBDeviceTool(ADBConfig(device_serial=serial))

        # Verify connectivity
        if not await device_tool.wait_for_device(timeout=5.0):
            raise FathomError(f"Device {serial or '(default)'} not found or offline.")

        capture_tool = ADBCaptureTool(config=ADBCaptureConfig(device_serial=serial))

        # Initialize Vision with Auth Resolution
        vision_configuration = GeminiConfig(
            model=self.settings.gemini_model,
            api_key=self.settings.gemini_api_key,
            location=self.settings.vertex_location,
            project_id=self.settings.vertex_project_id,
        )
        vision_tool = GeminiVisionTool(vision_configuration)

        # Initialize Workflow
        workflow = IntentWorkflow(
            workflow_id="runner-intent",
            intent=intent,
            vision=vision_tool,
            device=device_tool,
            capture=capture_tool,
            config=WorkflowConfig(max_steps=max_steps),
        )

        logger.info("Starting workflow execution", extra={"intent": intent})
        return await workflow.execute()
