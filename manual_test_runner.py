import asyncio
import os
import sys
from logging import INFO, basicConfig, getLogger

# Add src to path if needed for direct execution
sys.path.append(os.path.join(os.getcwd(), "src"))

from fathom.infrastructure.memory.ledger import Ledger
from fathom.schemas.configuration import WorkflowConfig
from fathom.settings.env import FathomSettings
from fathom.tools.capture.adb import ADBCaptureTool
from fathom.tools.device.adb import ADBDeviceTool
from fathom.tools.vision.gemini import GeminiVisionTool
from fathom.workflows.intent import IntentWorkflow

# Simple logger setup
basicConfig(level=INFO)
logger = getLogger(__name__)


async def main():
    # 1. Setup settings
    settings = FathomSettings()

    # 2. Instantiate tools
    # Using defaults/env vars. Assumes a device is connected or env vars are set.
    from fathom.tools.device.adb import ADBConfig

    adb_config = ADBConfig(device_serial=settings.android_serial)
    device = ADBDeviceTool(configuration=adb_config)
    from fathom.tools.capture.adb import ADBCaptureConfig

    capture_config = ADBCaptureConfig(device_serial=settings.android_serial)
    capture = ADBCaptureTool(config=capture_config)

    # Need API Key for Vision (usually in env GEMINI_API_KEY)
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY not found in settings/env. Vision tool might fail.")

    from fathom.infrastructure.llm.gemini import GeminiLLMClient
    from fathom.infrastructure.storage.local import LocalImageStorage
    from fathom.schemas.configuration import GeminiConfig

    gemini_config = GeminiConfig(api_key=settings.gemini_api_key)
    llm_client = GeminiLLMClient(configuration=gemini_config)

    local_storage = LocalImageStorage()
    cloud_storage = LocalImageStorage()  # Mock cloud for now

    from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider

    memory_provider = SQLiteMemoryProvider()
    ledger = Ledger()

    vision = GeminiVisionTool(
        model=llm_client, memory=memory_provider, ledger=ledger, local_storage=local_storage
    )

    # 3. Create Workflow
    print("Initializing IntentWorkflow...")
    workflow = IntentWorkflow(
        workflow_id="login-flow-test-001",
        device=device,
        capture=capture,
        vision=vision,
        memory=memory_provider,
        intent="Launch Make My Trip App and search for hotels in Goa",
        configuration=WorkflowConfig(
            max_steps=10,  # Keep it short for testing
            step_timeout=10.0,
        ),
    )

    # 4. Execute
    print("Executing workflow...")
    # This will test our stripped-down loop logic
    try:
        result = await workflow.execute()

        if result.success:
            print("✅ Workflow finished successfully!")
        else:
            print(f"❌ Workflow finished with failure: {result.completion_reason}")

        print(f"Steps taken: {result.steps_taken}")

    except Exception as e:
        print(f"💥 detailed error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
