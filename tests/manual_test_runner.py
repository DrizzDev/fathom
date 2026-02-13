import asyncio
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath("src"))

from fathom.agent.state import AgentState
from fathom.infrastructure.memory.ledger import Ledger
from fathom.workflows.intent import IntentWorkflow


async def test_agent_state():
    print("\n=== Testing AgentState Smart Context ===")
    state = AgentState(intent="test intent", max_steps=5)

    # Check if get_smart_context exists (it shouldn't yet, or will if we implemented it)
    if hasattr(state, "get_smart_context"):
        print("get_smart_context() found.")
        print(state.get_smart_context())
    else:
        print("get_smart_context() NOT found (as expected before Step 0.3).")


async def test_workflow_init():
    print("\n=== Testing IntentWorkflow Initialization ===")

    # Mock tools
    class MockTool:
        def __init__(self):
            self.provider = "mock"

    try:
        workflow = IntentWorkflow(
            workflow_id="test-1",
            intent="test intent",
            vision=MockTool(),
            device=MockTool(),
            capture=MockTool(),
            memory=Ledger(),
        )
        print("IntentWorkflow initialized successfully.")
    except Exception as e:
        print(f"IntentWorkflow initialization FAILED: {e}")


async def main():
    await test_agent_state()
    await test_workflow_init()
    await test_prompt_preprocessor()


async def test_prompt_preprocessor():
    print("\n=== Testing PromptPreprocessor ===")
    from fathom.services.prompt_preprocessor import PromptPreprocessor

    intent = "Type 'hello world' on the login screen"
    print(f"Intent: {intent}")

    hints = PromptPreprocessor.extract_hints(intent)
    print(f"Extracted hints: {hints}")

    if hints.get("typing_text") == "hello world" and hints.get("target_screen") == "login":
        print("[PASS] Hints extraction correct")
    else:
        print("[FAIL] Hints extraction incorrect")

    prefix = PromptPreprocessor.build_context_prefix(hints)
    print(f"Context Prefix:\n{prefix}")

    if "[HINT]" in prefix:
        print("[PASS] Context prefix built correctly")
    else:
        print("[FAIL] Context prefix missing hints")


if __name__ == "__main__":
    asyncio.run(main())
