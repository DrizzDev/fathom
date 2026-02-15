"""
Minimal example using the new Fathom builder API.

This demonstrates the simplest possible configuration with just device and LLM.
All other ports (memory, knowledge, signal, storage, telemetry) use defaults.
"""

import asyncio

from fathom.adapters import ADBDevice, GeminiLLM
from fathom.runtime import Fathom


async def main():
    """Run a simple automation with minimal configuration."""
    
    # Build Fathom with minimal configuration
    # Only device and llm are required - everything else gets sensible defaults
    runner = (
        Fathom.builder()
        .device(ADBDevice(serial="emulator-5554"))  # Required
        .llm(GeminiLLM(api_key="your-api-key"))     # Required
        .build()
    )
    
    # The runner now has:
    # - ADBDevice for device interactions
    # - GeminiLLM for AI reasoning
    # - SQLiteMemory (default) for session state
    # - SQLiteKnowledge (default) for app knowledge graph
    # - NoopSignal (default) for autonomous operation
    # - LocalStorage (default) for screenshots
    # - StructlogAdapter (default) for logging
    
    # Execute an intent
    result = await runner.run(
        intent="Open the settings app",
        max_steps=10,
        strategy="intent"
    )
    
    print(f"Execution result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
