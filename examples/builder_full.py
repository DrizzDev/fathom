"""
Full example using the new Fathom builder API.

This demonstrates explicit configuration of all seven ports.
"""

import asyncio

from fathom.adapters import (
    ADBDevice,
    GeminiLLM,
    LocalStorage,
    NoopSignal,
    SQLiteKnowledge,
    SQLiteMemory,
    StructlogAdapter,
)
from fathom.runtime import Fathom


async def main():
    """Run automation with full explicit configuration."""
    
    # Build Fathom with all ports explicitly configured
    # This gives you full control over every component
    runner = (
        Fathom.builder()
        .device(ADBDevice(serial="emulator-5554"))
        .llm(GeminiLLM(api_key="your-api-key", model="gemini-2.0-flash-exp"))
        .memory(SQLiteMemory(
            knowledge_path="custom/path/knowledge.db",
            ledger_path="custom/path/ledger.db"
        ))
        .knowledge(SQLiteKnowledge(database_path="custom/path/graph.db"))
        .signal(NoopSignal())  # Autonomous mode - no human intervention
        .storage(LocalStorage())
        .telemetry(StructlogAdapter(logger_name="my_app"))
        .build()
    )
    
    # Execute an intent
    result = await runner.run(
        intent="Search for 'hexagonal architecture' on Google",
        max_steps=20,
        strategy="intent"
    )
    
    print(f"Execution result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
