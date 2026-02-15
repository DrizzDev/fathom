# Migration Guide: Old to New Architecture

## Overview
Fathom has been re-architected using hexagonal architecture (ports and adapters pattern) for better testability, maintainability, and pluggability.

## Quick Start

### Old Way (Deprecated)
```bash
fathom-old run "your intent" --serial emulator-5554
```

### New Way (Recommended)
```bash
fathom run "your intent" --serial emulator-5554
```

## Command Changes

| Old Command | New Command | Status |
|------------|-------------|--------|
| `fathom-old run` | `fathom run` | ✅ Available |
| `fathom-old explore` | `fathom explore` | ✅ Available |

All command-line arguments remain the same!

## Code Migration

### If You're Using FathomRunner Directly

#### Old Code (Deprecated)
```python
from fathom.orchestration.runner import FathomRunner
from fathom.settings.env import FathomSettings

settings = FathomSettings()
runner = FathomRunner(settings)

result = await runner.run_intent(
    intent="Open Gmail",
    max_steps=20,
    use_xml=True,
    device_serial="emulator-5554"
)
```

#### New Code (Recommended)
```python
from fathom.runtime.builder import FathomBuilder
from fathom.schemas.configuration import (
    ADBConfig,
    GeminiConfig,
    ExecutionConfig,
    StrategyConfig
)

# Build runner with configuration
runner = (
    FathomBuilder()
    .device(ADBConfig(device_serial="emulator-5554"))
    .llm(GeminiConfig(
        api_key=None,  # Uses credentials file
        model="gemini-2.0-flash-exp",
        project_id="your-project",
        location="us-central1",
        credentials_path="/path/to/credentials.json"
    ))
    .memory()
    .signal()
    .storage()
    .telemetry()
    .execution(ExecutionConfig(
        max_retries=3,
        stability_wait=1000
    ))
    .strategy(StrategyConfig(
        max_steps=20,
        use_xml=True
    ))
    .build()
)

# Run intent
result = await runner.run_intent(intent="Open Gmail")
```

### If You're Using Workflows Directly

#### Old Code (Deprecated)
```python
from fathom.workflows.intent import IntentWorkflow
from fathom.tools.device.adb import ADBDeviceTool
from fathom.tools.vision.gemini import GeminiVisionTool

device = ADBDeviceTool(...)
vision = GeminiVisionTool(...)

workflow = IntentWorkflow(
    device=device,
    vision=vision,
    intent="Open Gmail",
    ...
)

result = await workflow.execute()
```

#### New Code (Recommended)
```python
from fathom.strategies.intent import IntentStrategy
from fathom.core.execution.engine import ExecutionEngine
from fathom.core.context.manager import ContextManager

# Use builder to create all components
runner = FathomBuilder()...build()

# Or create strategy directly with ports
strategy = IntentStrategy(
    engine=engine,
    context=context,
    intent="Open Gmail",
    device=device_port,
    llm=llm_port,
    memory=memory_port,
    storage=storage_port,
    telemetry=telemetry_port,
    max_steps=20,
    use_xml=True
)

result = await strategy.execute(max_steps=20)
```

## Key Architectural Changes

### 1. Ports Instead of Tools
**Old**: Direct tool dependencies
```python
device = ADBDeviceTool(config)
```

**New**: Port interfaces with adapters
```python
device_port: DevicePort = ADBDevice(config)
```

### 2. Builder Pattern
**Old**: Manual wiring
```python
runner = FathomRunner(settings)
```

**New**: Fluent builder API
```python
runner = FathomBuilder().device(...).llm(...).build()
```

### 3. Strategies Instead of Workflows
**Old**: Workflows with direct tool access
```python
workflow = IntentWorkflow(device=device, vision=vision, ...)
```

**New**: Strategies with port dependencies
```python
strategy = IntentStrategy(engine=engine, device=device_port, ...)
```

### 4. ExecutionEngine for Step Execution
**Old**: Workflows handle execution directly
```python
# Execution logic embedded in workflow
```

**New**: Centralized ExecutionEngine with 7-phase DAG
```python
engine = ExecutionEngine(device, llm, memory, signal, storage, telemetry)
result = await engine.execute_step(step)
```

## Configuration Changes

### Old Configuration (Settings)
```python
from fathom.settings.env import FathomSettings

settings = FathomSettings()
# Settings loaded from environment variables
```

### New Configuration (Explicit Configs)
```python
from fathom.schemas.configuration import (
    ADBConfig,
    GeminiConfig,
    ExecutionConfig,
    StrategyConfig
)

adb_config = ADBConfig(device_serial="emulator-5554")
gemini_config = GeminiConfig(
    model="gemini-2.0-flash-exp",
    credentials_path="/path/to/credentials.json"
)
execution_config = ExecutionConfig(max_retries=3)
strategy_config = StrategyConfig(max_steps=20)
```

## Benefits of New Architecture

1. **Testability**: Easy to mock ports for testing
2. **Pluggability**: Swap implementations without changing core logic
3. **Separation of Concerns**: Clear boundaries between layers
4. **Type Safety**: Strong typing with port interfaces
5. **Maintainability**: Easier to understand and modify
6. **Extensibility**: Add new adapters without touching core

## Deprecation Timeline

- **Current**: Both old and new code available
- **v2.0**: Old code marked as deprecated (warnings added)
- **v3.0**: Old code will be removed

## Getting Help

If you encounter issues during migration:
1. Check this guide
2. Review examples in `src/fathom/cli_new.py`
3. Look at new architecture docs in `documents/architecture/v2/ARCHITECTURE.md`
4. Open an issue on GitHub

## Example: Complete Migration

### Before (Old Code)
```python
from fathom.orchestration.runner import FathomRunner
from fathom.settings.env import FathomSettings

async def main():
    settings = FathomSettings()
    runner = FathomRunner(settings)
    
    result = await runner.run_intent(
        intent="Open Gmail and check unread count",
        max_steps=20,
        use_xml=True,
        device_serial="emulator-5554"
    )
    
    print(f"Success: {result.success}")
    print(f"Steps: {result.step_count}")
```

### After (New Code)
```python
from fathom.runtime.builder import FathomBuilder
from fathom.schemas.configuration import (
    ADBConfig,
    GeminiConfig,
    ExecutionConfig,
    StrategyConfig
)

async def main():
    runner = (
        FathomBuilder()
        .device(ADBConfig(device_serial="emulator-5554"))
        .llm(GeminiConfig(
            model="gemini-2.0-flash-exp",
            credentials_path="/path/to/credentials.json"
        ))
        .memory()
        .signal()
        .storage()
        .telemetry()
        .execution(ExecutionConfig(max_retries=3))
        .strategy(StrategyConfig(max_steps=20, use_xml=True))
        .build()
    )
    
    result = await runner.run_intent(
        intent="Open Gmail and check unread count"
    )
    
    print(f"Success: {result.success}")
    print(f"Steps: {result.step_count}")
```

## CLI Usage (No Changes Needed!)

The CLI interface remains the same, just use `fathom` instead of `fathom-old`:

```bash
# Old
fathom-old run "Open Gmail" --use-xml --serial emulator-5554 -v

# New (same arguments!)
fathom run "Open Gmail" --use-xml --serial emulator-5554 -v
```
