# Phase 6: Redundant Code Analysis

## Overview
The codebase now has TWO parallel implementations:
1. **NEW**: Hexagonal architecture (ports/adapters) - Active via `fathom` command
2. **OLD**: Direct tool wiring - Preserved via `fathom-old` command

## Old Code (Preserved for Backward Compatibility)

### Old Runner & Orchestration
- `src/fathom/orchestration/runner/fathom.py` - Old FathomRunner
- `src/fathom/orchestration/runner/workflow.py` - Old WorkflowRunner
- `src/fathom/orchestration/executor.py` - Old StepExecutor
- `src/fathom/orchestration/context.py` - Old context management
- `src/fathom/orchestration/__init__.py` - Exports old classes

### Old Workflows
- `src/fathom/workflows/base.py` - Base workflow class
- `src/fathom/workflows/intent.py` - Old IntentWorkflow
- `src/fathom/workflows/exploration.py` - Old ExplorationWorkflow
- `src/fathom/workflows/__init__.py` - Exports old workflows

### Old CLI
- `src/fathom/cli.py` - Old CLI using FathomRunner
- Entry point: `fathom-old` command in pyproject.toml

## New Code (Active)

### New Runner & Core
- `src/fathom/runtime/runner.py` - New FathomRunner (hexagonal)
- `src/fathom/runtime/builder.py` - Builder API
- `src/fathom/core/execution/engine.py` - ExecutionEngine (7-phase DAG)
- `src/fathom/core/context/manager.py` - ContextManager (3-tier)

### New Strategies
- `src/fathom/strategies/intent.py` - IntentStrategy (uses ports)
- `src/fathom/strategies/exploration.py` - ExplorationStrategy (uses ports)

### New CLI
- `src/fathom/cli_new.py` - New CLI using Builder API
- Entry point: `fathom` command in pyproject.toml

### Ports (Interfaces)
- `src/fathom/interfaces/device.py` - DevicePort
- `src/fathom/interfaces/llm.py` - LLMPort
- `src/fathom/interfaces/memory.py` - MemoryPort
- `src/fathom/interfaces/signal.py` - SignalPort
- `src/fathom/interfaces/storage.py` - StoragePort
- `src/fathom/interfaces/telemetry.py` - TelemetryPort

### Adapters (Implementations)
- `src/fathom/adapters/device/adb.py` - ADBDevice
- `src/fathom/adapters/llm/gemini.py` - GeminiLLM
- `src/fathom/adapters/memory/sqlite.py` - SQLiteMemory
- `src/fathom/adapters/signal/noop.py` - NoopSignal
- `src/fathom/adapters/storage/local.py` - LocalStorage
- `src/fathom/adapters/telemetry/structlog.py` - StructlogTelemetry

## Shared Code (Used by Both)

### Tools (Still used by old code)
- `src/fathom/tools/device/adb.py` - ADBDeviceTool
- `src/fathom/tools/capture/adb.py` - ADBCaptureTool
- `src/fathom/tools/vision/gemini.py` - GeminiVisionTool

### Infrastructure (Used by both)
- `src/fathom/infrastructure/llm/` - LLM clients
- `src/fathom/infrastructure/memory/` - Memory providers
- `src/fathom/infrastructure/storage/` - Storage providers

### Agent Components (Used by both)
- `src/fathom/agent/planner.py` - StepPlanner
- `src/fathom/agent/reasoner.py` - Reasoner
- `src/fathom/agent/state.py` - AgentState

### Schemas (Shared)
- `src/fathom/schemas/` - All domain models

### Services (Shared)
- `src/fathom/services/` - Audit, History, Resolution, UX

## Action Plan

### 6.1 Add Deprecation Warnings ✅
Add deprecation warnings to old code so users know to migrate:
- Old FathomRunner
- Old workflows
- Old CLI

### 6.2 Document Migration Path
Create migration guide for users of old code:
- How to switch from `fathom-old` to `fathom`
- API differences
- Configuration changes

### 6.3 Prevent Accidental Old Code Usage
Add checks to ensure new code doesn't import old code:
- Lint rule or test to detect imports from `orchestration/` or `workflows/`
- Exception: Backward compatibility shims are allowed

### 6.4 Mark Old Code as Legacy
Add clear markers in old code files:
- Module docstrings indicating "LEGACY - Use hexagonal architecture instead"
- Comments pointing to new equivalents

### 6.5 Future Cleanup (Not Now)
Eventually (after migration period):
- Remove old orchestration code
- Remove old workflows
- Remove old CLI
- Keep only the tools/infrastructure that are still useful

## Decision: Keep Old Code for Now

**Rationale**:
1. Backward compatibility for existing users
2. Safety net during migration
3. Reference implementation for comparison
4. No harm in keeping it (separate entry point)

**Strategy**:
- Mark as deprecated
- Document clearly
- Prevent new code from using it
- Plan removal for future major version
