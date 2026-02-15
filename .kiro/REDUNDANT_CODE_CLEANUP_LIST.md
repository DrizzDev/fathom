# Redundant Code Cleanup List

## Overview
After the hexagonal architecture migration is complete and stable, the following files and code can be safely deleted. This document provides a comprehensive list organized by category.

---

## 🔴 CRITICAL: Files to Delete (Old Architecture)

### 1. Old CLI
**Delete:**
- `src/fathom/cli.py` - Old CLI using direct tool wiring
- Entry point in `pyproject.toml`: `fathom-old = "fathom.cli:main"`

**Reason:** Replaced by `src/fathom/cli_new.py` (which should be renamed to `cli.py`)

---

### 2. Old Orchestration Layer
**Delete entire directory:**
- `src/fathom/orchestration/` (entire directory)
  - `src/fathom/orchestration/__init__.py`
  - `src/fathom/orchestration/context.py` - Old context management
  - `src/fathom/orchestration/executor.py` - Old StepExecutor
  - `src/fathom/orchestration/runner/__init__.py`
  - `src/fathom/orchestration/runner/fathom.py` - Old FathomRunner
  - `src/fathom/orchestration/runner/workflow.py` - Old WorkflowRunner

**Reason:** Replaced by:
- `src/fathom/runtime/runner.py` - New FathomRunner
- `src/fathom/runtime/builder.py` - Builder API
- `src/fathom/core/execution/engine.py` - ExecutionEngine
- `src/fathom/core/context/manager.py` - ContextManager

---

### 3. Old Workflows Layer
**Delete entire directory:**
- `src/fathom/workflows/` (entire directory)
  - `src/fathom/workflows/__init__.py`
  - `src/fathom/workflows/base.py` - BaseWorkflow
  - `src/fathom/workflows/intent.py` - Old IntentWorkflow
  - `src/fathom/workflows/exploration.py` - Old ExplorationWorkflow

**Reason:** Replaced by:
- `src/fathom/strategies/intent.py` - IntentStrategy (uses ports)
- `src/fathom/strategies/exploration.py` - ExplorationStrategy (uses ports)

---

### 4. Old Agent Strategies
**Delete entire directory:**
- `src/fathom/agent/strategies/` (entire directory)
  - `src/fathom/agent/strategies/__init__.py`
  - `src/fathom/agent/strategies/base.py` - Old ExecutionStrategy base
  - `src/fathom/agent/strategies/intent.py` - Old IntentStrategy (different from new one)
  - `src/fathom/agent/strategies/exploration.py` - Old ExplorationStrategy (different from new one)

**Reason:** Replaced by:
- `src/fathom/strategies/intent.py` - New IntentStrategy (hexagonal)
- `src/fathom/strategies/exploration.py` - New ExplorationStrategy (hexagonal)

**Note:** These are DIFFERENT from the new strategies in `src/fathom/strategies/`. The old ones use direct tool wiring, the new ones use ports.

---

### 5. Old Tool Abstractions (Partially)
**Delete:**
- `src/fathom/tools/base.py` - Old tool base classes
- `src/fathom/tools/definitions.py` - Old tool definitions
- `src/fathom/tools/device/base.py` - Old DeviceTool base
- `src/fathom/tools/capture/base.py` - Old CaptureTool base
- `src/fathom/tools/vision/base.py` - Old VisionTool base

**Keep (still used):**
- `src/fathom/tools/device/adb.py` - ADBDeviceTool (used by old code, but logic can be extracted)
- `src/fathom/tools/capture/adb.py` - ADBCaptureTool (used by old code)
- `src/fathom/tools/vision/gemini.py` - GeminiVisionTool (still used by NEW strategies via adapters)
- `src/fathom/tools/capture/hasher.py` - Utility functions (still useful)

**Reason:** Tool abstractions replaced by Port interfaces:
- `src/fathom/interfaces/device.py` - DevicePort
- `src/fathom/interfaces/llm.py` - LLMPort
- `src/fathom/interfaces/storage.py` - StoragePort

---

### 6. Mock Tools (Test Utilities)
**Delete:**
- `src/fathom/tools/device/mock.py` - Old mock device
- `src/fathom/tools/capture/mock.py` - Old mock capture
- `src/fathom/tools/vision/mock.py` - Old mock vision

**Reason:** Should create new mocks that implement Port interfaces instead

---

## 🟡 MEDIUM PRIORITY: Refactor Then Delete

### 7. Tool Implementations (Extract Logic)
**Current state:** These contain business logic that's still used

**Action required:**
1. Extract core logic from tools
2. Move logic to adapters or services
3. Delete tool wrappers

**Files:**
- `src/fathom/tools/device/adb.py` → Logic already in `src/fathom/adapters/device/adb.py`
- `src/fathom/tools/capture/adb.py` → Logic can be merged into device adapter
- `src/fathom/tools/vision/gemini.py` → Still used by strategies, needs refactoring

**Note:** `GeminiVisionTool` is still used by the NEW strategies through adapters. This needs careful refactoring.

---

### 8. Processing Module Re-exports
**Delete:**
- `src/fathom/tools/vision/processing/__init__.py` - Re-exports from `src/fathom/processing/`

**Reason:** Direct imports from `src/fathom/processing/` should be used instead

---

## 🟢 LOW PRIORITY: Documentation and Metadata

### 9. Old Schemas (If Unused)
**Check and potentially delete:**
- `src/fathom/schemas/orchestration.py` - ExecutionContext (if only used by old orchestration)

**Action:** Verify no new code uses these schemas, then delete

---

### 10. Old Exception Classes
**Check:**
- `src/fathom/exceptions.py` - Old exception classes
- `src/fathom/core/exceptions.py` - New exception classes

**Action:** Consolidate into one file, delete duplicates

---

## 📋 Cleanup Checklist

### Phase 1: Immediate Cleanup (Safe)
- [ ] Delete `src/fathom/cli.py` (old CLI)
- [ ] Remove `fathom-old` entry point from `pyproject.toml`
- [ ] Delete `src/fathom/orchestration/` directory
- [ ] Delete `src/fathom/workflows/` directory
- [ ] Delete `src/fathom/agent/strategies/` directory
- [ ] Delete mock tools: `tools/device/mock.py`, `tools/capture/mock.py`, `tools/vision/mock.py`
- [ ] Delete old tool base classes: `tools/base.py`, `tools/definitions.py`
- [ ] Delete `tools/vision/processing/__init__.py` (re-export shim)

### Phase 2: Refactor Then Delete
- [ ] Refactor `GeminiVisionTool` to not be needed by new strategies
- [ ] Extract any remaining logic from `tools/device/adb.py` to adapter
- [ ] Extract any remaining logic from `tools/capture/adb.py` to adapter
- [ ] Delete `src/fathom/tools/device/adb.py`
- [ ] Delete `src/fathom/tools/capture/adb.py`
- [ ] Delete `src/fathom/tools/device/base.py`
- [ ] Delete `src/fathom/tools/capture/base.py`
- [ ] Delete `src/fathom/tools/vision/base.py`

### Phase 3: Final Cleanup
- [ ] Consolidate exception classes (merge `exceptions.py` and `core/exceptions.py`)
- [ ] Delete unused schemas from `schemas/orchestration.py`
- [ ] Delete empty `tools/` directory if nothing remains
- [ ] Rename `cli_new.py` to `cli.py`
- [ ] Update all imports to use new names

---

## 🔍 Verification Steps

### Before Deleting Each File:
1. **Search for imports:**
   ```bash
   grep -r "from fathom.orchestration" src/
   grep -r "from fathom.workflows" src/
   grep -r "from fathom.agent.strategies" src/
   ```

2. **Check if only old code imports it:**
   - If only `cli.py` (old CLI) imports it → Safe to delete
   - If only `orchestration/` or `workflows/` imports it → Safe to delete
   - If new code (`strategies/`, `runtime/`, `core/`) imports it → Need to refactor first

3. **Run tests:**
   ```bash
   pytest tests/
   ```

4. **Verify CLI works:**
   ```bash
   fathom --help
   fathom run "test intent" --serial emulator-5554
   ```

---

## 📊 Size Reduction Estimate

### Files to Delete:
- **Old CLI:** 1 file (~500 lines)
- **Old Orchestration:** 5 files (~1,500 lines)
- **Old Workflows:** 4 files (~1,200 lines)
- **Old Agent Strategies:** 4 files (~800 lines)
- **Old Tool Abstractions:** 6 files (~600 lines)
- **Mock Tools:** 3 files (~300 lines)

**Total:** ~23 files, ~4,900 lines of code

### Directories to Delete:
- `src/fathom/orchestration/` (entire directory)
- `src/fathom/workflows/` (entire directory)
- `src/fathom/agent/strategies/` (entire directory)

---

## ⚠️ Important Notes

### DO NOT Delete:
- ✅ `src/fathom/agent/planner.py` - Still used by new strategies
- ✅ `src/fathom/agent/reasoner.py` - Still used by new strategies
- ✅ `src/fathom/agent/state.py` - Still used by new strategies
- ✅ `src/fathom/infrastructure/` - Still used by adapters
- ✅ `src/fathom/schemas/` - Shared domain models
- ✅ `src/fathom/services/` - Shared services
- ✅ `src/fathom/prompts/` - Still used by vision tools
- ✅ `src/fathom/processing/` - Still used by vision tools
- ✅ `src/fathom/tools/vision/gemini.py` - Still used by new strategies (needs refactoring)
- ✅ `src/fathom/tools/capture/hasher.py` - Utility functions

### Backward Compatibility Shims:
Currently, some `__init__.py` files re-export new classes for backward compatibility:
- `src/fathom/workflows/__init__.py` - Re-exports strategies
- `src/fathom/orchestration/__init__.py` - Re-exports runtime classes

**Action:** Delete these shims when deleting the directories.

---

## 🎯 Recommended Deletion Order

### Step 1: Delete Old CLI (Safest)
```bash
rm src/fathom/cli.py
# Remove fathom-old entry from pyproject.toml
```

### Step 2: Delete Old Orchestration
```bash
rm -rf src/fathom/orchestration/
```

### Step 3: Delete Old Workflows
```bash
rm -rf src/fathom/workflows/
```

### Step 4: Delete Old Agent Strategies
```bash
rm -rf src/fathom/agent/strategies/
```

### Step 5: Delete Mock Tools
```bash
rm src/fathom/tools/device/mock.py
rm src/fathom/tools/capture/mock.py
rm src/fathom/tools/vision/mock.py
```

### Step 6: Delete Old Tool Abstractions
```bash
rm src/fathom/tools/base.py
rm src/fathom/tools/definitions.py
rm src/fathom/tools/device/base.py
rm src/fathom/tools/capture/base.py
rm src/fathom/tools/vision/base.py
rm src/fathom/tools/vision/processing/__init__.py
```

### Step 7: Refactor and Delete Tool Implementations
(Requires code changes first)

### Step 8: Final Cleanup
```bash
# Rename cli_new.py to cli.py
mv src/fathom/cli_new.py src/fathom/cli.py

# Update pyproject.toml entry point
# fathom = "fathom.cli:main"
```

---

## 🧪 Testing After Each Deletion

After deleting each group of files, run:

```bash
# 1. Check imports
python -c "import fathom; print('Import OK')"

# 2. Run tests
pytest tests/

# 3. Test CLI
fathom --help
fathom run "test" --serial emulator-5554 --max-steps 1

# 4. Check for broken imports
grep -r "from fathom.orchestration" src/
grep -r "from fathom.workflows" src/
grep -r "from fathom.agent.strategies" src/
```

---

## 📝 Migration Guide for Users

If any external code depends on the old architecture, provide this migration guide:

### Old Code:
```python
from fathom.orchestration.runner import FathomRunner
from fathom.workflows.intent import IntentWorkflow

runner = FathomRunner(...)
result = await runner.run_workflow(IntentWorkflow(...))
```

### New Code:
```python
from fathom.runtime.builder import Fathom
from fathom.adapters.device.adb import ADBDevice
from fathom.adapters.llm.gemini import GeminiLLM

runner = (
    Fathom.builder()
    .device(ADBDevice(serial="emulator-5554"))
    .llm(GeminiLLM(api_key="..."))
    .build()
)

result = await runner.run_intent(intent="...", max_steps=20)
```

---

## 🎉 Expected Outcome

After cleanup:
- ✅ ~5,000 lines of redundant code removed
- ✅ 3 major directories deleted
- ✅ Cleaner architecture with single implementation
- ✅ Easier maintenance and testing
- ✅ No confusion about which code to use
- ✅ Smaller codebase, faster CI/CD

---

## 🚨 Final Warning

**DO NOT delete anything until:**
1. ✅ New architecture is fully tested in production
2. ✅ All users have migrated to new API
3. ✅ Backward compatibility period has ended
4. ✅ You have backups/git history to revert if needed

**Recommended timeline:**
- **Now:** Mark old code as deprecated
- **1-2 months:** Migration period, both APIs work
- **After migration:** Delete old code in phases
- **Major version bump:** Complete removal (e.g., v2.0.0)
