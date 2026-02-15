# Redundant Code Cleanup - Quick Summary

## 🎯 What to Delete After Migration is Complete

### 🔴 HIGH PRIORITY (Safe to Delete Immediately)

#### 1. Old CLI
- `src/fathom/cli.py`
- `fathom-old` entry in `pyproject.toml`

#### 2. Old Orchestration (Entire Directory)
- `src/fathom/orchestration/`
  - `orchestration/context.py`
  - `orchestration/executor.py`
  - `orchestration/runner/fathom.py`
  - `orchestration/runner/workflow.py`

#### 3. Old Workflows (Entire Directory)
- `src/fathom/workflows/`
  - `workflows/base.py`
  - `workflows/intent.py`
  - `workflows/exploration.py`

#### 4. Old Agent Strategies (Entire Directory)
- `src/fathom/agent/strategies/`
  - `agent/strategies/base.py`
  - `agent/strategies/intent.py`
  - `agent/strategies/exploration.py`

#### 5. Mock Tools
- `src/fathom/tools/device/mock.py`
- `src/fathom/tools/capture/mock.py`
- `src/fathom/tools/vision/mock.py`

#### 6. Old Tool Abstractions
- `src/fathom/tools/base.py`
- `src/fathom/tools/definitions.py`
- `src/fathom/tools/device/base.py`
- `src/fathom/tools/capture/base.py`
- `src/fathom/tools/vision/base.py`

---

### 🟡 MEDIUM PRIORITY (Refactor First)

#### 7. Tool Implementations (Extract Logic First)
- `src/fathom/tools/device/adb.py` - Logic already in adapter
- `src/fathom/tools/capture/adb.py` - Merge into device adapter
- `src/fathom/tools/vision/gemini.py` - Still used, needs refactoring

---

### 🟢 LOW PRIORITY (Verify First)

#### 8. Unused Schemas
- Check `src/fathom/schemas/orchestration.py` for unused classes

#### 9. Duplicate Exceptions
- Consolidate `src/fathom/exceptions.py` and `src/fathom/core/exceptions.py`

---

## 📊 Impact

### Files to Delete: ~23 files
### Lines to Remove: ~4,900 lines
### Directories to Delete: 3 major directories

---

## ✅ Verification Command

Before deleting, verify no new code uses old code:

```bash
# Check for imports of old code
grep -r "from fathom.orchestration" src/fathom/strategies/
grep -r "from fathom.orchestration" src/fathom/runtime/
grep -r "from fathom.orchestration" src/fathom/core/

grep -r "from fathom.workflows" src/fathom/strategies/
grep -r "from fathom.workflows" src/fathom/runtime/
grep -r "from fathom.workflows" src/fathom/core/

grep -r "from fathom.agent.strategies" src/fathom/strategies/
grep -r "from fathom.agent.strategies" src/fathom/runtime/
grep -r "from fathom.agent.strategies" src/fathom/core/
```

**Expected result:** No matches (or only matches in old code like `cli.py`)

---

## 🚀 Quick Deletion Script

```bash
#!/bin/bash
# Run this after verifying new architecture is stable

# Phase 1: Delete old architecture
rm src/fathom/cli.py
rm -rf src/fathom/orchestration/
rm -rf src/fathom/workflows/
rm -rf src/fathom/agent/strategies/

# Phase 2: Delete mock tools
rm src/fathom/tools/device/mock.py
rm src/fathom/tools/capture/mock.py
rm src/fathom/tools/vision/mock.py

# Phase 3: Delete old tool abstractions
rm src/fathom/tools/base.py
rm src/fathom/tools/definitions.py
rm src/fathom/tools/device/base.py
rm src/fathom/tools/capture/base.py
rm src/fathom/tools/vision/base.py
rm src/fathom/tools/vision/processing/__init__.py

# Phase 4: Rename new CLI
mv src/fathom/cli_new.py src/fathom/cli.py

# Phase 5: Update pyproject.toml
# Remove: fathom-old = "fathom.cli:main"
# Update: fathom = "fathom.cli:main" (was cli_new)

echo "Cleanup complete! Run tests to verify."
```

---

## ⚠️ DO NOT Delete

These are still used by the new architecture:

- ✅ `src/fathom/agent/planner.py`
- ✅ `src/fathom/agent/reasoner.py`
- ✅ `src/fathom/agent/state.py`
- ✅ `src/fathom/infrastructure/`
- ✅ `src/fathom/schemas/`
- ✅ `src/fathom/services/`
- ✅ `src/fathom/prompts/`
- ✅ `src/fathom/processing/`
- ✅ `src/fathom/adapters/`
- ✅ `src/fathom/interfaces/`
- ✅ `src/fathom/core/`
- ✅ `src/fathom/runtime/`
- ✅ `src/fathom/strategies/`

---

## 📅 Recommended Timeline

1. **Now:** Mark old code as deprecated (add warnings)
2. **1-2 months:** Migration period (both APIs work)
3. **After migration:** Delete in phases (test after each phase)
4. **Major version:** Complete removal (e.g., v2.0.0)

---

## 🧪 Test After Deletion

```bash
# 1. Import check
python -c "import fathom; print('OK')"

# 2. Run tests
pytest tests/

# 3. CLI check
fathom --help
fathom run "test" --serial emulator-5554 --max-steps 1

# 4. Verify no broken imports
grep -r "from fathom.orchestration" src/
grep -r "from fathom.workflows" src/
grep -r "from fathom.agent.strategies" src/
```

---

## 📖 Full Details

See `.kiro/REDUNDANT_CODE_CLEANUP_LIST.md` for:
- Detailed file-by-file breakdown
- Refactoring requirements
- Migration guide for users
- Step-by-step deletion order
- Testing procedures

---

## 🎉 Expected Result

After cleanup:
- Cleaner codebase (~5,000 lines removed)
- Single implementation (no confusion)
- Easier maintenance
- Faster CI/CD
- Better developer experience
