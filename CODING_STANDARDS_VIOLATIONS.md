# Coding Standards Violations

Analysis of hexagonal architecture files against `documents/playbook/coding.md`

**STATUS: ✅ ALL VIOLATIONS FIXED (100% COMPLIANT)**

---

## ✅ FIXED Violations

### 1. ✅ Inline Imports - FIXED
**Files**: `src/fathom/core/execution/engine.py`, `src/fathom/strategies/intent.py`
**Status**: Fixed - moved `import hashlib` to top of files

### 2. ✅ Domain-Specific Exceptions - FIXED
**File**: `src/fathom/core/exceptions.py`
**Status**: Created with `FathomError`, `ExecutionError`, `ConfigurationError`, `StrategyError`, `PortError`

### 3. ✅ Exception Handling - FIXED
**Files**: All strategy and core files
**Status**: Fixed - now catching specific exceptions (`ToolError`, `PortError`, `StrategyError`) before falling back to generic `Exception`

### 4. ✅ Magic Numbers - FIXED
**File**: `src/fathom/core/execution/engine.py`
**Status**: Extracted to constants:
- `VISUAL_HASH_LENGTH = 16`
- `DEFAULT_SWIPE_DISTANCE = 300`
- `DEFAULT_SCROLL_DISTANCE = 200`
- `DEFAULT_SWIPE_DURATION = 500`
- `BOUNDS_SWIPE_DISTANCE = 100`

### 5. ✅ Unused Imports - FIXED
**File**: `src/fathom/core/execution/engine.py`
**Status**: Removed unused imports (`Any`, `Dict`, `AnalysisResult`, `ScreenState`)

### 6. ✅ Type Annotation Inconsistency - FIXED
**File**: `src/fathom/core/execution/engine.py`
**Status**: Changed `tuple[int, int]` to `Tuple[int, int]` for consistency

### 7. ✅ Error Messages - FIXED
**File**: `src/fathom/runtime/builder.py`
**Status**: Changed from `ValueError` to `ConfigurationError` with descriptive messages

---

## Compliance Score: 100% ✅

All critical, medium, and minor violations have been fixed. The codebase now fully complies with `documents/playbook/coding.md`.
