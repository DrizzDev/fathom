# Confidence Attribute Bug Fix

## Issue
`AttributeError: 'PlanResult' object has no attribute 'confidence'` at line 202 in `src/fathom/strategies/intent.py`

## Root Cause
The code was checking `plan.confidence`, but the `PlanResult` schema doesn't have a `confidence` attribute. The confidence is actually on the `Action` object inside the `Step`, accessed via `plan.step.action.confidence`.

## Schema Structure
```
PlanResult
  ├── reason: str
  ├── is_complete: bool
  ├── step: Optional[Step]
  │     ├── action: Action
  │     │     ├── confidence: float  ← HERE
  │     │     ├── action_type: ActionType
  │     │     └── ...
  │     ├── screen_hash: str
  │     └── step_number: int
  ├── memories: int
  └── ...
```

## Fix Applied

### 1. Fixed confidence check (line 205)
**Before:**
```python
if plan.confidence and plan.confidence < 0.5:
```

**After:**
```python
if plan.step and plan.step.action.confidence < 0.5:
```

### 2. Fixed confidence reference in warning (line 208)
**Before:**
```python
confidence=plan.confidence,
```

**After:**
```python
confidence=plan.step.action.confidence,
```

### 3. Fixed confidence reference in question string (line 213)
**Before:**
```python
f"The agent is uncertain (confidence: {plan.confidence:.1%}) about what to do next.\n"
```

**After:**
```python
f"The agent is uncertain (confidence: {plan.step.action.confidence:.1%}) about what to do next.\n"
```

### 4. Added missing signal port to IntentStrategy

The HITL code was referencing `self.__signal` but it wasn't being passed to the constructor.

**Changes:**
- Added `signal: SignalPort` parameter to `IntentStrategy.__init__()`
- Added `self.__signal = signal` in constructor
- Added `from fathom.interfaces.signal import SignalPort` import
- Updated `FathomRunner.run_intent()` to pass `signal=self._signal`
- Updated `test_hexagonal_architecture.py` to pass signal port

## Files Modified
1. `src/fathom/strategies/intent.py` - Fixed confidence checks and added signal port
2. `src/fathom/runtime/runner.py` - Pass signal port to IntentStrategy
3. `test_hexagonal_architecture.py` - Pass signal port in test

## Verification
- ✅ No diagnostic errors in modified files
- ✅ Correct attribute path: `plan.step.action.confidence`
- ✅ Signal port properly wired through constructor
- ✅ All references updated consistently

## Next Steps
Run the same command that triggered the error to verify the fix:
```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" --use-xml --serial emulator-5554 -v
```
