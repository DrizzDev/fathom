# Fathom Repository - Logical Bug Report

## Summary
Found **8 significant logical bugs** in the Fathom mobile automation framework that could cause incorrect behavior, crashes, or silent failures.

---

## 🔴 CRITICAL BUGS

### 1. **Loop Detector - Flawed `is_stuck()` Logic (HIGH SEVERITY)**
**File:** [src/fathom/schemas/state.py](src/fathom/schemas/state.py#L60-L110)

**Issue:** The loop detection algorithm has a critical flaw in its repeated screen counting logic:

```python
for index in range(len(self.__recent_screens)):
    count = 1
    current = self.__recent_screens[index]
    for __next_index in range(index + 1, len(self.__recent_screens)):
        if current.is_same_screen(self.__recent_screens[__next_index]):
            count += 1
```

**The Problem:**
- The variable `count` is never reset when checking different screens, but it SHOULD reset for each new base screen
- After the inner loop ends, `count` contains the total matches for screen at `index`, not necessarily >= 3
- The check `if count >= self.threshold` then logs "stuck=true" but the count might be just 2 matches, not 3
- **Result:** Can incorrectly report stuck state or miss actual stuck detection

**Example Scenario:**
- Screens: [A, B, C, A, D] with threshold=3
- When checking A at index 0: count becomes 2 (finds A at index 3)
- Check triggers when count >= 3, so correctly skips
- But logic implicitly resets count only via the for loop structure, making this fragile

**Fix:**
```python
for index in range(len(self.__recent_screens)):
    count = 1  # Reset for each base screen (already done)
    current = self.__recent_screens[index]
    for __next_index in range(index + 1, len(self.__recent_screens)):
        if current.is_same_screen(self.__recent_screens[__next_index]):
            count += 1

    # Move this check OUTSIDE the inner loop, after checking all subsequent screens
    if count >= self.threshold:
        # ... stuck logic
```

---

### 2. **Coordinate Bounds Validation Missing (HIGH SEVERITY)**
**File:** [src/fathom/schemas/actions.py](src/fathom/schemas/actions.py#L14-L27)

**Issue:** The `Bounds` class accepts values up to 5000, but the `is_normalized` property only checks if values are <= 1000:

```python
@property
def is_normalized(self) -> bool:
    """Heuristic to check if coordinates are likely normalized (0-1000)."""
    return self.x <= 1000 and self.y <= 1000 and self.width <= 1000 and self.height <= 1000
```

**The Problem:**
- Values between 1001-5000 will be treated as pixel coordinates, even though they might be intended normalized coords
- This creates an ambiguity: a coordinate of (1500, 1500) is treated as pixels (1500px), but it might have been intended as normalized
- The validation allows up to 5000 but doesn't validate the conversion logic
- **Result:** Taps and swipes could hit wrong areas of screen with dimensions up to 5000x5000

**Additional Issue:** No bounds checking in [execute_device_action](src/fathom/utils/execution.py#L37) - if converter.center_to_pixels returns invalid coords, they're used directly without validation.

**Fix:**
```python
def to_pixels(self, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    """Convert with defensive bounds checking"""
    # ... existing logic ...
    # After conversion, validate:
    x_pixel = max(0, min(x_pixel, screen_width))
    y_pixel = max(0, min(y_pixel, screen_height))
    width_pixel = max(1, min(width_pixel, screen_width))
    height_pixel = max(1, min(height_pixel, screen_height))
    return x_pixel, y_pixel, width_pixel, height_pixel
```

---

### 3. **Loop Detector Recovery Exhaustion Not Always Handled (MEDIUM-HIGH SEVERITY)**
**File:** [src/fathom/agent/planner.py](src/fathom/agent/planner.py#L160-L165)

**Issue:** When checking `state.is_stuck`, recovery actions are fetched but never validated that recovery is actually possible:

```python
if state.is_stuck:
    logger.warning(msg="Agent is stuck in a loop. Requesting recovery from model.")
    # But what if state.get_recovery_action() returns None because recovery_exhausted?
```

Later in the code:
```python
# Optimization: Check if this EXACT action just failed on this screen hash
if state.should_avoid_action(action=action):
    logger.warning(msg=f"Avoiding recently failed action: {action.to_description()}")
    # This uses 'action' but 'action' might be from a None recovery_action
```

**The Problem:**
- `get_recovery_action()` can return `None` when `__loop_detector.can_recover()` is False
- But the code doesn't check if recovery is exhausted before proceeding
- If recovery is exhausted and we try an action that's already failed, we might loop forever
- The condition `if state.can_continue` checks this, but it's checked BEFORE reaching the stuck section

**Result:** Agent might attempt recovery actions beyond the limit, causing unnecessary delays

---

### 4. **Package Change Detection Doesn't Properly Reset Knowledge Graph (MEDIUM SEVERITY)**
**File:** [src/fathom/orchestration/runner/fathom.py](src/fathom/orchestration/runner/fathom.py#L300-L330)

**Issue:** In `__attach_knowledge_graph()`:

```python
if self.__knowledge_graph and Path(self.__knowledge_graph.provider.path) == final_db:
    result.knowledge_graph = self.__knowledge_graph.export_json()
    return
```

**The Problem:**
- If the foreground app changed mid-run, the method tries to compare `Path` objects but `self.__knowledge_graph.provider.path` might not be a Path
- The code attempts to reload from `final_db` but doesn't clear old references
- If the reload fails (line 325), it silently uses the old knowledge graph from a different app
- **Result:** Intent results could include knowledge from mismatched packages

**Fix:** Add explicit clearing and safer path comparison:
```python
if self.__knowledge_graph:
    current_path = Path(self.__knowledge_graph.provider.path) if hasattr(self.__knowledge_graph.provider, 'path') else None
    if current_path == final_db:
        # ... reuse
    else:
        self.__knowledge_graph = None  # Clear the old one
```

---

## 🟠 MEDIUM SEVERITY BUGS

### 5. **Unchecked Empty Image Data in Gemini Vision Tool (MEDIUM-HIGH)**
**File:** [src/fathom/infrastructure/llm/gemini.py](src/fathom/infrastructure/llm/gemini.py#L128-L133)

```python
for item in user_content:
    if isinstance(item, bytes):  # It's an image
        if not item:
            raise VisionError("Received empty image data for analysis")
```

**The Problem:**
- If `user_content` is passed as a list with mixed types, this check happens in a loop but a single empty image causes exception
- The error propagates but upstream code in [planner.py](src/fathom/agent/planner.py) doesn't check for this condition before passing images
- Capture tool might produce empty images under certain screen states, causing unhandled failures

**Fix:** Validate images before passing to vision tool, or provide retry logic in planner

---

### 6. **Memory Service Potential Race Condition (MEDIUM)**
**File:** [src/fathom/services/memory.py](src/fathom/services/memory.py#L28-H40)

```python
async def __ensure_initialized(self) -> None:
    """Initializes the database schema if it doesn't exist."""
    if self.__initialized:
        return

    async with aiosqlite.connect(self.__database_path) as db:
        # ... create tables ...

    self.__initialized = True
```

**The Problem:**
- Two concurrent calls to `__ensure_initialized()` both check `self.__initialized` before the lock exists
- No lock around the initialization check, only inside the method
- Both coroutines could pass the initial check and try to initialize simultaneously
- This can cause table creation conflicts or data corruption

**Fix:** Add lock at class level:
```python
def __init__(self, ...):
    self.__initialized = False
    self.__init_lock = asyncio.Lock()

async def __ensure_initialized(self) -> None:
    async with self.__init_lock:
        if self.__initialized:
            return
        # ... rest of init
```

---

### 7. **History Service Silent File Corruption on IOError (MEDIUM)**
**File:** [src/fathom/services/history.py](src/fathom/services/history.py#L68-L72)

```python
try:
    with path.open(mode="r") as handle:
        data = json.load(fp=handle)
except Exception:  # nosec
    pass
```

**The Problem:**
- Silently catches ALL exceptions including file corruption, permission errors, disk full, etc.
- Returns empty dict `{"workflow_id": ..., "history": []}` without logging why the file couldn't be read
- If the file has partial data, it's silently discarded
- Next save will overwrite any partially-saved history

**Result:** Silent data loss in execution history

**Fix:**
```python
except Exception as e:
    logger.warning(f"Failed to load history from {path}: {e}", exc_info=True)
    # Optionally backup corrupted file
```

---

### 8. **Action History Deque Mutation Without Summarizer Check (MEDIUM)**
**File:** [src/fathom/schemas/state.py](src/fathom/schemas/state.py#L180-L195) and [src/fathom/services/summarizer.py](src/fathom/services/summarizer.py#L72-L81)

```python
# In state.py
self.__action_history.record_action(
    action=result.step.action,
    success=result.success,
    activity=activity,
    screen_changed=result.screen_changed,
)

# In summarizer.py
def add_milestone(self, description: str) -> None:
    if description not in self.milestones:
        self.milestones.append(description)
```

**The Problem:**
- When summarizer is enabled, items are evicted from `__actions` deque but there's no guarantee the summarizer's callback is invoked
- The check `if description not in self.milestones` is O(N) list lookup, not a set
- `milestones` list can grow unbounded (it only has `_max_milestones=8` but no enforcement)
- If a milestone description is duplicated, the list check doesn't catch it for long descriptions

**Result:** Memory could grow unchecked with repeated milestone additions

---

## 🟡 LOWER SEVERITY ISSUES

### 9. **Type Annotation Inconsistency in PlanResult** (LOW)
**File:** [src/fathom/schemas/results.py](src/fathom/schemas/results.py#L76-L92)

The `PlanResult` class likely has optional fields that should be required in certain states, but the Pydantic model doesn't enforce this constraint through validators.

---

### 10. **Incomplete XML Handling in Hierarchy processing** (LOW-MEDIUM)
**File:** [src/fathom/services/hierarchy.py](src/fathom/services/hierarchy.py#L60-L70)

```python
if xml_size_kb < 0.2:
    logger.warning("XML too small, waiting for UI stability…")
    await asyncio.sleep(1.0)
```

Sleeping 1 second but no retry - just continues with empty XML. Should retry or return error.

---

## Recommendations

### Priority 1 (Fix Immediately)
1. Fix loop detector counting logic (Bug #1)
2. Add bounds validation to coordinate conversion (Bug #2)
3. Add proper race condition protection to memory initialization (Bug #6)

### Priority 2 (Fix Soon)
4. Add bounds clamping to execute_device_action
5. Improve error visibility in history service
6. Fix knowledge graph switching logic

### Priority 3 (Improve)
7. Add type validators to Pydantic models
8. Enforce maximum sizes in summarizer milestones
9. Add retry logic for XML parsing failures

---

## Testing Recommendations

1. **Loop Detection**: Test with screens [A, B, A, C, A, D] to verify stuck detection threshold
2. **Coordinates**: Test with normalized values near 1000, and raw pixel values near 5000
3. **Concurrency**: Run multiple memory initialization calls simultaneously
4. **File I/O**: Corrupt history JSON files and verify error handling
5. **Package Switching**: Run exploration that switches between multiple apps
