# Critical Issues Found in Hexagonal Architecture Migration

## 🚨 CRITICAL: IntentStrategy is Broken

### The Problem
The `IntentStrategy` in `src/fathom/strategies/intent.py` **WILL CRASH** when executed because:

```python
# Line 85 in strategies/intent.py
self.__planner = StepPlanner(vision_tool=None)  # ❌ BROKEN!
```

The StepPlanner requires a VisionTool, but we're passing `None`. When `plan_step()` is called, it will crash with `AttributeError: 'NoneType' object has no attribute 'analyze'`.

### Root Cause: Architecture Mismatch

We have TWO parallel systems:

**OLD System** (agent/strategies/intent.py):
- Uses: IVisionProvider → VisionTool → StepPlanner
- Works with: DeviceTool, CaptureTool, IMemoryProvider
- Status: ✅ Working (used by CLI)

**NEW System** (strategies/intent.py):
- Uses: LLMPort → ??? → StepPlanner
- Works with: DevicePort, StoragePort, MemoryPort
- Status: ❌ BROKEN (missing bridge)

The problem: We're trying to use OLD agent components (StepPlanner, Reasoner, AgentState) with NEW ports (LLMPort), but there's no bridge between them.

### Why This Happened

The migration strategy was:
1. ✅ Create new ports (DevicePort, LLMPort, etc.)
2. ✅ Create adapters that implement ports
3. ❌ **MISTAKE**: Try to reuse old agent components without adapting them

The old agent components expect `IVisionProvider` (old interface), but we have `LLMPort` (new interface).

### The Fix: Three Options

#### Option 1: Create IVisionProvider Adapter (RECOMMENDED)
Create an adapter that makes LLMPort look like IVisionProvider:

```python
# src/fathom/adapters/vision/llm_provider.py
class LLMVisionProvider(IVisionProvider):
    def __init__(self, llm: LLMPort, storage: StoragePort):
        self.__llm = llm
        self.__storage = storage
    
    async def analyze(self, system_instruction, user_content, tools):
        return await self.__llm.analyze(
            system_instruction=system_instruction,
            user_content=user_content,
            tools=tools
        )
    
    # ... implement other IVisionProvider methods
```

Then in IntentStrategy:
```python
from fathom.adapters.vision.llm_provider import LLMVisionProvider
from fathom.tools.vision.gemini import GeminiVisionTool

# Create provider adapter
provider = LLMVisionProvider(llm=self.__llm, storage=self.__storage)

# Create vision tool
vision_tool = GeminiVisionTool(
    model=provider,
    memory=self.__memory,  # Need to adapt this too
    ledger=self.__ledger,
    local_storage=self.__storage,  # Need to adapt this too
)

# Now StepPlanner works!
self.__planner = StepPlanner(vision_tool=vision_tool)
```

#### Option 2: Don't Reuse Old Agent Components
Rewrite IntentStrategy to NOT use StepPlanner, Reasoner, AgentState. Instead, implement the logic directly using the new ports. This is a LOT of work.

#### Option 3: Keep Using Old System
Don't migrate IntentStrategy at all. Keep using `agent/strategies/intent.py` through the CLI. The new hexagonal architecture is just for the builder API, but internally it still uses the old system.

This is actually what's happening now - the CLI uses the old system, and it works fine.

---

## 🟡 MINOR: ExplorationStrategy Package Name

### Problem
In `src/fathom/strategies/exploration.py`, line 395:

```python
def __compute_state(self, capture: ScreenCapture) -> ScreenState:
    visual_hash = hashlib.sha256(capture.image_data).hexdigest()[:16]
    package = "unknown"  # ❌ Always unknown
    return ScreenState(visual_hash=visual_hash, activity=package, ...)
```

### Impact
LOW - The visual_hash is the primary identifier, so exploration still works. We just lose activity names in the graph.

### Fix
Make it async and call the device port:

```python
async def __compute_state(self, capture: ScreenCapture) -> ScreenState:
    visual_hash = hashlib.sha256(capture.image_data).hexdigest()[:16]
    try:
        package = await self.__device.get_current_package()
    except:
        package = "unknown"
    return ScreenState(visual_hash=visual_hash, activity=package, ...)
```

---

## 📊 Current Status

### What Works ✅
- CLI (`fathom run`, `fathom explore`) - Uses old system
- All adapters have real logic (no placeholders)
- ExecutionEngine has real 7-phase DAG
- ContextManager has real 3-tier context
- Builder API creates FathomRunner
- ExplorationStrategy works (minor issue with package names)

### What's Broken ❌
- IntentStrategy through new builder API - Will crash
- Any code path that tries to use strategies/intent.py

### What's the Real Situation?

The hexagonal architecture migration is **INCOMPLETE**. We have:

1. ✅ New ports and adapters (working)
2. ✅ New core components (working)
3. ✅ New builder API (working)
4. ✅ ExplorationStrategy (mostly working)
5. ❌ IntentStrategy (broken - needs IVisionProvider adapter)

The CLI still works because it uses the OLD system (`agent/strategies/intent.py`), not the new one.

---

## 🎯 Recommended Next Steps

1. **Create IVisionProvider Adapter** (Option 1 above)
   - Wrap LLMPort to implement IVisionProvider interface
   - This allows reusing all the old agent components
   - Minimal code changes needed

2. **Fix ExplorationStrategy package names** (minor)
   - Make `__compute_state()` async
   - Call `await self.__device.get_current_package()`

3. **Test the new builder API end-to-end**
   - Try running: `python examples/builder_minimal.py`
   - It will likely crash when IntentStrategy tries to plan

---

## 🤔 Why Wasn't This Caught Earlier?

The migration followed the design document which said:
> "Copy-paste existing logic without modifications. Only update imports and structure."

But this approach doesn't work when there's an **interface mismatch**:
- Old code expects: IVisionProvider
- New code provides: LLMPort

These are different interfaces, so you can't just "copy-paste and update imports". You need an adapter layer.

The design document didn't account for this mismatch.
