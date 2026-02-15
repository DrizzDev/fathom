# Code Path Verification: Which Code Runs?

## Question
When running `fathom run "Ask GPT to do deep research about opencrawler(moltybot)" --use-xml --serial emulator-5554 -v`, which version of code is working? Is it the new code or the same old one?

## Answer: NEW CODE ✅

The `fathom` command runs the **NEW hexagonal architecture code**, not the old code.

## Proof: Complete Code Path Trace

### 1. Entry Point (pyproject.toml)
```toml
[project.scripts]
fathom = "fathom.cli_new:main"        # ← NEW CODE
fathom-old = "fathom.cli:main"        # ← OLD CODE (deprecated)
```

**Verification**: The `fathom` command points to `cli_new.py`, not `cli.py`.

### 2. CLI Layer (src/fathom/cli_new.py)
```python
from fathom.runtime.builder import Fathom  # ← NEW architecture

self.runner = (
    Fathom.builder()
    .device(ADBDevice(serial=serial))           # ← NEW adapter
    .llm(GeminiLLM(configuration=gemini_config)) # ← NEW adapter
    .memory(SQLiteMemory())                      # ← NEW adapter
    .knowledge(SQLiteKnowledge())                # ← NEW adapter
    .signal(NoopSignal())                        # ← NEW adapter
    .storage(LocalStorage())                     # ← NEW adapter
    .telemetry(StructlogAdapter())               # ← NEW adapter
    .build()
)

result = await self.runner.run_intent(...)  # ← NEW runner
```

**Verification**: Uses `Fathom.builder()` from the NEW runtime, not old `FathomRunner` from orchestration.

### 3. Builder Layer (src/fathom/runtime/builder.py)
```python
from fathom.runtime.runner import FathomRunner  # ← NEW runner

def build(self) -> FathomRunner:
    return FathomRunner(
        device=self._device,      # ← Port interface
        llm=self._llm,            # ← Port interface
        memory=self._memory,      # ← Port interface
        knowledge=self._knowledge, # ← Port interface
        signal=self._signal,      # ← Port interface
        storage=self._storage,    # ← Port interface
        telemetry=self._telemetry, # ← Port interface
        config=self._config,
    )
```

**Verification**: Creates NEW `FathomRunner` from `runtime/runner.py`, not old one from `orchestration/runner/fathom.py`.

### 4. Runner Layer (src/fathom/runtime/runner.py)
```python
from fathom.core.execution.engine import ExecutionEngine  # ← NEW core
from fathom.core.context.manager import ContextManager    # ← NEW core

class FathomRunner:
    def __init__(self, *, device: DevicePort, llm: LLMPort, ...):
        # Wire NEW core components
        self._engine = ExecutionEngine(...)      # ← NEW 7-phase DAG engine
        self._context_manager = ContextManager(...) # ← NEW 3-tier context
    
    async def run_intent(self, intent: str, ...):
        from fathom.strategies.intent import IntentStrategy  # ← NEW strategy
        
        strategy = IntentStrategy(
            engine=self._engine,           # ← NEW engine
            context=self._context_manager, # ← NEW context
            device=self._device,           # ← Port interface
            llm=self._llm,                 # ← Port interface
            ...
        )
        
        result = await strategy.execute(max_steps=max_steps)
```

**Verification**: Uses NEW `IntentStrategy` from `strategies/intent.py`, not old `IntentWorkflow` from `workflows/intent.py`.

### 5. Strategy Layer (src/fathom/strategies/intent.py)
```python
from fathom.core.execution.engine import ExecutionEngine  # ← NEW core

class IntentStrategy:
    def __init__(self, engine: ExecutionEngine, ...):
        self.__engine = engine  # ← NEW 7-phase DAG engine
    
    async def execute(self, max_steps: int):
        # Execute through NEW engine
        result = await self.__engine.execute_step(step=step)
```

**Verification**: Uses NEW `ExecutionEngine`, not old step executor.

### 6. Core Layer (src/fathom/core/execution/engine.py)
```python
class ExecutionEngine:
    """
    Core execution engine implementing the DAG-based execution flow.
    
    Phases: SignalCheck → Perceive → Reason → Act → Learn → Checkpoint → Evaluate
    """
    
    async def execute_step(self, step: Step, ...):
        # Phase 1: Signal Check
        await self.__check_signal()
        
        # Phase 2: Perceive
        pre_capture = await self.__perceive()
        
        # Phase 4: Act
        result = await self.__act(step=step)
        
        # Phase 5: Learn
        await self.__learn(...)
        
        # Phase 6: Checkpoint
        self.__checkpoint(step_result=step_result)
```

**Verification**: This is the NEW 7-phase DAG execution engine, not old execution logic.

## Comparison: Old vs New Code Paths

### OLD Code Path (fathom-old command)
```
fathom-old
  ↓
src/fathom/cli.py (OLD CLI)
  ↓
src/fathom/orchestration/runner/fathom.py (OLD FathomRunner)
  ↓
src/fathom/workflows/intent.py (OLD IntentWorkflow)
  ↓
Direct tool wiring (no ports/adapters)
  ↓
Old execution logic embedded in workflow
```

### NEW Code Path (fathom command) ✅
```
fathom
  ↓
src/fathom/cli_new.py (NEW CLI)
  ↓
src/fathom/runtime/builder.py (Builder API)
  ↓
src/fathom/runtime/runner.py (NEW FathomRunner)
  ↓
src/fathom/strategies/intent.py (NEW IntentStrategy)
  ↓
src/fathom/core/execution/engine.py (NEW ExecutionEngine)
  ↓
Port interfaces (DevicePort, LLMPort, etc.)
  ↓
Adapter implementations (ADBDevice, GeminiLLM, etc.)
```

## Key Differences

| Aspect | Old Code | New Code |
|--------|----------|----------|
| **Entry Point** | `cli.py` | `cli_new.py` |
| **Runner** | `orchestration/runner/fathom.py` | `runtime/runner.py` |
| **Workflow/Strategy** | `workflows/intent.py` | `strategies/intent.py` |
| **Execution** | Embedded in workflow | `core/execution/engine.py` |
| **Architecture** | Direct tool wiring | Ports & Adapters |
| **Dependencies** | Concrete tools | Port interfaces |
| **Testability** | Hard to mock | Easy to mock |
| **Pluggability** | Tightly coupled | Loosely coupled |

## How to Verify Yourself

### 1. Check Entry Point
```bash
grep "^\[project.scripts\]" -A 2 pyproject.toml
```
Output shows `fathom = "fathom.cli_new:main"` ✅

### 2. Check Imports in CLI
```bash
grep "from fathom.runtime" src/fathom/cli_new.py
```
Output shows `from fathom.runtime.builder import Fathom` ✅

### 3. Check Runner Import
```bash
grep "from fathom.runtime.runner import FathomRunner" src/fathom/runtime/builder.py
```
Output shows NEW runner is imported ✅

### 4. Check Strategy Import
```bash
grep "from fathom.strategies.intent import IntentStrategy" src/fathom/runtime/runner.py
```
Output shows NEW strategy is imported ✅

### 5. Add Debug Print (Optional)
Add this to `src/fathom/runtime/runner.py` in `run_intent()`:
```python
async def run_intent(self, intent: str, ...):
    print("🎯 USING NEW HEXAGONAL ARCHITECTURE!")
    ...
```

Then run your command and you'll see the message.

## Conclusion

**100% CONFIRMED**: When you run `fathom run "..."`, you are using the **NEW hexagonal architecture code**.

The old code is only accessible via `fathom-old` command and is marked as deprecated.

## Architecture Components Used

When you run `fathom run`, these NEW components are active:

✅ **NEW CLI**: `src/fathom/cli_new.py`
✅ **NEW Builder**: `src/fathom/runtime/builder.py`
✅ **NEW Runner**: `src/fathom/runtime/runner.py`
✅ **NEW Strategy**: `src/fathom/strategies/intent.py`
✅ **NEW Engine**: `src/fathom/core/execution/engine.py` (7-phase DAG)
✅ **NEW Context**: `src/fathom/core/context/manager.py` (3-tier)
✅ **NEW Ports**: All 7 port interfaces
✅ **NEW Adapters**: All 7 adapter implementations

The hexagonal architecture is **FULLY ACTIVE** and working! 🎉
