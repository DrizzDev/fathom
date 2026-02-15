# Design Document: Fathom Hexagonal Architecture Migration

## Overview

This design specifies an incremental migration strategy to transform Fathom from its current architecture to a hexagonal architecture (ports and adapters pattern) as defined in documents/architecture/v2/ARCHITECTURE.md. The migration preserves all existing functionality while establishing clean architectural boundaries that separate business logic from infrastructure concerns.

The key principle is **incremental migration without breaking changes**. The old code continues to work while new architecture is built alongside it. Once the new architecture is stable and tested, we gradually migrate modules one at a time.

**CRITICAL: Existing logic must NOT be modified.** When migrating code (especially Gemini, LLM, Prompt, Tool Definition modules), copy-paste the same code and only restructure/update imports as required. The business logic, algorithms, and behavior must remain identical.

### Migration Philosophy

1. **Build alongside, not replace**: Create new directories and interfaces without touching existing code
2. **Dual-mode operation**: Both old and new code paths work simultaneously during migration
3. **One module at a time**: Migrate individual modules incrementally with full test coverage
4. **Backward compatibility**: Maintain compatibility shims until all dependents are migrated
5. **Rollback safety**: Each migration step can be rolled back independently
6. **Logic preservation**: Copy-paste existing code, only change imports and structure

## Architecture

### Target Directory Structure

```
src/fathom/
├── interfaces/          # Port definitions (ABCs)
│   ├── device.py       # DevicePort
│   ├── llm.py          # LLMPort
│   ├── memory.py       # MemoryPort
│   ├── knowledge.py    # KnowledgePort
│   ├── signal.py       # SignalPort
│   ├── storage.py      # StoragePort
│   └── telemetry.py    # TelemetryPort
├── adapters/           # Concrete implementations
│   ├── device/
│   │   ├── adb.py      # ADBDevice adapter
│   │   └── mock.py     # MockDevice adapter
│   ├── llm/
│   │   ├── gemini.py   # GeminiLLM adapter
│   │   └── mock.py     # MockLLM adapter
│   ├── memory/
│   │   └── sqlite.py   # SQLiteMemory adapter
│   ├── knowledge/
│   │   └── sqlite.py   # SQLiteKnowledge adapter
│   ├── signal/
│   │   └── noop.py     # NoopSignal adapter
│   ├── storage/
│   │   ├── local.py    # LocalStorage adapter
│   │   └── cloud.py    # CloudStorage adapter
│   └── telemetry/
│       └── structlog.py # StructlogAdapter
├── core/               # Business logic
│   ├── execution/      # Execution engine
│   ├── context/        # Context management
│   └── models/         # Domain models
├── runtime/            # Composition root
│   ├── builder.py      # Fathom.builder() API
│   └── runner.py       # Execution orchestration
├── strategies/         # Execution strategies
│   ├── intent.py       # Intent-based strategy
│   └── exploration.py  # Exploration strategy
├── processing/         # UI processing (moved from tools/vision/processing)
│   ├── annotator.py
│   ├── drawer.py
│   ├── geometry.py
│   └── parsers/
├── prompts/            # Proprietary prompt logic (preserved)
├── services/           # Business services (preserved)
├── schemas/            # Pydantic models (preserved)
└── [legacy dirs]       # orchestration/, tools/, workflows/ (deprecated but functional)
```

### Import Dependency Rules

The architecture enforces strict import rules to maintain clean boundaries:

| Module | Can Import | Cannot Import |
|--------|-----------|---------------|
| interfaces/ | schemas/ | core/, adapters/, runtime/, processing/ |
| adapters/ | interfaces/, schemas/, external libs | core/, runtime/ |
| core/ | interfaces/, schemas/ | adapters/, runtime/ |
| strategies/ | core/, interfaces/, schemas/ | adapters/, runtime/ |
| processing/ | schemas/ | core/, adapters/, interfaces/, runtime/ |
| runtime/ | ALL | (composition root) |

## Components and Interfaces

### Port Definitions (interfaces/)

Each port is defined as an abstract base class (ABC) with Protocol for runtime checking.

#### DevicePort

```python
from abc import ABC, abstractmethod
from typing import Tuple

class DevicePort(ABC):
    """Port for mobile device interactions."""
    
    @abstractmethod
    async def tap(self, x: int, y: int) -> ExecutionResult:
        """Tap at screen coordinates."""
        pass
    
    @abstractmethod
    async def type_text(self, text: str) -> ExecutionResult:
        """Type text into focused element."""
        pass
    
    @abstractmethod
    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> ExecutionResult:
        """Swipe from (x1,y1) to (x2,y2)."""
        pass
    
    @abstractmethod
    async def back(self) -> ExecutionResult:
        """Press back button."""
        pass
    
    @abstractmethod
    async def home(self) -> ExecutionResult:
        """Press home button."""
        pass
    
    @abstractmethod
    async def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions (width, height)."""
        pass
    
    @abstractmethod
    async def capture_screen(self) -> bytes:
        """Capture screenshot as PNG bytes."""
        pass
    
    @abstractmethod
    async def get_current_package(self) -> str:
        """Get current foreground package name."""
        pass
    
    @abstractmethod
    async def wait_for_device(self, timeout: float) -> bool:
        """Wait for device to be ready."""
        pass
```

#### LLMPort

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class LLMPort(ABC):
    """Port for language model interactions."""
    
    @abstractmethod
    async def analyze(
        self,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Analyze content with LLM.
        
        Args:
            system_instruction: System prompt
            user_content: List of text strings and image bytes
            tools: Optional tool definitions for function calling
        
        Returns:
            AnalysisResult with reasoning, action, and metrics
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources."""
        pass
```

#### MemoryPort

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class MemoryPort(ABC):
    """Port for session state and cross-run memory."""
    
    @abstractmethod
    async def set(self, key: str, value: str) -> None:
        """Store key-value pair in session."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Retrieve value by key."""
        pass
    
    @abstractmethod
    async def get_all(self) -> Dict[str, str]:
        """Get all session data."""
        pass
    
    @abstractmethod
    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None:
        """Store screen observation for future recall."""
        pass
    
    @abstractmethod
    async def store_experience(self, visual_hash: str, action: Action, success: bool) -> None:
        """Store action outcome for learning."""
        pass
    
    @abstractmethod
    async def retrieve_knowledge(self, visual_hash: str) -> Dict[str, Any]:
        """Retrieve everything known about a screen."""
        pass
```

#### KnowledgePort

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class KnowledgePort(ABC):
    """Port for application knowledge graph."""
    
    @abstractmethod
    async def add_screen(self, screen_id: str, metadata: Dict[str, Any]) -> None:
        """Add screen node to graph."""
        pass
    
    @abstractmethod
    async def add_transition(self, from_screen: str, to_screen: str, action: Action) -> None:
        """Add transition edge between screens."""
        pass
    
    @abstractmethod
    async def find_path(self, from_screen: str, to_screen: str) -> Optional[List[Action]]:
        """Find action sequence to reach target screen."""
        pass
    
    @abstractmethod
    async def get_neighbors(self, screen_id: str) -> List[str]:
        """Get screens reachable from given screen."""
        pass
```

#### SignalPort

```python
from abc import ABC, abstractmethod
from typing import Optional

class SignalPort(ABC):
    """Port for human-in-the-loop control signals."""
    
    @abstractmethod
    async def check_signal(self) -> Optional[str]:
        """Check for control signal (PAUSE, RESUME, INJECT, ASK)."""
        pass
    
    @abstractmethod
    async def wait_for_resume(self) -> None:
        """Block until RESUME signal received."""
        pass
    
    @abstractmethod
    async def request_input(self, prompt: str) -> str:
        """Request human input with prompt."""
        pass
```

#### StoragePort

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class StoragePort(ABC):
    """Port for artifact persistence."""
    
    @abstractmethod
    async def save(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact and return identifier.
        
        Args:
            data: Binary data to store
            metadata: Optional metadata for organizing storage
        
        Returns:
            Storage identifier (path, URL, etc.)
        """
        pass
```

#### TelemetryPort

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class TelemetryPort(ABC):
    """Port for telemetry and observability."""
    
    @abstractmethod
    def debug(self, message: str, **context: Any) -> None:
        """Log debug message with context."""
        pass
    
    @abstractmethod
    def info(self, message: str, **context: Any) -> None:
        """Log info message with context."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **context: Any) -> None:
        """Log warning message with context."""
        pass
    
    @abstractmethod
    def error(self, message: str, **context: Any) -> None:
        """Log error message with context."""
        pass
```

### Adapter Implementations (adapters/)

Adapters implement ports by wrapping existing infrastructure or external libraries.

**IMPORTANT: Adapters should wrap existing code, not rewrite it.** The adapter's job is to implement the port interface by delegating to existing infrastructure code. Copy-paste existing logic if needed, but do not modify the business logic.

#### ADBDevice Adapter

Wraps existing `tools/device/adb.py` and `tools/capture/adb.py` functionality:

```python
from fathom.interfaces.device import DevicePort
from fathom.tools.device.adb import ADBDeviceTool
from fathom.tools.capture.adb import ADBCaptureTool

class ADBDevice(DevicePort):
    """ADB adapter for Android devices."""
    
    def __init__(self, serial: Optional[str] = None):
        self._device = ADBDeviceTool(configuration=ADBConfig(device_serial=serial))
        self._capture = ADBCaptureTool(config=ADBCaptureConfig(device_serial=serial))
    
    async def tap(self, x: int, y: int) -> ExecutionResult:
        return await self._device.tap(x=x, y=y)
    
    async def capture_screen(self) -> bytes:
        capture = await self._capture.capture()
        return capture.image_data
    
    # ... implement other methods by delegating to existing tools
```

#### GeminiLLM Adapter

Wraps existing `infrastructure/llm/gemini.py` **without modifying its logic**:

```python
from fathom.interfaces.llm import LLMPort
from fathom.infrastructure.llm.gemini import GeminiLLMClient

class GeminiLLM(LLMPort):
    """Gemini adapter for LLM interactions."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp"):
        config = GeminiConfig(api_key=api_key, model=model)
        # Use existing GeminiLLMClient as-is, no logic changes
        self._client = GeminiLLMClient(configuration=config)
    
    async def analyze(
        self,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        # Delegate directly to existing client
        return await self._client.analyze(
            system_instruction=system_instruction,
            user_content=user_content,
            tools=tools
        )
    
    async def cleanup(self) -> None:
        await self._client.cleanup()
```

#### SQLiteMemory Adapter

Combines existing `infrastructure/memory/sqlite.py` and `infrastructure/memory/ledger.py` **without modifying their logic**:

```python
from fathom.interfaces.memory import MemoryPort
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.infrastructure.memory.ledger import Ledger

class SQLiteMemory(MemoryPort):
    """SQLite adapter for memory persistence."""
    
    def __init__(self, path: str = "assets/memory/knowledge.db"):
        # Use existing implementations as-is
        self._provider = SQLiteMemoryProvider(database_path=path)
        self._ledger = Ledger()
    
    async def set(self, key: str, value: str) -> None:
        await self._ledger.set(key, value)
    
    async def get(self, key: str) -> Optional[str]:
        return await self._ledger.get(key)
    
    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None:
        await self._provider.store_observation(screen, description)
    
    # ... delegate to existing implementations
```

### Core Business Logic (core/)

The core layer contains business logic independent of infrastructure.

#### Execution Engine

```python
# core/execution/engine.py

class ExecutionEngine:
    """
    Core execution engine implementing the DAG-based execution flow.
    
    Phases: SignalCheck → Perceive → Reason → Act → Learn → Checkpoint → Evaluate
    """
    
    def __init__(
        self,
        device: DevicePort,
        llm: LLMPort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
    ):
        self._device = device
        self._llm = llm
        self._memory = memory
        self._signal = signal
        self._storage = storage
        self._telemetry = telemetry
    
    async def execute_step(self, context: ExecutionContext) -> StepResult:
        """Execute one step of the execution DAG."""
        
        # Phase 1: Signal Check
        signal = await self._signal.check_signal()
        if signal == "PAUSE":
            await self._signal.wait_for_resume()
        
        # Phase 2: Perceive
        screenshot = await self._device.capture_screen()
        await self._storage.save(screenshot, metadata={"step": context.step_number})
        
        # Phase 3: Reason
        analysis = await self._llm.analyze(
            system_instruction=context.system_prompt,
            user_content=[context.user_prompt, screenshot],
            tools=context.tools
        )
        
        # Phase 4: Act
        result = await self._execute_action(analysis.action)
        
        # Phase 5: Learn
        await self._memory.store_experience(
            visual_hash=context.screen_hash,
            action=analysis.action,
            success=result.success
        )
        
        # Phase 6: Checkpoint
        self._telemetry.info("Step completed", step=context.step_number, success=result.success)
        
        # Phase 7: Evaluate
        terminal = self._is_terminal(analysis.action, result)
        
        return StepResult(
            action=analysis.action,
            success=result.success,
            terminal=terminal,
            metrics=analysis.metrics
        )
    
    async def _execute_action(self, action: Action) -> ExecutionResult:
        """Execute device action based on action type."""
        # Implementation delegates to DevicePort
        pass
    
    def _is_terminal(self, action: Action, result: ExecutionResult) -> bool:
        """Determine if execution should terminate."""
        # Implementation checks for COMPLETE action or max steps
        pass
```

#### Context Management

```python
# core/context/manager.py

class ContextManager:
    """
    Manages three-tier versioned context (GCC-inspired).
    
    Tiers:
    - roadmap: Original intent + milestones
    - milestones: Summaries of completed sub-goals
    - trace: Fine-grained OTA log (every Observe-Thought-Action cycle)
    """
    
    def __init__(self, memory: MemoryPort):
        self._memory = memory
        self._roadmap: str = ""
        self._milestones: List[str] = []
        self._trace: List[Dict[str, Any]] = []
    
    async def commit(self, observation: str, thought: str, action: Action) -> None:
        """Commit OTA cycle to trace."""
        self._trace.append({
            "observation": observation,
            "thought": thought,
            "action": action.model_dump()
        })
    
    async def branch(self, milestone: str) -> None:
        """Create milestone and compress trace."""
        self._milestones.append(milestone)
        # Compress trace into milestone summary
        self._trace = []
    
    async def recall(self, tier: str) -> Any:
        """Retrieve context from specified tier."""
        if tier == "roadmap":
            return self._roadmap
        elif tier == "milestones":
            return self._milestones
        elif tier == "trace":
            return self._trace
        raise ValueError(f"Unknown tier: {tier}")
```

### Runtime Composition (runtime/)

The runtime layer is the composition root that wires everything together.

#### Builder API

```python
# runtime/builder.py

class FathomBuilder:
    """
    Fluent builder for Fathom configuration.
    
    Methods are order-independent. Validation happens at build().
    """
    
    def __init__(self):
        self._device: Optional[DevicePort] = None
        self._llm: Optional[LLMPort] = None
        self._memory: Optional[MemoryPort] = None
        self._knowledge: Optional[KnowledgePort] = None
        self._signal: Optional[SignalPort] = None
        self._storage: Optional[StoragePort] = None
        self._telemetry: Optional[TelemetryPort] = None
    
    def device(self, device: DevicePort) -> "FathomBuilder":
        """Configure device port."""
        self._device = device
        return self
    
    def llm(self, llm: LLMPort) -> "FathomBuilder":
        """Configure LLM port."""
        self._llm = llm
        return self
    
    def memory(self, memory: MemoryPort) -> "FathomBuilder":
        """Configure memory port."""
        self._memory = memory
        return self
    
    def knowledge(self, knowledge: KnowledgePort) -> "FathomBuilder":
        """Configure knowledge port."""
        self._knowledge = knowledge
        return self
    
    def signal(self, signal: SignalPort) -> "FathomBuilder":
        """Configure signal port."""
        self._signal = signal
        return self
    
    def storage(self, storage: StoragePort) -> "FathomBuilder":
        """Configure storage port."""
        self._storage = storage
        return self
    
    def telemetry(self, telemetry: TelemetryPort) -> "FathomBuilder":
        """Configure telemetry port."""
        self._telemetry = telemetry
        return self
    
    def build(self) -> "FathomRunner":
        """
        Build configured Fathom instance.
        
        Validates required ports and applies defaults.
        """
        # Validate required ports
        if not self._device:
            raise ValueError("device() is required")
        if not self._llm:
            raise ValueError("llm() is required")
        
        # Apply defaults for optional ports
        if not self._memory:
            self._memory = SQLiteMemory()
        if not self._knowledge:
            self._knowledge = SQLiteKnowledge()
        if not self._signal:
            self._signal = NoopSignal()
        if not self._storage:
            self._storage = LocalStorage()
        if not self._telemetry:
            self._telemetry = StructlogAdapter()
        
        return FathomRunner(
            device=self._device,
            llm=self._llm,
            memory=self._memory,
            knowledge=self._knowledge,
            signal=self._signal,
            storage=self._storage,
            telemetry=self._telemetry
        )


class Fathom:
    """Main entry point for Fathom library."""
    
    @staticmethod
    def builder() -> FathomBuilder:
        """Create a new builder instance."""
        return FathomBuilder()
```

#### Runner

```python
# runtime/runner.py

class FathomRunner:
    """
    Executes Fathom workflows with configured ports.
    """
    
    def __init__(
        self,
        device: DevicePort,
        llm: LLMPort,
        memory: MemoryPort,
        knowledge: KnowledgePort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
    ):
        self._device = device
        self._llm = llm
        self._memory = memory
        self._knowledge = knowledge
        self._signal = signal
        self._storage = storage
        self._telemetry = telemetry
        
        # Build core components
        self._engine = ExecutionEngine(
            device=device,
            llm=llm,
            memory=memory,
            signal=signal,
            storage=storage,
            telemetry=telemetry
        )
        self._context_manager = ContextManager(memory=memory)
    
    async def run(
        self,
        intent: str,
        max_steps: int = 20,
        strategy: str = "intent"
    ) -> ExecutionResult:
        """
        Execute workflow with given intent.
        
        Args:
            intent: User intent to accomplish
            max_steps: Maximum execution steps
            strategy: Execution strategy ("intent" or "exploration")
        
        Returns:
            ExecutionResult with outcome and metrics
        """
        self._telemetry.info("Starting execution", intent=intent, max_steps=max_steps)
        
        # Initialize context
        await self._context_manager.commit(
            observation="",
            thought=f"User intent: {intent}",
            action=Action(action_type=ActionType.WAIT)
        )
        
        # Select strategy
        if strategy == "intent":
            strategy_impl = IntentStrategy(
                engine=self._engine,
                context=self._context_manager,
                intent=intent
            )
        else:
            strategy_impl = ExplorationStrategy(
                engine=self._engine,
                context=self._context_manager
            )
        
        # Execute
        result = await strategy_impl.execute(max_steps=max_steps)
        
        self._telemetry.info("Execution completed", success=result.success, steps=result.steps)
        return result
```

### Strategies (strategies/)

Strategies implement different execution approaches.

```python
# strategies/intent.py

class IntentStrategy:
    """Intent-based execution strategy."""
    
    def __init__(
        self,
        engine: ExecutionEngine,
        context: ContextManager,
        intent: str
    ):
        self._engine = engine
        self._context = context
        self._intent = intent
    
    async def execute(self, max_steps: int) -> ExecutionResult:
        """Execute intent-based workflow."""
        for step in range(max_steps):
            # Build execution context with intent
            exec_context = ExecutionContext(
                step_number=step,
                intent=self._intent,
                roadmap=await self._context.recall("roadmap"),
                milestones=await self._context.recall("milestones"),
                trace=await self._context.recall("trace")
            )
            
            # Execute step
            result = await self._engine.execute_step(exec_context)
            
            # Update context
            await self._context.commit(
                observation=result.observation,
                thought=result.reasoning,
                action=result.action
            )
            
            # Check termination
            if result.terminal:
                return ExecutionResult(success=True, steps=step+1)
        
        return ExecutionResult(success=False, steps=max_steps)
```

### Processing Module (processing/)

The processing module contains UI element processing logic moved from `tools/vision/processing/`.

```python
# processing/annotator.py
# Moved from tools/vision/processing/annotator.py
# Logic preserved, only imports updated

# processing/drawer.py
# Moved from tools/vision/processing/drawer.py
# Logic preserved, only imports updated

# processing/geometry.py
# Moved from tools/vision/processing/geometry.py
# Logic preserved, only imports updated

# processing/parsers/
# Moved from tools/vision/processing/parsers/
# Logic preserved, only imports updated
```

## Data Models

All existing Pydantic models in `schemas/` are preserved without changes:

- `schemas/actions.py`: Action, ActionType
- `schemas/screens.py`: ScreenState, ScreenCapture
- `schemas/results.py`: AnalysisResult, ExecutionResult, StepResult
- `schemas/steps.py`: Step
- `schemas/orchestration.py`: ExecutionContext
- `schemas/configuration.py`: All configuration models

These schemas are used by all layers (interfaces, adapters, core, runtime, strategies).


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Port Interface Compliance

*For any* adapter class in the adapters/ directory, it SHALL implement its corresponding port interface from interfaces/ with all required methods.

**Validates: Requirements 1.3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

### Property 2: Builder Method Chaining

*For any* builder method (device, llm, memory, knowledge, signal, storage, log), calling the method SHALL return the same builder instance to enable method chaining.

**Validates: Requirements 4.3**

### Property 3: Builder Order Independence

*For any* two different orderings of builder method calls with the same arguments, calling build() SHALL produce functionally equivalent FathomRunner instances.

**Validates: Requirements 4.4**

### Property 4: Required Port Validation

*For any* builder instance where device() or llm() has not been called, calling build() SHALL raise a ValueError with a descriptive message indicating which required port is missing.

**Validates: Requirements 4.5, 4.6, 11.5**

### Property 5: Default Port Assignment

*For any* optional port (memory, knowledge, signal, storage, log) that is not explicitly configured, calling build() SHALL assign the default adapter for that port.

**Validates: Requirements 11.1, 11.4**

### Property 6: Legacy Code Backward Compatibility

*For any* existing import statement from orchestration/, tools/, services/, or prompts/ directories, the import SHALL continue to work either directly or through re-export shims.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 9.4**

### Property 7: Proprietary Code Preservation

*For any* function or class in proprietary modules (prompts/, tools/definitions.py, services/parsing.py), the function signature and logic SHALL remain identical after migration, with only import statements modified.

**Validates: Requirements 6.1, 6.4, 6.5, 6.6**

### Property 8: Processing Module Import Updates

*For any* module that depends on processing code, after migration the module SHALL import from processing/ instead of tools/vision/processing/ and SHALL function identically.

**Validates: Requirements 7.6, 7.7**

### Property 9: Core Layer Import Restrictions

*For any* Python file in core/, the file SHALL only import from interfaces/, schemas/, or standard library, and SHALL NOT import from adapters/ or runtime/.

**Validates: Requirements 8.1, 8.2**

### Property 10: Interfaces Layer Import Restrictions

*For any* Python file in interfaces/, the file SHALL only import from schemas/ or standard library, and SHALL NOT import from core/, adapters/, or runtime/.

**Validates: Requirements 8.3, 8.4**

### Property 11: Strategies Layer Import Restrictions

*For any* Python file in strategies/, the file SHALL only import from core/, interfaces/, schemas/, or standard library, and SHALL NOT import from adapters/ or runtime/.

**Validates: Requirements 8.5, 8.6**

### Property 12: Adapters Layer Import Restrictions

*For any* Python file in adapters/, the file SHALL only import from interfaces/, schemas/, external libraries, or standard library, and SHALL NOT import from core/ or runtime/.

**Validates: Requirements 8.7, 8.8**

### Property 13: Processing Layer Import Restrictions

*For any* Python file in processing/, the file SHALL only import from schemas/ or standard library, and SHALL NOT import from core/, adapters/, interfaces/, or runtime/.

**Validates: Requirements 8.10**

### Property 14: Execution Phase Sequence Preservation

*For any* execution step, the phases SHALL execute in the exact order: SignalCheck → Perceive → Reason → Act → Learn → Checkpoint → Evaluate.

**Validates: Requirements 10.1, 10.2, 10.3**

### Property 15: HITL Signal Handling

*For any* HITL signal (PAUSE, RESUME, INJECT, ASK), the execution engine SHALL handle the signal correctly according to its semantics (pause execution, resume execution, inject action, request input).

**Validates: Requirements 10.4**

### Property 16: Workflow Compatibility

*For any* existing workflow (IntentWorkflow, ExplorationWorkflow), after migration the workflow SHALL produce equivalent results for the same inputs.

**Validates: Requirements 10.5**

### Property 17: Schema Preservation

*For any* Pydantic model in schemas/, the model definition SHALL remain unchanged after migration, and SHALL be importable by all architecture layers.

**Validates: Requirements 12.2, 12.3, 12.4**

### Property 18: Explicit Port Configuration

*For any* of the seven ports (device, llm, memory, knowledge, signal, storage, log), the builder SHALL accept an explicit adapter instance for that port and use it instead of the default.

**Validates: Requirements 11.3**

## Error Handling

### Port Configuration Errors

The builder validates port configuration at build() time:

- **Missing required ports**: Raises `ValueError` with message "device() is required" or "llm() is required"
- **Invalid port type**: Raises `TypeError` if provided adapter doesn't implement the port interface
- **Initialization failures**: Adapters raise specific exceptions (e.g., `VisionError`, `ToolError`) with descriptive messages

### Execution Errors

The execution engine handles errors at each phase:

- **Device errors**: Retry with exponential backoff (max 2 retries), then fail step with error message
- **LLM errors**: Retry with exponential backoff (max retries configurable), handle quota errors with longer delays
- **Memory errors**: Log error and continue execution (non-critical)
- **Signal errors**: Log error and treat as no signal (fail-safe to autonomous mode)

### Migration Errors

During incremental migration:

- **Import errors**: Backward compatibility shims catch import errors and redirect to new locations
- **Missing modules**: Clear error messages indicate which module needs migration
- **Test failures**: Migration step is rolled back, error is logged with failing test details

## Testing Strategy

### Dual Testing Approach

The testing strategy uses both unit tests and property-based tests:

- **Unit tests**: Verify specific examples, edge cases, and integration points
- **Property tests**: Verify universal properties across all inputs using randomized testing

### Unit Testing Focus

Unit tests should focus on:

- Specific examples demonstrating correct behavior (e.g., builder with minimal config)
- Integration points between components (e.g., adapter wrapping existing infrastructure)
- Edge cases and error conditions (e.g., missing required ports)
- Migration compatibility (e.g., old imports still work)

Avoid writing too many unit tests for behavior that property tests cover comprehensively.

### Property-Based Testing Configuration

- **Library**: Use `hypothesis` for Python property-based testing
- **Iterations**: Minimum 100 iterations per property test
- **Tagging**: Each property test must reference its design document property
- **Tag format**: `# Feature: fathom-hexagonal-rearch, Property {number}: {property_text}`

### Property Test Examples

```python
from hypothesis import given, strategies as st
import hypothesis

# Feature: fathom-hexagonal-rearch, Property 2: Builder Method Chaining
@given(st.text())
@hypothesis.settings(max_examples=100)
def test_builder_method_chaining_returns_self(serial):
    """For any builder method, calling it returns the same builder instance."""
    builder = Fathom.builder()
    device = ADBDevice(serial=serial)
    
    result = builder.device(device)
    assert result is builder

# Feature: fathom-hexagonal-rearch, Property 3: Builder Order Independence
@given(st.text(), st.text())
@hypothesis.settings(max_examples=100)
def test_builder_order_independence(serial, api_key):
    """Different orderings of builder calls produce equivalent instances."""
    device = ADBDevice(serial=serial)
    llm = GeminiLLM(api_key=api_key)
    
    # Order 1: device then llm
    runner1 = Fathom.builder().device(device).llm(llm).build()
    
    # Order 2: llm then device
    runner2 = Fathom.builder().llm(llm).device(device).build()
    
    # Both should have same configuration
    assert type(runner1._device) == type(runner2._device)
    assert type(runner1._llm) == type(runner2._llm)

# Feature: fathom-hexagonal-rearch, Property 9: Core Layer Import Restrictions
@hypothesis.settings(max_examples=100)
def test_core_layer_import_restrictions():
    """For any file in core/, it only imports from interfaces/ and schemas/."""
    core_files = Path("src/fathom/core").rglob("*.py")
    
    for file_path in core_files:
        with open(file_path) as f:
            content = f.read()
        
        # Parse imports
        imports = extract_imports(content)
        
        for imp in imports:
            # Allow imports from interfaces, schemas, standard library
            assert (
                imp.startswith("fathom.interfaces") or
                imp.startswith("fathom.schemas") or
                is_stdlib(imp)
            ), f"{file_path} imports forbidden module: {imp}"
```

### Migration Testing Strategy

Each migration step follows this testing pattern:

1. **Pre-migration baseline**: Run full test suite, record results
2. **Create new component**: Write unit tests for new component
3. **Integration test**: Test new component with existing code
4. **Backward compatibility test**: Verify old imports still work
5. **Property test**: Verify architectural properties hold
6. **Full regression**: Run complete test suite
7. **Rollback test**: Verify rollback procedure works

### Test Coverage Requirements

- **Port interfaces**: 100% coverage (all methods tested)
- **Adapters**: 90% coverage (focus on port implementation)
- **Core logic**: 95% coverage (critical business logic)
- **Runtime**: 90% coverage (builder and runner)
- **Strategies**: 90% coverage (execution strategies)
- **Migration shims**: 100% coverage (backward compatibility critical)

## Migration Execution Plan

### Phase 1: Foundation (Weeks 1-2)

1. Create directory structure (interfaces/, adapters/, core/, runtime/, strategies/, processing/)
2. Define all seven port interfaces in interfaces/
3. Write unit tests for port interfaces
4. Verify import restrictions with linting rules

### Phase 2: Adapters (Weeks 3-4)

1. Implement ADBDevice adapter wrapping existing tools/device/adb.py
2. Implement GeminiLLM adapter wrapping existing infrastructure/llm/gemini.py
3. Implement SQLiteMemory adapter wrapping existing infrastructure/memory/
4. Implement remaining adapters (SQLiteKnowledge, NoopSignal, LocalStorage, StructlogAdapter)
5. Write unit tests for each adapter
6. Write property tests for adapter compliance

### Phase 3: Runtime (Week 5)

1. Implement FathomBuilder with fluent API
2. Implement FathomRunner with port wiring
3. Write unit tests for builder validation
4. Write property tests for builder behavior
5. Create minimal working example

### Phase 4: Core Migration (Weeks 6-7)

1. Create core/execution/engine.py with ExecutionEngine
2. Create core/context/manager.py with ContextManager
3. Migrate execution logic from orchestration/executor.py
4. Write unit tests for core components
5. Write property tests for execution phases
6. Maintain backward compatibility shims in orchestration/

### Phase 5: Strategies (Week 8)

1. Create strategies/intent.py with IntentStrategy
2. Create strategies/exploration.py with ExplorationStrategy
3. Migrate strategy logic from workflows/
4. Write unit tests for strategies
5. Write property tests for strategy behavior
6. Maintain backward compatibility shims in workflows/

### Phase 6: Processing Migration (Week 9)

1. Create processing/ directory
2. Move tools/vision/processing/annotator.py to processing/annotator.py
3. Move tools/vision/processing/drawer.py to processing/drawer.py
4. Move tools/vision/processing/geometry.py to processing/geometry.py
5. Move tools/vision/processing/parsers/ to processing/parsers/
6. Update imports in dependent code
7. Write property tests for processing preservation
8. Maintain backward compatibility shims in tools/vision/processing/

### Phase 7: Proprietary Code Migration (Week 10)

1. Move prompts/ to new location (if needed)
2. Move tools/definitions.py to new location (if needed)
3. Move services/parsing.py to new location (if needed)
4. Update only import statements
5. Write property tests for code preservation
6. Verify all function signatures unchanged

### Phase 8: Cleanup (Week 11)

1. Remove backward compatibility shims (after all dependents migrated)
2. Mark legacy directories as deprecated
3. Update documentation
4. Final full test suite run
5. Performance benchmarking

### Rollback Procedures

Each phase has a rollback procedure:

1. **Revert commits**: Use git to revert to pre-migration state
2. **Restore shims**: Ensure backward compatibility shims are in place
3. **Run tests**: Verify system works in pre-migration state
4. **Document issues**: Record what went wrong for next attempt

### Success Criteria

Migration is complete when:

1. All seven ports are defined and implemented
2. Builder API works with minimal and full configuration
3. All existing tests pass
4. All property tests pass
5. Legacy code works through compatibility shims
6. New code follows import restrictions
7. Documentation is updated
8. Performance is equivalent or better
