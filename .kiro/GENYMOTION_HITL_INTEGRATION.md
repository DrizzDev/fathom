# Genymotion HITL Integration Plan

## 🎯 Architecture Overview

Your Genymotion service uses **Temporal workflows** for orchestration, which is PERFECT for HITL because:
- ✅ Workflows can receive signals (pause/resume/inject)
- ✅ Long-running execution is supported
- ✅ State is persisted across signals
- ✅ Multiple users can interact with the same workflow

## 📋 Current Setup Analysis

### What You Have
```
Genymotion Service (FastAPI)
├── POST /crawler/intent          → Start workflow
├── POST /runs/{id}/pause         → Signal: pause
├── POST /runs/{id}/resume        → Signal: resume
└── POST /runs/{id}/inject        → Signal: inject (context)

Temporal Workflow
└── FathomWorkflow.run()
    └── Signals: pause, resume, inject
```

### What's Missing
1. **Temporal workflow implementation** (`fathom.runtime.temporal.workflow`)
2. **Signal-aware signal adapter** (to receive Temporal signals)
3. **Workflow-to-Fathom bridge** (connect Temporal signals to HITL)

---

## 🏗️ Implementation Plan

### Phase 1: Create Temporal Workflow (NEW)
Create `src/fathom/runtime/temporal/workflow.py`

### Phase 2: Create Temporal Signal Adapter (NEW)
Create `src/fathom/adapters/signal/temporal.py`

### Phase 3: Update Genymotion Manager
Update `services/crawler/manager.py` to use new workflow

### Phase 4: Test End-to-End
Test the full flow: Start → Pause → Inject → Resume

---

## 📁 File Structure

```
src/fathom/
├── runtime/
│   └── temporal/
│       ├── __init__.py
│       ├── workflow.py      # NEW: Temporal workflow
│       └── activities.py    # NEW: Temporal activities
├── adapters/
│   └── signal/
│       ├── interactive.py   # Existing: Terminal-based HITL
│       └── temporal.py      # NEW: Temporal-based HITL
└── ...

genymotion_project/
├── services/
│   └── crawler/
│       └── manager.py       # UPDATE: Use new workflow
└── ...
```

---

## 🔧 Implementation Details

### 1. Temporal Workflow (`src/fathom/runtime/temporal/workflow.py`)

```python
"""Temporal workflow for Fathom execution with HITL support."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from logging import getLogger
from typing import Any, Dict

from temporalio import workflow
from temporalio.common import RetryPolicy

from fathom.runtime.runner import FathomRunner
from fathom.runtime.builder import FathomBuilder
from fathom.schemas.results import ExecutionResult

logger = getLogger(__name__)


@workflow.defn
class FathomWorkflow:
    """
    Temporal workflow for executing Fathom tasks with HITL support.
    
    Supports signals:
    - pause: Pause execution
    - resume: Resume execution
    - inject: Inject user context
    """

    def __init__(self) -> None:
        """Initialize workflow state."""
        self._paused = False
        self._injected_context: str | None = None
        self._cancelled = False

    @workflow.run
    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom intent with HITL support.
        
        Args:
            request: Agent run request containing:
                - session_id: Device session ID
                - intent: User intent
                - enricher_url: URL for enrichment service
                - planner_configuration: LLM config
        
        Returns:
            Execution result with success, steps, duration, etc.
        """
        workflow.logger.info(f"Starting Fathom workflow for session {request.get('session_id')}")
        
        try:
            # Execute Fathom with Temporal signal adapter
            result = await workflow.execute_activity(
                execute_fathom_intent,
                args=[request, workflow.info().workflow_id],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=RetryPolicy(
                    maximum_attempts=1,  # No retries for HITL workflows
                ),
            )
            
            workflow.logger.info(f"Workflow completed: {result}")
            return result
            
        except Exception as e:
            workflow.logger.exception(f"Workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "steps": 0,
                "duration": 0,
            }

    @workflow.signal
    async def pause(self) -> None:
        """Signal to pause execution."""
        workflow.logger.info("Received pause signal")
        self._paused = True

    @workflow.signal
    async def resume(self) -> None:
        """Signal to resume execution."""
        workflow.logger.info("Received resume signal")
        self._paused = False

    @workflow.signal
    async def inject(self, context: str) -> None:
        """Signal to inject user context."""
        workflow.logger.info(f"Received inject signal: {context}")
        self._injected_context = context

    @workflow.signal
    async def cancel(self) -> None:
        """Signal to cancel execution."""
        workflow.logger.info("Received cancel signal")
        self._cancelled = True


@workflow.defn
async def execute_fathom_intent(
    request: Dict[str, Any],
    workflow_id: str,
) -> Dict[str, Any]:
    """
    Activity to execute Fathom intent.
    
    This runs as a Temporal activity so it can be monitored and cancelled.
    """
    from fathom.adapters.signal.temporal import TemporalSignalAdapter
    from fathom.runtime.config import FathomConfig
    
    # Create config with Temporal signal adapter
    config = FathomConfig(
        device_id=request["session_id"],
        credentials_path=None,  # Will use credentials_json
        credentials_json=request["planner_configuration"]["credentials_json"],
        project_id=request["planner_configuration"]["project_id"],
        location=request["planner_configuration"]["location"],
        model=request["planner_configuration"]["model"],
        max_steps=100,  # From request limit
        use_xml=False,
        interactive=True,  # Enable HITL
    )
    
    # Create Temporal signal adapter
    signal_adapter = TemporalSignalAdapter(workflow_id=workflow_id)
    
    # Build runner with Temporal signal adapter
    builder = FathomBuilder(config=config)
    builder.with_signal_adapter(signal_adapter)
    runner = builder.build()
    
    try:
        # Execute intent
        result = await runner.run_intent(
            intent=request["intent"],
            workflow_id=workflow_id,
        )
        
        return {
            "success": result.success,
            "steps": result.steps,
            "duration": result.duration,
            "error": result.error,
            "metrics": result.metrics.to_dict() if result.metrics else None,
        }
        
    finally:
        await runner.cleanup()
```

### 2. Temporal Signal Adapter (`src/fathom/adapters/signal/temporal.py`)

```python
"""Temporal-based signal adapter for HITL in distributed workflows."""

from __future__ import annotations

import asyncio
from typing import Optional
from logging import getLogger

from temporalio.client import Client

from fathom.interfaces.signal import SignalPort
from fathom.constants import SignalType

logger = getLogger(__name__)


class TemporalSignalAdapter(SignalPort):
    """
    Signal adapter that receives signals from Temporal workflows.
    
    This enables HITL in distributed microservice environments where
    the user interacts via HTTP API instead of terminal.
    """

    def __init__(self, workflow_id: str, client: Optional[Client] = None) -> None:
        """
        Initialize Temporal signal adapter.
        
        Args:
            workflow_id: The Temporal workflow ID
            client: Optional Temporal client (will create if not provided)
        """
        self.__workflow_id = workflow_id
        self.__client = client
        self.__paused = False
        self.__injected_context: Optional[str] = None
        self.__pause_requested = False
        
        logger.info(f"TemporalSignalAdapter initialized for workflow {workflow_id}")

    async def check_signal(self) -> Optional[str]:
        """
        Check for control signal from Temporal workflow.
        
        This is called periodically by the execution loop.
        """
        # In Temporal, signals are received via workflow.signal decorators
        # The workflow state is shared, so we check the workflow's state
        
        # For now, we'll use a simple flag-based approach
        # The workflow will set these flags via signals
        
        if self.__pause_requested:
            return SignalType.ASK.value
        
        return None

    def is_pause_requested(self) -> bool:
        """Check if pause is requested (for immediate cancellation)."""
        return self.__pause_requested

    async def wait_for_resume(self) -> None:
        """
        Block until RESUME signal received from Temporal.
        
        This polls the workflow state until resume is signaled.
        """
        logger.info(f"Workflow {self.__workflow_id} paused, waiting for resume signal")
        
        self.__paused = True
        
        # Poll for resume signal
        while self.__paused:
            await asyncio.sleep(0.5)  # Poll every 500ms
        
        logger.info(f"Workflow {self.__workflow_id} resumed")

    async def request_input(self, *, prompt: str) -> str:
        """
        Request human input with prompt.
        
        In Temporal mode, this is not used because the user provides
        input via the /inject endpoint, not interactively.
        """
        logger.warning("request_input called in Temporal mode - not supported")
        return ""

    def pause(self) -> None:
        """Pause execution (called by Temporal signal)."""
        self.__paused = True
        self.__pause_requested = True
        logger.info(f"Workflow {self.__workflow_id} pause requested")

    def resume(self) -> None:
        """Resume execution (called by Temporal signal)."""
        self.__paused = False
        self.__pause_requested = False
        logger.info(f"Workflow {self.__workflow_id} resume requested")

    def inject_context(self, context: str) -> None:
        """Inject user context (called by Temporal signal)."""
        self.__injected_context = context
        logger.info(f"Context injected into workflow {self.__workflow_id}: {context}")

    def get_injected_context(self) -> Optional[str]:
        """Get injected context and clear it."""
        context = self.__injected_context
        self.__injected_context = None
        return context

    def has_injected_context(self) -> bool:
        """Check if there's injected context available."""
        return self.__injected_context is not None
```

### 3. Update FathomBuilder (`src/fathom/runtime/builder.py`)

Add method to inject custom signal adapter:

```python
class FathomBuilder:
    # ... existing code ...
    
    def with_signal_adapter(self, adapter: SignalPort) -> FathomBuilder:
        """
        Use a custom signal adapter instead of the default.
        
        This allows using Temporal signals, WebSocket signals, etc.
        """
        self.__signal_adapter = adapter
        return self
    
    def build(self) -> FathomRunner:
        # ... existing code ...
        
        # Use custom signal adapter if provided
        if self.__signal_adapter:
            signal = self.__signal_adapter
        elif self.__config.interactive:
            signal = InteractiveSignal()
        else:
            signal = NoOpSignal()
        
        # ... rest of build logic ...
```

### 4. Update Temporal Workflow Registration

In your Genymotion service, you need to register the workflow:

```python
# services/temporal/worker.py (NEW FILE)
from temporalio.client import Client
from temporalio.worker import Worker

from fathom.runtime.temporal.workflow import FathomWorkflow, execute_fathom_intent


async def start_worker(client: Client, task_queue: str):
    """Start Temporal worker for Fathom workflows."""
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[FathomWorkflow],
        activities=[execute_fathom_intent],
    )
    
    await worker.run()
```

### 5. Update Signal Handling in Workflow

The workflow needs to communicate with the activity:

```python
# In FathomWorkflow class

@workflow.signal
async def pause(self) -> None:
    """Signal to pause execution."""
    workflow.logger.info("Received pause signal")
    # Send signal to activity via shared state or activity cancellation
    # For now, we'll use a simpler approach with the adapter
    pass  # The adapter will handle this
```

---

## 🔄 Complete Flow

### 1. User Starts Workflow
```
POST /crawler/intent
{
  "intent": "Search for open source memory options",
  "session": "emulator-5554",
  "limit": 100,
  "org_id": 123
}

Response:
{
  "identity": "uuid-1234",
  "status": "initiated",
  "message": "Crawler started..."
}
```

### 2. Workflow Executes
```
Temporal Workflow (uuid-1234)
└── Activity: execute_fathom_intent
    └── FathomRunner with TemporalSignalAdapter
        └── IntentStrategy (executing steps)
```

### 3. User Pauses
```
POST /runs/uuid-1234/pause

→ Temporal signal: workflow.pause()
→ TemporalSignalAdapter.pause()
→ IntentStrategy detects pause
→ LLM call cancelled
→ Execution paused
```

### 4. User Injects Context
```
POST /runs/uuid-1234/inject
{
  "context": "Actually search for indian climate instead"
}

→ Temporal signal: workflow.inject(context)
→ TemporalSignalAdapter.inject_context(context)
→ Context stored in adapter
```

### 5. User Resumes
```
POST /runs/uuid-1234/resume

→ Temporal signal: workflow.resume()
→ TemporalSignalAdapter.resume()
→ IntentStrategy checks for injected context
→ Finds: "Actually search for indian climate instead"
→ Adds to LLM prompt with PRIORITY formatting
→ Execution continues with new context
```

---

## 🚀 Migration Steps

### Step 1: Create Temporal Workflow Files
```bash
mkdir -p src/fathom/runtime/temporal
touch src/fathom/runtime/temporal/__init__.py
touch src/fathom/runtime/temporal/workflow.py
touch src/fathom/runtime/temporal/activities.py
```

### Step 2: Create Temporal Signal Adapter
```bash
touch src/fathom/adapters/signal/temporal.py
```

### Step 3: Update FathomBuilder
Add `with_signal_adapter()` method to `src/fathom/runtime/builder.py`

### Step 4: Update Genymotion Manager
Update `services/crawler/manager.py` to use new workflow

### Step 5: Register Temporal Worker
Create worker registration in Genymotion service

### Step 6: Test
```bash
# Start Temporal worker
python -m services.temporal.worker

# Test the flow
curl -X POST http://localhost:8000/crawler/intent \
  -H "Content-Type: application/json" \
  -d '{"intent": "Open Chrome", "session": "emulator-5554", "limit": 10, "org_id": 1}'

# Get workflow ID from response
WORKFLOW_ID="uuid-from-response"

# Pause
curl -X POST http://localhost:8000/runs/$WORKFLOW_ID/pause

# Inject
curl -X POST http://localhost:8000/runs/$WORKFLOW_ID/inject \
  -H "Content-Type: application/json" \
  -d '{"context": "Wait 5 seconds before proceeding"}'

# Resume
curl -X POST http://localhost:8000/runs/$WORKFLOW_ID/resume
```

---

## 🎯 Key Benefits

### For End Users
- ✅ Pause execution at ANY time via API
- ✅ Inject context/guidance via API
- ✅ Resume execution via API
- ✅ No terminal access needed
- ✅ Works in distributed/cloud environments

### For Your Architecture
- ✅ Leverages existing Temporal infrastructure
- ✅ Signals are persisted (survives restarts)
- ✅ Multiple users can interact with same workflow
- ✅ Full observability via Temporal UI
- ✅ Scales horizontally

### For Fathom
- ✅ HITL works in microservice mode
- ✅ Context injection with priority formatting
- ✅ Immediate LLM cancellation on pause
- ✅ Sub-goal and modified intent support

---

## 📊 Comparison: Terminal vs Temporal HITL

| Feature | Terminal (InteractiveSignal) | Temporal (TemporalSignalAdapter) |
|---------|------------------------------|----------------------------------|
| **Pause trigger** | Type "pause" + Enter | POST /runs/{id}/pause |
| **Context injection** | Interactive menu | POST /runs/{id}/inject |
| **Resume** | Menu option | POST /runs/{id}/resume |
| **User interface** | Terminal | HTTP API |
| **Multi-user** | ❌ Single user | ✅ Multiple users |
| **Distributed** | ❌ Local only | ✅ Works across services |
| **State persistence** | ❌ Lost on crash | ✅ Persisted by Temporal |
| **Observability** | Terminal logs | Temporal UI + logs |

---

## 🔍 Next Steps

1. **I'll create the implementation files** if you confirm this approach
2. **Test locally** with a simple intent
3. **Deploy to Genymotion** service
4. **Monitor** via Temporal UI
5. **Iterate** based on user feedback

**Should I proceed with creating the implementation files?** 🚀
