# Design: Temporal Workflow-Activity Communication

## 1. Architecture Overview

### 1.1 Current Problem
The workflow and activity have separate state:
- Workflow: `self._paused`, `self._cancelled`, `self._injected_context`
- Activity's TemporalSignalAdapter: `self.__paused`, `self.__pause_requested`, `self.__injected_context`

When a user sends a signal (e.g., pause), the workflow updates its state, but the activity doesn't know about it because they don't share state.

### 1.2 Solution: Workflow Handle Queries
The activity will query the workflow's state using Temporal's workflow handle API:

```python
# In activity
from temporalio.client import Client
from temporalio import workflow

# Get workflow handle
handle = workflow.get_external_workflow_handle(workflow_id)

# Query workflow state
state = await handle.query(FathomWorkflow.get_state)

# Check if paused
if state["paused"]:
    # Take action
```

### 1.3 Signal Flow
```
┌─────────┐     HTTP      ┌──────────────┐    Signal    ┌──────────┐
│  User   │────────────────▶│  Genymotion  │─────────────▶│ Workflow │
└─────────┘   POST /pause   │     API      │              │  State   │
                            └──────────────┘              └────┬─────┘
                                                               │
                                                          Updates:
                                                          _paused=True
                                                               │
┌──────────┐    Query       ┌──────────────┐    Returns  ┌────▼─────┐
│ Activity │◀───────────────│ Workflow     │◀────────────│ get_state│
│          │   get_state()  │   Handle     │   {paused:  │  Query   │
│          │                │              │    True}    └──────────┘
└────┬─────┘                └──────────────┘
     │
     ▼
Detects pause,
calls wait_for_resume()
```

## 2. Component Design

### 2.1 FathomWorkflow (workflow.py)
**No changes needed** - already has:
- Signal handlers: `pause()`, `resume()`, `inject()`, `cancel()`
- State variables: `_paused`, `_cancelled`, `_injected_context`
- Query method: `get_state()` returns current state

### 2.2 TemporalSignalAdapter (adapters/signal/temporal.py)
**Major changes needed** - replace placeholder logic with real queries:

#### 2.2.1 Constructor
```python
def __init__(self, workflow_id: str) -> None:
    self.__workflow_id = workflow_id
    self.__workflow_handle = None  # Lazy-initialized
```

#### 2.2.2 Get Workflow Handle (new private method)
```python
async def __get_workflow_handle(self):
    """Get or create workflow handle for querying state."""
    if self.__workflow_handle is None:
        # Get handle from activity context
        handle = workflow.get_external_workflow_handle(self.__workflow_id)
        self.__workflow_handle = handle
    return self.__workflow_handle
```

#### 2.2.3 Query Workflow State (new private method)
```python
async def __query_workflow_state(self) -> Dict[str, Any]:
    """Query current workflow state."""
    try:
        handle = await self.__get_workflow_handle()
        state = await handle.query(FathomWorkflow.get_state)
        return state
    except Exception as e:
        logger.error(f"Failed to query workflow state: {e}")
        return {"paused": False, "cancelled": False, "has_context": False}
```

#### 2.2.4 check_signal() - Replace placeholder
```python
async def check_signal(self) -> Optional[str]:
    """Check for control signal from Temporal workflow."""
    state = await self.__query_workflow_state()
    
    if state.get("cancelled"):
        return SignalType.CANCEL.value
    
    if state.get("paused"):
        return SignalType.ASK.value
    
    return None
```

#### 2.2.5 is_pause_requested() - Replace placeholder
```python
def is_pause_requested(self) -> bool:
    """Check if pause is requested (synchronous for LLM cancellation)."""
    # For immediate cancellation during LLM calls, we need sync check
    # Use asyncio.run() to query workflow state synchronously
    import asyncio
    try:
        state = asyncio.run(self.__query_workflow_state())
        return state.get("paused", False)
    except Exception:
        return False
```

#### 2.2.6 wait_for_resume() - Replace placeholder
```python
async def wait_for_resume(self) -> None:
    """Block until RESUME signal received from Temporal."""
    logger.info(f"Workflow {self.__workflow_id} paused, waiting for resume")
    
    while True:
        # Query workflow state
        state = await self.__query_workflow_state()
        
        if not state.get("paused"):
            logger.info(f"Workflow {self.__workflow_id} resumed")
            break
        
        # Send heartbeat to keep activity alive
        try:
            activity.heartbeat("Paused - waiting for resume")
        except RuntimeError:
            pass
        
        # Poll every 500ms
        await asyncio.sleep(0.5)
```

#### 2.2.7 get_injected_context() - Replace placeholder
```python
def get_injected_context(self) -> Optional[str]:
    """Get injected context from workflow."""
    import asyncio
    try:
        state = asyncio.run(self.__query_workflow_state())
        
        # If workflow has context, retrieve it via signal
        if state.get("has_context"):
            # Get the actual context value
            handle = asyncio.run(self.__get_workflow_handle())
            # Workflow needs a query method to return and clear context
            context = asyncio.run(handle.query(FathomWorkflow.get_injected_context))
            return context
    except Exception as e:
        logger.error(f"Failed to get injected context: {e}")
    
    return None
```

#### 2.2.8 has_injected_context() - Replace placeholder
```python
def has_injected_context(self) -> bool:
    """Check if there's injected context available."""
    import asyncio
    try:
        state = asyncio.run(self.__query_workflow_state())
        return state.get("has_context", False)
    except Exception:
        return False
```

#### 2.2.9 Remove unused methods
Remove these methods (workflow handles signals directly):
- `pause()` - Not needed, workflow handles signal
- `resume()` - Not needed, workflow handles signal
- `inject_context()` - Not needed, workflow handles signal

### 2.3 FathomWorkflow - Add Query for Context
**Add new query method** to return and clear injected context:

```python
@workflow.query
def get_injected_context(self) -> Optional[str]:
    """
    Query and clear injected context.
    
    Returns:
        The injected context, or None if no context
    """
    context = self._injected_context
    self._injected_context = None
    return context
```

### 2.4 Activities (activities.py)
**Minor changes** - remove unused import:
- Remove `import json` (unused)

## 3. Error Handling

### 3.1 Workflow Not Found
```python
from temporalio.exceptions import WorkflowNotFoundError

try:
    state = await handle.query(FathomWorkflow.get_state)
except WorkflowNotFoundError:
    logger.error(f"Workflow {workflow_id} not found")
    return default_state
```

### 3.2 Activity Timeout During Pause
- Send heartbeats every 500ms during `wait_for_resume()`
- Heartbeat timeout set to 30s in workflow (already configured)
- Activity will stay alive as long as heartbeats continue

### 3.3 Query Failures
- Catch all exceptions in `__query_workflow_state()`
- Return safe default state: `{"paused": False, "cancelled": False, "has_context": False}`
- Log errors for debugging

## 4. Performance Considerations

### 4.1 Query Frequency
- Poll workflow state every 500ms during execution
- Only query when needed (not on every step)
- Cache workflow handle (don't recreate on every query)

### 4.2 Heartbeat Frequency
- Send heartbeats every 500ms during pause
- Heartbeat timeout: 30s (configured in workflow)
- Prevents activity timeout during long pauses

### 4.3 Synchronous Queries
- `is_pause_requested()` needs to be synchronous for LLM cancellation
- Use `asyncio.run()` to run async query in sync context
- This is acceptable because it's only called during LLM analysis (not frequently)

## 5. Testing Strategy

### 5.1 Unit Tests
- Test `TemporalSignalAdapter` methods with mocked workflow handle
- Test query error handling
- Test state transitions

### 5.2 Integration Tests
- Start real Temporal workflow
- Send signals via workflow handle
- Verify activity detects state changes
- Test pause/resume/inject/cancel flows

### 5.3 Manual Testing
- Deploy to Genymotion environment
- Test HTTP API endpoints
- Verify end-to-end signal flow

## 6. Migration Path

### 6.1 Backward Compatibility
- No breaking changes to public APIs
- Existing code using `FathomBuilder` continues to work
- Only internal implementation changes

### 6.2 Deployment
1. Update `TemporalSignalAdapter` implementation
2. Add `get_injected_context()` query to `FathomWorkflow`
3. Remove unused imports from activities
4. Deploy to Temporal workers
5. Test with Genymotion API

## 7. Correctness Properties

### Property 1: Signal Detection Latency
**Property**: When a signal is sent to the workflow, the activity detects it within 1 second.

**Rationale**: Users expect responsive pause/resume behavior. 1 second is acceptable latency for HITL interactions.

**Test Strategy**: 
- Send pause signal
- Measure time until activity detects it
- Assert latency < 1000ms

### Property 2: State Consistency
**Property**: The activity's view of workflow state is eventually consistent with the workflow's actual state.

**Rationale**: Distributed systems have eventual consistency. The activity should converge to the correct state within a bounded time.

**Test Strategy**:
- Set workflow state (e.g., paused=True)
- Query from activity repeatedly
- Assert activity sees correct state within 1 second

### Property 3: Context Delivery
**Property**: When context is injected, it is delivered to the activity exactly once.

**Rationale**: Context should not be lost or duplicated. The workflow clears context after delivery.

**Test Strategy**:
- Inject context via signal
- Query from activity
- Assert context is returned
- Query again
- Assert context is None (cleared)

### Property 4: Heartbeat Continuity
**Property**: During pause, the activity sends heartbeats at least every 1 second.

**Rationale**: Temporal requires heartbeats to keep activities alive. Missing heartbeats cause timeout.

**Test Strategy**:
- Pause workflow
- Monitor heartbeat timestamps
- Assert gap between heartbeats < 1 second

### Property 5: Graceful Cancellation
**Property**: When cancel signal is sent, the activity stops within 2 seconds without errors.

**Rationale**: Cancellation should be clean and timely. 2 seconds allows current step to complete.

**Test Strategy**:
- Start workflow
- Send cancel signal
- Measure time until activity exits
- Assert exit time < 2000ms
- Assert no exceptions raised

## 8. Implementation Order

1. Add `get_injected_context()` query to `FathomWorkflow`
2. Implement `__get_workflow_handle()` in `TemporalSignalAdapter`
3. Implement `__query_workflow_state()` in `TemporalSignalAdapter`
4. Replace `check_signal()` implementation
5. Replace `is_pause_requested()` implementation
6. Replace `wait_for_resume()` implementation
7. Replace `get_injected_context()` implementation
8. Replace `has_injected_context()` implementation
9. Remove unused methods (`pause()`, `resume()`, `inject_context()`)
10. Remove unused imports from activities
11. Test with unit tests
12. Test with integration tests
13. Deploy and test with Genymotion API
