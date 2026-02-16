# Requirements: Temporal Workflow-Activity Communication

## 1. Overview

Fix the Temporal integration so that workflow signals (pause/resume/inject/cancel) properly communicate with the activity executing Fathom. Currently, the workflow and activity have separate state that doesn't synchronize.

## 2. User Stories

### 2.1 As a user, I want to pause a running Fathom workflow via HTTP API
**Given** a Fathom workflow is executing in Temporal  
**When** I send `POST /runs/{workflow_id}/pause`  
**Then** the activity should detect the pause signal within 500ms and pause execution

### 2.2 As a user, I want to resume a paused workflow via HTTP API
**Given** a Fathom workflow is paused  
**When** I send `POST /runs/{workflow_id}/resume`  
**Then** the activity should detect the resume signal and continue execution

### 2.3 As a user, I want to inject context into a running workflow via HTTP API
**Given** a Fathom workflow is executing  
**When** I send `POST /runs/{workflow_id}/inject` with context  
**Then** the activity should receive the context and include it in the next LLM prompt

### 2.4 As a user, I want to cancel a running workflow via HTTP API
**Given** a Fathom workflow is executing  
**When** I send `POST /runs/{workflow_id}/cancel`  
**Then** the activity should detect the cancel signal and stop execution gracefully

## 3. Acceptance Criteria

### 3.1 Workflow State Management
- Workflow maintains state for: `_paused`, `_cancelled`, `_injected_context`
- Workflow exposes `get_state()` query to read current state
- Workflow signal handlers update state immediately

### 3.2 Activity State Synchronization
- Activity queries workflow state via workflow handle
- Activity polls workflow state every 100-500ms during execution
- Activity detects state changes within 500ms maximum

### 3.3 Signal Flow Architecture
```
User → HTTP API → Temporal Signal → Workflow State Update
                                         ↓
Activity → Query Workflow State → Detect Change → Take Action
```

### 3.4 TemporalSignalAdapter Implementation
- `check_signal()`: Queries workflow state, returns SignalType.ASK if paused
- `is_pause_requested()`: Queries workflow state, returns True if paused
- `wait_for_resume()`: Polls workflow state until `_paused` becomes False
- `get_injected_context()`: Queries workflow state, returns and clears `_injected_context`
- `has_injected_context()`: Queries workflow state, returns True if context exists

### 3.5 No Placeholder Logic
- Remove all comments like "In production, this would..."
- Remove all placeholder implementations
- Implement actual workflow state queries using Temporal APIs

### 3.6 Error Handling
- Handle `WorkflowNotFoundError` gracefully
- Handle activity timeout during pause (keep activity alive with heartbeats)
- Handle network errors when querying workflow state

### 3.7 Performance
- State queries should not block execution significantly (<50ms per query)
- Use efficient polling intervals (100-500ms)
- Send heartbeats during long pauses to prevent activity timeout

## 4. Technical Constraints

### 4.1 Temporal APIs to Use
- `workflow.get_external_workflow_handle()`: Get handle to query workflow from activity
- `handle.query(FathomWorkflow.get_state)`: Query workflow state
- `activity.heartbeat()`: Keep activity alive during pauses

### 4.2 Architecture Rules
- Follow hexagonal architecture: `interfaces/` not `ports/`
- Keep `core/` and `runtime/` separate
- Plug-and-play pattern: Builder accepts any SignalPort implementation

### 4.3 Code Quality
- No placeholder/dummy logic
- No boilerplate code
- Production-grade implementation
- Proper error handling and logging

## 5. Out of Scope

- Workflow execution logic (already implemented)
- Activity execution logic (already implemented)
- HTTP API endpoints (already implemented in Genymotion)
- Builder pattern changes (already correct)

## 6. Dependencies

- `temporalio` Python SDK
- Existing Fathom interfaces and implementations
- Genymotion HTTP API (external, already implemented)

## 7. Success Metrics

- User can pause/resume/inject/cancel workflows via HTTP API
- State changes detected within 500ms
- No placeholder logic remains in codebase
- All error cases handled gracefully
