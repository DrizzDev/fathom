# Tasks: Temporal Workflow-Activity Communication

## 1. Update FathomWorkflow
- [x] 1.1 Add `get_injected_context()` query method to return and clear injected context

## 2. Implement Workflow Handle Management in TemporalSignalAdapter
- [x] 2.1 Add `__workflow_handle` instance variable to constructor
- [x] 2.2 Implement `__get_workflow_handle()` private method using `workflow.get_external_workflow_handle()`
- [x] 2.3 Implement `__query_workflow_state()` private method with error handling

## 3. Replace Placeholder Logic in TemporalSignalAdapter
- [x] 3.1 Replace `check_signal()` to query workflow state and return appropriate signal type
- [x] 3.2 Replace `is_pause_requested()` to synchronously query workflow state
- [x] 3.3 Replace `wait_for_resume()` to poll workflow state with heartbeats
- [x] 3.4 Replace `get_injected_context()` to query workflow for context
- [x] 3.5 Replace `has_injected_context()` to query workflow state

## 4. Clean Up TemporalSignalAdapter
- [x] 4.1 Remove unused `pause()` method
- [x] 4.2 Remove unused `resume()` method
- [x] 4.3 Remove unused `inject_context()` method
- [x] 4.4 Remove unused instance variables (`__paused`, `__pause_requested`, `__injected_context`)

## 5. Clean Up Activities
- [x] 5.1 Remove unused `import json` from activities.py

## 6. Testing
- [ ] 6.1 Write unit tests for TemporalSignalAdapter with mocked workflow handle
- [ ] 6.2 Write integration test for pause/resume flow
- [ ] 6.3 Write integration test for context injection flow
- [ ] 6.4 Write integration test for cancel flow
- [ ] 6.5 Test signal detection latency (Property 1)
- [ ] 6.6 Test state consistency (Property 2)
- [ ] 6.7 Test context delivery (Property 3)
- [ ] 6.8 Test heartbeat continuity (Property 4)
- [ ] 6.9 Test graceful cancellation (Property 5)

## 7. Documentation
- [ ] 7.1 Update TemporalSignalAdapter docstring with correct architecture
- [ ] 7.2 Add usage examples to module docstring
- [ ] 7.3 Document error handling behavior
