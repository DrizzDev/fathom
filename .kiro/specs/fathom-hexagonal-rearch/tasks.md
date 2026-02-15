# Implementation Plan: Fathom Hexagonal Architecture Migration

## Overview

This plan implements an incremental migration to hexagonal architecture. The approach builds new architecture alongside existing code, maintains backward compatibility throughout, and migrates modules one at a time with full test coverage.

**Key Principle**: Copy-paste existing logic without modifications. Only update imports and structure.

## Tasks

- [x] 1. Create hexagonal architecture foundation
  - Create directory structure: interfaces/, adapters/, core/, runtime/, strategies/, processing/
  - Set up import linting rules to enforce architectural boundaries
  - Create __init__.py files for all new directories
  - _Requirements: 1.1_

- [ ] 2. Define port interfaces
  - [x] 2.1 Define DevicePort interface
    - Create interfaces/device.py with DevicePort ABC
    - Define methods: tap(), type_text(), swipe(), back(), home(), get_screen_size(), capture_screen(), get_current_package(), wait_for_device()
    - Add Protocol decorator for runtime checking
    - _Requirements: 2.1_
  
  - [x] 2.2 Define LLMPort interface
    - Create interfaces/llm.py with LLMPort ABC
    - Define methods: analyze(), cleanup()
    - _Requirements: 2.2_
  
  - [x] 2.3 Define MemoryPort interface
    - Create interfaces/memory.py with MemoryPort ABC
    - Define methods: set(), get(), get_all(), store_observation(), store_experience(), retrieve_knowledge()
    - _Requirements: 2.3_
  
  - [x] 2.4 Define KnowledgePort interface
    - Create interfaces/knowledge.py with KnowledgePort ABC
    - Define methods: add_screen(), add_transition(), find_path(), get_neighbors()
    - _Requirements: 2.4_
  
  - [x] 2.5 Define SignalPort interface
    - Create interfaces/signal.py with SignalPort ABC
    - Define methods: check_signal(), wait_for_resume(), request_input()
    - _Requirements: 2.5_
  
  - [x] 2.6 Define StoragePort interface
    - Create interfaces/storage.py with StoragePort ABC
    - Define method: save()
    - _Requirements: 2.6_
  
  - [x] 2.7 Define TelemetryPort interface
    - Create interfaces/telemetry.py with TelemetryPort ABC
    - Define methods: debug(), info(), warning(), error()
    - _Requirements: 2.7_
  
  - [ ]* 2.8 Write property test for port interface compliance
    - **Property 1: Port Interface Compliance**
    - **Validates: Requirements 1.3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

- [ ] 3. Implement adapter for ADBDevice
  - [x] 3.1 Create ADBDevice adapter
    - Create adapters/device/adb.py
    - Wrap existing tools/device/adb.py and tools/capture/adb.py
    - Implement DevicePort by delegating to existing ADBDeviceTool and ADBCaptureTool
    - Copy-paste existing logic, do not modify
    - _Requirements: 3.1_
  
  - [ ]* 3.2 Write unit tests for ADBDevice adapter
    - Test each DevicePort method delegates correctly
    - Test initialization with serial parameter
    - _Requirements: 3.1_

- [ ] 4. Implement adapter for GeminiLLM
  - [x] 4.1 Create GeminiLLM adapter
    - Create adapters/llm/gemini.py
    - Wrap existing infrastructure/llm/gemini.py
    - Implement LLMPort by delegating to existing GeminiLLMClient
    - Copy-paste existing logic, do not modify
    - _Requirements: 3.2_
  
  - [ ]* 4.2 Write unit tests for GeminiLLM adapter
    - Test analyze() delegates to GeminiLLMClient
    - Test cleanup() delegates correctly
    - Test initialization with api_key and model parameters
    - _Requirements: 3.2_

- [ ] 5. Implement adapter for SQLiteMemory
  - [x] 5.1 Create SQLiteMemory adapter
    - Create adapters/memory/sqlite.py
    - Wrap existing infrastructure/memory/sqlite.py and infrastructure/memory/ledger.py
    - Implement MemoryPort by delegating to SQLiteMemoryProvider and Ledger
    - Copy-paste existing logic, do not modify
    - _Requirements: 3.3_
  
  - [ ]* 5.2 Write unit tests for SQLiteMemory adapter
    - Test session methods (set, get, get_all) delegate to Ledger
    - Test memory methods (store_observation, store_experience, retrieve_knowledge) delegate to SQLiteMemoryProvider
    - _Requirements: 3.3_

- [ ] 6. Implement remaining adapters
  - [x] 6.1 Create SQLiteKnowledge adapter
    - Create adapters/knowledge/sqlite.py
    - Implement KnowledgePort using SQLite and rustworkx
    - _Requirements: 3.4_
  
  - [x] 6.2 Create NoopSignal adapter
    - Create adapters/signal/noop.py
    - Implement SignalPort with no-op methods for autonomous operation
    - _Requirements: 3.5_
  
  - [x] 6.3 Create LocalStorage adapter
    - Create adapters/storage/local.py
    - Wrap existing infrastructure/storage/local.py
    - Copy-paste existing logic, do not modify
    - _Requirements: 3.6_
  
  - [x] 6.4 Create StructlogAdapter for telemetry
    - Create adapters/telemetry/structlog.py
    - Implement TelemetryPort using structlog
    - _Requirements: 3.7_
  
  - [ ]* 6.5 Write unit tests for remaining adapters
    - Test SQLiteKnowledge graph operations
    - Test NoopSignal returns None for all signals
    - Test LocalStorage delegates to existing implementation
    - Test StructlogAdapter logs correctly
    - _Requirements: 3.4, 3.5, 3.6, 3.7_

- [ ] 7. Checkpoint - Ensure all adapter tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement builder API
  - [x] 8.1 Create FathomBuilder class
    - Create runtime/builder.py
    - Implement builder methods: device(), llm(), memory(), knowledge(), signal(), storage(), telemetry()
    - Each method returns self for chaining
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 8.2 Implement build() validation
    - Validate device() and llm() are required
    - Raise ValueError with descriptive message if missing
    - Apply defaults for optional ports (memory, knowledge, signal, storage, telemetry)
    - Return FathomRunner instance
    - _Requirements: 4.5, 4.6, 4.7, 11.1, 11.4, 11.5_
  
  - [x] 8.3 Create Fathom entry point
    - Create Fathom class with static builder() method
    - _Requirements: 4.1_
  
  - [ ]* 8.4 Write property test for builder method chaining
    - **Property 2: Builder Method Chaining**
    - **Validates: Requirements 4.3**
  
  - [ ]* 8.5 Write property test for builder order independence
    - **Property 3: Builder Order Independence**
    - **Validates: Requirements 4.4**
  
  - [ ]* 8.6 Write property test for required port validation
    - **Property 4: Required Port Validation**
    - **Validates: Requirements 4.5, 4.6, 11.5**
  
  - [ ]* 8.7 Write property test for default port assignment
    - **Property 5: Default Port Assignment**
    - **Validates: Requirements 11.1, 11.4**
  
  - [ ]* 8.8 Write property test for explicit port configuration
    - **Property 18: Explicit Port Configuration**
    - **Validates: Requirements 11.3**
  
  - [ ]* 8.9 Write unit tests for builder API
    - Test minimal configuration (device + llm only)
    - Test full configuration (all seven ports)
    - Test error messages for missing required ports
    - _Requirements: 4.1, 4.2, 4.5, 4.6, 11.2_

- [ ] 9. Implement core execution engine
  - [ ] 9.1 Create ExecutionEngine class
    - Create core/execution/engine.py
    - Implement execute_step() with seven phases: SignalCheck → Perceive → Reason → Act → Learn → Checkpoint → Evaluate
    - Accept ports as constructor parameters (device, llm, memory, signal, storage, telemetry)
    - Copy logic from orchestration/executor.py, only update imports
    - _Requirements: 10.1, 10.2, 10.3_
  
  - [ ] 9.2 Implement HITL signal handling
    - Add signal checking in SignalCheck phase
    - Handle PAUSE, RESUME, INJECT, ASK signals
    - Copy logic from existing code, do not modify
    - _Requirements: 10.4_
  
  - [ ]* 9.3 Write property test for execution phase sequence
    - **Property 14: Execution Phase Sequence Preservation**
    - **Validates: Requirements 10.1, 10.2, 10.3**
  
  - [ ]* 9.4 Write property test for HITL signal handling
    - **Property 15: HITL Signal Handling**
    - **Validates: Requirements 10.4**
  
  - [ ]* 9.5 Write unit tests for execution engine
    - Test each phase executes in order
    - Test PAUSE signal pauses execution
    - Test action execution delegates to DevicePort
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 10. Implement context management
  - [ ] 10.1 Create ContextManager class
    - Create core/context/manager.py
    - Implement three-tier context: roadmap, milestones, trace
    - Implement methods: commit(), branch(), recall()
    - _Requirements: 10.1_
  
  - [ ]* 10.2 Write unit tests for context management
    - Test commit() adds to trace
    - Test branch() creates milestone and compresses trace
    - Test recall() retrieves correct tier
    - _Requirements: 10.1_

- [ ] 11. Checkpoint - Ensure all core tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement FathomRunner
  - [x] 12.1 Create FathomRunner class
    - Create runtime/runner.py
    - Accept all seven ports as constructor parameters
    - Wire ExecutionEngine and ContextManager
    - Implement run() method with intent and strategy parameters
    - _Requirements: 4.7, 11.2_
  
  - [ ]* 12.2 Write unit tests for FathomRunner
    - Test runner wires all ports correctly
    - Test run() executes with intent strategy
    - Test run() executes with exploration strategy
    - _Requirements: 4.7, 11.2_

- [ ] 13. Implement execution strategies
  - [ ] 13.1 Create IntentStrategy
    - Create strategies/intent.py
    - Migrate logic from workflows/intent.py
    - Copy-paste existing logic, only update imports
    - _Requirements: 10.5_
  
  - [ ] 13.2 Create ExplorationStrategy
    - Create strategies/exploration.py
    - Migrate logic from workflows/exploration.py
    - Copy-paste existing logic, only update imports
    - _Requirements: 10.5_
  
  - [ ]* 13.3 Write property test for workflow compatibility
    - **Property 16: Workflow Compatibility**
    - **Validates: Requirements 10.5**
  
  - [ ]* 13.4 Write unit tests for strategies
    - Test IntentStrategy executes intent-based workflow
    - Test ExplorationStrategy executes exploration workflow
    - _Requirements: 10.5_

- [ ] 14. Migrate processing module
  - [ ] 14.1 Move annotator module
    - Copy tools/vision/processing/annotator.py to processing/annotator.py
    - Update imports only, preserve all logic
    - _Requirements: 7.2, 7.7_
  
  - [ ] 14.2 Move drawer module
    - Copy tools/vision/processing/drawer.py to processing/drawer.py
    - Update imports only, preserve all logic
    - _Requirements: 7.3, 7.7_
  
  - [ ] 14.3 Move geometry module
    - Copy tools/vision/processing/geometry.py to processing/geometry.py
    - Update imports only, preserve all logic
    - _Requirements: 7.4, 7.7_
  
  - [ ] 14.4 Move parsers directory
    - Copy tools/vision/processing/parsers/ to processing/parsers/
    - Update imports only, preserve all logic
    - _Requirements: 7.5, 7.7_
  
  - [ ] 14.5 Update imports in dependent code
    - Find all imports from tools/vision/processing
    - Update to import from processing/
    - _Requirements: 7.6_
  
  - [ ]* 14.6 Write property test for processing module preservation
    - **Property 8: Processing Module Import Updates**
    - **Validates: Requirements 7.6, 7.7**
  
  - [ ]* 14.7 Write unit tests for processing modules
    - Test annotator functions work identically
    - Test drawer functions work identically
    - Test geometry functions work identically
    - Test parsers work identically
    - _Requirements: 7.7_

- [ ] 15. Create backward compatibility shims
  - [ ] 15.1 Create re-export shims in orchestration/
    - Add imports from new runtime/ and core/
    - Re-export with original names
    - _Requirements: 5.1, 5.5, 9.4_
  
  - [ ] 15.2 Create re-export shims in tools/
    - Add imports from new adapters/device/
    - Re-export with original names
    - _Requirements: 5.2, 5.5, 9.4_
  
  - [ ] 15.3 Create re-export shims in workflows/
    - Add imports from new strategies/
    - Re-export with original names
    - _Requirements: 5.4, 5.5, 9.4_
  
  - [ ] 15.4 Create re-export shims in tools/vision/processing/
    - Add imports from new processing/
    - Re-export with original names
    - _Requirements: 5.2, 5.5, 9.4_
  
  - [ ]* 15.5 Write property test for legacy code backward compatibility
    - **Property 6: Legacy Code Backward Compatibility**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 9.4**

- [ ] 16. Migrate proprietary code
  - [ ] 16.1 Verify prompts/ location
    - Check if prompts/ needs to be moved
    - If moving, copy-paste code with only import updates
    - _Requirements: 6.1, 6.4, 6.5, 6.6_
  
  - [ ] 16.2 Verify tools/definitions.py location
    - Check if tools/definitions.py needs to be moved
    - If moving, copy-paste code with only import updates
    - _Requirements: 6.2, 6.5, 6.6_
  
  - [ ] 16.3 Verify services/parsing.py location
    - Check if services/parsing.py needs to be moved
    - If moving, copy-paste code with only import updates
    - _Requirements: 6.3, 6.5, 6.6_
  
  - [ ]* 16.4 Write property test for proprietary code preservation
    - **Property 7: Proprietary Code Preservation**
    - **Validates: Requirements 6.1, 6.4, 6.5, 6.6**

- [ ] 17. Enforce import restrictions
  - [ ] 17.1 Add import linting rules
    - Configure ruff or mypy to enforce import restrictions
    - Add rules for each layer (core, interfaces, strategies, adapters, processing)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.10_
  
  - [ ]* 17.2 Write property tests for import restrictions
    - **Property 9: Core Layer Import Restrictions**
    - **Property 10: Interfaces Layer Import Restrictions**
    - **Property 11: Strategies Layer Import Restrictions**
    - **Property 12: Adapters Layer Import Restrictions**
    - **Property 13: Processing Layer Import Restrictions**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.10**

- [ ] 18. Verify schema preservation
  - [ ] 18.1 Verify schemas/ directory unchanged
    - Check all Pydantic models in schemas/ are unchanged
    - Verify schemas/ is importable by all layers
    - _Requirements: 12.1, 12.2, 12.3, 12.4_
  
  - [ ]* 18.2 Write property test for schema preservation
    - **Property 17: Schema Preservation**
    - **Validates: Requirements 12.2, 12.3, 12.4**

- [ ] 19. Run full test suite
  - [ ] 19.1 Run existing test suite
    - Execute all existing tests
    - Verify all tests pass with new architecture
    - _Requirements: 5.6_
  
  - [ ] 19.2 Run all property tests
    - Execute all property-based tests (minimum 100 iterations each)
    - Verify all properties hold
    - _Requirements: All property requirements_

- [ ] 20. Create minimal working example
  - [ ] 20.1 Write example script
    - Create example showing minimal configuration (device + llm)
    - Create example showing full configuration (all seven ports)
    - Demonstrate order-independent builder API
    - _Requirements: 4.1, 4.2, 4.4, 11.1, 11.2_
  
  - [ ]* 20.2 Test examples execute successfully
    - Run minimal example
    - Run full example
    - Verify both produce expected results
    - _Requirements: 11.2_

- [ ] 21. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties with minimum 100 iterations
- Unit tests validate specific examples and edge cases
- **CRITICAL**: Copy-paste existing logic without modifications when migrating code
- Only update imports and structure, never modify business logic
- Use hypothesis library for property-based testing in Python
- Tag each property test with: `# Feature: fathom-hexagonal-rearch, Property {number}: {property_text}`
