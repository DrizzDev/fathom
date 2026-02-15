# Requirements Document

## Introduction

This document specifies requirements for incrementally re-architecting Fathom using hexagonal architecture (ports and adapters pattern) as defined in documents/architecture/v2/ARCHITECTURE.md. The migration must preserve all existing functionality while establishing a clean architectural foundation that separates business logic from infrastructure concerns.

## Glossary

- **Fathom**: The mobile device automation library being re-architected
- **Hexagonal_Architecture**: An architectural pattern that isolates core business logic from external concerns through ports and adapters
- **Port**: An abstract interface (ABC) defining a contract for external dependencies
- **Adapter**: A concrete implementation of a port that connects to specific technologies
- **Core**: The business logic layer containing domain models and orchestration
- **Runtime**: The composition root that wires together ports, adapters, and core components
- **Builder**: A fluent API for configuring Fathom instances with order-independent method chaining
- **Legacy_Code**: Existing working code in orchestration/, tools/, services/, prompts/ directories
- **Proprietary_Code**: Business-critical code in prompts/, tools/definitions.py, services/parsing.py that must be preserved
- **Processing_Module**: Code for UI element annotation, drawing, geometry, and parsing
- **Migration**: The incremental process of moving from current architecture to hexagonal architecture

## Requirements

### Requirement 1: Establish Hexagonal Architecture Foundation

**User Story:** As a developer, I want a clean hexagonal architecture structure, so that business logic is decoupled from infrastructure concerns.

#### Acceptance Criteria

1. THE System SHALL organize code into five top-level directories: interfaces/, adapters/, core/, runtime/, strategies/
2. THE System SHALL define all port contracts as abstract base classes in the interfaces/ directory
3. THE System SHALL implement concrete adapters in the adapters/ directory
4. THE System SHALL place business logic and domain models in the core/ directory
5. THE System SHALL place composition and wiring logic in the runtime/ directory
6. THE System SHALL place execution strategies in the strategies/ directory

### Requirement 2: Define Seven Core Ports

**User Story:** As a developer, I want well-defined port interfaces, so that I can swap implementations without changing business logic.

#### Acceptance Criteria

1. THE System SHALL define a DevicePort interface for mobile device actions and perception
2. THE System SHALL define an LLMPort interface for language model completions with text, vision, and tool support
3. THE System SHALL define a MemoryPort interface for session state and cross-run memory
4. THE System SHALL define a KnowledgePort interface for application knowledge graphs
5. THE System SHALL define a SignalPort interface for human-in-the-loop control signals
6. THE System SHALL define a StoragePort interface for artifact persistence
7. THE System SHALL define a TelemetryPort interface for telemetry and observability

### Requirement 3: Implement Default Adapters

**User Story:** As a developer, I want working default adapters, so that Fathom works out-of-the-box with minimal configuration.

#### Acceptance Criteria

1. THE System SHALL provide an ADBDevice adapter implementing DevicePort for Android Debug Bridge
2. THE System SHALL provide a GeminiLLM adapter implementing LLMPort for Google Gemini API
3. THE System SHALL provide a SQLiteMemory adapter implementing MemoryPort for local persistence
4. THE System SHALL provide a SQLiteKnowledge adapter implementing KnowledgePort using SQLite and rustworkx
5. THE System SHALL provide a NoopSignal adapter implementing SignalPort for autonomous operation
6. THE System SHALL provide a LocalStorage adapter implementing StoragePort for filesystem persistence
7. THE System SHALL provide a StructlogAdapter implementing TelemetryPort for telemetry and observability

### Requirement 4: Create Fluent Builder API

**User Story:** As a developer, I want a fluent builder API, so that I can configure Fathom instances with readable, order-independent code.

#### Acceptance Criteria

1. THE System SHALL provide a Fathom.builder() static method that returns a builder instance
2. THE System SHALL provide builder methods using bare noun names: device(), llm(), memory(), knowledge(), signal(), storage(), telemetry()
3. WHEN any builder method is called, THE System SHALL return self for method chaining
4. THE System SHALL allow builder methods to be called in any order
5. WHEN build() is called, THE System SHALL validate that required ports are configured
6. WHEN build() is called with missing required ports, THE System SHALL raise a descriptive error
7. THE System SHALL return a configured Fathom instance from build()

### Requirement 5: Preserve Legacy Code Functionality

**User Story:** As a developer, I want existing code to continue working during migration, so that we maintain system stability.

#### Acceptance Criteria

1. WHILE migration is in progress, THE System SHALL keep orchestration/ directory functional
2. WHILE migration is in progress, THE System SHALL keep tools/ directory functional
3. WHILE migration is in progress, THE System SHALL keep services/ directory functional
4. WHILE migration is in progress, THE System SHALL keep prompts/ directory functional
5. WHEN legacy code imports are updated, THE System SHALL maintain backward compatibility through re-exports
6. THE System SHALL ensure all existing tests continue to pass

### Requirement 6: Migrate Proprietary Code Without Logic Changes

**User Story:** As a developer, I want proprietary code preserved exactly, so that business-critical functionality remains unchanged.

#### Acceptance Criteria

1. THE System SHALL move prompts/ module to new location with only import path changes
2. THE System SHALL move tools/definitions.py to new location with only import path changes
3. THE System SHALL move services/parsing.py to new location with only import path changes
4. THE System SHALL preserve all logic in prompts/base.py, prompts/factory.py, prompts/gemini.py, prompts/preprocessor.py, prompts/templates.py
5. WHEN proprietary code is moved, THE System SHALL update only import statements
6. THE System SHALL maintain all existing function signatures in proprietary code

### Requirement 7: Migrate Processing Module

**User Story:** As a developer, I want the processing module properly organized, so that UI processing logic is cleanly separated.

#### Acceptance Criteria

1. THE System SHALL create a processing/ directory at the appropriate architecture level
2. THE System SHALL move tools/vision/processing/annotator.py to processing/annotator.py
3. THE System SHALL move tools/vision/processing/drawer.py to processing/drawer.py
4. THE System SHALL move tools/vision/processing/geometry.py to processing/geometry.py
5. THE System SHALL move tools/vision/processing/parsers/ to processing/parsers/
6. WHEN processing modules are moved, THE System SHALL update import paths in dependent code
7. THE System SHALL preserve all processing logic without modifications

### Requirement 8: Enforce Import Rules

**User Story:** As a developer, I want enforced import rules, so that architectural boundaries are maintained.

#### Acceptance Criteria

1. THE System SHALL allow core/ to import from interfaces/ and schemas/ only
2. THE System SHALL prevent core/ from importing from adapters/ or runtime/
3. THE System SHALL allow interfaces/ to import from schemas/ only
4. THE System SHALL prevent interfaces/ from importing from core/, adapters/, or runtime/
5. THE System SHALL allow strategies/ to import from core/, interfaces/, and schemas/
6. THE System SHALL prevent strategies/ from importing from adapters/ or runtime/
7. THE System SHALL allow adapters/ to import from interfaces/, schemas/, and external libraries only
8. THE System SHALL prevent adapters/ from importing from core/ or runtime/
9. THE System SHALL allow runtime/ to import from all modules
10. THE System SHALL allow processing/ to import from schemas/ only

### Requirement 9: Implement Incremental Migration Strategy

**User Story:** As a developer, I want a safe incremental migration path, so that we can migrate without breaking production.

#### Acceptance Criteria

1. THE System SHALL create new architecture directories alongside existing code
2. THE System SHALL implement new ports and adapters before migrating core logic
3. WHEN new components are ready, THE System SHALL migrate one module at a time
4. THE System SHALL maintain backward compatibility shims during migration
5. WHEN a module is migrated, THE System SHALL update its dependents incrementally
6. THE System SHALL run full test suite after each migration step
7. THE System SHALL allow rollback to previous state if tests fail

### Requirement 10: Maintain Execution Engine Compatibility

**User Story:** As a developer, I want the execution engine to work with new architecture, so that automation workflows continue functioning.

#### Acceptance Criteria

1. THE System SHALL preserve the execution loop structure during migration
2. WHEN execution phases are refactored, THE System SHALL maintain the same phase sequence
3. THE System SHALL ensure SignalCheck, Perceive, Reason, Act, Learn, Checkpoint, Evaluate phases continue working
4. THE System SHALL preserve HITL branching logic for PAUSE, RESUME, INJECT, ASK operations
5. WHEN execution engine is migrated, THE System SHALL maintain compatibility with existing workflows

### Requirement 11: Support Minimal and Full Configuration

**User Story:** As a developer, I want flexible configuration options, so that I can use Fathom with minimal or full setup.

#### Acceptance Criteria

1. WHEN only device() and llm() are configured, THE System SHALL use default adapters for other ports
2. WHEN build() is called with minimal configuration, THE System SHALL create a working Fathom instance
3. THE System SHALL allow explicit configuration of all seven ports
4. WHEN a port is not explicitly configured, THE System SHALL use the default adapter for that port
5. THE System SHALL validate that device() and llm() are always provided

### Requirement 12: Preserve Existing Schemas

**User Story:** As a developer, I want existing schemas preserved, so that data structures remain consistent.

#### Acceptance Criteria

1. THE System SHALL keep the schemas/ directory in its current location
2. THE System SHALL preserve all Pydantic models in schemas/
3. WHEN new architecture is implemented, THE System SHALL continue using existing schemas
4. THE System SHALL allow schemas/ to be imported by all architecture layers

### Requirement 13: Document Migration Path

**User Story:** As a developer, I want clear migration documentation, so that I understand how to complete the transition.

#### Acceptance Criteria

1. THE System SHALL provide a migration guide documenting the incremental approach
2. THE System SHALL document which modules to migrate in which order
3. THE System SHALL document how to update imports when modules are migrated
4. THE System SHALL document how to test each migration step
5. THE System SHALL document rollback procedures for failed migrations
