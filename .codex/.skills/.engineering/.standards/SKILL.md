# Engineering & Architecture Standard

Use this skill for any coding, architecture, refactor, implementation-plan, PRD, design-review, or code-audit task in this workspace unless the user explicitly says otherwise.

## Non-negotiable goals

All systems must be:

- production-grade
- efficient and performant
- modular and extensible
- vendor-neutral
- open-source friendly
- architecturally disciplined
- safe for long-term evolution

Readability, maintainability, and performance must coexist. Do not trade one away casually.

## 1. Architecture

Use a strict layered architecture:

- Domain: business logic, rules, entities, policies, invariants
- Application: use-cases, orchestration, workflows
- Infrastructure: databases, networks, messaging, file systems, external SDKs
- Adapters: translate external systems into internal interfaces

Rules:

- Domain must not depend on any other layer.
- Application depends only on Domain.
- Infrastructure and Adapters depend inward only.
- All cross-layer communication must use explicit interfaces or protocols.
- Business decisions belong only in Domain.
- Application coordinates execution but does not make business decisions.
- No layer may bypass another.
- Abstractions must not leak across boundaries.

## 2. Core principles

- production-grade reliability
- efficient use of time and memory
- clear modular architecture
- strict SOLID adherence
- vendor neutrality
- deterministic behavior where possible
- architecture/design before library selection
- performance, readability, and maintainability are all mandatory

## 3. Naming

- no unnecessary abbreviations
- prefer precise, domain-driven names
- names must be self-explanatory

Formats:

- Class: `PascalCase`
- Function / Method: `snake_case`
- Private Method: `__snake_case`
- Protected / Internal: `_snake_case`
- Constant: `UPPER_SNAKE_CASE`
- Module: `snake_case`

## 4. Imports

- imports at the top of the file
- no circular imports
- order imports as:
  - standard library
  - third-party
  - internal modules

## 5. Typing and documentation

- strong type hints are mandatory on function signatures, method signatures, and class attributes
- every class needs one concise docstring stating purpose
- every function/method needs one concise docstring stating what it does
- doc-strings describe intent and contract, not implementation detail
- avoid redundant inline comments
- `Any` is prohibited except at the outermost boundary where type is genuinely unknowable

## 6. Classes and encapsulation

- default to classes for behavior
- module-level functions are only for pure stateless transformations
- classes must have one clearly articulable responsibility
- expose the minimum public surface
- use `__` for truly private implementation details
- use `_` for protected/internal members
- prefer composition over inheritance
- never use class attributes as a substitute for injected dependencies

## 7. Data models

- use Pydantic `BaseModel` for anything crossing a boundary
- never use raw `dict` or `tuple` instead of a typed model
- prefer immutable domain entities/value objects where possible
- validate at the boundary; Domain may assume valid data

## 8. Keyword arguments

- all internal calls must use keyword arguments
- positional arguments are allowed only for standard library, third-party APIs, or strong idiomatic conventions

## 9. Exceptions

- catch the most specific exception possible
- name exception variables descriptively, e.g. `exception`
- prefer domain-specific exception types
- never swallow errors silently
- re-raise or wrap with context at boundaries
- error messages must be actionable

## 10. Domain purity and state

Domain must never contain:

- HTTP/network calls
- database access
- file system operations
- environment reads
- logging or telemetry
- global state or singletons
- framework-specific imports

Rules:

- Domain is stateless by default
- inject all configuration
- pass all state explicitly
- Domain must be testable with no infrastructure

## 11. Extensibility and composition

- components must be replaceable without changing calling code
- no hardcoded providers in core logic
- all integrations must be swappable through interfaces
- prefer composition over inheritance
- preferred patterns: Strategy, Adapter, Factory, Repository, Port/Adapter

## 12. API design

- avoid breaking changes
- maintain backward compatibility where possible
- all inputs/outputs must use explicit typed schemas
- APIs must fail clearly and deterministically
- internal types must not leak through public surfaces

## 13. Configuration

- never hardcode configuration
- inject configuration through constructors or initialization
- Domain remains configuration-agnostic
- validate configuration schemas at the boundary

## 14. Dependency policy

- use open-source libraries with permissive licenses
- avoid hidden paid-tier lock-in
- avoid vendor lock-in in core/application layers
- isolate all external systems behind interfaces/adapters
- never import a vendor SDK directly into Domain or Application

## 15. Reuse policy

- inspect existing code before writing new code
- reuse when correct and well-structured
- refactor when logic is recoverable but structure is wrong
- rewrite only when fundamentally flawed
- never duplicate as a shortcut

## 16. Deterministic behavior

- avoid hidden side effects
- avoid implicit/global state
- same input should produce same output where practical
- inject time, randomness, environment, flags, locale, and other non-deterministic inputs

## 17. Boundary validation

- validate and normalize all external input before Application or Domain logic
- fail early with explicit structured errors
- never trust external data
- Domain should only see validated typed data
- validation belongs in Adapter or Infrastructure, not Domain

## 18. Idempotency

- retryable/re-playable operations must be idempotent
- prevent duplicate or corrupted state on re-entry
- use idempotency keys or natural deduplication where needed
- document which operations are idempotent

## 19. Fail fast

- detect invalid state and invariant violations immediately
- raise explicit structured exceptions with diagnostic context
- never silently recover from unexpected conditions
- prefer hard failures in development over hidden corruption

## 20. Concurrency

- avoid shared mutable state across concurrent paths
- prefer message passing and event-driven coordination
- protect critical sections explicitly
- ensure thread safety of shared resources
- avoid unnecessary locking
- concurrency boundaries must be explicit and documented

## 21. Resource safety

- every acquired resource must have a guaranteed release path
- use context managers / structured lifecycle management
- avoid unbounded growth in queues, caches, pools, and in-memory collections
- resource exhaustion must fail explicitly

## 22. Performance awareness

- identify hot paths during design
- avoid unnecessary allocations and copies on hot paths
- avoid repeated expensive operations; cache/batch deliberately
- choose the right data structure
- avoid avoidable `O(n^2)` paths
- performance must be measurable in production
- do not optimize blindly; measure first

## 23. Observability

Minimum expectations:

- structured machine-parseable logging at significant boundaries
- actionable error messages with context
- correlation identifiers
- metrics for throughput, error rate, and latency
- tracing at latency/failure/retry boundaries

Rules:

- observability is part of design, not an afterthought
- logging belongs only in outer layers
- silent failures are unacceptable
- logs must be actionable without requiring source-code access

## 24. Isolation of external systems

Dependency direction must be:

`External System -> Adapter -> Interface -> Application / Domain`

Rules:

- Domain and Application depend only on interfaces
- all external interactions go through adapters
- every external dependency must be replaceable, mockable, and testable

## 25. Backward compatibility

- public interfaces stay stable once exposed
- prefer additive change
- plan and phase breaking changes explicitly
- lifecycle: extend -> deprecate -> remove
- define versioning before first external consumer

## 26. Migration safety

- schema changes must be backward compatible first
- data migrations should be reversible where feasible
- deployments must not break in-flight work
- prefer phased migration over hard cut over
- every migration needs a rollback path

## 27. System evolution

- avoid locking the future architecture to today's implementation
- prefer composition
- keep boundaries clear and enforced
- keep coupling low
- every module/boundary must have a one-sentence role

## 28. Security awareness

- validate all external input
- do not leak internal details to untrusted consumers
- defensively validate external responses
- never log sensitive data
- authn/authz must be enforced at the boundary

## 29. Testing philosophy

Levels:

- unit tests for Domain logic
- integration tests for infrastructure/adapter seams
- end-to-end tests for workflows

Rules:

- Domain tests must be infrastructure-free
- tests must be deterministic
- test observable behavior, not internals
- critical path coverage is non-negotiable

## 30. Controlled complexity

- avoid deep inheritance
- avoid monolithic classes
- break logic into small testable components
- refactor to reduce complexity when needed

## 31. Ownership and responsibility

- every component must have a narrow one-sentence role
- split modules that do unrelated work
- respect ownership boundaries
- do not reach into internals across boundaries

## 32. Style rules

Prohibited:

- emojis in code/comments/doc-strings
- AI-style filler comments
- TODO without a ticket/issue reference
- logging or telemetry inside Domain
- global mutable state
- magic numbers or strings without named constants

## How to apply this skill

When implementing or reviewing:

- first evaluate whether the current code respects the layer boundaries
- identify any violations of typing, boundary validation, determinism, or dependency policy
- prefer corrective refactors before adding new logic on top of bad structure
- if a requested change conflicts with these standards, call out the conflict explicitly
- when producing plans or PRDs, map proposed modules and data flow back to these rules

## Fathom-specific reminder

For Fathom work, below are strictly compulsory to follow

- Proper layered and nested structure, be it folder, files, or classes, entities, it's fields, and always use single word names for folder, files, etc...
- Proper usages of design pattern, considering all basic coding concepts like OOPS, SOLID, SRP, OCP, LSP, Hexagonal Architecture
- Using Pydantic for entities with Fields, 1 line descriptions, and defining it at correct files/location (schemas)
- No hardcoded constants/values, define things inside constants (Use IntEnum, StrEnum, etc...)
- Don't use any unit(ms, s, etc..) as prefix/suffix for any variables/fields, use description to convey what it means
- Use elegant, beautiful, sort, generic names for functions, variables, fields (Eg: For Pydantic/Fields, I prefer nested fields, Eg: ocr.enabled and not ocr_enabled, etc...)
- Always use __ and not _ for private methods/variables, and use __methods to make code readable, making function non-bloated, etc...
- Doc string should be in format of """\n Crisp, to the point doc-string\n""" and """doc-string"""
- UTs should follow same file/folder structure, and each UT should be class, no function based UTs, and UT should cover actual real cases and not vague, random, fake cases
- I don't like standalone functions at all, everything should be class with proper functions (instance method, class/static methods depending on its nature)
- Nothing should violate HEXAGONAL ARCHITECTURE In any case, We've to maintain SRP, OOPS, Other basic coding standards at all cost, this is non-negotiable
```
