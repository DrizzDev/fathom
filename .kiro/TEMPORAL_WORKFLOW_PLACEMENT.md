# Temporal Workflow Placement Decision

## 🤔 The Question

Where should `FathomWorkflow` live?

**Option A**: In Fathom project (`src/fathom/runtime/temporal/`)
**Option B**: In Genymotion project (host service)

---

## 📊 Analysis

### Option A: In Fathom Project ✅ RECOMMENDED

**Location**: `src/fathom/runtime/temporal/workflow.py`

#### Pros
1. **Reusability**: Any service can use Fathom with Temporal (not just Genymotion)
2. **Versioning**: Workflow evolves with Fathom's capabilities
3. **Testing**: Can test workflow with Fathom's test suite
4. **Documentation**: Workflow docs live with Fathom docs
5. **Maintenance**: Single source of truth for Fathom execution
6. **Distribution**: Ships with Fathom package
7. **Consistency**: All Fathom integrations (CLI, Temporal, gRPC) in one place

#### Cons
1. **Temporal dependency**: Fathom would depend on `temporalio` package
   - **Solution**: Make it optional with extras: `pip install fathom[temporal]`
2. **Coupling**: Fathom becomes aware of Temporal
   - **Solution**: Keep it in `runtime/temporal/` as an optional integration

#### Structure
```
src/fathom/
├── runtime/
│   ├── runner.py           # Core runner (no Temporal)
│   ├── builder.py          # Core builder (no Temporal)
│   ├── config.py           # Core config (no Temporal)
│   └── temporal/           # OPTIONAL Temporal integration
│       ├── __init__.py
│       ├── workflow.py     # FathomWorkflow
│       └── activities.py   # Temporal activities
├── adapters/
│   └── signal/
│       ├── interactive.py  # Terminal HITL
│       ├── noop.py         # No HITL
│       └── temporal.py     # Temporal HITL
└── ...
```

#### Package Structure
```python
# setup.py or pyproject.toml
[project.optional-dependencies]
temporal = [
    "temporalio>=1.0.0",
]

# Users install with:
pip install fathom[temporal]
```

---

### Option B: In Genymotion Project

**Location**: `genymotion_project/workflows/fathom_workflow.py`

#### Pros
1. **No coupling**: Fathom stays pure, no Temporal dependency
2. **Flexibility**: Genymotion can customize workflow
3. **Independence**: Genymotion controls workflow versioning

#### Cons
1. **Duplication**: Every service needs to implement their own workflow
2. **Inconsistency**: Different services might implement differently
3. **Maintenance**: Bug fixes need to be applied in multiple places
4. **Testing**: Each service tests their own workflow
5. **Documentation**: Scattered across services
6. **Versioning**: Workflow version might not match Fathom version
7. **Expertise**: Each team needs Temporal + Fathom expertise

#### Structure
```
genymotion_project/
├── workflows/
│   └── fathom_workflow.py  # Custom FathomWorkflow
├── services/
│   └── crawler/
│       └── manager.py       # Uses custom workflow
└── ...
```

---

## 🎯 Recommendation: Option A (In Fathom)

### Reasoning

1. **Fathom is the execution engine** - The workflow is just a wrapper around Fathom's execution
2. **Temporal is an integration pattern** - Like CLI, gRPC, or HTTP - it's a way to invoke Fathom
3. **Reusability** - Other teams/services can use Fathom with Temporal without reimplementing
4. **Maintenance** - Single source of truth, easier to maintain and evolve
5. **Optional dependency** - Can be installed only when needed

### Implementation Strategy

#### 1. Make Temporal Optional
```python
# pyproject.toml
[project.optional-dependencies]
temporal = ["temporalio>=1.0.0"]
cli = ["click>=8.0.0", "rich>=13.0.0"]
all = ["temporalio>=1.0.0", "click>=8.0.0", "rich>=13.0.0"]
```

#### 2. Graceful Import Handling
```python
# src/fathom/runtime/temporal/__init__.py
try:
    from temporalio import workflow
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    workflow = None

if TEMPORAL_AVAILABLE:
    from .workflow import FathomWorkflow
    from .activities import execute_fathom_intent
    
    __all__ = ["FathomWorkflow", "execute_fathom_intent"]
else:
    __all__ = []
```

#### 3. Clear Documentation
```python
# src/fathom/runtime/temporal/workflow.py
"""
Temporal workflow integration for Fathom.

This module provides a Temporal workflow wrapper for Fathom execution,
enabling distributed, long-running, and HITL-capable mobile automation.

Installation:
    pip install fathom[temporal]

Usage:
    from fathom.runtime.temporal import FathomWorkflow
    
    # Register with Temporal worker
    worker = Worker(
        client,
        task_queue="fathom-tasks",
        workflows=[FathomWorkflow],
    )
"""
```

#### 4. Genymotion Integration
```python
# genymotion_project/services/crawler/manager.py
from fathom.runtime.temporal import FathomWorkflow  # Import from Fathom

class CrawlerManager:
    async def run_intent(self, identity: str, request: IntentRequest):
        # Use Fathom's workflow directly
        await client.start_workflow(
            FathomWorkflow.run,
            args=[agent_request.model_dump()],
            id=identity,
            task_queue=task_queue
        )
```

---

## 🏗️ Proposed Structure

### Fathom Project
```
src/fathom/
├── runtime/
│   ├── __init__.py
│   ├── runner.py           # Core (no Temporal dependency)
│   ├── builder.py          # Core (no Temporal dependency)
│   ├── config.py           # Core (no Temporal dependency)
│   └── temporal/           # Optional integration
│       ├── __init__.py     # Graceful import handling
│       ├── workflow.py     # FathomWorkflow
│       └── activities.py   # execute_fathom_intent
├── adapters/
│   └── signal/
│       ├── __init__.py
│       ├── interactive.py  # Terminal HITL
│       ├── noop.py         # No HITL
│       └── temporal.py     # Temporal HITL (requires temporalio)
└── ...
```

### Genymotion Project
```
genymotion_project/
├── services/
│   └── crawler/
│       └── manager.py      # Imports FathomWorkflow from Fathom
└── ...
```

---

## 📦 Installation Scenarios

### Scenario 1: CLI Only
```bash
pip install fathom[cli]
# Can use: fathom run "intent" --interactive
# Cannot use: Temporal workflows
```

### Scenario 2: Temporal Only
```bash
pip install fathom[temporal]
# Can use: FathomWorkflow in Temporal
# Cannot use: CLI commands
```

### Scenario 3: Everything
```bash
pip install fathom[all]
# Can use: CLI + Temporal + everything
```

### Scenario 4: Core Only
```bash
pip install fathom
# Can use: FathomRunner programmatically
# Cannot use: CLI or Temporal
```

---

## 🔄 Migration Path

### Phase 1: Add Temporal to Fathom
1. Create `src/fathom/runtime/temporal/`
2. Add `temporalio` as optional dependency
3. Implement `FathomWorkflow` and `TemporalSignalAdapter`
4. Add tests
5. Add documentation

### Phase 2: Update Genymotion
1. Add `fathom[temporal]` to requirements
2. Import `FathomWorkflow` from Fathom
3. Remove any custom workflow code
4. Test end-to-end

### Phase 3: Document
1. Add Temporal integration guide to Fathom docs
2. Add examples for other services
3. Add troubleshooting guide

---

## 🎓 Analogy

Think of it like this:

**Fathom** = Database engine (PostgreSQL)
**Temporal Integration** = Connection driver (psycopg2)
**Genymotion** = Application using the database

You wouldn't implement the PostgreSQL driver in your application - you'd use the official driver that ships with PostgreSQL. Same principle here.

---

## ✅ Decision: Option A

**FathomWorkflow should live in Fathom project** as an optional integration.

### Benefits
- ✅ Single source of truth
- ✅ Reusable across services
- ✅ Versioned with Fathom
- ✅ Tested with Fathom
- ✅ Documented with Fathom
- ✅ Optional dependency (no forced coupling)

### Implementation
```
src/fathom/runtime/temporal/
├── __init__.py          # Graceful imports
├── workflow.py          # FathomWorkflow
└── activities.py        # execute_fathom_intent

src/fathom/adapters/signal/
└── temporal.py          # TemporalSignalAdapter

pyproject.toml:
[project.optional-dependencies]
temporal = ["temporalio>=1.0.0"]
```

### Genymotion Usage
```python
# Just import and use
from fathom.runtime.temporal import FathomWorkflow

await client.start_workflow(FathomWorkflow.run, ...)
```

---

## 🚀 Next Steps

1. **Create the Temporal integration in Fathom**
2. **Make it optional with extras**
3. **Update Genymotion to use it**
4. **Document the integration**
5. **Test end-to-end**

**Shall I proceed with implementing Option A?** 🎯
