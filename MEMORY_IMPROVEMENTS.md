# Memory System Reliability Improvements

## Overview
This document outlines the pragmatic improvements made to the existing memory system to increase reliability without major architectural changes.

## Changes Made

### 1. Fixed Critical Bugs
- **Summarization Bug**: Fixed `FunctionCall.get()` error that was causing GCC summarization to fail
- **Shadow Buffer Size**: Increased from 3 to 7 items for better context continuity
- **Cross-Screen Memory**: Changed from visual_hash-based to ALL persistent memory retrieval

### 2. Added Memory Tool Instructions
**Location**: `src/fathom/core/prompts/templates.py`

Added comprehensive instructions to the LLM on when and how to use memory tools:
- `store_memory`: Proactively store progress after completing sub-goals
- `recall_memory`: Check what's already done before starting new actions
- `memory_updates` field in `execute_ui`: Track progress inline with actions

**Key Addition**:
```
MEMORY STRATEGY:
- Store progress IMMEDIATELY after completing each sub-goal
- Recall memory BEFORE starting new actions to check what's already done
- Use memory to maintain state across screen transitions
- Memory persists across the entire workflow - use it to avoid repeating work
```

### 3. Enhanced Logging & Visibility
**Locations**: 
- `src/fathom/infrastructure/memory/ledger.py`
- `src/fathom/strategies/graph/intent/nodes.py`
- `src/fathom/core/services/vision.py`
- `src/fathom/core/prompts/gemini.py`

Added detailed logging at every memory operation:
- `[LEDGER] SET` - When memory is stored
- `[LEDGER] GET` - When memory is retrieved
- `[LEDGER] GET_ALL` - When all memory is retrieved (shows keys)
- `[NODE: RECORD]` - When memory updates are stored from actions
- `[VISION]` - When memory is retrieved for LLM context
- `[H3]` - When memory ledger is added to prompt

**Benefits**:
- Can trace exactly what memory is stored and when
- Can verify memory is being retrieved and passed to LLM
- Can debug memory-related issues quickly

### 4. Added Validation & Error Handling
**Location**: `src/fathom/infrastructure/memory/ledger.py`

Added validation to prevent silent failures:
- Key must be non-empty string
- Value must be string type
- Exceptions are logged and re-raised
- Try-catch blocks around all DB operations

### 5. Added Health Check Method
**Location**: `src/fathom/infrastructure/memory/ledger.py`

New `health_check()` method returns:
- `healthy`: Boolean status
- `table_exists`: Verify schema is correct
- `entry_count`: Number of stored entries
- `oldest_entry`: Timestamp of oldest entry
- `newest_entry`: Timestamp of newest entry
- `database_path`: Location of SQLite file

**Usage**:
```python
health = await ledger.health_check()
if not health["healthy"]:
    logger.error(f"Memory system unhealthy: {health}")
```

### 6. Memory Flow Verification
Verified the complete flow works correctly:

1. **Storage**: `record_node` → `memory.set()` → `ledger.set()` → SQLite
2. **Retrieval**: `vision.analyze()` → `memory.get_all()` → `ledger.get_all()` → SQLite
3. **Context Building**: `prompt_builder.build_user_context(memory=all_memory)`
4. **Prompt Injection**: `payload.append(context)` → LLM sees memory

## How This Solves Memory Issues

### Issue 1: Agent Repeating Actions (Amnesia Loop)
**Root Cause**: LLM had no instructions to use memory tools
**Solution**: 
- Added explicit instructions to use `store_memory` after sub-goals
- Added instructions to use `recall_memory` before actions
- Memory is now injected into every prompt with clear label

### Issue 2: Agent Losing Context After GCC Branching
**Root Cause**: 
- Summarization was crashing (FunctionCall bug)
- Shadow buffer was too small (3 items)
**Solution**:
- Fixed summarization bug
- Increased shadow buffer to 7 items
- Memory persists independently of GCC state

### Issue 3: Memory Only Available on Same Screen
**Root Cause**: Memory retrieval was visual_hash-based
**Solution**: Changed to `get_all()` which retrieves ALL memory regardless of screen

### Issue 4: Silent Failures
**Root Cause**: No logging or validation
**Solution**: 
- Added comprehensive logging at every step
- Added validation to catch errors early
- Added health check for diagnostics

## Testing Recommendations

### 1. Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
```

### 2. Monitor Memory Operations
Look for these log patterns:
```
[LEDGER] SET | key=selected_days | value_length=11
[LEDGER] GET_ALL | total_entries=3 | keys=['selected_days', 'date_range_set', 'service_type']
[VISION] Memory Retrieved | persistent_memories=3 | persistent_keys=['selected_days', ...]
[H3] Memory Ledger Added | ledger_length=85
```

### 3. Verify Memory in Prompts
Check that `<MEMORY_LEDGER>` appears in LLM prompts with actual data:
```
<MEMORY_LEDGER>
Persistent memory (use store_memory/recall_memory tools):
[selected_days:Mon,Tue,Fri, date_range_set:true, service_type:bathroom_cleaning]
</MEMORY_LEDGER>
```

### 4. Run Health Check
Periodically check memory system health:
```python
health = await memory._Ledger__ledger.health_check()
print(health)
```

## Next Steps If Issues Persist

### Short Term
1. **Improve Prompt Engineering**: Add more specific examples of when to use memory
2. **Add Memory Validation**: Verify memory is actually being used by LLM
3. **Enhance GCC Milestones**: Make them more actionable with specific state info

### Medium Term
1. **Add Memory Queries**: `get_by_prefix()`, `get_by_pattern()` for better retrieval
2. **Add Memory Expiry**: Optional TTL for temporary state
3. **Add Memory Namespacing**: Separate workflow-specific vs global memory

### Long Term (Only If Needed)
1. **Add Event Types**: Simple `type` column to categorize entries
2. **Add Structured Logging**: Log each step with metadata
3. **Consider Event Sourcing**: Only if scaling becomes an issue

## Key Principles

1. **Simplicity First**: Use the simplest solution that solves the problem
2. **Measure Before Optimizing**: Add logging to understand actual behavior
3. **Iterate Based on Data**: Make changes based on observed issues, not speculation
4. **Avoid Over-Engineering**: Don't add complexity until it's proven necessary

## Summary

These improvements make the existing memory system more reliable through:
- **Better Instructions**: LLM knows when/how to use memory
- **Better Visibility**: Comprehensive logging at every step
- **Better Reliability**: Validation and error handling
- **Better Diagnostics**: Health checks and detailed logs

The system now has a solid foundation for reliable memory operations without requiring major architectural changes.
