# GCC (Git-Context-Controller) Implementation

## Overview

GCC provides hierarchical context management for long-running agent workflows (100-150+ steps). It balances semantic compression with context continuity through a three-tier architecture.

## Architecture

### Tier 1: Milestones (Semantic Summaries)
- **Purpose**: High-level summaries of completed work segments
- **Created**: Every 15 steps via background LLM summarization
- **Example**: "Successfully configured a custom schedule by selecting Monday, Tuesday, and Friday"
- **Benefit**: Provides semantic understanding without token overhead

### Tier 2: Shadow Buffer (Recent Context Window)
- **Purpose**: Maintains detailed recent history during summarization
- **Size**: Last 3 items (configurable via `context_window` parameter)
- **Lifecycle**:
  - Populated by `prepare_summarization()` when branching occurs
  - Trimmed by `commit()` after milestone creation
  - Merged with active log in `get_context()`
- **Benefit**: Ensures no context gaps during async summarization

### Tier 3: Active Log (Current Actions)
- **Purpose**: Uncommitted actions in current segment
- **Cleared**: When branching occurs (moved to shadow buffer)
- **Benefit**: Fresh, detailed trace of recent steps

## Context Flow

```
Step 1-15: Actions accumulate in active_log
           ↓
Step 15:   Branching triggered (trace >= 15)
           ↓
           prepare_summarization():
           - Move 15 items from active_log → shadow_buffer
           - Clear active_log
           - Return segment for summarization
           ↓
           Background Task:
           - LLM summarizes 15 steps
           - Creates milestone: "Completed X, Y, Z"
           ↓
           commit():
           - Create milestone node
           - Keep last 3 items in shadow_buffer
           - Clear the rest
           ↓
Step 16:   LLM sees:
           - Milestones: ["Completed X, Y, Z"]
           - Shadow Buffer: [step 13, 14, 15]
           - Active Log: [step 16]
           ↓
Step 17-30: Process continues...
```

## Configuration

### Branching Threshold
```python
BRANCHING_THRESHOLD = 15  # Steps before creating milestone
```

**Rationale**:
- For 100-150 step workflows → 7-10 milestones
- Balances compression benefits with context freshness
- Too low (5): Frequent branching, context churn
- Too high (50): Loses compression benefits

### Context Window
```python
context_window = 3  # Items to keep in shadow_buffer
```

**Rationale**:
- 3 items provides ~3 steps of recent detailed history
- Prevents context loss during milestone creation
- Ensures continuity between segments

## Benefits Over Simple History

### Scalability
- **Simple**: O(n) token growth, fails at 100+ steps
- **GCC**: O(log n) token growth via semantic compression

### Context Quality
- **Simple**: Only last 8 raw actions
- **GCC**: Semantic milestones + recent detailed trace

### Cache Efficiency
- **Simple**: History changes every step (poor cache hits)
- **GCC**: Milestones are stable (better cache hits)

### Long-Term Memory
- **Simple**: Loses context beyond last 8 steps
- **GCC**: Maintains semantic understanding of entire session

## Example: 100-Step Workflow

### Without GCC (Simple History)
```
LLM sees: [step 93, 94, 95, 96, 97, 98, 99, 100]
- No context about steps 1-92
- Can't understand overall progress
- May repeat earlier actions
```

### With GCC
```
LLM sees:
Milestones:
- Navigated to schedule screen
- Configured custom schedule (Mon, Tue, Fri)
- Selected date range (6 days from now, 15 days duration)
- Set duration to 90 minutes
- Selected evening time slot
- Configured UPI payment

Shadow Buffer:
- Step 98: Screen: payment -> TAP:UPI option
- Step 99: Screen: payment -> TYPE:8105944810@ybl
- Step 100: Screen: payment -> TAP:Confirm button

Active Log:
- Step 101: [current step]
```

## Debugging

### Logging
All GCC operations are logged with `[GCC]` prefix:
```
[GCC] prepare_summarization(): branch.log_length=15, shadow_buffer_length_before=3
[GCC] commit() called: shadow_buffer_length_before=18
[GCC] commit(): kept last 3 items, shadow_buffer_length_after=3
[GCC] get_context(): shadow_buffer_length=3, branch.log_length=1, total_trace_length=4
```

### Common Issues

**Issue**: Agent repeats actions
- **Cause**: Context window too small or branching too frequent
- **Fix**: Increase `context_window` or `BRANCHING_THRESHOLD`

**Issue**: Token limit exceeded
- **Cause**: Branching threshold too high
- **Fix**: Decrease `BRANCHING_THRESHOLD`

**Issue**: Agent loses context
- **Cause**: Summarization losing critical details
- **Fix**: Improve summarization prompt or increase context window

## Future Enhancements

1. **Adaptive Branching**: Adjust threshold based on task complexity
2. **Selective Summarization**: Keep critical steps in full detail
3. **Multi-Level Milestones**: Hierarchical summaries (high/medium/low level)
4. **Context Pruning**: Remove redundant information from shadow buffer
5. **Semantic Deduplication**: Merge similar actions in trace

## References

- [GCC Paper](https://arxiv.org/pdf/2508.00031v1) - Original research from Google
- LangGraph State Management - Inspiration for shadow buffer pattern
