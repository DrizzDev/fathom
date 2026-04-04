# PRD: Sub-Goal Replanning

## Problem

Fathom's agent decomposes a high-level intent into sequential sub-goals at the start of a session. When a sub-goal becomes unachievable — due to app state changes, incorrect decomposition, or the agent getting stuck — the system has no mechanism to recover. The agent loops on the same sub-goal until it hits `max_steps`, wasting time and API calls.

## Current Behavior

1. Intent is decomposed into sub-goals **once** at session start
2. Sub-goals execute sequentially: GROUND → ANALYZE → EXECUTE → RECORD
3. Sub-goal advancement requires a 2-signal gate: `llm_signaled` + `effective_action`
4. If the gate never passes, the agent loops indefinitely on the same sub-goal
5. If the VERIFY node rejects a sub-goal 3+ times, there's no fallback
6. The original decomposition is immutable — no adaptation to runtime conditions

## Solution

Add a replanning system that re-decomposes the **remaining** sub-goals when the agent is stuck, using the current screenshot for visual context.

## Triggers

| Trigger | Where | Condition |
|---------|-------|-----------|
| **3 verification failures** | VERIFY node | Sub-goal signals pass the 2-signal gate → VERIFY rejects 3 times → replan |
| **15 actions without advancement** | RECORD node | The LLM never sets `sub_goal_completed=true` for 15 consecutive actions → replan |

Both counters (`sub_goal_verify_failures`, `sub_goal_action_count`) are persisted in the `AgentState` checkpoint so they survive graph iteration boundaries.

## Replanning Flow

```
Trigger fires (VERIFY 3rd rejection OR RECORD 15-action stuck)
  │
  ├─ Collect remaining (unfinished) sub-goal descriptions
  │   → Join with ". " into a single intent string
  │
  ├─ Capture current screenshot (from VERIFY capture or fresh perception call)
  │
  ├─ Call IntentDecomposer.decompose(
  │     intent=remaining_intent,
  │     screenshot=capture.image
  │   )
  │   → System instruction appends: "A screenshot of the current screen is attached.
  │     Plan sub-goals starting from this screen. Do NOT include steps to reach
  │     this screen — the agent is already here."
  │   → Decomposer returns fresh List[SubGoal]
  │
  ├─ AgentState.replace_remaining_sub_goals(new_sub_goals)
  │   → Preserves already-completed sub-goals
  │   → Replaces everything from current index onward
  │   → Resets action count, verify failure count
  │   → Re-indexes new sub-goals (completed_count + 0, 1, 2...)
  │   → Marks first new sub-goal as IN_PROGRESS
  │
  ├─ Reset completion flag, clear stale guidance
  │
  └─ Return IS_COMPLETE=False, SHOULD_RETRY=True
       → Routes to GROUND → agent continues with fresh sub-goals
```

## Components Modified

### `AgentState` (`core/agent/state.py`)

**New fields:**
- `sub_goal_verify_failures: int` — verification rejection count per sub-goal
- `sub_goal_action_count: int` — actions executed on current sub-goal (existed, now exposed + persisted)

**New methods:**
- `replace_remaining_sub_goals(new_sub_goals)` — replaces unfinished sub-goals, preserves completed
- `record_verify_failure()` — increment verify failure counter
- `sub_goal_verify_failures` (property) — read counter
- `sub_goal_action_count` (property) — read counter

**Checkpoint persistence:**
- Both counters added to `to_checkpoint()`, `from_checkpoint()`, `__restore_from_data()`
- Counters reset to 0 on sub-goal advancement and on replanning

### `IntentDecomposer` (`core/services/decomposer.py`)

**Changed signature:**
```python
async def decompose(
    self,
    intent: str,
    *,
    screenshot: Optional[bytes] = None,  # NEW
) -> List[SubGoal]:
```

- When `screenshot` is provided, appends it to the prompt parts and adds screen context to the system instruction
- Initial decomposition (session start) passes no screenshot — unchanged
- Replanning passes the current screenshot for visual grounding

### `IntentNodeProvider` (`strategies/graph/intent/nodes.py`)

**New method:**
- `__replan_remaining_sub_goals(capture)` — orchestrates the replan flow

**New constant:**
- `__MAX_ACTIONS_PER_SUBGOAL = 15`

**Modified methods:**
- `__evaluate_subgoal_completion()` — now async, adds stuck detection (15 actions) in both the no-analysis and signal-gate-failure paths
- `verify()` — adds 3rd-failure replanning trigger after `record_verify_failure()`

### Routing (unchanged)

- RECORD → VERIFY: when `IS_COMPLETE=True` (sub-goal signals passed)
- VERIFY → GROUND: when verification rejects or replanning succeeds
- VERIFY → END: when verification passes (first completion signal exits)
- No new edges or nodes in the graph

## Constraints

1. **No extra LLM calls for screen description** — the screenshot is passed directly to the decomposer
2. **Completed sub-goals are preserved** — replanning only replaces unfinished sub-goals
3. **Counters persist across checkpoints** — replanning triggers are not reset by graph iteration boundaries
4. **Replanning is idempotent** — if it fails (exception), returns `None` and falls through to standard rejection feedback
5. **Single constant for stuck threshold** — `__MAX_ACTIONS_PER_SUBGOAL = 15` used in both detection paths

## Success Criteria

1. Agent recovers from stuck sub-goals within 15 actions instead of running to `max_steps`
2. Replanning produces sub-goals grounded in the current screen state
3. No regression on sub-goals that complete normally (2-signal gate + VERIFY flow unchanged)
4. Completed work is never lost — already-done sub-goals survive replanning
5. All pre-commit checks pass (ruff, mypy, bandit)
6. Existing tests pass

## Out of Scope

- Per-sub-goal `done_when` criteria (discussed but not implemented)
- `describe_screen` tool (removed — screenshot passed directly)
- Multi-provider support (single decomposer, single LLM)
- Configurable stuck threshold (hardcoded at 15)
