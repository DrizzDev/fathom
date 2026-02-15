# HITL Reality Check - What Actually Works

## The Truth

After multiple attempts, here's what actually works reliably:

### ✅ What Works: Automatic Pause

**Agent automatically pauses when uncertain (confidence < 50%)**

```
❓ Agent Question
┌────────────────────────────────────────────────────────────┐
│ The agent is uncertain (confidence: 45%) about what to do  │
│ next.                                                       │
│                                                             │
│ Current screen: com.example.app.LoginActivity              │
│ Intent: Login to the app                                   │
│ Suggested action: Tap on Element at [500, 800]            │
│                                                             │
│ What should the agent do?                                  │
└────────────────────────────────────────────────────────────┘
Your answer: _
```

**This works perfectly:**
- Agent detects low confidence
- Pauses automatically
- Asks for help
- User types answer
- Agent uses answer to make better decision

### ❌ What Doesn't Work: Manual Pause

**Manual pause (pressing a key to pause) doesn't work reliably due to:**

1. **Terminal I/O conflicts** - Rich library manipulates terminal
2. **stdin conflicts** - Can't read key press and use Prompt.ask() simultaneously
3. **Timing issues** - Can only check between steps, not during LLM calls (3-10 seconds)
4. **Platform limitations** - termios is Unix-only, doesn't work on Windows

**Attempted solutions that failed:**
- Keyboard listener with termios → Terminal conflicts
- File-based control → Too complex, not user-friendly
- Background thread with stdin → stdin conflicts with Rich

## What You Can Do

### Option 1: Use Automatic Pause (Recommended)

Just use `--interactive` and let the agent ask questions when uncertain:

```bash
fathom run "Your intent" --interactive -s emulator-5554
```

**When agent is uncertain, it will pause and ask for help automatically.**

### Option 2: Lower Confidence Threshold

Make the agent ask questions more frequently by lowering the confidence threshold:

```python
# In src/fathom/strategies/intent.py
# Change from:
if plan.step and plan.step.action.confidence < 0.5:

# To:
if plan.step and plan.step.action.confidence < 0.7:  # Ask more often
```

### Option 3: API-Based Control (Future)

For production workflows, implement API-based pause/resume:

```python
# Check API endpoint instead of key press
if await self.__check_api_pause_request():
    await self.__handle_pause()
```

This avoids all terminal I/O issues.

## Current Implementation

### What's Implemented

1. **Automatic Pause** ✅
   - Agent pauses when confidence < 50%
   - User provides guidance
   - Agent re-analyzes with guidance

2. **Context Injection** ✅
   - User can inject context when paused
   - Context is added to LLM prompt
   - Agent makes better decisions

3. **Interactive Menu** ✅
   - Resume execution
   - Inject additional context
   - Cancel execution

### What's NOT Implemented

1. **Manual Pause** ❌
   - Can't press a key to pause
   - Too many technical limitations
   - Not reliable enough for production

## Recommendation

**Use automatic pause (confidence-based) for now.**

It works reliably and provides the core HITL functionality:
- Agent asks for help when uncertain
- User provides guidance
- Agent uses guidance to make better decisions

For manual control in production, implement API-based pause/resume that doesn't rely on terminal I/O.

## Keys You Actually Use

### When Agent Asks Question:
- **Type your answer** and press **Enter**

### When Paused (Menu Appears):
- **Press '1'** and **Enter** - Resume
- **Press '2'** and **Enter** - Inject context (then type context and press Enter)
- **Press '3'** and **Enter** - Cancel

### Anytime:
- **Press Ctrl+C** - Cancel execution immediately

## Summary

✅ Automatic pause works perfectly
❌ Manual pause doesn't work reliably
✅ Context injection works
✅ Interactive menu works
✅ Agent uses guidance effectively

**Bottom line:** Use `--interactive` and let the agent ask questions when uncertain. That's what works.
