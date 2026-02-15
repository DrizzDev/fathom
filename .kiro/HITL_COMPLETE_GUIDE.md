# Human-in-the-Loop (HITL) - Complete Implementation Guide

## Overview

The HITL system is now **FULLY IMPLEMENTED** with production-grade code. It provides real-time interaction between the agent and human operator with TWO interaction modes:

1. **Automatic Pause**: Agent automatically pauses when uncertain (confidence < 50%)
2. **Manual Pause**: User can pause execution at ANY time using file-based control

You can:
- **Pause execution** at any time and provide additional context
- **Inject guidance** to help the agent make better decisions
- **Answer agent questions** when it's uncertain or stuck
- **Resume execution** with the injected context affecting LLM reasoning

---

## How HITL Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User (Human Operator)                     │
│  - Creates .fathom_pause to pause                           │
│  - Creates .fathom_context to inject context                │
│  - Creates .fathom_resume to resume                         │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                      FileWatcher                             │
│  - Watches for .fathom_pause file                           │
│  - Reads .fathom_context file                               │
│  - Detects .fathom_resume file                              │
│  - Runs in background thread                                 │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│              InteractiveSignal (Signal Adapter)              │
│  - Integrates FileWatcher                                    │
│  - Pause/Resume control                                      │
│  - Context injection                                         │
│  - Question/Answer handling                                  │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                    IntentStrategy                            │
│  - Checks signal every step                                  │
│  - Detects agent uncertainty (confidence < 0.5)             │
│  - Requests human input                                      │
│  - Re-analyzes with injected context                        │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                   ContextManager                             │
│  - Stores user guidance                                      │
│  - Injects into LLM context                                 │
│  - Persists to memory                                        │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                      LLM (Gemini)                            │
│  - Receives context with user guidance                       │
│  - Re-reasons with additional information                    │
│  - Makes better decisions                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Features Implemented

### 1. Automatic Pause (Agent Questions) ✅

**How it works**:
- Agent detects low confidence (< 0.5) in its decision
- Automatically pauses and asks user for help
- User provides guidance via interactive menu
- Agent re-analyzes with guidance and makes better decision

**Example**:
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
Your answer: Tap the blue "Sign in" button at the bottom
```

### 2. Manual Pause (File-Based) ✅

**How it works**:
- User creates `.fathom_pause` file to pause execution at ANY time
- User creates `.fathom_context` file to inject context
- User creates `.fathom_resume` file to resume execution
- FileWatcher detects files and triggers pause/resume

**Commands**:
```bash
# Pause execution
touch .fathom_pause

# Inject context (while paused)
echo "Your context here" > .fathom_context

# Resume execution
touch .fathom_resume
```

**Complete Example**:
```bash
# Terminal 1: Start execution
fathom run "Login to app" --interactive --serial emulator-5554

# Terminal 2: Pause and inject context
touch .fathom_pause
echo "Use test@example.com and password123" > .fathom_context
touch .fathom_resume
```

### 3. Context Injection ✅

**How it works**:
- User provides additional information during pause
- Context is stored in ContextManager
- Context is added to LLM prompt in next analysis
- LLM re-reasons with the new information

**Example Context**:
```
USER CONTEXT:
- The login button is at the bottom of the screen
- Use test@example.com as the email
- Skip the tutorial screens
```

### 4. Context Persistence ✅

**How it works**:
- All user guidance is stored in memory
- Guidance persists across steps
- Guidance is included in LLM context
- Agent uses guidance for better decisions

---

## Usage Examples

### Example 1: Basic Interactive Mode

```bash
# Enable interactive mode with -i flag
fathom run "Login to Gmail" --use-xml --serial emulator-5554 -i

# Output:
🤝 Interactive HITL Mode Enabled
• Agent will ask questions when uncertain (confidence < 50%)
• You can pause execution at ANY time using files

┌─ Manual Pause Instructions ─────────────────┐
│                                              │
│ To pause execution:                          │
│   touch .fathom_pause                        │
│                                              │
│ To inject context (while paused):            │
│   echo "Your context here" > .fathom_context │
│                                              │
│ To resume execution:                         │
│   touch .fathom_resume                       │
└──────────────────────────────────────────────┘

# Agent starts executing...
```

### Example 2: Manual Pause During Execution

```bash
# Terminal 1: Start execution
fathom run "Book a flight to NYC" --use-xml --serial emulator-5554 -i

# Terminal 2: Pause and inject context
touch .fathom_pause

# Agent output in Terminal 1:
⏸️  Execution Paused
Paused by file (.fathom_pause detected)
Waiting for resume signal...

# Terminal 2: Inject context
echo "Use departure date of March 15th" > .fathom_context

# Agent output in Terminal 1:
✓ Context injected from file: Use departure date of March 15th

# Terminal 2: Resume
touch .fathom_resume

# Agent output in Terminal 1:
▶️  Resuming execution...
```

### Example 3: Agent Asks Question (Automatic Pause)

```bash
fathom run "Complete checkout" --use-xml --serial emulator-5554 -i

# Agent encounters uncertainty:
❓ Agent Question
┌────────────────────────────────────────────────────────────┐
│ The agent is uncertain (confidence: 35%) about what to do  │
│ next.                                                       │
│                                                             │
│ Current screen: CheckoutActivity                           │
│ Intent: Complete checkout                                  │
│ Suggested action: Tap on Element at [500, 800]            │
│                                                             │
│ What should the agent do?                                  │
└────────────────────────────────────────────────────────────┘
Your answer: Use credit card ending in 1234
✓ Answer recorded: Use credit card ending in 1234

# Agent re-analyzes with your guidance and continues
```

### Example 4: Multiple Context Injections

```bash
# Terminal 1: Start
fathom run "Complete registration" --interactive -s emulator-5554

# Terminal 2: First injection
touch .fathom_pause
echo "Use email: test@example.com" > .fathom_context
touch .fathom_resume

# Wait for agent to process...

# Terminal 2: Second injection
touch .fathom_pause
echo "Use password: Test123!" > .fathom_context
touch .fathom_resume

# Agent uses both contexts in its reasoning
```

---

## How Context Affects LLM Reasoning

### Without User Guidance

```
LLM Prompt:
-----------
Intent: Login to the app
Current Screen: LoginActivity
Available Actions: [tap, type, swipe]
History: [previous actions...]

What should the agent do next?
```

### With User Guidance

```
LLM Prompt:
-----------
Intent: Login to the app
Current Screen: LoginActivity
Available Actions: [tap, type, swipe]
History: [previous actions...]

USER CONTEXT:
- The login button is at the bottom of the screen
- Use test@example.com as the email
- Skip the tutorial screens

What should the agent do next?
```

**Result**: LLM makes better decisions with the additional context!

---

## Why File-Based Pause?

We chose file-based control because:

1. **No terminal conflicts**: Works with Rich library and stdin
2. **Works during LLM calls**: Can pause even when agent is thinking
3. **Simple**: Just create/delete files
4. **Scriptable**: Easy to automate
5. **No special keys**: Works on all platforms
6. **Reliable**: No timing issues or race conditions

Previous attempts with keyboard listeners (Ctrl+P) had issues:
- Terminal I/O conflicts with Rich library
- Couldn't pause during LLM analysis (3-10 second delay)
- stdin conflicts between listener and Prompt.ask()
- Couldn't enter context properly when paused

---

## Implementation Files

### Core HITL Components

1. **FileWatcher** (`src/fathom/adapters/signal/file_watcher.py`)
   - Background thread watching for control files
   - Detects `.fathom_pause`, `.fathom_context`, `.fathom_resume`
   - Reads context from files
   - Automatic cleanup on stop

2. **InteractiveSignal** (`src/fathom/adapters/signal/interactive.py`)
   - Integrates FileWatcher
   - Handles automatic pause (agent questions)
   - Handles manual pause (file-based)
   - Context injection from both sources
   - Interactive menu for agent questions

3. **IntentStrategy** (`src/fathom/strategies/intent.py`)
   - Signal checking every step
   - Uncertainty detection (confidence < 0.5)
   - Automatic question asking
   - Context injection into LLM
   - Re-analysis with guidance

4. **ContextManager** (`src/fathom/core/context/manager.py`)
   - `inject_user_guidance()` - Store guidance
   - `get_user_guidance()` - Retrieve guidance
   - Persistence to memory

5. **CLI** (`src/fathom/cli_new.py`)
   - `--interactive` / `-i` flag
   - Signal adapter selection
   - Interactive mode messaging

---

## Testing Commands

### Test 1: Basic Interactive Mode

```bash
fathom run "Open Gmail" --use-xml --serial emulator-5554 -i -v
```

**Expected**:
- Interactive mode message displayed
- Manual pause instructions shown
- Agent executes normally
- If uncertain, asks for help

### Test 2: Manual Pause Test

```bash
# Terminal 1
fathom run "Login to Gmail" --use-xml --serial emulator-5554 -i

# Terminal 2
touch .fathom_pause
echo "Use test@example.com" > .fathom_context
touch .fathom_resume
```

**Expected**:
- Agent pauses when `.fathom_pause` detected
- Context injected from `.fathom_context`
- Agent resumes when `.fathom_resume` detected
- Agent uses context in next analysis

### Test 3: Agent Question Test

```bash
# Use complex intent likely to trigger uncertainty
fathom run "Login to Gmail with test@example.com and navigate to settings" \
  --use-xml --serial emulator-5554 -i -v
```

**Expected**:
- Agent may ask questions during execution
- You can provide guidance via interactive menu
- Agent uses your guidance for better decisions

---

## Troubleshooting

### Pause not working?

Check:
1. Are you in the correct directory where fathom is running?
2. Is `.fathom_pause` file created?
3. Is interactive mode enabled (`--interactive`)?
4. Check file permissions

### Context not injected?

Check:
1. Is `.fathom_context` file created?
2. Does it contain text?
3. Did you resume after injecting?
4. Check file encoding (should be UTF-8)

### Files not cleaned up?

The FileWatcher automatically cleans up files when:
- Execution completes
- Execution is cancelled
- FileWatcher is stopped

If files remain, you can manually delete them:
```bash
rm -f .fathom_pause .fathom_context .fathom_resume
```

### Agent not asking questions?

Check:
1. Is interactive mode enabled (`--interactive`)?
2. Is the agent actually uncertain? (Check logs for confidence scores)
3. Is the task too easy? (Agent might be confident)

---

## Benefits of HITL

1. **Better Decisions**: Agent makes informed choices with human guidance
2. **Error Recovery**: Human can correct agent when it goes wrong
3. **Learning**: Agent learns from human feedback
4. **Transparency**: Human sees agent's reasoning and uncertainty
5. **Control**: Human maintains control over critical decisions
6. **Efficiency**: Agent handles routine tasks, human handles edge cases
7. **Flexibility**: Can pause at ANY time, not just when agent is uncertain

---

## Status

✅ **FULLY IMPLEMENTED** - Production-ready HITL system
✅ **Automatic Pause** - Agent asks questions when uncertain
✅ **Manual Pause** - File-based pause at any time
✅ **Context Injection** - Affects LLM reasoning
✅ **Context Persistence** - Stored in memory
✅ **CLI Integration** - `--interactive` flag
✅ **Documentation** - Complete guide
✅ **No Terminal Conflicts** - File-based approach is reliable

**Ready for production use!** 🎉
