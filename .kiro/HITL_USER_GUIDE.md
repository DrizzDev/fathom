# HITL (Human-in-the-Loop) User Guide

## Quick Start

Your command is correct! Just add the `-i` or `--interactive` flag:

```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml \
  --serial emulator-5554 \
  --verbose \
  --interactive
```

Or shorter:
```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml -s emulator-5554 -v -i
```

---

## How HITL Works - 3 Ways to Interact

### 1. 🤖 Agent Asks You Questions (Automatic)

**When it happens**: The agent automatically asks for help when it's uncertain (confidence < 50%)

**What you'll see**:
```
❓ Agent Question
┌────────────────────────────────────────────────────────────┐
│ The agent is uncertain (confidence: 45%) about what to do  │
│ next.                                                       │
│                                                             │
│ Current screen: com.google.android.gm.LoginActivity        │
│ Intent: Ask GPT to do deep research about opencrawler      │
│ Suggested action: Tap on Element at [500, 800]            │
│                                                             │
│ What should the agent do? (Provide guidance or press       │
│ Enter to continue)                                          │
└────────────────────────────────────────────────────────────┘
Your answer: _
```

**What to do**:
- Type your guidance and press Enter
- Or just press Enter to let the agent continue with its suggestion

**Example answers**:
```
Your answer: Tap the blue "Sign in" button at the bottom
Your answer: Use the search bar at the top to search for "opencrawler"
Your answer: Scroll down to find the settings menu
Your answer: Type "moltybot" in the search field
```

**What happens next**:
- Your guidance is added to the agent's context
- The agent re-analyzes the screen with your guidance
- The agent makes a better decision based on your input

---

### 2. ⏸️ Manual Pause (Currently Not Implemented via Keyboard)

**Note**: The pause mechanism is implemented in the code but there's no keyboard shortcut yet. The agent will pause automatically when it asks questions (method #1 above).

**Future enhancement**: We could add Ctrl+P to pause, but for now, the agent pauses automatically when uncertain.

---

### 3. 💡 Context Injection During Agent Questions

When the agent asks you a question, you can provide rich context that will be used for ALL future decisions:

**Example**:
```
Your answer: The app requires login. Use username "test@example.com" and password "test123". After login, navigate to the search section and look for "opencrawler" or "moltybot".
```

This guidance will be:
1. Stored in the agent's memory
2. Added to the LLM prompt for the next analysis
3. Used to make better decisions going forward

---

## Real-World Example Walkthrough

### Scenario: Your Command

```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml -s emulator-5554 -v -i
```

### What Will Happen:

**Step 1: Agent Starts**
```
🤝 Interactive mode enabled
You can pause execution and provide guidance at any time

╭─────────────────────────────────────────╮
│ Fathom Agent                            │
│ Intent: Ask GPT to do deep research     │
│         about opencrawler(moltybot)     │
╰─────────────────────────────────────────╯

Agent working...
```

**Step 2: Agent Captures Screen**
```
[INFO] Captured screen: com.android.launcher
[INFO] Analyzing screen with LLM...
```

**Step 3: Agent Might Ask for Help**

If the agent is uncertain about which app to open or how to proceed:

```
❓ Agent Question
┌────────────────────────────────────────────────────────────┐
│ The agent is uncertain (confidence: 40%) about what to do  │
│ next.                                                       │
│                                                             │
│ Current screen: com.android.launcher                       │
│ Intent: Ask GPT to do deep research about opencrawler      │
│ Suggested action: Tap on Element at [540, 960]            │
│                                                             │
│ What should the agent do?                                  │
└────────────────────────────────────────────────────────────┘
Your answer: _
```

**Your Response Options**:

**Option A: Provide Specific Guidance**
```
Your answer: Open the ChatGPT app (green icon with white logo). Then ask it to research "opencrawler" and "moltybot" projects.
```

**Option B: Just Press Enter**
```
Your answer: [press Enter]
```
The agent will continue with its suggested action.

**Step 4: Agent Uses Your Guidance**
```
✓ Answer recorded: Open the ChatGPT app (green icon with white logo)...

[INFO] Re-analyzing with user guidance...
[INFO] LLM Context includes:
      USER GUIDANCE:
      - Open the ChatGPT app (green icon with white logo)
      - Then ask it to research "opencrawler" and "moltybot" projects

[INFO] Executing: Tap on ChatGPT app icon
```

**Step 5: Agent Continues**

The agent will continue executing steps. If it gets stuck again, it will ask for help again.

---

## Tips for Providing Good Guidance

### ✅ Good Guidance Examples

**Be Specific**:
```
Your answer: Tap the blue "Sign in" button at the bottom right corner
```

**Provide Context**:
```
Your answer: The app needs login first. Use test@example.com and password test123. After login, go to the search tab.
```

**Give Step-by-Step Instructions**:
```
Your answer: First open ChatGPT app. Then type "research opencrawler and moltybot projects" in the chat. Wait for the response.
```

**Describe Visual Elements**:
```
Your answer: Look for the green icon with a white chat bubble. It's in the second row of apps.
```

### ❌ Avoid Vague Guidance

**Too Vague**:
```
Your answer: Just do it
Your answer: Click something
Your answer: I don't know
```

**Better**:
```
Your answer: Try tapping the search icon at the top of the screen
Your answer: Scroll down to see more options
Your answer: Go back and try a different approach
```

---

## How Context Affects the Agent

### Without Your Guidance

The LLM sees:
```
Intent: Ask GPT to do deep research about opencrawler(moltybot)
Current Screen: com.android.launcher
Available Actions: [tap, type, swipe, scroll]
Screen Elements: [App icons, search bar, dock]

What should the agent do next?
```

### With Your Guidance

The LLM sees:
```
Intent: Ask GPT to do deep research about opencrawler(moltybot)
Current Screen: com.android.launcher
Available Actions: [tap, type, swipe, scroll]
Screen Elements: [App icons, search bar, dock]

USER GUIDANCE:
- Open the ChatGPT app (green icon with white logo)
- Then ask it to research "opencrawler" and "moltybot" projects

What should the agent do next?
```

**Result**: The LLM makes a much better decision! 🎯

---

## Advanced Usage

### Multiple Guidance Inputs

You can provide guidance multiple times during execution:

**First Question**:
```
Your answer: Open ChatGPT app
```

**Second Question** (later in execution):
```
Your answer: Type the research query in the chat input field
```

**Third Question**:
```
Your answer: Wait for the response and scroll down to read it
```

All guidance is accumulated and used by the agent!

---

## Troubleshooting

### "Agent never asks questions"

**Possible reasons**:
1. Agent is very confident (confidence > 50%) - this is good!
2. The task is straightforward
3. You forgot the `-i` flag

**Solution**: Make sure you use `--interactive` or `-i` flag

### "I want to pause manually"

**Current limitation**: Manual pause via keyboard is not implemented yet.

**Workaround**: Wait for the agent to ask a question (it will when uncertain)

**Future enhancement**: We can add Ctrl+P keyboard shortcut

### "Agent ignores my guidance"

**Check**:
1. Did you press Enter after typing?
2. Was your guidance specific enough?
3. Check the logs to see if guidance was recorded

**Example log**:
```
[INFO] User guidance received: Open ChatGPT app
[INFO] Injecting guidance into context
[INFO] Re-analyzing with user guidance
```

---

## Command Reference

### Basic Command
```bash
fathom run "YOUR INTENT" -i
```

### Full Command with All Options
```bash
fathom run "YOUR INTENT" \
  --interactive \              # Enable HITL
  --use-xml \                  # Use XML for better element detection
  --serial emulator-5554 \     # Device serial
  --verbose \                  # Show detailed logs
  --max-steps 30               # Max steps (default: 20)
```

### Short Form
```bash
fathom run "YOUR INTENT" -i -x -s emulator-5554 -v
```

---

## What Happens Behind the Scenes

### 1. Agent Detects Uncertainty
```python
if plan.step.action.confidence < 0.5:
    # Agent is uncertain - ask for help
```

### 2. Agent Asks Question
```python
user_guidance = await signal.request_input(prompt=question)
```

### 3. Guidance Stored
```python
await context.inject_user_guidance(guidance=user_guidance)
```

### 4. Agent Re-analyzes
```python
plan = await planner.plan_step(
    state=state,
    screen=screen,
    additional_context=f"USER GUIDANCE: {user_guidance}"
)
```

### 5. LLM Gets Enhanced Context
```
Original Prompt + USER GUIDANCE → Better Decision
```

---

## Benefits of Interactive Mode

1. **Better Accuracy**: Agent makes informed decisions with your help
2. **Error Recovery**: You can correct the agent when it goes wrong
3. **Learning**: Agent learns from your feedback
4. **Transparency**: You see when the agent is uncertain
5. **Control**: You maintain control over critical decisions
6. **Efficiency**: Agent handles routine tasks, you handle edge cases

---

## Example Session

```bash
$ fathom run "Ask GPT to do deep research about opencrawler(moltybot)" -i -x -s emulator-5554 -v

🤝 Interactive mode enabled
You can pause execution and provide guidance at any time

╭─────────────────────────────────────────╮
│ Fathom Agent                            │
│ Intent: Ask GPT to do deep research     │
│         about opencrawler(moltybot)     │
╰─────────────────────────────────────────╯

[INFO] Step 1: Capturing screen...
[INFO] Screen: com.android.launcher
[INFO] Analyzing with LLM...

❓ Agent Question
┌────────────────────────────────────────────────────────────┐
│ The agent is uncertain (confidence: 35%) about what to do  │
│ next.                                                       │
│                                                             │
│ Current screen: com.android.launcher                       │
│ Intent: Ask GPT to do deep research about opencrawler      │
│ Suggested action: Tap on Element at [540, 960]            │
│                                                             │
│ What should the agent do?                                  │
└────────────────────────────────────────────────────────────┘
Your answer: Open the ChatGPT app (green icon). Then ask it to research opencrawler and moltybot projects in detail.

✓ Answer recorded: Open the ChatGPT app (green icon)...

[INFO] Re-analyzing with user guidance...
[INFO] Step 2: Executing - Tap on ChatGPT app
[INFO] Step 3: Capturing screen...
[INFO] Screen: com.openai.chatgpt
[INFO] Step 4: Executing - Tap on chat input field
[INFO] Step 5: Executing - Type "Please do deep research about opencrawler and moltybot projects"
[INFO] Step 6: Executing - Tap send button

✓ Execution Summary
┌─────────────────┬──────────────────────────────────────┐
│ Metric          │ Value                                │
├─────────────────┼──────────────────────────────────────┤
│ Status          │ Success                              │
│ Reason          │ Goal successfully achieved           │
│ Steps Taken     │ 6                                    │
└─────────────────┴──────────────────────────────────────┘
```

---

## Summary

**To use HITL**:
1. Add `-i` or `--interactive` flag to your command
2. Wait for the agent to ask questions when uncertain
3. Provide specific, helpful guidance
4. Press Enter to submit your answer
5. The agent will use your guidance to make better decisions

**Your command is perfect**:
```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml --serial emulator-5554 --verbose --interactive
```

Just run it and wait for the agent to ask for help! 🚀
