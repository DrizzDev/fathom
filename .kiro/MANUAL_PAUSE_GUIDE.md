# Manual Pause Feature - Complete Guide

## ✅ NOW IMPLEMENTED!

The manual pause feature is now fully implemented. You can press **Ctrl+P** at ANY time during execution to pause and inject context.

---

## How to Use Manual Pause

### Step 1: Run with Interactive Mode

```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml --serial emulator-5554 --verbose --interactive
```

### Step 2: Press Ctrl+P Anytime

While the agent is executing, press **Ctrl+P** to pause:

```
[INFO] Step 3: Analyzing screen...
[INFO] Executing: Tap on Element at [540, 960]

⏸️  Manual pause requested (Ctrl+P)
⏸️  Execution Paused
The agent is waiting for your input...

┌─────────────────────────────────────────┐
│ HITL Control                            │
│                                         │
│ Options:                                │
│   1. Resume execution                   │
│   2. Inject additional context          │
│   3. Cancel execution                   │
└─────────────────────────────────────────┘
Choose an option [1/2/3] (1): _
```

### Step 3: Inject Context

Choose option 2 to inject context:

```
Choose an option [1/2/3] (1): 2

💡 Inject Additional Context
Provide additional information to help the agent make better decisions.
Examples:
  - 'The login button is at the bottom of the screen'
  - 'Use test@example.com as the email'
  - 'Skip the tutorial screens'

Enter context (or press Enter to skip): The ChatGPT app is the green icon in the second row. After opening it, type "research opencrawler and moltybot projects in detail"
✓ Context injected: The ChatGPT app is the green icon...
```

### Step 4: Resume Execution

Choose option 1 to resume:

```
Choose an option [1/2/3] (1): 1
▶️  Resuming execution...

[INFO] Context injected by user
[INFO] Re-analyzing with user context...
[INFO] LLM Context now includes:
      USER CONTEXT:
      - The ChatGPT app is the green icon in the second row
      - After opening it, type "research opencrawler and moltybot projects in detail"

[INFO] Executing with enhanced context...
```

---

## Two Ways to Provide Context

### 1. Manual Pause (Ctrl+P) - NEW! ✅

**When**: You want to pause at ANY specific moment

**How**: Press Ctrl+P

**Use case**: 
- You see the agent is about to make a mistake
- You want to provide context before a critical step
- You have information that will help the agent

**Example**:
```
Agent is about to tap wrong element...
→ Press Ctrl+P
→ Inject context: "Don't tap that. The correct button is at the bottom right"
→ Resume
→ Agent uses your context and taps the correct element
```

### 2. Automatic Questions (Agent-Initiated)

**When**: Agent is uncertain (confidence < 50%)

**How**: Agent automatically pauses and asks

**Use case**:
- Agent doesn't know what to do
- Agent has low confidence
- Agent is stuck

**Example**:
```
❓ Agent Question
The agent is uncertain (confidence: 35%) about what to do next.
Your answer: Open the ChatGPT app
```

---

## Complete Workflow Example

### Your Command:
```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml -s emulator-5554 -v -i
```

### Execution Flow:

**1. Agent Starts**
```
🤝 Interactive mode enabled
💡 Tip: Press Ctrl+P at any time to pause and provide context

╭─────────────────────────────────────────╮
│ Fathom Agent                            │
│ Intent: Ask GPT to do deep research     │
│         about opencrawler(moltybot)     │
╰─────────────────────────────────────────╯

Agent working...
```

**2. Agent Captures Screen**
```
[INFO] Step 1: Capturing screen...
[INFO] Current screen: com.android.launcher
[INFO] Analyzing with LLM...
```

**3. You Press Ctrl+P (Manual Pause)**
```
[You press Ctrl+P]

⏸️  Manual pause requested (Ctrl+P)
⏸️  Execution Paused

┌─────────────────────────────────────────┐
│ HITL Control                            │
│ Options:                                │
│   1. Resume execution                   │
│   2. Inject additional context          │
│   3. Cancel execution                   │
└─────────────────────────────────────────┘
Choose an option [1/2/3] (1): 2
```

**4. You Inject Context**
```
Enter context: The ChatGPT app is the green icon with white logo. Open it and ask "Please do deep research about opencrawler and moltybot projects"

✓ Context injected
```

**5. You Resume**
```
Choose an option [1/2/3] (1): 1
▶️  Resuming execution...

[INFO] Context injected by user
[INFO] Re-analyzing with user context...
```

**6. Agent Uses Your Context**
```
[INFO] LLM Context includes:
      USER CONTEXT:
      - The ChatGPT app is the green icon with white logo
      - Open it and ask "Please do deep research about opencrawler and moltybot projects"

[INFO] Confidence: 90% (HIGH - much better with your context!)
[INFO] Executing: Tap on ChatGPT app icon
```

**7. Agent Continues with Better Decisions**
```
[INFO] Step 2: Opening ChatGPT app...
[INFO] Step 3: Tapping chat input field...
[INFO] Step 4: Typing research query...
[INFO] Step 5: Sending message...

✓ Success! Goal achieved
```

---

## When to Use Manual Pause

### ✅ Good Use Cases:

**1. Prevent Mistakes**
```
Agent is about to tap the wrong button
→ Ctrl+P
→ "Don't tap that. The correct button is the blue one at the bottom"
```

**2. Provide Credentials**
```
Agent reached login screen
→ Ctrl+P
→ "Use username test@example.com and password test123"
```

**3. Give Navigation Hints**
```
Agent is on home screen
→ Ctrl+P
→ "The app you need is in the second row, third icon"
```

**4. Clarify Intent**
```
Agent seems confused about the goal
→ Ctrl+P
→ "The goal is to search for 'opencrawler', not to open settings"
```

**5. Skip Steps**
```
Agent is about to go through tutorial
→ Ctrl+P
→ "Skip the tutorial by tapping the X button at the top right"
```

### ❌ Don't Need Manual Pause:

**1. Agent is Confident**
```
Agent confidence: 85%
→ Let it continue, it knows what to do
```

**2. Agent Already Asked**
```
Agent asked a question
→ Just answer the question, no need to pause
```

**3. Everything is Going Well**
```
Agent is making good progress
→ Let it work, don't interrupt
```

---

## How Context Affects Execution

### Without Your Context:
```
Agent sees: Home screen with many apps
Agent thinks: "I need to open an app, but which one?"
Agent confidence: 30% (LOW)
Agent action: Tap random app (might be wrong)
```

### With Your Context (Manual Pause):
```
[You press Ctrl+P]
Your context: "Open the ChatGPT app - green icon in second row"

Agent sees: Home screen with many apps
Agent thinks: "User said ChatGPT app is green icon in second row"
Agent confidence: 90% (HIGH)
Agent action: Tap ChatGPT app (correct!)
```

---

## Technical Details

### How It Works:

**1. Keyboard Listener**
- Background thread listens for Ctrl+P
- Non-blocking, doesn't interfere with execution
- Works on Unix/Linux/macOS (requires termios)

**2. Signal Check**
- Every step, agent checks for signals
- If Ctrl+P was pressed, PAUSE signal is returned
- Agent pauses immediately

**3. Context Injection**
- User provides context via interactive menu
- Context is stored in ContextManager
- Context is added to LLM prompt

**4. Resume**
- Agent continues from exact same point
- Next LLM analysis includes your context
- Better decisions throughout execution

### Code Flow:

```python
# Phase 1: Signal Check (in ExecutionEngine)
signal = await self.__signal.check_signal()

if signal == SignalType.PAUSE:
    # User pressed Ctrl+P
    await self.__signal.wait_for_resume()
    # User provided context
    injected_context = self.__signal.get_injected_context()
    
# Phase 3: Reason (in IntentStrategy)
if injected_context:
    # Add to context manager
    await self.__context.inject_user_guidance(injected_context)
    
    # Re-analyze with context
    plan = await self.__perform_analysis(
        state=state,
        screen=screen,
        additional_context=f"USER CONTEXT: {injected_context}"
    )
```

---

## Troubleshooting

### Q: Ctrl+P doesn't work

**A**: Check your system:
- Works on: Unix, Linux, macOS
- Doesn't work on: Windows (requires different implementation)
- Workaround: Wait for agent to ask questions (automatic pause)

### Q: How do I know if manual pause is available?

**A**: Look for this message when starting:
```
💡 Tip: Press Ctrl+P at any time to pause and provide context
```

If you don't see it, manual pause is not available on your system.

### Q: Can I pause multiple times?

**A**: Yes! Press Ctrl+P as many times as you want. Each time you can inject more context.

### Q: What if I press Ctrl+P by accident?

**A**: Just choose option 1 (Resume execution) without injecting context. The agent will continue normally.

---

## Comparison: Manual vs Automatic Pause

| Feature | Manual Pause (Ctrl+P) | Automatic Pause (Agent Questions) |
|---------|----------------------|-----------------------------------|
| **Trigger** | You press Ctrl+P | Agent confidence < 50% |
| **Timing** | Anytime you want | When agent is uncertain |
| **Control** | Full control | Agent decides |
| **Use Case** | Prevent mistakes, provide hints | Agent needs help |
| **Frequency** | As many times as you want | Only when uncertain |

**Best Practice**: Use both!
- Let agent ask questions when uncertain (automatic)
- Press Ctrl+P when you see a problem (manual)

---

## Summary

✅ **Manual pause is NOW IMPLEMENTED**

✅ **Press Ctrl+P anytime** to pause execution

✅ **Inject context** to help agent make better decisions

✅ **Resume execution** with enhanced context

✅ **Works alongside** automatic agent questions

✅ **Full control** over execution flow

---

## Your Command (Ready to Use!)

```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml --serial emulator-5554 --verbose --interactive
```

**During execution**:
- Press **Ctrl+P** to pause anytime
- Inject context when needed
- Let agent ask questions when uncertain
- Achieve your goal with human-agent collaboration! 🚀
