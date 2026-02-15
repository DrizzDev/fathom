# Your Command Walkthrough

## Your Command
```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml \
  --serial emulator-5554 \
  --verbose \
  --interactive
```

## What Will Happen Step-by-Step

### 1. Startup (First 2 seconds)

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

**What's happening**:
- Fathom connects to your emulator (emulator-5554)
- Loads the hexagonal architecture
- Initializes Gemini LLM with your credentials
- Enables interactive mode (HITL)
- Starts the execution loop

---

### 2. First Screen Capture

```
[INFO] Phase 1: Signal Check - No signals
[INFO] Phase 2: Perceive - Capturing screen...
[INFO] Screen captured: 1080x2400 pixels
[INFO] Current activity: com.android.launcher3.Launcher
[INFO] Phase 3: Reason - Analyzing with LLM...
```

**What's happening**:
- Agent captures screenshot of your device
- Detects current app (probably home screen)
- Sends screenshot + intent to Gemini LLM
- LLM analyzes what to do next

---

### 3. First Decision (Likely Uncertain)

Since your intent is complex ("Ask GPT to do deep research..."), the agent will likely be uncertain about the first step.

```
[INFO] LLM Analysis complete
[INFO] Confidence: 35% (LOW - will ask for help)
[INFO] Suggested action: Tap on Element at [540, 960]

❓ Agent Question
┌────────────────────────────────────────────────────────────┐
│ The agent is uncertain (confidence: 35%) about what to do  │
│ next.                                                       │
│                                                             │
│ Current screen: com.android.launcher3.Launcher             │
│ Intent: Ask GPT to do deep research about                  │
│         opencrawler(moltybot)                               │
│ Suggested action: Tap on Element at [540, 960]            │
│                                                             │
│ What should the agent do? (Provide guidance or press       │
│ Enter to continue)                                          │
└────────────────────────────────────────────────────────────┘
Your answer: _
```

**What's happening**:
- Agent detected low confidence (35% < 50%)
- Agent automatically pauses and asks for your help
- Waiting for your input

---

### 4. Your Response

**Option A: Provide Guidance** (Recommended)

```
Your answer: Open the ChatGPT app (green icon with white logo). Then ask it to do deep research about opencrawler and moltybot projects.
```

Press Enter.

**What happens next**:
```
✓ Answer recorded: Open the ChatGPT app (green icon with white logo)...

[INFO] User guidance received
[INFO] Injecting guidance into context
[INFO] Re-analyzing with user guidance...
[INFO] LLM Context now includes:
      USER GUIDANCE:
      - Open the ChatGPT app (green icon with white logo)
      - Then ask it to do deep research about opencrawler and moltybot projects

[INFO] LLM Analysis complete
[INFO] Confidence: 85% (HIGH - much better!)
[INFO] Planned action: Tap on ChatGPT app icon at [540, 1200]
```

**Option B: Just Press Enter**

```
Your answer: [press Enter]
```

**What happens next**:
```
[INFO] User chose to continue with suggested action
[INFO] Executing: Tap on Element at [540, 960]
```

Agent will tap whatever it thinks is best (might be wrong).

---

### 5. Execution Continues

Assuming you provided guidance (Option A):

```
[INFO] Phase 4: Act - Executing tap on ChatGPT app
[INFO] Device command: adb shell input tap 540 1200
[INFO] Command executed successfully
[INFO] Phase 5: Learn - Storing experience in memory
[INFO] Phase 6: Checkpoint - Step 1 completed in 2.3s
[INFO] Phase 7: Evaluate - Screen changed: Yes

[INFO] Step 1 complete
[INFO] Steps taken: 1/20
```

---

### 6. Next Screen (ChatGPT App)

```
[INFO] Phase 2: Perceive - Capturing screen...
[INFO] Current activity: com.openai.chatgpt.MainActivity
[INFO] Phase 3: Reason - Analyzing with LLM...
[INFO] LLM Context includes previous USER GUIDANCE
[INFO] Confidence: 75% (GOOD)
[INFO] Planned action: Tap on chat input field
```

**What's happening**:
- Agent captured new screen (ChatGPT app)
- Your previous guidance is STILL in the context
- Agent is more confident now (75%)
- No question asked (confidence > 50%)

```
[INFO] Phase 4: Act - Executing tap on chat input field
[INFO] Step 2 complete
```

---

### 7. Typing the Query

```
[INFO] Phase 2: Perceive - Capturing screen...
[INFO] Current activity: com.openai.chatgpt.MainActivity
[INFO] Phase 3: Reason - Analyzing with LLM...
[INFO] Confidence: 80%
[INFO] Planned action: Type "Please do deep research about opencrawler and moltybot projects"

[INFO] Phase 4: Act - Executing type action
[INFO] Device command: adb shell input text "Please do deep research..."
[INFO] Step 3 complete
```

---

### 8. Sending the Message

```
[INFO] Phase 2: Perceive - Capturing screen...
[INFO] Phase 3: Reason - Analyzing with LLM...
[INFO] Confidence: 90%
[INFO] Planned action: Tap on send button

[INFO] Phase 4: Act - Executing tap on send button
[INFO] Step 4 complete
```

---

### 9. Waiting for Response (Might Ask Again)

```
[INFO] Phase 2: Perceive - Capturing screen...
[INFO] Current activity: com.openai.chatgpt.MainActivity
[INFO] Phase 3: Reason - Analyzing with LLM...
[INFO] Confidence: 40% (LOW - will ask for help)

❓ Agent Question
┌────────────────────────────────────────────────────────────┐
│ The agent is uncertain (confidence: 40%) about what to do  │
│ next.                                                       │
│                                                             │
│ Current screen: com.openai.chatgpt.MainActivity            │
│ Intent: Ask GPT to do deep research about                  │
│         opencrawler(moltybot)                               │
│ Suggested action: Wait for response                        │
│                                                             │
│ What should the agent do?                                  │
└────────────────────────────────────────────────────────────┘
Your answer: _
```

**Your response**:
```
Your answer: Wait for ChatGPT to finish responding, then scroll down to read the full research results.
```

---

### 10. Completion

```
[INFO] Phase 3: Reason - Analyzing with LLM...
[INFO] Goal completion detected: Yes
[INFO] Reason: ChatGPT has provided deep research about opencrawler and moltybot

[INFO] Execution complete!

✓ Execution Summary
┌─────────────────┬──────────────────────────────────────┐
│ Metric          │ Value                                │
├─────────────────┼──────────────────────────────────────┤
│ Status          │ Success                              │
│ Reason          │ Goal successfully achieved           │
│ Steps Taken     │ 6                                    │
└─────────────────┴──────────────────────────────────────┘

✓ Timing Audit
┌─────────────────┬──────────────┬──────────────┐
│ Operation       │ Total Time   │ Avg/Step     │
├─────────────────┼──────────────┼──────────────┤
│ screenshot      │ 1.2s         │ 0.2s         │
│ analysis        │ 8.4s         │ 1.4s         │
└─────────────────┴──────────────┴──────────────┘

✓ Resource Usage (Tokens)
┌─────────────────┬──────────────┐
│ Metric          │ Value        │
├─────────────────┼──────────────┤
│ Prompt Tokens   │ 12,450       │
│ Completion      │ 890          │
│ Cached Tokens   │ 8,200        │
│ Total Tokens    │ 13,340       │
└─────────────────┴──────────────┘
```

---

## Summary of Interactions

### You Will Be Asked Questions When:

1. **First step** - Agent doesn't know which app to open
   - Your guidance: "Open ChatGPT app"

2. **After sending message** - Agent doesn't know if it should wait or do something else
   - Your guidance: "Wait for response and read results"

3. **Any uncertain moment** - Confidence < 50%
   - Your guidance: Specific instructions

### Your Guidance Will:

1. Be stored in agent's memory
2. Be added to LLM context for ALL future steps
3. Help agent make better decisions
4. Increase agent's confidence
5. Lead to successful completion

---

## Key Takeaways

✅ **Your command is perfect** - Just run it!

✅ **Agent will ask for help** - When uncertain (confidence < 50%)

✅ **Provide specific guidance** - "Open ChatGPT app" not "do something"

✅ **Guidance persists** - Used for all future decisions

✅ **You can answer multiple times** - As many times as agent asks

✅ **Just press Enter** - If you want agent to continue without guidance

---

## Ready to Run!

```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml --serial emulator-5554 --verbose --interactive
```

The agent will guide you through the process by asking questions when needed! 🚀
