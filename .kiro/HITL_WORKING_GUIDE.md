# HITL Working Guide - After Fixes

## ✅ What's Fixed

All critical bugs are now fixed:
- ✅ Metrics validation error
- ✅ Terminal I/O conflicts
- ✅ Context injection issues
- ✅ Execution crashes

## 🎯 How HITL Works Now

### Automatic Pause (Production-Ready)

The agent **automatically pauses and asks questions** when uncertain (confidence < 50%).

**No manual pause needed** - the agent knows when it needs help!

---

## Your Command

```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml --serial emulator-5554 --verbose --interactive
```

---

## What Happens

### 1. Agent Starts
```
🤝 Interactive mode enabled
🤝 Agent will ask questions when uncertain (confidence < 50%)

╭─────────────────────────────────────────╮
│ Fathom Agent                            │
│ Intent: Ask GPT to do deep research     │
│         about opencrawler(moltybot)     │
╰─────────────────────────────────────────╯

Agent working...
```

### 2. Agent Executes Steps
```
[INFO] Step 1: Capturing screen...
[INFO] Analyzing with LLM...
[INFO] Confidence: 90% (HIGH)
[INFO] Executing: Tap on ChatGPT app
```

### 3. Agent Asks When Uncertain
```
[INFO] Step 2: Analyzing...
[INFO] Confidence: 35% (LOW - will ask for help)

❓ Agent Question
┌────────────────────────────────────────────────────────────┐
│ The agent is uncertain (confidence: 35%) about what to do  │
│ next.                                                       │
│                                                             │
│ Current screen: com.openai.chatgpt                         │
│ Intent: Ask GPT to do deep research about opencrawler      │
│ Suggested action: Tap on Element at [540, 960]            │
│                                                             │
│ What should the agent do? (Provide guidance or press       │
│ Enter to continue)                                          │
└────────────────────────────────────────────────────────────┘
Your answer: _
```

### 4. You Provide Guidance
```
Your answer: Tap the chat input field at the bottom, then type "Please do deep research about opencrawler and moltybot projects"

✓ Answer recorded: Tap the chat input field...
```

### 5. Agent Uses Your Guidance
```
[INFO] User guidance received
[INFO] Injecting guidance into context
[INFO] Re-analyzing with user guidance...
[INFO] LLM Context now includes:
      USER GUIDANCE:
      - Tap the chat input field at the bottom
      - Then type "Please do deep research about opencrawler and moltybot projects"

[INFO] Confidence: 85% (HIGH - much better!)
[INFO] Executing: Tap on chat input field
```

### 6. Agent Continues
```
[INFO] Step 3: Typing message...
[INFO] Step 4: Sending message...
[INFO] Step 5: Waiting for response...

✓ Execution Summary
┌─────────────────┬──────────────────────────────────────┐
│ Metric          │ Value                                │
├─────────────────┼──────────────────────────────────────┤
│ Status          │ Success                              │
│ Reason          │ Goal successfully achieved           │
│ Steps Taken     │ 5                                    │
└─────────────────┴──────────────────────────────────────┘
```

---

## When Agent Asks Questions

The agent asks questions when:

1. **Low Confidence** (< 50%):
   - Doesn't know which app to open
   - Unsure about which element to tap
   - Multiple possible actions

2. **Stuck Detection**:
   - Same screen visited multiple times
   - No progress being made

3. **Ambiguous Intent**:
   - Intent is complex or unclear
   - Multiple ways to achieve goal

---

## How to Provide Good Guidance

### ✅ Good Examples:

**Be Specific**:
```
Your answer: Tap the blue "Sign in" button at the bottom right corner
```

**Provide Context**:
```
Your answer: The ChatGPT app is the green icon in the second row. Open it and type "research opencrawler and moltybot"
```

**Give Step-by-Step**:
```
Your answer: First tap the chat input field at the bottom. Then type the research query. Then tap the send button on the right.
```

**Describe Visually**:
```
Your answer: Look for the green icon with white logo. It's in the second row, third position.
```

### ❌ Avoid:

**Too Vague**:
```
Your answer: Just do it
Your answer: Click something
```

**Not Helpful**:
```
Your answer: I don't know
Your answer: Whatever
```

---

## Tips for Success

### 1. Let Agent Work
- Don't worry if agent doesn't ask immediately
- Agent is confident when it knows what to do
- Questions mean agent needs help

### 2. Be Ready to Help
- Watch the screen
- Think about what agent should do
- Prepare guidance in advance

### 3. Provide Rich Context
- Don't just answer the immediate question
- Provide context for future steps
- Think ahead about what agent will need

### 4. Trust the Agent
- If agent is confident (> 50%), let it work
- If agent asks, provide guidance
- Agent learns from your guidance

---

## Example Session

```bash
$ fathom run "Ask GPT to research opencrawler" -i -x -s emulator-5554 -v

🤝 Interactive mode enabled
🤝 Agent will ask questions when uncertain

╭─────────────────────────────────────────╮
│ Fathom Agent                            │
│ Intent: Ask GPT to research opencrawler │
╰─────────────────────────────────────────╯

[INFO] Step 1: Analyzing home screen...
[INFO] Confidence: 40% (LOW)

❓ Agent Question
Your answer: Open the ChatGPT app - it's the green icon with white logo

✓ Answer recorded

[INFO] Re-analyzing with guidance...
[INFO] Confidence: 90% (HIGH)
[INFO] Step 2: Opening ChatGPT app...
[INFO] Step 3: Tapping chat input...
[INFO] Step 4: Typing research query...
[INFO] Step 5: Sending message...

✓ Success! Goal achieved in 5 steps
```

---

## Troubleshooting

### Q: Agent never asks questions?
**A**: Good! It means agent is confident. Your intent is clear and agent knows what to do.

### Q: Agent asks too many questions?
**A**: Provide more detailed guidance each time. Include context for future steps.

### Q: Can I pause manually?
**A**: Not currently. Manual pause (Ctrl+P) was removed due to technical issues. Agent will ask when it needs help.

### Q: How do I cancel execution?
**A**: Press Ctrl+C. Agent will cleanup and exit gracefully.

---

## What Changed

### Before (Broken):
- ❌ Manual pause (Ctrl+P) - didn't work
- ❌ Terminal conflicts
- ❌ Metrics validation errors
- ❌ Execution crashes

### After (Fixed):
- ✅ Automatic pause - works reliably
- ✅ No terminal conflicts
- ✅ Metrics validation works
- ✅ Execution completes successfully

---

## Summary

**Current Status**: ✅ Production-Ready

**How It Works**:
1. Run with `--interactive` flag
2. Agent asks questions when uncertain
3. You provide guidance
4. Agent uses guidance for better decisions
5. Goal achieved successfully

**Key Point**: You don't need to do anything special. Just run the command and answer questions when agent asks!

---

## Your Command (Ready to Use!)

```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" \
  --use-xml --serial emulator-5554 --verbose --interactive
```

**The agent will guide you through by asking questions when needed!** 🚀
