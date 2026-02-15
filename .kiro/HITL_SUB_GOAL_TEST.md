# Quick Test Guide - HITL Sub-Goal Support

## 🎯 What Was Fixed

The HITL context injection now properly supports:
- ✅ Sub-goals
- ✅ Modified intents
- ✅ Priority instructions

## 🧪 Test Your Original Case

### Step 1: Start Execution
```bash
conda run -n Fathom-ENV fathom run "Ask GPT to do deep research about opencrawler(moltybot)" --interactive -s emulator-5554
```

### Step 2: Pause Immediately
```
pause [Enter]
```

### Step 3: Inject Modified Intent
```
Option: 2
Instruction: Actually search for indian climate instead of opencrawler
```

### Step 4: Resume
```
Option: 1
```

### Expected Result
The agent should now:
1. Type "indian climate" in ChatGPT (NOT "opencrawler")
2. Submit the search
3. Wait for response

---

## 🔍 What to Look For

### In Console
```
📝 Adding your context to LLM prompt:
Actually search for indian climate instead of opencrawler

🤖 Restarting analysis with your context...
```

### In Logs (if you check)
```
User Payload (Text Parts):
[...,
 '============================================================',
 '🎯 USER INSTRUCTION (PRIORITY):',
 'Actually search for indian climate instead of opencrawler',
 'Note: This user instruction takes priority. If it conflicts with the original goal, follow this instruction instead.',
 '============================================================',
 ...]
```

### On Device
The agent should type "indian climate" in the ChatGPT input field.

---

## 🧪 Additional Test Cases

### Test 2: Sub-Goal
```bash
# Start
fathom run "Submit the registration form" --interactive -s emulator-5554

# Pause
pause [Enter]

# Inject sub-goal
Option: 2
Instruction: First scroll down to review all fields, then submit

# Resume
Option: 1

# Expected: Agent scrolls down first, then submits
```

### Test 3: Guidance (Should Still Work)
```bash
# Start
fathom run "Ask ChatGPT about Python" --interactive -s emulator-5554

# Pause after typing
pause [Enter]

# Inject guidance
Option: 2
Instruction: Wait for ChatGPT to finish generating the full response

# Resume
Option: 1

# Expected: Agent waits longer before next action
```

---

## ✅ Success Criteria

The fix is working if:
1. ✅ Agent follows your modified intent (searches for "indian climate")
2. ✅ Console shows "USER INSTRUCTION (PRIORITY)" message
3. ✅ Agent behavior changes based on your instruction
4. ✅ No errors or crashes

---

## 🐛 If It Doesn't Work

### Check 1: Verify Code Changes
```bash
# Check if changes are applied
grep -A 5 "USER INSTRUCTION (PRIORITY)" src/fathom/strategies/intent.py
```

Should show the new formatting code.

### Check 2: Check Logs
```bash
# Run with verbose logging
conda run -n Fathom-ENV fathom run "your intent" --interactive -s emulator-5554 --verbose
```

Look for the priority context in the LLM payload.

### Check 3: Verify Signal Adapter
```bash
# Check if interactive signal has the new guidance
grep -A 10 "Modified intent" src/fathom/adapters/signal/interactive.py
```

Should show the updated examples.

---

## 📊 Comparison

### Before This Fix
```
LLM sees:
Goal: Research opencrawler
USER CONTEXT: search for indian climate

Result: Agent researches opencrawler (ignores context)
```

### After This Fix
```
LLM sees:
Goal: Research opencrawler
🎯 USER INSTRUCTION (PRIORITY):
search for indian climate
Note: This takes priority. If it conflicts, follow this instead.

Result: Agent searches for indian climate (follows priority instruction)
```

---

## 🎉 Ready to Test!

Run the test and let me know if:
1. The agent follows your modified intent
2. You see the priority message in console
3. The behavior matches your expectation

**This should now work exactly as you wanted!** 🚀
