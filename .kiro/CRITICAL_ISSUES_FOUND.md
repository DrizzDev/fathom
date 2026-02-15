# Critical Issues Found During Testing

## Issues Identified

### 1. ❌ Ctrl+P Delay
**Problem**: Significant delay between pressing Ctrl+P and pause happening
**Root Cause**: 
- Keyboard listener checks every 0.1 seconds
- Signal check only happens between steps
- If LLM is analyzing (takes 3-10 seconds), pause won't happen until analysis completes

**Impact**: User presses Ctrl+P but nothing happens for several seconds

### 2. ❌ Context Input Not Working
**Problem**: Menu appeared but couldn't enter context
**Root Cause**: Terminal settings conflict with Rich library's Prompt.ask()

**Impact**: User can't inject context even when paused

### 3. ❌ Metrics Validation Error
**Problem**: `ValidationError: metrics should be Dict[str, Dict[str, float]] but got Dict[str, int]`
**Root Cause**: Metrics from ExecutionMetrics.to_dict() returns flat integers, but IntentResult expects nested dictionaries

**Impact**: Execution crashes at the end with validation error

### 4. ❌ Cancellation Not Working
**Problem**: Pressing Ctrl+C doesn't stop execution immediately
**Root Cause**: Signal handler not properly integrated with async execution

**Impact**: User has to wait or force kill the process

## Root Cause Analysis

### Keyboard Listener Issues

The current implementation has fundamental problems:

1. **Terminal Mode Conflicts**:
   ```python
   tty.setcbreak(fd)  # Sets terminal to cbreak mode
   # But Rich library also manipulates terminal
   # This causes conflicts
   ```

2. **Blocking During LLM Calls**:
   ```python
   # Signal check happens here
   signal = await self.__signal.check_signal()
   
   # But then LLM analysis takes 3-10 seconds
   plan = await self.__perform_analysis(...)  # BLOCKS HERE
   
   # User presses Ctrl+P during this time
   # But check_signal() won't be called again until analysis completes
   ```

3. **Input Conflicts**:
   ```python
   # Keyboard listener is reading stdin
   char = sys.stdin.read(1)
   
   # But Rich Prompt also tries to read stdin
   answer = Prompt.ask("Your answer")
   
   # They conflict with each other
   ```

## Recommended Solutions

### Solution 1: Simpler Approach (Recommended)

**Don't use keyboard listener at all**. Instead:

1. **Use threading.Event for pause requests**
2. **Check pause flag more frequently** (every 0.5s during LLM calls)
3. **Use a separate thread to listen for input** without terminal manipulation

### Solution 2: Alternative Approach

**Use signal-based pause** (Unix signals):
- Send SIGUSR1 to pause
- Cleaner, no terminal conflicts
- But requires user to know process ID

### Solution 3: File-Based Pause

**Watch a file for pause requests**:
- User creates `.pause` file to pause
- Agent checks file existence
- Simple, no conflicts
- But less user-friendly

## Immediate Fixes Needed

### Fix 1: Metrics Format (CRITICAL - Blocks Execution)

```python
# In IntentStrategy.get_progress()
def get_progress(self) -> Dict[str, Any]:
    metrics_dict = self.__metrics.to_dict()
    
    # Convert flat metrics to nested format
    formatted_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            formatted_metrics[key] = {
                "total": float(value),
                "avg": 0.0,
                "count": 0
            }
        else:
            formatted_metrics[key] = value
    
    return {
        "intent": self.__intent,
        "step_count": self.__state.step_count,
        "is_complete": self.__state.is_complete,
        "context": self.__context.get_full_context(),
        "metrics": formatted_metrics,  # Use formatted metrics
    }
```

### Fix 2: Remove Keyboard Listener (CRITICAL - Causes Conflicts)

The keyboard listener approach has too many issues. We should:

1. Remove `keyboard_listener.py`
2. Remove keyboard listener from `InteractiveSignal`
3. Keep only automatic pause (agent questions)
4. Document that manual pause requires a different approach

### Fix 3: Better Cancellation

```python
# In CLI, handle Ctrl+C properly
try:
    result = await self.runner.run_intent(...)
except KeyboardInterrupt:
    console.print("\n[yellow]Cancelling...[/yellow]")
    if self.runner:
        self.runner.cancel()
    # Give it 2 seconds to cleanup
    await asyncio.sleep(2)
    raise
```

## Decision Required

We have two options:

### Option A: Remove Manual Pause (Recommended)
- Remove keyboard listener
- Keep automatic pause (agent questions)
- Fix metrics issue
- Document limitation
- **Pros**: Stable, no conflicts, works reliably
- **Cons**: No manual pause capability

### Option B: Implement File-Based Pause
- Remove keyboard listener
- Watch for `.pause` file
- User creates file to pause
- **Pros**: No terminal conflicts, works
- **Cons**: Less user-friendly

### Option C: Implement Signal-Based Pause
- Remove keyboard listener
- Use SIGUSR1 signal
- User sends signal to pause
- **Pros**: Clean, Unix-standard
- **Cons**: Requires knowing PID, Unix-only

## Recommendation

**Go with Option A** for now:
1. Remove keyboard listener (it's causing more problems than it solves)
2. Fix metrics issue (critical blocker)
3. Keep automatic pause (works well)
4. Document that manual pause will be added in future with better implementation

The automatic pause (agent asks questions when uncertain) is working well and provides the core HITL functionality. Manual pause can be added later with a better implementation that doesn't conflict with terminal I/O.
