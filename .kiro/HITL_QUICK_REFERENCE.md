# HITL Quick Reference

## Start Interactive Mode

```bash
fathom run "Your intent" --interactive --serial emulator-5554
```

## Manual Pause (File-Based)

### Pause
```bash
touch .fathom_pause
```

### Inject Context
```bash
echo "Your context here" > .fathom_context
```

### Resume
```bash
touch .fathom_resume
```

## Complete Example

```bash
# Terminal 1: Start
fathom run "Login to app" -i -s emulator-5554

# Terminal 2: Control
touch .fathom_pause
echo "Use test@example.com and password123" > .fathom_context
touch .fathom_resume
```

## Automatic Pause

Agent automatically pauses when uncertain (confidence < 50%) and asks questions.

## Context Examples

```bash
# Credentials
echo "Email: test@example.com, Password: Test123!" > .fathom_context

# Navigation
echo "Settings is in the hamburger menu at top-left" > .fathom_context

# Instructions
echo "Skip all tutorial screens by clicking Skip button" > .fathom_context

# Corrections
echo "Don't click that button, use the one at the bottom" > .fathom_context
```

## Cleanup

```bash
rm -f .fathom_pause .fathom_context .fathom_resume
```

## Status

✅ Fully implemented and production-ready
✅ Works at ANY time during execution
✅ No terminal conflicts
✅ Context affects LLM reasoning
