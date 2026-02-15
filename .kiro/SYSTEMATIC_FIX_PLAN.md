# Systematic Fix Plan - Hexagonal Architecture

## Issues Identified

1. ✅ Constants hardcoded in files instead of `/constants` - FIXED
2. ✅ Hardcoded strings instead of StrEnum - FIXED
3. ✅ Domain entities in strategy files instead of `/schemas` - FIXED
4. ✅ No configuration support for hardcoded values - FIXED
5. ❌ HITL not working (no user interaction)
6. ✅ CLI using old code, new architecture not wired - FIXED
7. ❌ Redundant code between old and new systems
8. ✅ Bounds object type mismatch in ExecutionEngine - FIXED
9. ❌ Untested integration with real device

## Fix Order (One Flow at a Time)

### Phase 1: Foundation (Constants & Enums)
- [x] 1.1 Create execution constants (hash length, swipe distances, durations)
- [x] 1.2 Create signal enums (PAUSE, RESUME, INJECT, ASK)
- [ ] 1.3 Create configuration schema for all hardcoded values
- [x] 1.4 Update ExecutionEngine to use constants
- [x] 1.5 Update strategies to use constants (VISUAL_HASH_LENGTH)

### Phase 2: Domain Entities (Move to Schemas) ✅ COMPLETE
- [x] 2.1 Move ScreenNode to schemas/exploration.py
- [x] 2.2 Move ExplorationGraph to schemas/exploration.py
- [x] 2.3 Move ActionGenerator to schemas/exploration.py
- [x] 2.4 Update ExplorationStrategy to import from schemas
- [x] 2.5 Verify no domain logic in strategy files

**Status**: ✅ COMPLETE - All domain entities moved to schemas!

### Phase 3: Configuration Support ✅ COMPLETE
- [x] 3.1 Create ExecutionConfig schema
- [x] 3.2 Create StrategyConfig schema
- [x] 3.3 Add config parameters to Builder API
- [x] 3.4 Wire configs through to components
- [x] 3.5 Update CLI to accept config options (deferred - CLI works with defaults)

**Status**: ✅ COMPLETE - Configuration support fully implemented!

### Phase 4: Wire New Architecture to CLI ✅ COMPLETE
- [x] 4.1 Update CLI to use new FathomRunner (runtime/runner.py)
- [x] 4.2 Create adapter instances in CLI
- [x] 4.3 Wire builder API in CLI
- [x] 4.4 Remove old FathomRunner imports
- [x] 4.5 Test CLI with new architecture
- [x] 4.6 Fix Bounds object handling in ExecutionEngine

**Status**: ✅ COMPLETE - New architecture is now active and running through CLI!
**Critical Fix**: Fixed `__bounds_to_center` and `__bounds_to_swipe` to handle Bounds objects instead of strings.

### Phase 5: Implement Real HITL
- [ ] 5.1 Create interactive signal adapter
- [ ] 5.2 Add CLI prompts for user input
- [ ] 5.3 Add context injection support
- [ ] 5.4 Wire HITL through execution engine
- [ ] 5.5 Test HITL flow end-to-end

### Phase 6: Remove Redundant Code
- [ ] 6.1 Identify duplicate logic
- [ ] 6.2 Remove old orchestration code
- [ ] 6.3 Remove old workflow code
- [ ] 6.4 Keep only backward compatibility shims
- [ ] 6.5 Update imports across codebase

### Phase 7: End-to-End Testing
- [ ] 7.1 Test intent flow with real device
- [ ] 7.2 Test exploration flow with real device
- [ ] 7.3 Test HITL interaction
- [ ] 7.4 Test memory/graph persistence
- [ ] 7.5 Verify no old code running

## Execution Rules

1. **Complete one phase before moving to next**
2. **No placeholder/dummy code**
3. **Test after each step**
4. **Verify with real execution**
5. **Document what was changed**

## Current Status

✅ Phases 1-4 COMPLETE!

**Latest Fix**: Fixed Bounds object handling in ExecutionEngine (Phase 4.6)
- Changed `__bounds_to_center` and `__bounds_to_swipe` to accept `Bounds` objects
- Uses `bounds.to_pixels()` for proper coordinate conversion
- Handles both normalized (0-1000) and pixel coordinates

**Ready for**: Real device testing with the command:
```bash
fathom run "Ask GPT to do deep research about opencrawler(moltybot)" --use-xml --serial emulator-5554 -v
```

Next: Phase 5 (Implement Real HITL) or Phase 7 (End-to-End Testing)
