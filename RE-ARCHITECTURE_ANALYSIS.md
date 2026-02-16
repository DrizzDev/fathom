# Fathom Re-Architecture Analysis & Plan

## 1. Root Cause Analysis: The Coordinate Displacement Issue

The critical failure in the previous re-architecture attempt was a breakdown in the **Logical-to-Physical coordinate mapping**.

### The Mechanism of Failure:
1.  **Scaling Mismatch**: The original `AndroidParser` was hardcoded to a scale factor of `1.0`. In high-resolution emulators (like the 1080x2340 used in tests), the XML logical coordinates (e.g., 0-1080) often don't match the screenshot's physical pixels if any resizing occurred.
2.  **Mapping Corruption**: I incorrectly stored **physical (scaled)** coordinates in the `label_map` during the grounding phase.
3.  **Circular Error**:
    *   `IntentStrategy` took those physical coordinates and normalized them to 0-1000.
    *   `ExecutionEngine` then took those 0-1000 values and converted them *back* to pixels using the device's width/height.
    *   This circular conversion (Physical -> Normalized -> Physical) with different scale assumptions introduced massive rounding errors and displacements.

### The Fix:
*   **Logical-Only Grounding**: The `label_map` must strictly store **logical** coordinates (as they appear in the XML).
*   **Proven Converters**: All pixel translation MUST be delegated to the existing `CoordinateConverter` utility, which is designed to handle the specific normalization logic Fathom's LLM expects.

---

## 2. Proposed Architectural Vision: Clean Hexagonal

The goal is to move from a Tool-based architecture to a **Port & Adapter** (Hexagonal) architecture.

### Core Principles:
1.  **Ports (Interfaces)**: Defined in `src/fathom/interfaces/`. These define *what* the system can do (e.g., `DevicePort.tap`, `LLMPort.generate`).
2.  **Adapters (Infrastructure)**: Defined in `src/fathom/adapters/`. These implement the ports (e.g., `ADBDevice`, `GeminiLLM`).
3.  **Core Services (Domain)**: Defined in `src/fathom/core/services/`. These contain the business logic (e.g., `VisionService`, `HierarchyService`).
4.  **Runtime (Wiring)**: Uses a `Builder` pattern to inject specific adapters into the engine.

---

## 3. Implementation Flow-Wise Plan

### Phase 1: The Foundation (Interfaces & Mapping)
*   **Consolidate DevicePort**: Merge perception (screenshot/XML) and action (tap/swipe) into a single `DevicePort`. Separation of these two causes unnecessary sync issues.
*   **Coordinate Integrity**: Restore `src/fathom/utils/coordinates.py` and ensure `ADBConfig` includes all required distance/duration fields.

### Phase 2: Perception & Grounding ("The Eyes")
*   **HierarchyService**: Migrates grounding logic. It must strictly populate the `label_map` with **logical** coordinates from the XML.
*   **AndroidParser Fix**: Implement dynamic scale factor calculation (`screenshot_width / xml_width`) to ensure annotations align perfectly with reality.

### Phase 3: Reasoning & Resolution ("The Brain")
*   **IntentStrategy**: Refactor to use the new `VisionService`.
*   **Coordinate Resolution**: Re-implement `__resolve_coordinates` to map numeric labels from the LLM back to the 0-1000 normalized scale using the logical grounding data.
*   **Async Tracing**: Move visual tracing (saving action screenshots) to a background task to prevent execution latency.

### Phase 4: Execution Engine ("The Hands")
*   **ExecutionEngine**: Orchestrates the 7-phase DAG (Signal -> Perceive -> Reason -> Act -> Learn -> Checkpoint -> Evaluate).
*   **Statelessness**: Ensure the engine holds no device state; it simply pipes data between ports.

---

## 4. Verification Strategy
To avoid "buggy code" during the transition:
1.  **Logic-First Move**: Copy the exact code from `tools/device/adb.py` to `adapters/device/adb.py`. No refactoring of internal method logic until the move is verified.
2.  **Unit Test Parity**: Ensure that `CoordinateConverter` tests pass with the same inputs before and after the move.
3.  **Visual Audit**: Every action must be verified against the `assets/traces/` output to ensure the "Red Circle" (tap location) matches the intended UI element.
