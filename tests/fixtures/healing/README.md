# Healing Replay Fixtures

Replay fixtures used by the Phase 12 healing-runtime acceptance tests.

Each fixture pins one named regression observed in production. Phase 12
(audit entries 036-046) populates the per-scenario directories. This
file is the authoritative manifest for what each fixture covers and how
its files are laid out, so the fixtures themselves remain replayable
without re-deriving intent from source logs.

## Source Material

Raw run dumps from previous executions live under `logs/` (gitignored).
The replay fixtures here are derived from those dumps but committed and
deterministic — fixtures must never depend on `logs/` at test time.

Mapping from raw log to fixture directory:

| Fixture id | Raw log             | Scenario pinned                              |
|------------|---------------------|----------------------------------------------|
| `001`      | n/a (synthetic)     | yVKnb-style coachmark two-screen oscillation |
| `002`      | `logs/healing.txt`  | Cosmetic-replan counter loss on scroll task (Swiggy search log) |
| `003`      | `logs/healing__2.txt` | Pixel-scrim overlay with no manifest dialog |
| `004`      | `logs/healing__1.txt` | Overlay-dismiss thrash before heart-tap (Swiggy favourites log) |
| `005`      | `logs/3.txt`        | Scroll loop crossing per-task attempt budget |

The yVKnb scenario has no log dump; it is reproduced from the
oscillation pattern already pinned in
`tests/unit/schemas/test_loop_detector_fixtures.py`.

## Per-Scenario Layout

Every scenario directory uses the same file layout so the replay loader
is one implementation, not five:

```text
tests/fixtures/healing/<fixture-id>/
    intent.txt          Single-line natural-language intent (the run goal).
    steps.json          Ordered list of replayed steps (schema below).
    frames/             PNG screenshot per step (`step_<index>.png`).
    manifests/          XML manifest per step (`step_<index>.xml`).
    expected.json       Expected runtime outcome (schema below).
```

Optional files:

```text
    dimensions.json     Screen pixel dimensions; defaults to the first
                        frame's PNG header when absent.
    overlay_mask/       Pre-computed scrim mask PNGs for pixel-overlay
                        scenarios only.
    ocr/                Pre-computed Document AI responses (JSON) for
                        OCR-bound scenarios; replayed verbatim so tests
                        never call the real provider.
```

### `steps.json` Schema

A frozen list of replay steps. The replay harness substitutes the
named adapters with deterministic stubs that read these entries.

```json
[
  {
    "index": 0,
    "intent_segment": "Open Swiggy app",
    "model_output": {
      "tool": "execute_ui",
      "arguments": { "action": { "type": "TAP", "label_id": "L3" }, "task_status": "PARTIAL" }
    },
    "frame": "frames/step_0.png",
    "manifest": "manifests/step_0.xml"
  }
]
```

### `expected.json` Schema

The pinned acceptance contract for the replay. Each field is asserted
by the Phase 12 test that loads the fixture.

```json
{
  "terminal_status": "SUCCEEDED|BOUNDED_FAILURE|ESCALATED",
  "max_step_count": 14,
  "max_repeated_no_effect": 2,
  "block_reasons": ["OVERLAY_BLOCKS"],
  "recoveries_invoked": ["OverlayRecovery"],
  "raw_llm_coordinates_executed": 0
}
```

Field meanings:

- `terminal_status` — one of the three accepted run terminations.
- `max_step_count` — hard cap on total steps executed; the run must
  terminate at or before this count.
- `max_repeated_no_effect` — hard cap on consecutive
  `BlockReason.REPEATED_NO_EFFECT` records before bounded failure.
- `block_reasons` — supervision block reasons that MUST appear at
  least once during the run.
- `recoveries_invoked` — recovery strategy names that MUST run at
  least once during the run.
- `raw_llm_coordinates_executed` — pin against legacy regression;
  always 0 for new fixtures.

## Determinism Contract

Replay fixtures must be:

- Hermetic — no network, no real device, no real model call. All
  external boundaries (LLM, OCR, ensemble localizer, device
  executor) are bound to stub adapters that read from the fixture.
- Stable — once committed, a fixture must not change in lockstep
  with implementation. Schema-additive changes are allowed; pinned
  fields are not loosened without a separate audit entry.
- Self-contained — every byte the replay harness needs lives in
  the scenario directory. No cross-scenario imports.

## Phase 0 Status

This README is the only file in `tests/fixtures/healing/` at the end of
audit entry 021. Per-scenario directories ship under audit entries
036-046, gated by the dependencies named in the completion plan
(OCR adapter for `overlay_scrim`, ensemble localizer for `overlay_thrash`,
pixel-overlay detector for `overlay_scrim`).
