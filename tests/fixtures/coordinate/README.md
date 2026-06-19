# Coordinate-dispatch fixtures

Pinned screen captures + element hierarchies that exercise
:class:`fathom.utils.coordinates.CoordinateConverter` against real
production failure modes. The cases primarily pin the iOS retina
coordinate-space mismatch (XML logical -> drawer scaled to
device-pixel -> converter previously clamped against logical screen,
yielding a garbage dispatched tap coordinate).

## Layout

```
coordinate/
├── ios/                          # platform tier
│   ├── tap/                      # action-type tier
│   │   ├── 001/                  # numbered case
│   │   │   ├── case.yaml         # self-describing metadata
│   │   │   ├── hierarchy.xml     # captured XML hierarchy
│   │   │   └── screenshot.png    # captured screenshot
│   │   ├── 002/
│   │   └── ...
│   ├── swipe/                    # reserved
│   └── type/                     # reserved
├── android/
│   ├── tap/
│   ├── swipe/
│   └── type/
├── output/                       # renderer writes annotated + traced PNGs here (gitignored)
└── README.md
```

## Adding a new case

1. Pick the platform tier (`ios/` or `android/`).
2. Pick the action-type tier (`tap/`, `swipe/`, `type/`, ...).
3. Create the next numbered directory (e.g. `005/`).
4. Drop `case.yaml`, `hierarchy.xml`, `screenshot.png` inside.

The test loader walks every numbered directory under each action-type
tier; no Python edit is required to register a new case.
