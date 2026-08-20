# Agent Instructions

These instructions are mandatory for every contributor and coding agent — human or model —
working in this repository. Read them in full before you change any tracked file, and never
rely on remembered context in their place.

Two documents carry the detailed rules, and both are binding. Read both before you design or
edit anything:

- [standards/engineering.md](standards/engineering.md) — the engineering standard: hexagonal
  architecture, naming, typing, data models, logging, and the P0 rules that must never be
  broken.
- [standards/review.md](standards/review.md) — the checklist every change is judged against
  before it merges, and how to run an independent review.

When this file and a linked standard disagree, the standard wins. When any rule is unclear or
a material decision is unresolved, stop and ask rather than guess.

## 1. Understand the work

Before changing tracked files:

1. read the request and identify the expected outcome;
2. inspect the current working tree;
3. read the files that own the affected behavior;
4. read the applicable standards and accepted decisions;
5. identify affected boundaries, state, failures, compatibility, security, and tests;
6. stop and ask when a material product or architecture decision is unresolved.

Never treat remembered context as repository evidence. Re-read relevant source after context
compaction, a handoff, a user correction, or a material scope change.

## 2. Design before editing

- Resolve every owner, boundary, dependency, state, failure, file, test, and verification
  decision before starting an implementation slice.
- Apply SOLID, separation of concerns, high cohesion, low coupling, dependency inversion, and
  composition over inheritance where applicable.
- Preserve the repository's hexagonal layering — the core is pure and depends only on typed
  ports; devices, models, and storage plug in as adapters. This direction is enforced by the
  tests under `tests/architecture/`.
- Give every module and type one clear responsibility.
- Use strict single-word names; represent multi-word concepts through meaningful nesting. A
  leading underscore marks non-public identifiers.
- Use `UPPER_SNAKE_CASE` only for internal semantic string values; preserve values defined by
  an external protocol or public contract.
- Use keyword arguments for project-owned calls that pass multiple values.
- Model entities, validated value objects, and anything crossing a boundary with Pydantic
  `BaseModel`. Never use `@dataclass`. Validate at the boundary; the domain may assume valid data.
- Do not introduce a dependency, abstraction, interface, or framework without a demonstrated
  requirement.
- Treat each slice as a fresh repository review — re-read the applicable standards and owning
  code instead of relying on conversation memory.

## 3. Make the smallest complete change

- Stay within the requested scope; preserve unrelated and pre-existing work.
- Keep domain policy out of adapters and infrastructure; keep constants, contracts, models,
  errors, ports, use cases, adapters, and validators in their owning layer.
- Use explicit, typed contracts at every boundary. Raw `dict`/`tuple` must not cross layers.
- Keep secrets, credentials, and captured user data inside their approved boundaries.
- Do not add speculative features or future-facing scaffolding.
- Keep each module focused on one responsibility rather than growing without bound.
- Do not write unnecessary comments; a justified technical comment is normally one line.

## 4. Verify with evidence

1. run the narrowest relevant checks while editing;
2. run the repository checks for the changed area — `ruff check .`, `ruff format --check .`,
   `mypy src`, and `pytest`;
3. test real integrations when the result depends on real provider behavior (live tests are
   opt-in: `FATHOM_RUN_LIVE_TESTS=1`);
4. inspect the fresh complete diff and working-tree status;
5. review the diff against the design decisions above.

Never label a failure pre-existing, flaky, or unrelated without reproduction and baseline
evidence. A required check that does not yet exist must be reported as unavailable, not
simulated through prose.

## 5. Report honestly

Report the outcome, the changed files, the decisions made, the checks run and their exact
scope, and any unresolved risks or blocked checks. Do not claim approval, completeness,
compliance, or production readiness that has not been demonstrated.
