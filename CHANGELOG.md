# Changelog

All notable changes to Fathom are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Changes landing on the main branch that have not yet been cut into a release are listed
here.

## [1.1.0] - 2026-08-20

Initial public release under the Business Source License 1.1.

### Added

- Hexagonal agent core that turns a plain-language intent into grounded device actions on
  both Android and iOS, driven entirely by a vision-language model.
- Pluggable ports and adapters for device, model, and perception, wired through a
  `FathomBuilder` so any backend can be swapped without touching the core.
- Perception ensemble with opt-in OCR, icon, computer-vision, keyboard, and overlay passes,
  plus multi-member localization voting for more reliable grounding.
- Intent decomposition into ordered sub-goals and a LangGraph state machine with
  check-pointing, so a run can pause, resume, and replan mid-flight.
- Evidence-driven completion that advances only on observed success, command success, or
  captured proof, rather than on model self-reports.
- Human-in-the-loop support to ask the operator, pause, and resume through a signal port.
- Conversation and observability layer recording threads, messages, tasks, and scripts,
  with an in-memory default and an optional Postgres backend.
- Priority and adaptive inference that sheds load and recovers when the model provider
  slows down or starts rejecting calls.
- Typed `FathomConfiguration` surface plus environment overrides for every optional
  subsystem, documented in `.env.template`.

[Unreleased]: https://github.com/DrizzDev/fathom/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/DrizzDev/fathom/releases/tag/v1.1.0
