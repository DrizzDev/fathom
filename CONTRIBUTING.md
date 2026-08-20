# Contributing to Fathom

Thanks for your interest in improving Fathom. This guide covers how to build, test, and
submit changes.

## License and sign-off

Fathom is distributed under the Business Source License 1.1 (converting to Apache-2.0 on the
date in [LICENSE](LICENSE)). By contributing, you agree that your contribution is licensed
under those same terms.

Every commit must carry a Developer Certificate of Origin sign-off. Add one by committing
with `-s`:

```sh
git commit -s -m "Your message"
```

This appends a `Signed-off-by: Your Name <you@example.com>` line, which certifies that you
wrote the change or otherwise have the right to submit it under the project license (see
[DCO](DCO)). Pull requests whose commits are not signed off will not pass the sign-off check.

## Prerequisites

- **Python 3.11+**
- A vision-language model — a Gemini API key, or Vertex AI credentials
- For device work: Android `adb` / platform-tools, or Xcode and the iOS SDK (WebDriverAgent)

## Build and verify

```sh
git clone git@github.com:DrizzDev/fathom.git
cd fathom
pip install -e .
pip install -r requirements.txt
pre-commit install
```

CI runs `pre-commit` on every pull request. Run the same checks locally before opening one:

```sh
pre-commit run --all-files   # ruff, formatting, mypy, and repository checks
pytest                       # tests
```

Live tests (which call a real model) are opt-in and skipped by default; set
`FATHOM_RUN_LIVE_TESTS=1` to run them.

## Standards

This codebase follows the engineering standards and review checklist in [`AGENTS.md`](AGENTS.md).
In short:

- One primary responsibility per file; dependencies point toward the owner of policy.
- Boundaries use explicit, typed contracts; the hexagonal layering is enforced by the tests
  under `tests/architecture/`.
- Entities and anything crossing a boundary use Pydantic `BaseModel` — never `@dataclass`.
- Focused tests cover real behavior and failure paths, not mocks for their own sake.
- Documentation changes with public behavior, configuration, or architecture.
- Never commit secrets, real `.env` files, device serials, screenshots, or captured
  hierarchies.

## Submitting changes

1. Open a topic branch from the default branch.
2. Keep the change focused; add tests for new behavior and failure paths.
3. Sign off every commit (`git commit -s`).
4. Make sure `ruff`, `mypy`, and `pytest` pass.
5. Open a pull request and fill in the template, describing the motivation, the change, and
   any compatibility or migration considerations. A maintainer will review it; CI must pass
   before merge.

External contributors work from a fork: fork the repository, push your branch to your fork,
and open a pull request against `main`.

## Reporting bugs and requesting features

Use GitHub Issues, with clear reproduction steps and your environment (OS, Python version,
device/emulator). For security vulnerabilities, do **not** open a public issue — see
[SECURITY.md](SECURITY.md) and use private reporting instead.
