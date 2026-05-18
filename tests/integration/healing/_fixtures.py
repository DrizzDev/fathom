from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field


class TerminalStatus(StrEnum):
    """
    Terminal run statuses accepted by a Phase 12 replay fixture.

    Only three terminations are valid: ``SUCCEEDED`` (the agent
    completed the intent), ``BOUNDED_FAILURE`` (healing or supervision
    bounded the run and it terminated cleanly), and ``ESCALATED`` (the
    run was routed to the human via ASK_USER).
    """

    ESCALATED = "ESCALATED"
    SUCCEEDED = "SUCCEEDED"
    BOUNDED_FAILURE = "BOUNDED_FAILURE"


class FixtureStep(BaseModel):
    """
    One replay step.

    Holds the frozen LLM output the harness will replay at this step
    and the relative paths to the screenshot and XML manifest that
    seeded the original capture.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0, description="Zero-based step index in the replay sequence.")
    intent_segment: str = Field(description="Single-line description of the intent for this step.")
    model_output: Dict[str, Any] = Field(description="Frozen LLM tool call replayed at this step.")
    frame: str = Field(description="Relative path to the step screenshot inside the fixture dir.")
    manifest: str = Field(
        description="Relative path to the step XML manifest inside the fixture dir."
    )


class FixtureExpectation(BaseModel):
    """
    Pinned acceptance contract for one replay fixture.

    Each field encodes one runtime invariant the replay harness must
    assert: the terminal status, hard caps on step count and consecutive
    no-effect blocks, and the supervision/recovery names that must
    appear (or stay absent) during the run. ``raw_llm_coordinates_executed``
    is the legacy-regression pin and stays at zero for every new fixture.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    terminal_status: TerminalStatus = Field(description="Required run termination status.")
    max_step_count: int = Field(ge=1, description="Hard cap on steps the run may execute.")
    max_repeated_no_effect: int = Field(
        ge=0,
        description="Cap on consecutive REPEATED_NO_EFFECT records before bounded failure.",
    )
    block_reasons: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Supervision block reasons that must appear at least once.",
    )
    recoveries_invoked: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Recovery strategy names that must dispatch at least once.",
    )
    raw_llm_coordinates_executed: int = Field(
        ge=0,
        description="Pin against the legacy regression; always zero for new fixtures.",
    )


class FixtureTrace(BaseModel):
    """
    Typed, hermetic record of one healing-runtime replay fixture.

    Returned by :meth:`FixtureLoader.load`. The harness consumes the
    ``steps`` to drive the IntentNodeProvider stub through the
    GROUND→VERIFY cycle and asserts the run output against
    :attr:`expected`. ``directory`` is included so step frame and
    manifest paths can be resolved relative to it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Scenario identifier (directory name under fixtures/healing).")
    intent: str = Field(description="Single-line run intent loaded from intent.txt.")
    steps: Tuple[FixtureStep, ...] = Field(
        description="Ordered replay steps loaded from steps.json."
    )
    expected: FixtureExpectation = Field(
        description="Pinned acceptance contract loaded from expected.json."
    )
    directory: Path = Field(description="Absolute path of the fixture directory.")


class FixtureLoader:
    """
    Hermetic loader for Phase 12 healing-runtime replay fixtures.

    Each scenario directory under ``tests/fixtures/healing/`` follows
    the layout documented in the fixtures README — ``intent.txt``,
    ``steps.json``, ``expected.json``, plus ``frames/`` and
    ``manifests/`` subdirectories. The loader resolves the directory,
    validates the JSON payloads against the typed Pydantic models, and
    returns a frozen :class:`FixtureTrace`.
    """

    __FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "healing"

    @classmethod
    def root(cls) -> Path:
        """
        Return the absolute path to the healing-fixtures root directory.
        """

        return cls.__FIXTURES_ROOT

    @classmethod
    def scenarios(cls) -> Tuple[str, ...]:
        """
        Return the alphabetically-sorted list of available scenario names.
        """

        return tuple(
            sorted(entry.name for entry in cls.__FIXTURES_ROOT.iterdir() if entry.is_dir()),
        )

    @classmethod
    def load(cls, *, name: str) -> FixtureTrace:
        """
        Load and validate one named replay fixture into a typed FixtureTrace.
        """

        directory = cls.__FIXTURES_ROOT / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Healing fixture '{name}' not found at {directory!s}")

        intent = (directory / "intent.txt").read_text(encoding="utf-8").strip()
        raw_steps = json.loads((directory / "steps.json").read_text(encoding="utf-8"))
        raw_expected = json.loads((directory / "expected.json").read_text(encoding="utf-8"))

        steps: List[FixtureStep] = [FixtureStep.model_validate(entry) for entry in raw_steps]
        expectation = FixtureExpectation.model_validate(raw_expected)

        return FixtureTrace(
            name=name,
            intent=intent,
            steps=tuple(steps),
            expected=expectation,
            directory=directory,
        )
