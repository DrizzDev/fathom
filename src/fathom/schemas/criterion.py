from __future__ import annotations

from enum import StrEnum
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class CriterionVerdict(StrEnum):
    """
    Tri-state outcome of evaluating whether a sub-goal's criterion is
    observably satisfied on the current screen.

    ``UNCLEAR`` is a first-class state, not a degenerate ``UNSATISFIED``.
    Treating ambiguity as failure produces silent false negatives; the
    completion gate falls back to the implicit-completion streak guard
    only on ``UNCLEAR``, never on ``UNSATISFIED``.
    """

    UNCLEAR = "unclear"
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"


class CriterionSource(StrEnum):
    """
    Which checker layer produced the verdict.

    Carried on every decision so RCA can tell whether the gate trusted a
    cheap symbolic match, paid for an LLM call, or replayed a cached judgement.
    """

    LLM = "llm"
    CACHE = "cache"
    SYMBOLIC = "symbolic"


class CriterionDecision(BaseModel):
    """
    Outcome of a single criterion-satisfaction check with provenance.

    The verdict drives the completion gate; the source, confidence and
    evidence drive telemetry and post-incident review. ``evidence`` is
    intentionally short — matched element text or a one-line LLM
    rationale — to keep logs readable and avoid leaking screen content.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: CriterionVerdict = Field(
        description="Criterion satisfaction outcome.",
    )
    source: CriterionSource = Field(
        description=(
            "Layer that produced the verdict (symbolic match, LLM call, or cache replay)."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the verdict on [0, 1]. Symbolic matches use a "
            "deterministic hit-ratio score; LLM verdicts use a fixed "
            "per-outcome confidence pending structured-output upgrade."
        ),
    )
    evidence: Tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Short evidence snippets (matched token list, one-line LLM "
            "rationale) for telemetry. Bounded; never raw screen content."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-text annotation. Used to explain UNCLEAR "
            "verdicts (missing evidence, LLM error, empty criterion)."
        ),
    )
