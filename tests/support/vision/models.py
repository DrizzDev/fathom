from __future__ import annotations

from enum import StrEnum
from math import ceil
from typing import Optional, Sequence, Tuple

from pydantic import Field

from fathom.constants.assessment import ShadowDivergenceKind, VisualVerdict
from fathom.constants.success import SuccessKind
from fathom.schemas.base.common import NonBlank, SealedModel


class LabelSource(StrEnum):
    """
    Whether a case's ground-truth label was assigned by human inspection or produced programmatically.
    """

    HUMAN = "HUMAN"
    PROGRAMMATIC = "PROGRAMMATIC"


class CaseAttempt(SealedModel):
    """
    One live repetition of one case: the model output, the deterministic shadow admission, and its labels.
    """

    rep: int = Field(ge=0, description="Zero-based repetition index for this case.")
    assertion: NonBlank = Field(description="Exact assertion threaded into the prompt this attempt.")
    expected_verdict: VisualVerdict = Field(description="Oracle verdict for an observed goal.")
    expected_admission: bool = Field(description="Whether the effective rule should advance this attempt.")
    verdict: Optional[VisualVerdict] = Field(default=None, description="Model verdict, or None when absent.")
    confidence: Optional[float] = Field(default=None, description="Model confidence; recorded for calibration only.")
    evidence: Optional[str] = Field(default=None, description="Concise visible evidence the model cited.")
    action_type: Optional[str] = Field(default=None, description="Proposed action type, or None when absent.")
    action_target: Optional[str] = Field(default=None, description="Proposed action target, or None when absent.")
    action_present: bool = Field(description="Whether the model proposed an action on the same response.")
    foreground_package: Optional[str] = Field(default=None, description="Foreground package on the captured screen.")
    authority_package: Optional[str] = Field(default=None, description="Bound TargetAuthority package, or None.")
    divergences: Tuple[ShadowDivergenceKind, ...] = Field(
        default=(), description="Every deterministic shadow divergence recorded for this attempt."
    )
    schema_malformed: bool = Field(default=False, description="Whether the assessment payload failed its schema.")
    missing: bool = Field(default=False, description="Whether an observed goal produced no assessment at all.")
    raw_false_positive: bool = Field(description="Verdict SATISFIED while the oracle says not satisfied.")
    raw_false_negative: bool = Field(description="Verdict withheld while the oracle says satisfied.")
    admitted: bool = Field(description="Whether the effective shadow rule would advance the goal this attempt.")
    effective_false_positive: bool = Field(description="Admitted advance while the oracle says not satisfied.")
    effective_false_negative: bool = Field(description="Not admitted while the oracle says satisfied.")
    latency_ms: float = Field(ge=0.0, description="Wall-clock latency of the single production call.")


class LatencySummary(SealedModel):
    """
    Distribution of single-call latencies in milliseconds.
    """

    samples: int = Field(ge=0, description="Number of latency samples.")
    min_ms: float = Field(ge=0.0, description="Fastest observed call.")
    p50_ms: float = Field(ge=0.0, description="Median call latency.")
    p95_ms: float = Field(ge=0.0, description="95th-percentile call latency.")
    max_ms: float = Field(ge=0.0, description="Slowest observed call.")

    @classmethod
    def from_samples(cls, *, samples: Sequence[float]) -> "LatencySummary":
        """
        Build a summary using nearest-rank percentiles; an empty input yields all zeros.
        """

        if not samples:
            return cls(samples=0, min_ms=0.0, p50_ms=0.0, p95_ms=0.0, max_ms=0.0)

        ordered = sorted(samples)
        count = len(ordered)
        return cls(
            samples=count,
            min_ms=round(ordered[0], 1),
            p50_ms=round(ordered[cls.__rank(count=count, quantile=0.50)], 1),
            p95_ms=round(ordered[cls.__rank(count=count, quantile=0.95)], 1),
            max_ms=round(ordered[-1], 1),
        )

    @staticmethod
    def __rank(*, count: int, quantile: float) -> int:
        """
        Return the nearest-rank index into an ascending sequence for a quantile.
        """

        return min(count - 1, max(0, ceil(quantile * count) - 1))


class CaseReport(SealedModel):
    """
    Aggregated outcome for one case across all repetitions.
    """

    name: NonBlank = Field(description="Case identifier.")
    app: NonBlank = Field(description="Application exercised by the screen.")
    scenario: NonBlank = Field(description="Screen type / adversarial trap this case represents.")
    goal_kind: SuccessKind = Field(description="Success kind of the active goal.")
    provenance: NonBlank = Field(description="Source of the pixels (trace run path or a controlled tag).")
    label_source: LabelSource = Field(description="How this case's ground-truth was labeled.")
    critical_negative: bool = Field(description="Whether an effective false positive here blocks cutover.")
    expected_verdict: VisualVerdict = Field(description="Oracle verdict.")
    expected_admission: bool = Field(description="Whether the effective rule should advance this case.")
    attempts: Tuple[CaseAttempt, ...] = Field(description="Every repetition's full record.")
    latency: LatencySummary = Field(description="Latency distribution across repetitions.")
    raw_false_positive: int = Field(ge=0, description="Repetitions with a raw verdict false positive.")
    raw_false_negative: int = Field(ge=0, description="Repetitions with a raw verdict false negative.")
    effective_false_positive: int = Field(ge=0, description="Repetitions admitted to advance against the oracle.")
    effective_false_negative: int = Field(ge=0, description="Repetitions wrongly not admitted for a satisfied goal.")
    missing: int = Field(ge=0, description="Repetitions with no assessment for an observed goal.")
    schema_failures: int = Field(ge=0, description="Repetitions whose assessment failed its schema.")

    @classmethod
    def assemble(
        cls,
        *,
        name: str,
        app: str,
        scenario: str,
        goal_kind: SuccessKind,
        provenance: str,
        label_source: LabelSource,
        critical_negative: bool,
        expected_verdict: VisualVerdict,
        expected_admission: bool,
        attempts: Sequence[CaseAttempt],
    ) -> "CaseReport":
        """
        Fold the per-attempt records into a single case report.
        """

        return cls(
            name=name,
            app=app,
            scenario=scenario,
            goal_kind=goal_kind,
            provenance=provenance,
            label_source=label_source,
            critical_negative=critical_negative,
            expected_verdict=expected_verdict,
            expected_admission=expected_admission,
            attempts=tuple(attempts),
            latency=LatencySummary.from_samples(samples=[attempt.latency_ms for attempt in attempts]),
            raw_false_positive=sum(1 for attempt in attempts if attempt.raw_false_positive),
            raw_false_negative=sum(1 for attempt in attempts if attempt.raw_false_negative),
            effective_false_positive=sum(1 for attempt in attempts if attempt.effective_false_positive),
            effective_false_negative=sum(1 for attempt in attempts if attempt.effective_false_negative),
            missing=sum(1 for attempt in attempts if attempt.missing),
            schema_failures=sum(1 for attempt in attempts if attempt.schema_malformed),
        )


class Totals(SealedModel):
    """
    Corpus-wide counts across every case and repetition.
    """

    attempts: int = Field(ge=0, description="Total repetitions executed.")
    raw_false_positive: int = Field(ge=0, description="Total raw verdict false positives.")
    raw_false_negative: int = Field(ge=0, description="Total raw verdict false negatives.")
    effective_false_positive: int = Field(ge=0, description="Total effective false-positive advancements.")
    effective_false_negative: int = Field(ge=0, description="Total effective false-negative retentions.")
    missing: int = Field(ge=0, description="Total observed-goal calls with no assessment.")
    schema_failures: int = Field(ge=0, description="Total assessments that failed their schema.")

    @classmethod
    def from_cases(cls, *, cases: Sequence[CaseReport]) -> "Totals":
        """
        Sum the case reports into corpus-wide totals.
        """

        return cls(
            attempts=sum(len(case.attempts) for case in cases),
            raw_false_positive=sum(case.raw_false_positive for case in cases),
            raw_false_negative=sum(case.raw_false_negative for case in cases),
            effective_false_positive=sum(case.effective_false_positive for case in cases),
            effective_false_negative=sum(case.effective_false_negative for case in cases),
            missing=sum(case.missing for case in cases),
            schema_failures=sum(case.schema_failures for case in cases),
        )


class EvaluationReport(SealedModel):
    """
    The complete shadow-gate evidence: totals, latency, per-case reports, and the acceptance decision.
    """

    model: NonBlank = Field(description="Model identifier used for the run.")
    reps_per_case: int = Field(gt=0, description="Repetitions executed per case.")
    apps: Tuple[str, ...] = Field(description="Distinct applications covered by the corpus.")
    provenances: Tuple[str, ...] = Field(description="Distinct pixel sources covered by the corpus.")
    totals: Totals = Field(description="Corpus-wide counts.")
    latency: LatencySummary = Field(description="Latency distribution across all calls.")
    latency_budget_ms: float = Field(gt=0.0, description="The p95 latency budget the gate is held to.")
    critical_effective_false_positive: int = Field(
        ge=0, description="Effective false-positive advancements on critical-negative cases."
    )
    unresolved_raw_false_positive: Tuple[str, ...] = Field(
        description="Case names whose raw false positive was not neutralized by a deterministic veto."
    )
    acceptance_passed: bool = Field(description="Whether every cutover acceptance condition held.")
    cases: Tuple[CaseReport, ...] = Field(description="Every case's aggregated report.")

    @classmethod
    def assemble(
        cls,
        *,
        model: str,
        reps_per_case: int,
        latency_budget_ms: float,
        cases: Sequence[CaseReport],
    ) -> "EvaluationReport":
        """
        Assemble the corpus report and evaluate cutover acceptance from the case records.
        """

        totals = Totals.from_cases(cases=cases)
        latency = LatencySummary.from_samples(
            samples=[attempt.latency_ms for case in cases for attempt in case.attempts]
        )
        critical_fp = sum(
            case.effective_false_positive for case in cases if case.critical_negative
        )
        unresolved = tuple(
            case.name
            for case in cases
            if case.raw_false_positive > 0 and case.effective_false_positive > 0
        )
        positives_advance = all(
            case.effective_false_negative == 0 for case in cases if case.expected_admission
        )
        acceptance = (
            critical_fp == 0
            and totals.effective_false_positive == 0
            and not unresolved
            and positives_advance
            and totals.missing == 0
            and totals.schema_failures == 0
            and latency.p95_ms <= latency_budget_ms
        )
        return cls(
            model=model,
            reps_per_case=reps_per_case,
            apps=tuple(sorted({case.app for case in cases})),
            provenances=tuple(sorted({case.provenance for case in cases})),
            totals=totals,
            latency=latency,
            latency_budget_ms=latency_budget_ms,
            critical_effective_false_positive=critical_fp,
            unresolved_raw_false_positive=unresolved,
            acceptance_passed=acceptance,
            cases=tuple(cases),
        )
