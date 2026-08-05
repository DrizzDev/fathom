from __future__ import annotations

import os
from pathlib import Path

import pytest

from fathom.core.agent.candidate import ShadowCandidate
from fathom.core.agent.tools import DEFAULT_TOOL_POLICIES, ToolScope
from fathom.core.services.vision import VisionService
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.tools import ToolPolicyContext
from tests.fixtures.vision.corpus import CASES
from tests.live.core.services.test_vision import NoopTelemetry
from tests.support.vision.evaluator import AssessmentEvaluator
from tests.support.vision.models import EvaluationReport
from tests.support.vision.repository import ReportWriter, VisionCaseRepository

pytestmark = pytest.mark.release

REPORT_PATH = Path("debug/vision_assessment/report.json")
LATENCY_BUDGET_MS = 8000.0


class TestVisionAssessmentGate:
    """
    Drives the real model over the multi-app real-pixel corpus and evaluates the effective shadow admission rule.
    """

    async def test_effective_shadow_admission_has_no_false_positive_advancement(
        self,
        llm: LLMPort,
        memory_port_stub: MemoryPort,
    ) -> None:
        """
        The effective rule must not advance any negative case, still advance positives, and stay in latency budget.
        """

        capabilities = RuntimeCapabilities(hitl=HITLCapability(enabled=False))
        tools = ToolScope(policies=DEFAULT_TOOL_POLICIES).compute(
            context=ToolPolicyContext(capabilities=capabilities)
        )
        service = VisionService(
            llm=llm,
            memory=memory_port_stub,
            telemetry=NoopTelemetry(),
            use_cache=False,
            capabilities=capabilities,
            session_id="live-assessment",
        )
        evaluator = AssessmentEvaluator(
            service=service,
            repository=VisionCaseRepository(),
            candidate=ShadowCandidate(),
            tools=tools,
            memory=memory_port_stub,
            reps=int(os.getenv("FATHOM_ASSESS_REPS", "3")),
        )

        cases = await evaluator.evaluate(cases=CASES)
        report = EvaluationReport.assemble(
            model=llm.model_name,
            reps_per_case=int(os.getenv("FATHOM_ASSESS_REPS", "3")),
            latency_budget_ms=LATENCY_BUDGET_MS,
            cases=cases,
        )
        ReportWriter().write(report=report, path=REPORT_PATH)
        await llm.cleanup()

        self.__print(report=report)

        self.__assertions(report=report)

    @staticmethod
    def __print(*, report: EvaluationReport) -> None:
        """
        Emit a compact human-readable table alongside the persisted typed report.
        """

        print("\n=== EFFECTIVE SHADOW ADMISSION — real model, real-pixel corpus ===")
        print(
            f"model={report.model} reps/case={report.reps_per_case} apps={list(report.apps)} "
            f"cases={len(report.cases)} attempts={report.totals.attempts}"
        )
        print(
            f"raw FP/FN={report.totals.raw_false_positive}/{report.totals.raw_false_negative}  "
            f"effective FP/FN={report.totals.effective_false_positive}/{report.totals.effective_false_negative}  "
            f"missing={report.totals.missing} schema_fail={report.totals.schema_failures}  "
            f"latency p50/p95/max={report.latency.p50_ms}/{report.latency.p95_ms}/{report.latency.max_ms}ms"
        )
        for case in report.cases:
            print(
                f"{case.name:30} {case.app:12} exp_adv={str(case.expected_admission):5} "
                f"rawFP={case.raw_false_positive} effFP={case.effective_false_positive} "
                f"effFN={case.effective_false_negative} miss={case.missing} "
                f"verdicts={[attempt.verdict.value if attempt.verdict else 'MISSING' for attempt in case.attempts]}"
            )
        print(f"acceptance_passed={report.acceptance_passed} "
              f"critical_effFP={report.critical_effective_false_positive} "
              f"unresolved_rawFP={list(report.unresolved_raw_false_positive)}")

    def __assertions(self, *, report: EvaluationReport) -> None:
        """
        Enforce every cutover acceptance condition as hard assertions.
        """

        assert report.totals.missing == 0, "An observed goal produced no assessment; the boundary failed to decode."
        assert report.totals.schema_failures == 0, "An assessment failed its schema at the production boundary."
        assert report.totals.effective_false_positive == 0, (
            f"Effective false-positive advancement remains: {list(report.unresolved_raw_false_positive)}."
        )
        assert report.critical_effective_false_positive == 0, "A critical-negative case advanced under the shadow rule."
        assert not report.unresolved_raw_false_positive, (
            f"Raw false positives not neutralized by a deterministic veto: {list(report.unresolved_raw_false_positive)}."
        )
        positives_retained = [
            case.name
            for case in report.cases
            if case.expected_admission and case.effective_false_negative > 0
        ]
        assert not positives_retained, f"Positive cases failed to advance: {positives_retained}."
        assert report.latency.p95_ms <= report.latency_budget_ms, (
            f"p95 latency {report.latency.p95_ms}ms exceeds budget {report.latency_budget_ms}ms."
        )
        assert report.acceptance_passed, "Cutover acceptance did not hold."
