from __future__ import annotations

import hashlib
from time import perf_counter
from typing import List, Tuple

from fathom.constants.assessment import VisualVerdict
from fathom.constants.success import SuccessKind
from fathom.constants.turn.advancement import AdvanceKind, ObservationPhase
from fathom.core.agent.candidate import ShadowCandidate
from fathom.core.agent.shadow import ShadowAssessor
from fathom.core.context.manager import ContextManager
from fathom.core.services.vision import SubGoalContext, VisionService
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.success import ObservedSuccess, Success
from fathom.schemas.target import TargetAuthority
from fathom.schemas.tools import AllowedTools
from tests.builders.success import SuccessFixtures
from tests.fixtures.vision.corpus import VisionCase
from tests.support.vision.models import CaseAttempt, CaseReport
from tests.support.vision.repository import VisionCaseRepository

STATIC_SCREEN = "static-corpus-screen"


class AssessmentEvaluator:
    """
    Runs each corpus case through the production single-call boundary and records complete per-attempt evidence.
    """

    def __init__(
        self,
        *,
        service: VisionService,
        repository: VisionCaseRepository,
        candidate: ShadowCandidate,
        tools: AllowedTools,
        memory: MemoryPort,
        reps: int,
    ) -> None:
        """
        Bind the production service, the fixture repository, the shared advancement candidate, and the run shape.
        """

        self.__service = service
        self.__repository = repository
        self.__candidate = candidate
        self.__assessor = ShadowAssessor()
        self.__tools = tools
        self.__memory = memory
        self.__reps = reps

    async def evaluate(self, *, cases: Tuple[VisionCase, ...]) -> Tuple[CaseReport, ...]:
        """
        Evaluate every case across all repetitions and fold the attempts into case reports.
        """

        return tuple([await self.__case(case=case) for case in cases])

    async def __case(self, *, case: VisionCase) -> CaseReport:
        """
        Run one case for every repetition and assemble its report.
        """

        success = self.__success(case=case)
        sub_goal = self.__sub_goal(case=case)
        capture = self.__capture(case=case)
        authority = (
            TargetAuthority.requested(package=case.authority_package)
            if case.authority_package is not None
            else TargetAuthority.unbound()
        )

        attempts: List[CaseAttempt] = []
        for rep in range(self.__reps):
            context = ContextManager(memory=self.__memory, workflow_id=f"assess-{case.name}-{rep}")
            started = perf_counter()
            analysis = await self.__service.analyze(
                intent=case.assertion,
                tools=self.__tools,
                capture=capture,
                context_manager=context,
                visual_hash=hashlib.sha256(f"{case.name}:{rep}".encode()).hexdigest()[:16],
                screen_width=capture.width,
                screen_height=capture.height,
                sub_goal_info=sub_goal,
            )
            latency_ms = (perf_counter() - started) * 1000.0
            attempts.append(
                self.__attempt(
                    case=case,
                    rep=rep,
                    analysis=analysis,
                    latency_ms=latency_ms,
                    success=success,
                    authority=authority,
                    foreground_package=capture.activity,
                )
            )

        return CaseReport.assemble(
            name=case.name,
            app=case.app,
            scenario=case.scenario,
            goal_kind=case.goal_kind,
            provenance=case.provenance,
            label_source=case.label_source,
            critical_negative=case.critical_negative,
            expected_verdict=case.expected_verdict,
            expected_admission=case.expected_admission,
            attempts=attempts,
        )

    def __attempt(
        self,
        *,
        case: VisionCase,
        rep: int,
        analysis: AnalysisResult,
        latency_ms: float,
        success: Success,
        authority: TargetAuthority,
        foreground_package: str,
    ) -> CaseAttempt:
        """
        Build the full record for one repetition, including the deterministic admission decision and labels.
        """

        assessment = analysis.visual_assessment
        malformed = analysis.assessment_malformed
        verdict = assessment.verdict if assessment is not None else None
        action = analysis.action
        action_present = action is not None

        candidate = self.__candidate.decide(
            success=success,
            phase=ObservationPhase.POST_DISPATCH,
            assessment=assessment,
            malformed=malformed,
            action_present=action_present,
            screen=STATIC_SCREEN,
            authority=authority,
            foreground=foreground_package,
            execution=None,
        )
        admitted = candidate.kind in (AdvanceKind.ADVANCE, AdvanceKind.SATISFIED_PRIOR)
        divergences = tuple(
            divergence.kind
            for divergence in self.__assessor.assess(
                success=success,
                assessment=assessment,
                action_present=action_present,
                assessment_malformed=malformed,
                authority=authority,
                foreground_package=foreground_package,
            )
        )
        satisfied = verdict is VisualVerdict.SATISFIED
        missing = isinstance(success, ObservedSuccess) and assessment is None and not malformed
        raw_false_positive = satisfied and not case.truth_satisfied
        raw_false_negative = case.truth_satisfied and not satisfied

        return CaseAttempt(
            rep=rep,
            assertion=case.assertion,
            expected_verdict=case.expected_verdict,
            expected_admission=case.expected_admission,
            verdict=verdict,
            confidence=assessment.confidence if assessment is not None else None,
            evidence=assessment.evidence if assessment is not None else None,
            action_type=action.action_type.value if action is not None else None,
            action_target=action.target if action is not None else None,
            action_present=action_present,
            foreground_package=foreground_package,
            authority_package=authority.package,
            divergences=divergences,
            schema_malformed=malformed,
            missing=missing,
            raw_false_positive=raw_false_positive,
            raw_false_negative=raw_false_negative,
            admitted=admitted,
            effective_false_positive=admitted and not case.expected_admission,
            effective_false_negative=not admitted and case.expected_admission,
            latency_ms=latency_ms,
        )

    @staticmethod
    def __success(*, case: VisionCase) -> Success:
        """
        Build the active goal's typed success from the case's declared goal kind.
        """

        if case.goal_kind is SuccessKind.OBSERVED:
            return SuccessFixtures.observed(assertion=case.assertion)
        if case.goal_kind is SuccessKind.COMMAND:
            return SuccessFixtures.command()
        return SuccessFixtures.capture()

    @staticmethod
    def __sub_goal(*, case: VisionCase) -> SubGoalContext:
        """
        Thread the assertion only for observed goals; command and capture goals request no assessment.
        """

        context = SubGoalContext(
            index=0,
            total=2,
            description=f"Reach the state: {case.assertion}",
            durable=case.goal_kind is not SuccessKind.OBSERVED,
        )
        if case.goal_kind is SuccessKind.OBSERVED:
            context["assertion"] = case.assertion
        return context

    def __capture(self, *, case: VisionCase) -> ScreenCapture:
        """
        Wrap the committed pixels with a truthful foreground package and real dimensions.
        """

        data = self.__repository.image_bytes(case=case)
        width, height = self.__repository.dimensions(case=case)
        return ScreenCapture(
            width=width,
            height=height,
            activity=case.package,
            image=data,
            timestamp=0,
        )
