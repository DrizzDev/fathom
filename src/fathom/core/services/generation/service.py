from __future__ import annotations

import time
from logging import getLogger
from typing import Tuple

from fathom.constants.flow import IssueCode, Limit
from fathom.core.dialect.policy import Policy
from fathom.core.exceptions import LanguageComplianceError
from fathom.core.services.generation.binder import LaunchBinder
from fathom.interfaces.dialect import Dialect
from fathom.interfaces.evidence import EvidenceSource
from fathom.interfaces.generator import FlowGenerator
from fathom.schemas.flow import Evidence, Flow, Issue, LaunchNode, RunObjective
from fathom.schemas.generation import GenerationResult, ScriptReview

logger = getLogger(__name__)


class ScriptGenerationService:
    """
    Generates a validated Drizz script from a run's evidence, repairing within a bounded budget.
    """

    def __init__(
        self,
        *,
        policy: Policy,
        dialect: Dialect,
        binder: LaunchBinder,
        evidence: EvidenceSource,
        generator: FlowGenerator,
    ) -> None:
        """
        Bind the evidence source, generator, launch binder, policy, and dialect.
        """

        self.__policy = policy
        self.__binder = binder
        self.__dialect = dialect
        self.__evidence = evidence
        self.__generator = generator

    async def generate(self, *, run: str, objective: RunObjective) -> GenerationResult:
        """
        Read evidence, generate and gate a flow with bounded repair, returning the script text.
        """

        feedback: Tuple[Issue, ...] = ()
        evidence = await self.__evidence.read(run=run, objective=objective)

        started = time.perf_counter()
        logger.info(
            "script quality generation started",
            extra={
                "event": "script.quality.started",
                "workflow.id": run,
                "script.source": "quality",
                "script.partial": evidence.partial,
                "script.package": objective.package,
                "script.step_count": len(evidence.steps),
            },
        )

        for attempt in range(1, Limit.MAX_REPAIR_ATTEMPTS + 2):
            logger.info(
                "script quality attempt started",
                extra={
                    "event": "script.quality.attempt.started",
                    "workflow.id": run,
                    "script.attempt": attempt,
                    "script.issue_codes": [issue.code.value for issue in feedback],
                },
            )

            generated = await self.__generator.generate(evidence=evidence, feedback=feedback)
            self.__log_generated_flow(run=run, attempt=attempt, flow=generated)

            flow = self.__binder.bind(flow=generated, evidence=evidence)
            self.__log_binder(run=run, flow=flow, evidence=evidence)

            issues, text = self.__gate(run=run, flow=flow, evidence=evidence)

            if not issues:
                logger.info(
                    "script quality generation succeeded",
                    extra={
                        "event": "script.quality.generated",
                        "workflow.id": run,
                        "script.source": "quality",
                        "script.attempt": attempt,
                        "script.partial": evidence.partial,
                        "script.line_count": len(text.splitlines()),
                        "script.discarded_count": len(evidence.discarded),
                        "duration.ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                )
                return GenerationResult(
                    text=text,
                    attempts=attempt,
                    review=ScriptReview(
                        reason=evidence.reason,
                        partial=evidence.partial,
                        discarded=evidence.discarded,
                    ),
                )

            logger.info(
                "script quality attempt failed the gate",
                extra={
                    "event": "script.quality.attempt.failed",
                    "workflow.id": run,
                    "script.attempt": attempt,
                    "script.issue_count": len(issues),
                    "script.issue_codes": [issue.code.value for issue in issues],
                },
            )
            feedback = issues

        logger.warning(
            "script quality generation exhausted the repair budget",
            extra={
                "event": "script.quality.failed",
                "workflow.id": run,
                "script.source": "quality",
                "script.issue_count": len(feedback),
                "script.issue_codes": [issue.code.value for issue in feedback],
                "duration.ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )
        raise LanguageComplianceError(self.__failure(feedback=feedback))

    def __log_generated_flow(self, *, run: str, attempt: int, flow: Flow) -> None:
        """
        Record the model's flow shape before binding and gating.
        """

        logger.info(
            "script quality attempt produced a flow",
            extra={
                "event": "script.quality.attempt.generated_flow",
                "workflow.id": run,
                "script.attempt": attempt,
                "script.node_count": len(flow.nodes),
                "script.launch_count": sum(
                    1 for node in flow.nodes if isinstance(node, LaunchNode)
                ),
            },
        )

    def __log_binder(self, *, run: str, flow: Flow, evidence: Evidence) -> None:
        """
        Record how many launch nodes the binder stamped from evidence markers.
        """

        markers = sum(1 for step in evidence.steps if step.launch is not None)
        launches = sum(1 for node in flow.nodes if isinstance(node, LaunchNode))
        stamped = min(markers, launches)

        logger.info(
            "script launch binder applied",
            extra={
                "event": "script.binder.applied",
                "workflow.id": run,
                "script.marker_count": markers,
                "script.launch_count": launches,
                "script.stamped_count": stamped,
                "script.untouched_count": launches - stamped,
            },
        )

    def __gate(self, *, run: str, flow: Flow, evidence: Evidence) -> Tuple[Tuple[Issue, ...], str]:
        """
        Run the fidelity gate then the rendered-syntax gate, returning all issues and the text.
        """

        fidelity = self.__policy.evaluate(flow=flow, evidence=evidence)
        self.__log_policy(run=run, issues=fidelity.issues)

        logger.info(
            "script render started",
            extra={
                "event": "script.render.started",
                "workflow.id": run,
                "script.source": "quality",
                "script.node_count": len(flow.nodes),
            },
        )
        try:
            text = self.__dialect.renderer.render(flow=flow)
        except LanguageComplianceError as exception:
            logger.warning(
                "script render failed",
                extra={
                    "event": "script.render.failed",
                    "workflow.id": run,
                    "script.source": "quality",
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return fidelity.issues + (self.__unrenderable(exception=exception),), ""

        logger.info(
            "script render generated text",
            extra={
                "event": "script.render.generated",
                "workflow.id": run,
                "script.source": "quality",
                "script.line_count": len(text.splitlines()),
            },
        )

        syntax = self.__dialect.checker.check(text=text)
        self.__log_check(run=run, issues=syntax.issues)

        return fidelity.issues + syntax.issues, text

    def __log_policy(self, *, run: str, issues: Tuple[Issue, ...]) -> None:
        """
        Record the fidelity-gate verdict with issue codes.
        """

        if issues:
            logger.warning(
                "script policy gate failed",
                extra={
                    "event": "script.policy.failed",
                    "workflow.id": run,
                    "script.source": "quality",
                    "script.issue_count": len(issues),
                    "script.issue_codes": [issue.code.value for issue in issues],
                },
            )
            return

        logger.info(
            "script policy gate passed",
            extra={
                "event": "script.policy.passed",
                "workflow.id": run,
                "script.source": "quality",
            },
        )

    def __log_check(self, *, run: str, issues: Tuple[Issue, ...]) -> None:
        """
        Record the syntax-checker verdict, carrying the line-level diagnostic message on failure.
        """

        if issues:
            logger.warning(
                "script syntax check failed",
                extra={
                    "event": "script.check.failed",
                    "workflow.id": run,
                    "script.source": "quality",
                    "script.issue_codes": [issue.code.value for issue in issues],
                    "script.issue_messages": [issue.message for issue in issues],
                },
            )
            return

        logger.info(
            "script syntax check passed",
            extra={
                "event": "script.check.passed",
                "workflow.id": run,
                "script.source": "quality",
            },
        )

    def __unrenderable(self, *, exception: LanguageComplianceError) -> Issue:
        """
        Turn a renderer compliance failure into a repairable issue.
        """

        return Issue(code=IssueCode.UNRENDERABLE_VALUE, message=str(exception))

    def __failure(self, *, feedback: Tuple[Issue, ...]) -> str:
        """
        Build an actionable message listing the issues that survived the repair budget.
        """

        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in feedback)
        return f"Script generation failed after {Limit.MAX_REPAIR_ATTEMPTS} repairs: {detail}"
