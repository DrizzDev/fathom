from __future__ import annotations

from typing import Tuple

from fathom.constants.flow import IssueCode
from fathom.constants.generation import ScriptSource, ScriptStatus
from fathom.core.dialect.policy import Policy
from fathom.core.exceptions import LanguageComplianceError
from fathom.core.services.generation.projector import DeterministicFlowGenerator
from fathom.interfaces.dialect import Dialect
from fathom.schemas.flow import Evidence, Issue
from fathom.schemas.generation import (
    BaselineArtifact,
    ScriptFileMetadata,
    ScriptReview,
    SkippedStep,
)


class BaselineScriptService:
    """
    Produces a deterministic baseline script from evidence, gated by policy and syntax: faithful or loud fail.
    """

    __PARTIAL_REASON = "No scriptable grounded terminal validation was recorded."

    def __init__(
        self,
        *,
        policy: Policy,
        dialect: Dialect,
        generator: DeterministicFlowGenerator,
    ) -> None:
        """
        Bind the deterministic projector, the fidelity policy, and the dialect renderer/checker.
        """

        self.__policy = policy
        self.__dialect = dialect
        self.__generator = generator

    def build(self, *, evidence: Evidence) -> BaselineArtifact:
        """
        Project, gate on policy then syntax (both blocking), and render; never an empty success.
        """

        report = self.__generator.project(evidence=evidence)

        review = ScriptReview(
            partial=report.flow.partial,
            reason=evidence.reason or (self.__PARTIAL_REASON if report.flow.partial else None),
            discarded=evidence.discarded,
        )

        policy_evidence = evidence.model_copy(update={"partial": report.flow.partial})
        fidelity = self.__policy.evaluate(flow=report.flow, evidence=policy_evidence)

        if fidelity.issues:
            return self.__failed(issues=fidelity.issues, review=review, skipped=report.skipped)

        try:
            text = self.__dialect.renderer.render(flow=report.flow)
        except LanguageComplianceError as exception:
            issue = Issue(code=IssueCode.UNRENDERABLE_VALUE, message=str(exception))
            return self.__failed(issues=(issue,), review=review, skipped=report.skipped)

        if not text.strip():
            issue = Issue(
                code=IssueCode.EMPTY_SCRIPT,
                message="No evidence step could be scripted; the baseline flow is empty.",
            )
            return self.__failed(issues=(issue,), review=review, skipped=report.skipped)

        syntax = self.__dialect.checker.check(text=text)
        if syntax.issues:
            return self.__failed(issues=syntax.issues, review=review, skipped=report.skipped)

        metadata = ScriptFileMetadata(
            review=review,
            skipped=report.skipped,
            source=ScriptSource.BASELINE,
            status=ScriptStatus.GENERATED,
        )
        return BaselineArtifact(text=text, metadata=metadata)

    @staticmethod
    def __failed(
        *,
        review: ScriptReview,
        issues: Tuple[Issue, ...],
        skipped: Tuple[SkippedStep, ...],
    ) -> BaselineArtifact:
        """
        Build a failed baseline artifact carrying the blocking issues and projection diagnostics.
        """

        metadata = ScriptFileMetadata(
            issues=issues,
            review=review,
            skipped=skipped,
            status=ScriptStatus.FAILED,
            source=ScriptSource.BASELINE,
        )
        return BaselineArtifact(text=None, metadata=metadata)
