from __future__ import annotations

from typing import List, Tuple

from fathom.constants.flow import IssueCode
from fathom.constants.generation import ScriptSource, ScriptStatus
from fathom.core.dialect.policy import Policy
from fathom.core.exceptions import LanguageComplianceError
from fathom.core.services.generation.commands import ScriptCommandBuilder
from fathom.core.services.generation.projector import DeterministicFlowGenerator
from fathom.interfaces.dialect import Dialect
from fathom.schemas.flow import BranchNode, Evidence, FlowNode, Issue
from fathom.schemas.generation import (
    BaselineArtifact,
    ScriptFileMetadata,
    ScriptLineage,
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
        self.__commands = ScriptCommandBuilder(dialect=dialect)

    def build(self, *, evidence: Evidence) -> BaselineArtifact:
        """
        Project, gate on policy then syntax (both blocking), and render; never an empty success.
        """

        report = self.__generator.project(evidence=evidence)

        review = ScriptReview(
            partial=report.flow.partial,
            discarded=evidence.discarded,
            lineage=self.__lineage(nodes=report.flow.nodes),
            commands=self.__commands.build(flow=report.flow),
            reason=evidence.reason or (self.__PARTIAL_REASON if report.flow.partial else None),
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

    def __lineage(self, *, nodes: Tuple[FlowNode, ...]) -> Tuple[ScriptLineage, ...]:
        """
        Return evidence source steps for each flattened baseline node.
        """

        records: List[ScriptLineage] = []

        for position, node in enumerate(self.__flatten(nodes=nodes)):
            records.append(
                ScriptLineage(
                    node_index=position,
                    verified_by=("execution",),
                    source_steps=node.source_steps,
                )
            )

        return tuple(records)

    def __flatten(self, *, nodes: Tuple[FlowNode, ...]) -> Tuple[FlowNode, ...]:
        """
        Flatten top-level and branch-body nodes in rendered node order.
        """

        flattened: List[FlowNode] = []

        for node in nodes:
            flattened.append(node)
            if isinstance(node, BranchNode):
                flattened.extend(self.__flatten(nodes=node.body))

        return tuple(flattened)

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
