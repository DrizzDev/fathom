from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from fathom.constants.authoring import AuthoringKind
from fathom.constants.flow import IssueCode
from fathom.core.dialect.policy import Policy
from fathom.core.exceptions import LanguageComplianceError
from fathom.interfaces.dialect import Dialect
from fathom.schemas.authoring import AuthoringTask
from fathom.schemas.flow import (
    BranchNode,
    CheckNode,
    Evidence,
    EvidenceStep,
    Flow,
    FlowNode,
    Issue,
    ScrollUntilNode,
    StoreNode,
    TapNode,
    TypeNode,
)
from fathom.schemas.generation import ScriptLineage


class AuthoringReview:
    """
    Deterministic review result for an authored flow.
    """

    def __init__(
        self,
        *,
        text: str,
        issues: Tuple[Issue, ...],
        advisories: Tuple[Issue, ...] = (),
        lineage: Tuple[ScriptLineage, ...] = (),
    ) -> None:
        """
        Bind review issues, advisories, lineage, and rendered script text.
        """

        self.__text = text
        self.__issues = issues
        self.__advisories = advisories
        self.__lineage_records = lineage

    @property
    def issues(self) -> Tuple[Issue, ...]:
        """
        Return deterministic review issues.
        """

        return self.__issues

    @property
    def text(self) -> str:
        """
        Return rendered script text when render succeeded.
        """

        return self.__text

    @property
    def advisories(self) -> Tuple[Issue, ...]:
        """
        Return non-blocking authoring quality notes.
        """

        return self.__advisories

    @property
    def lineage(self) -> Tuple[ScriptLineage, ...]:
        """
        Return per-node evidence provenance records.
        """

        return self.__lineage_records

    @property
    def accepted(self) -> bool:
        """
        Return whether the reviewed flow can be published.
        """

        return not self.__issues and bool(self.__text)


class AuthoringReviewer:
    """
    Reviews authored flows with hard-truth policy and dialect checks.
    """

    def __init__(self, *, policy: Policy, dialect: Dialect) -> None:
        """
        Bind deterministic policy and target dialect.
        """

        self.__policy = policy
        self.__dialect = dialect

    def review(self, *, task: AuthoringTask, flow: Flow) -> AuthoringReview:
        """
        Render, syntax-check, and policy-review an authored flow.
        """

        issues: Tuple[Issue, ...] = ()
        evidence = self.__evidence(task=task)

        if task.kind is AuthoringKind.RUN and evidence is not None:
            issues = self.__policy.evaluate(flow=flow, evidence=evidence).issues

        try:
            text = self.__dialect.renderer.render(flow=flow).strip()
        except LanguageComplianceError as exception:
            return AuthoringReview(
                text="",
                issues=issues + (self.__render_issue(exception=exception),),
            )

        syntax = self.__dialect.checker.check(text=text)
        return AuthoringReview(
            text=text,
            issues=issues + syntax.issues,
            lineage=self.__lineage(flow=flow, evidence=evidence),
            advisories=self.__advisories(flow=flow, evidence=evidence),
        )

    def __advisories(self, *, flow: Flow, evidence: Optional[Evidence]) -> Tuple[Issue, ...]:
        """
        Return non-blocking quality notes derived from evidence provenance.
        """

        if evidence is None:
            return ()

        index = {step.index: step for step in evidence.steps if step.launch is None}
        advisories: List[Issue] = []

        for position, node in enumerate(self.__flatten(nodes=flow.nodes)):
            if self.__uses_unconfirmed_claim(node=node, index=index):
                advisories.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.UNCONFIRMED_TARGET_CLAIM,
                        message=(
                            f"Node {position} uses a planner target claim that was not confirmed "
                            "by an available evidence anchor; prefer screen or artifact evidence."
                        ),
                    )
                )

        return tuple(advisories)

    def __lineage(self, *, flow: Flow, evidence: Optional[Evidence]) -> Tuple[ScriptLineage, ...]:
        """
        Return per-node evidence provenance labels for metadata.
        """

        index = {} if evidence is None else {step.index: step for step in evidence.steps}
        records: List[ScriptLineage] = []

        for position, node in enumerate(self.__flatten(nodes=flow.nodes)):
            steps = tuple(step for step in node.source_steps if step in index)
            records.append(
                ScriptLineage(
                    source_steps=steps,
                    node_index=position,
                    verified_by=self.__verified_by(node=node, index=index),
                    screen_authored=self.__uses_unconfirmed_claim(node=node, index=index),
                )
            )

        return tuple(records)

    def __verified_by(self, *, node: FlowNode, index: Dict[int, EvidenceStep]) -> Tuple[str, ...]:
        """
        Return evidence channel labels supporting a node.
        """

        labels: List[str] = []

        if any(step in index for step in node.source_steps):
            labels.append("execution")

        if isinstance(node, CheckNode) and node.assertion_ids:
            labels.append("completion_assertion")

        for source in node.source_steps:
            step = index.get(source)
            if step is None:
                continue

            if step.capture is not None and isinstance(node, StoreNode):
                labels.append("capture")

            if step.target.anchors.visual:
                labels.append("visual")

            if step.target.anchors.accessibility:
                labels.append("accessibility")

        return tuple(dict.fromkeys(labels))

    def __flatten(self, *, nodes: Tuple[FlowNode, ...]) -> Tuple[FlowNode, ...]:
        """
        Flatten top-level and branch-body nodes for review.
        """

        flattened: List[FlowNode] = []

        for node in nodes:
            flattened.append(node)
            if isinstance(node, BranchNode):
                flattened.extend(self.__flatten(nodes=node.body))

        return tuple(flattened)

    @classmethod
    def __uses_unconfirmed_claim(cls, *, node: FlowNode, index: Dict[int, EvidenceStep]) -> bool:
        """
        Return whether a node target exactly uses an unconfirmed planner target claim.
        """

        for source in node.source_steps:
            step = index.get(source)
            if step is not None and cls.__matches_claim(node=node, step=step):
                return True

        return False

    @staticmethod
    def __matches_claim(*, node: FlowNode, step: EvidenceStep) -> bool:
        """
        Return whether node text equals the step's unconfirmed target claim.
        """

        claim = step.target.claim.text
        if claim is None or step.target.claim.verified:
            return False

        if isinstance(node, TapNode):
            return node.selector.text == claim

        if isinstance(node, TypeNode):
            return node.field.text == claim

        if isinstance(node, ScrollUntilNode):
            return node.target == claim

        if isinstance(node, CheckNode):
            return any(check.subject == claim for check in node.checks)

        return False

    @staticmethod
    def __render_issue(*, exception: LanguageComplianceError) -> Issue:
        """
        Convert an unrenderable flow into deterministic review feedback.
        """

        return Issue(
            code=IssueCode.UNRENDERABLE_VALUE,
            message=f"Flow could not be rendered: {exception}",
        )

    @staticmethod
    def __evidence(*, task: AuthoringTask) -> Optional[Evidence]:
        """
        Return normalized execution evidence from the task view.
        """

        if task.evidence.run is not None:
            return task.evidence.run.source

        if task.evidence.step is not None:
            return task.evidence.step.source

        if task.evidence.repair is not None:
            return task.evidence.repair.source

        return None
