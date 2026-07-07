from __future__ import annotations

from typing import Dict, List, Tuple

from fathom.constants.flow import CheckKind, LaunchProvenance
from fathom.schemas.flow import (
    BranchNode,
    Check,
    CheckNode,
    CompletionAssertion,
    Evidence,
    Flow,
    FlowNode,
    LaunchNode,
)


class AuthoringMaterializer:
    """
    Fills evidence-owned bookkeeping on authored flows before deterministic review.
    """

    def materialize(self, *, flow: Flow, evidence: Evidence) -> Flow:
        """
        Return a flow whose internal provenance metadata is derived from evidence.
        """

        nodes = self.__nodes(nodes=flow.nodes, evidence=evidence)
        return flow.model_copy(update={"nodes": nodes})

    def __nodes(self, *, nodes: Tuple[FlowNode, ...], evidence: Evidence) -> Tuple[FlowNode, ...]:
        """
        Materialize each node while preserving authored user-facing text.
        """

        return tuple(self.__node(node=node, evidence=evidence) for node in nodes)

    def __node(self, *, node: FlowNode, evidence: Evidence) -> FlowNode:
        """
        Materialize one node using deterministic evidence facts.
        """

        if isinstance(node, LaunchNode):
            return self.__launch(node=node, evidence=evidence)

        if isinstance(node, CheckNode):
            return self.__check(node=node, evidence=evidence)

        if isinstance(node, BranchNode):
            return node.model_copy(
                update={"body": self.__nodes(nodes=node.body, evidence=evidence)}
            )

        return node

    def __launch(self, *, node: LaunchNode, evidence: Evidence) -> LaunchNode:
        """
        Attach normalized launcher source steps when package and provenance already match.
        """

        if node.source_steps:
            return node

        if node.provenance != LaunchProvenance.LAUNCHER_TRANSITION:
            return node

        matches = [
            step.launch
            for step in evidence.steps
            if step.launch is not None
            and step.launch.package == node.package
            and step.launch.provenance is node.provenance
        ]

        if len(matches) != 1:
            return node

        return node.model_copy(update={"source_steps": matches[0].source_steps})

    def __check(self, *, node: CheckNode, evidence: Evidence) -> CheckNode:
        """
        Attach completion assertion identifiers and source steps when assertions match the node.
        """

        if not evidence.assertions:
            return node

        checks = self.__check_keys(checks=node.checks)
        assertions = {
            (assertion.kind, assertion.subject): assertion for assertion in evidence.assertions
        }

        if not checks or any(key not in assertions for key in checks):
            return node

        assertion_ids = tuple(assertions[key].id for key in checks)
        source_steps = self.__assertion_source_steps(
            assertions=tuple(assertions[key] for key in checks),
            evidence=evidence,
        )
        updates: Dict[str, object] = {}

        if not node.assertion_ids:
            updates["assertion_ids"] = assertion_ids

        if source_steps and node.source_steps != source_steps:
            updates["source_steps"] = source_steps

        if not updates:
            return node

        return node.model_copy(update=updates)

    @staticmethod
    def __check_keys(*, checks: Tuple[Check, ...]) -> Tuple[Tuple[CheckKind, str], ...]:
        """
        Return stable assertion matching keys for a check group.
        """

        return tuple((check.kind, check.subject) for check in checks)

    @staticmethod
    def __assertion_source_steps(
        *,
        assertions: Tuple[CompletionAssertion, ...],
        evidence: Evidence,
    ) -> Tuple[int, ...]:
        """
        Return valid source steps for matched completion assertions.
        """

        steps: List[int] = []
        valid = {step.index for step in evidence.steps if step.launch is None}
        fallback = evidence.steps[-1].index if evidence.steps else None

        for assertion in assertions:
            step = assertion.step_index if assertion.step_index in valid else fallback
            if step is not None and step not in steps:
                steps.append(step)

        return tuple(steps)
