from __future__ import annotations

from typing import Dict, List, Optional, Tuple, cast

from fathom.constants.flow import CheckKind
from fathom.schemas.flow import (
    BranchNode,
    Check,
    CheckNode,
    CompletionAssertion,
    Evidence,
    EvidenceStep,
    Flow,
    FlowNode,
    Guard,
    LaunchNode,
    LeafNode,
    StepLaunch,
)


class LaunchEvidenceCursor:
    """
    Provides ordered launch markers for authored launch materialization.
    """

    def __init__(self, *, evidence: Evidence) -> None:
        """
        Bind the ordered launch markers from normalized evidence.
        """

        self.__position = 0
        self.__markers = tuple(step.launch for step in evidence.steps if step.launch is not None)

    def match(self, *, node: LaunchNode) -> Optional[StepLaunch]:
        """
        Return the next marker when it matches the authored launch package.
        """

        if self.__position >= len(self.__markers):
            return None

        marker = self.__markers[self.__position]

        if marker.package != node.package:
            return None

        self.__position += 1
        return marker


class AuthoringMaterializer:
    """
    Fills evidence-owned bookkeeping on authored flows before deterministic review.
    """

    def materialize(self, *, flow: Flow, evidence: Evidence) -> Flow:
        """
        Return a flow whose internal provenance metadata is derived from evidence.
        """

        cursor = LaunchEvidenceCursor(evidence=evidence)
        nodes = self.__nodes(nodes=flow.nodes, evidence=evidence, launches=cursor)

        return flow.model_copy(update={"nodes": nodes})

    def __nodes(
        self,
        *,
        evidence: Evidence,
        nodes: Tuple[FlowNode, ...],
        launches: LaunchEvidenceCursor,
    ) -> Tuple[FlowNode, ...]:
        """
        Materialize each node while preserving authored user-facing text.
        """

        materialized = tuple(
            self.__node(node=node, evidence=evidence, launches=launches) for node in nodes
        )
        return self.__guarded(nodes=materialized, evidence=evidence)

    def __node(
        self, *, node: FlowNode, evidence: Evidence, launches: LaunchEvidenceCursor
    ) -> FlowNode:
        """
        Materialize one node using deterministic evidence facts.
        """

        if isinstance(node, LaunchNode):
            return self.__launch(node=node, launches=launches)

        if isinstance(node, CheckNode):
            return self.__check(node=node, evidence=evidence)

        if isinstance(node, BranchNode):
            return node.model_copy(
                update={"body": self.__nodes(nodes=node.body, evidence=evidence, launches=launches)}
            )

        return node

    def __guarded(self, *, nodes: Tuple[FlowNode, ...], evidence: Evidence) -> Tuple[FlowNode, ...]:
        """
        Return nodes with missing IF wrappers restored from conditional evidence.
        """

        output: List[FlowNode] = []
        branch: Optional[Guard] = None
        body: List[LeafNode] = []

        for node in nodes:
            guard = self.__guard(node=node, evidence=evidence)
            if guard is None:
                self.__flush(output=output, guard=branch, body=body)
                branch = None
                body = []
                output.append(node)
                continue

            if branch is not None and branch != guard:
                self.__flush(output=output, guard=branch, body=body)
                body = []

            branch = guard
            body.append(cast("LeafNode", node))

        self.__flush(output=output, guard=branch, body=body)
        return tuple(output)

    @staticmethod
    def __flush(
        *,
        output: List[FlowNode],
        guard: Optional[Guard],
        body: List[LeafNode],
    ) -> None:
        """
        Append the pending guarded branch when present.
        """

        if guard is None or not body:
            return

        source_steps: List[int] = []
        for node in body:
            for step in node.source_steps:
                if step not in source_steps:
                    source_steps.append(step)

        output.append(
            BranchNode(
                source_steps=tuple(source_steps),
                guard=guard,
                body=tuple(body),
            )
        )

    def __guard(self, *, node: FlowNode, evidence: Evidence) -> Optional[Guard]:
        """
        Return the recorded branch guard required by a node, if any.
        """

        if isinstance(node, (BranchNode, LaunchNode)):
            return None

        guarded: List[Tuple[str, int]] = []
        for index in node.source_steps:
            step = self.__step(evidence=evidence, index=index)
            if step is None or not step.guard.conditional:
                continue

            condition = step.guard.condition
            if condition:
                guarded.append((condition, step.index))

        if not guarded:
            return None

        conditions = set(guarded)
        if len(conditions) != 1:
            return None

        condition, source_step = next(iter(conditions))
        return Guard(condition=condition, source_step=source_step)

    @staticmethod
    def __step(*, evidence: Evidence, index: int) -> Optional[EvidenceStep]:
        """
        Return the evidence step for an index, if present.
        """

        return next((step for step in evidence.steps if step.index == index), None)

    @staticmethod
    def __launch(*, node: LaunchNode, launches: LaunchEvidenceCursor) -> LaunchNode:
        """
        Attach normalized launch provenance from the corresponding ordered marker.
        """

        marker = launches.match(node=node)

        if marker is None:
            return node

        if node.provenance == marker.provenance and node.source_steps == marker.source_steps:
            return node

        return node.model_copy(
            update={
                "provenance": marker.provenance,
                "source_steps": marker.source_steps,
            }
        )

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
            evidence=evidence,
            assertions=tuple(assertions[key] for key in checks),
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
