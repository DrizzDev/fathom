from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, cast

from fathom.constants import GESTURE_SCROLL_DIRECTION, ActionType
from fathom.constants.flow import CheckKind, EvidenceMarker
from fathom.constants.generation import SkipReason
from fathom.interfaces.generator import FlowGenerator
from fathom.schemas.flow import (
    BackNode,
    BranchNode,
    Check,
    CheckNode,
    Evidence,
    EvidenceStep,
    Flow,
    FlowNode,
    Guard,
    Issue,
    LaunchNode,
    LeafNode,
    ScrollNode,
    ScrollUntilNode,
    Selector,
    StoreNode,
    TapNode,
    TypeNode,
    WaitNode,
)
from fathom.schemas.generation import ProjectionReport, SkippedStep

Projected = Union[LeafNode, SkipReason]
ScrollLeaf = Union[ScrollNode, ScrollUntilNode]


class DeterministicFlowGenerator(FlowGenerator):
    """
    Projects recorded evidence into a faithful Flow by fixed per-step rules, with no LLM or judgment.
    """

    __VALIDATION: str = "validation"

    async def generate(self, *, evidence: Evidence, feedback: Tuple[Issue, ...] = ()) -> Flow:
        """
        Project evidence into a Flow; feedback is ignored because projection is deterministic.
        """

        _ = feedback
        return self.project(evidence=evidence).flow

    def project(self, *, evidence: Evidence) -> ProjectionReport:
        """
        Project evidence into a Flow plus the steps that could not be faithfully scripted.
        """

        nodes: List[FlowNode] = []
        skipped: List[SkippedStep] = []

        body: List[LeafNode] = []
        condition: Optional[str] = None

        guard_step = 0

        for step in evidence.steps:
            outcome = self.__leaf(step=step)

            if isinstance(outcome, SkipReason):
                skipped.append(SkippedStep(index=step.index, action=step.action, reason=outcome))
                continue

            guard = self.__guard_condition(step=step)

            if guard is None:
                nodes.extend(self.__flush(condition=condition, step=guard_step, body=body))
                body, condition = [], None
                nodes.append(outcome)
                continue

            if body and guard != condition:
                nodes.extend(self.__flush(condition=condition, step=guard_step, body=body))
                body = []

            if not body:
                condition, guard_step = guard, step.index

            body.append(outcome)

        nodes.extend(self.__flush(condition=condition, step=guard_step, body=body))
        nodes = self.__merge_scroll_attempts(nodes=nodes, evidence=evidence)
        nodes = self.__apply_completion_assertions(nodes=nodes, evidence=evidence)

        flow = Flow(
            intent=evidence.intent,
            partial=evidence.partial
            or self.__has_blocking_skip(skipped=tuple(skipped))
            or self.__has_unconfirmed_target(nodes=tuple(nodes), evidence=evidence)
            or not self.__has_terminal_validation(nodes=tuple(nodes), evidence=evidence),
            package=evidence.package,
            nodes=tuple(nodes),
        )
        return ProjectionReport(flow=flow, skipped=tuple(skipped))

    @staticmethod
    def __has_blocking_skip(*, skipped: Tuple[SkippedStep, ...]) -> bool:
        """
        Return whether projection skipped evidence that makes the script partial.
        """

        return any(step.reason is SkipReason.MISSING_TARGET for step in skipped)

    @classmethod
    def __has_unconfirmed_target(cls, *, nodes: Tuple[FlowNode, ...], evidence: Evidence) -> bool:
        """
        Return whether a baseline node had to use an unconfirmed planner target claim.
        """

        index = {step.index: step for step in evidence.steps if step.launch is None}

        for node in nodes:
            for source in node.source_steps:
                step = index.get(source)
                if step is not None and cls.__uses_unconfirmed_claim(node=node, step=step):
                    return True

        return False

    @staticmethod
    def __uses_unconfirmed_claim(*, node: FlowNode, step: EvidenceStep) -> bool:
        """
        Return whether a node target equals an unconfirmed planner claim.
        """

        claim = step.target.claim.text
        if claim is None or step.target.claim.verified:
            return False

        if isinstance(node, (TapNode,)):
            return node.selector.text == claim

        if isinstance(node, TypeNode):
            return node.field.text == claim

        if isinstance(node, ScrollUntilNode):
            return node.target == claim

        if isinstance(node, CheckNode):
            return any(check.subject == claim for check in node.checks)

        return False

    @classmethod
    def __apply_completion_assertions(
        cls, *, nodes: List[FlowNode], evidence: Evidence
    ) -> List[FlowNode]:
        """
        End completed evidence with the verifier assertions when they are available.
        """

        if not evidence.assertions or not evidence.steps:
            return nodes

        terminal = CheckNode(
            source_steps=cls.__assertion_source_steps(evidence=evidence),
            assertion_ids=tuple(assertion.id for assertion in evidence.assertions),
            checks=tuple(
                Check(kind=assertion.kind, subject=assertion.subject)
                for assertion in evidence.assertions
            ),
        )

        if nodes and isinstance(nodes[-1], CheckNode):
            return [*nodes[:-1], terminal]

        return [*nodes, terminal]

    @staticmethod
    def __assertion_source_steps(*, evidence: Evidence) -> Tuple[int, ...]:
        """
        Return valid provenance steps for verifier assertions.
        """

        steps: List[int] = []
        valid = {step.index for step in evidence.steps if step.launch is None}
        fallback = evidence.steps[-1].index

        for assertion in evidence.assertions:
            step = assertion.step_index if assertion.step_index in valid else fallback
            if step not in steps:
                steps.append(step)

        return tuple(steps)

    @classmethod
    def __merge_scroll_attempts(
        cls, *, nodes: List[FlowNode], evidence: Evidence
    ) -> List[FlowNode]:
        """
        Coalesce consecutive scroll attempts from the same episode into one replay command.
        """

        merged: List[FlowNode] = []
        index = {step.index: step for step in evidence.steps if step.launch is None}

        for node in nodes:
            if not merged or not cls.__same_scroll_episode(
                previous=merged[-1], current=node, index=index
            ):
                merged.append(node)
                continue

            current = cast("ScrollLeaf", node)
            previous = cast("ScrollLeaf", merged[-1])

            merged[-1] = ScrollNode(
                direction=previous.direction,
                source_steps=cls.__combined_sources(
                    previous=previous.source_steps, current=current.source_steps
                ),
            )

        return merged

    @staticmethod
    def __combined_sources(
        *, previous: Tuple[int, ...], current: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Combine source steps in order without duplicates.
        """

        combined: List[int] = []

        for step in (*previous, *current):
            if step not in combined:
                combined.append(step)

        return tuple(combined)

    @staticmethod
    def __same_scroll_episode(
        *, previous: FlowNode, current: FlowNode, index: Dict[int, EvidenceStep]
    ) -> bool:
        """
        Return whether two scroll nodes are consecutive attempts for one recorded goal.
        """

        if not isinstance(previous, (ScrollNode, ScrollUntilNode)):
            return False

        if not isinstance(current, (ScrollNode, ScrollUntilNode)):
            return False

        if previous.direction != current.direction:
            return False

        previous_goals = {
            recorded.goal.index
            for step in previous.source_steps
            if (recorded := index.get(step)) is not None and recorded.goal is not None
        }
        current_goals = {
            recorded.goal.index
            for step in current.source_steps
            if (recorded := index.get(step)) is not None and recorded.goal is not None
        }
        return bool(previous_goals & current_goals)

    @classmethod
    def __has_terminal_validation(cls, *, nodes: Tuple[FlowNode, ...], evidence: Evidence) -> bool:
        """
        Return whether the flow ends in a check grounded in a successful validation step.
        """

        if not nodes or not isinstance(nodes[-1], CheckNode):
            return False

        validations = {
            step.index
            for step in evidence.steps
            if step.event == cls.__VALIDATION and step.outcome.success
        }
        if bool(set(nodes[-1].source_steps) & validations):
            return True

        assertion_ids = getattr(nodes[-1], "assertion_ids", ())
        known = {assertion.id for assertion in evidence.assertions}

        return bool(assertion_ids) and set(assertion_ids).issubset(known)

    @staticmethod
    def __flush(*, condition: Optional[str], step: int, body: List[LeafNode]) -> List[FlowNode]:
        """
        Emit a single branch wrapping the accumulated conditional leaves, or nothing when empty.
        """

        if not body:
            return []

        return [
            BranchNode(
                source_steps=(step,),
                body=tuple(body),
                guard=Guard(condition=condition or "", source_step=step),
            )
        ]

    def __leaf(self, *, step: EvidenceStep) -> Projected:
        """
        Project one evidence step into its leaf node, or a skip reason when it must not be scripted.
        """

        if step.launch is not None:
            return LaunchNode(
                package=step.launch.package,
                provenance=step.launch.provenance,
                source_steps=step.launch.source_steps,
            )

        if step.guard.condition == EvidenceMarker.RECOVERY:
            return SkipReason.RECOVERY

        if not step.outcome.success:
            return SkipReason.FAILED

        if step.event == self.__VALIDATION:
            subject = self.__subject(step=step)
            if subject is None:
                return SkipReason.MISSING_TARGET

            return CheckNode(
                source_steps=(step.index,),
                checks=(Check(kind=CheckKind.VISIBLE, subject=subject),),
            )

        action = self.__action(value=step.action)
        if action is None:
            return SkipReason.UNSUPPORTED

        return self.__node(action=action, step=step)

    def __node(self, *, action: ActionType, step: EvidenceStep) -> Projected:
        """
        Map a successful, supported action to its leaf node, or a skip reason when unsupported.
        """

        sources: Tuple[int, ...] = (step.index,)

        if action is ActionType.TAP:
            selector = self.__selector(step=step)
            return (
                TapNode(selector=selector, source_steps=sources)
                if selector
                else SkipReason.MISSING_TARGET
            )

        if action in (ActionType.TYPE, ActionType.TEXT):
            selector = self.__selector(step=step)
            if selector is None:
                return SkipReason.MISSING_TARGET

            if not step.text:
                return SkipReason.MISSING_TEXT

            return TypeNode(text=step.text, field=selector, source_steps=sources)

        if action is ActionType.BACK:
            return BackNode(source_steps=sources)

        if action in GESTURE_SCROLL_DIRECTION:
            return self.__scroll(action=action, step=step)

        if action is ActionType.WAIT:
            return (
                WaitNode(subject=step.wait.subject, source_steps=sources)
                if step.wait.subject
                else SkipReason.MISSING_WAIT_SUBJECT
            )

        if action is ActionType.STORE:
            capture = step.capture

            if capture is not None and capture.success and capture.value:
                return StoreNode(value=capture.value, name=capture.name, source_steps=sources)

            return SkipReason.MISSING_CAPTURE

        return SkipReason.UNSUPPORTED

    def __scroll(self, *, action: ActionType, step: EvidenceStep) -> LeafNode:
        """
        Project a scroll gesture into a scroll-until when a target was recorded, else a bare scroll.
        """

        sources: Tuple[int, ...] = (step.index,)
        direction = GESTURE_SCROLL_DIRECTION[action]

        if step.target.scroll:
            return ScrollUntilNode(
                direction=direction,
                source_steps=sources,
                target=step.target.scroll,
            )

        return ScrollNode(direction=direction, source_steps=sources)

    @staticmethod
    def __selector(*, step: EvidenceStep) -> Optional[Selector]:
        """
        Build a target selector from the recorded export phrase or raw name, or None when neither exists.
        """

        text = DeterministicFlowGenerator.__target_text(step=step)

        if not text:
            return None

        return Selector(
            text=text, position=step.target.generalized if step.target.positional else None
        )

    @staticmethod
    def __target_text(*, step: EvidenceStep) -> Optional[str]:
        """
        Return the safest deterministic target phrase for a recorded step.
        """

        anchors = (*step.target.anchors.accessibility, *step.target.anchors.visual)
        if anchors:
            return anchors[0]

        if step.target.structure.role:
            return step.target.structure.role

        return step.target.export or step.target.name or step.target.claim.text

    @staticmethod
    def __subject(*, step: EvidenceStep) -> Optional[str]:
        """
        Return the structured assertion subject for a validation step.
        """

        return step.target.export

    @staticmethod
    def __guard_condition(*, step: EvidenceStep) -> Optional[str]:
        """
        Return a concrete branch condition for a conditional step, or None when unguarded.
        """

        if step.launch is not None or not step.guard.conditional:
            return None

        if step.guard.condition:
            return step.guard.condition

        return step.target.export or step.target.name or step.target.generalized

    @staticmethod
    def __action(*, value: str) -> Optional[ActionType]:
        """
        Resolve the recorded action string to an ActionType, or None when it is not a known command.
        """

        try:
            return ActionType(value)
        except ValueError:
            return None
