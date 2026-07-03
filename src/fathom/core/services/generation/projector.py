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

        flow = Flow(
            intent=evidence.intent,
            partial=evidence.partial
            or not self.__has_terminal_validation(nodes=tuple(nodes), evidence=evidence),
            package=evidence.package,
            nodes=tuple(nodes),
        )
        return ProjectionReport(flow=flow, skipped=tuple(skipped))

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
        return bool(set(nodes[-1].source_steps) & validations)

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
                direction=direction, target=step.target.scroll, source_steps=sources
            )

        return ScrollNode(direction=direction, source_steps=sources)

    @staticmethod
    def __selector(*, step: EvidenceStep) -> Optional[Selector]:
        """
        Build a target selector from the recorded export phrase or raw name, or None when neither exists.
        """

        text = step.target.export or step.target.name

        if not text:
            return None

        return Selector(
            text=text, position=step.target.generalized if step.target.positional else None
        )

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

        target = step.target.export or step.target.name or step.target.generalized
        if target:
            return f"{target} is visible"

        return step.guard.condition

    @staticmethod
    def __action(*, value: str) -> Optional[ActionType]:
        """
        Resolve the recorded action string to an ActionType, or None when it is not a known command.
        """

        try:
            return ActionType(value)
        except ValueError:
            return None
