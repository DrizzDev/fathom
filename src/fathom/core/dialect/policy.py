from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

from fathom.constants import GESTURE_ACTION_TYPES, GESTURE_SCROLL_DIRECTION, ActionType
from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.constants.flow import EvidenceMarker, IssueCode, LaunchProvenance, NodeKind
from fathom.schemas.flow import (
    BranchNode,
    CheckNode,
    Evidence,
    EvidenceStep,
    Flow,
    FlowNode,
    Issue,
    LaunchNode,
    MapNode,
    Report,
    ScrollNode,
    ScrollUntilNode,
    StoreNode,
    TapNode,
    TypeNode,
    WaitNode,
)


class Policy:
    """
    Validates a flow against its recorded evidence for fidelity before rendering.
    """

    __VALIDATION = "validation"

    def __init__(self, *, launchers: FrozenSet[str] = LAUNCHER_PACKAGES) -> None:
        """
        Bind the launcher package set rejected as launch targets.
        """

        self.__launchers = launchers

    def evaluate(self, *, flow: Flow, evidence: Evidence) -> Report:
        """
        Return fidelity issues between the flow and the recorded evidence.
        """

        index = {step.index: step for step in evidence.steps if step.launch is None}
        indexed = list(enumerate(self.__flatten(nodes=flow.nodes)))

        issues: List[Issue] = []
        issues.extend(self.__completion(flow=flow, evidence=evidence))
        issues.extend(self.__recovery(indexed=indexed, index=index))
        issues.extend(self.__provenance(indexed=indexed, index=index))
        issues.extend(self.__grounded_conditions(indexed=indexed, index=index))
        issues.extend(self.__unguarded_conditional(nodes=flow.nodes, index=index))
        issues.extend(self.__target_content(indexed=indexed, index=index))
        issues.extend(self.__type_content(indexed=indexed, index=index))
        issues.extend(self.__wait_content(indexed=indexed, index=index))
        issues.extend(self.__validation_content(indexed=indexed, index=index))
        issues.extend(self.__ungrounded_scrolls(indexed=indexed, index=index))
        issues.extend(self.__redundant_scrolls(indexed=indexed, index=index))
        issues.extend(self.__ungrounded_stores(indexed=indexed, index=index))
        issues.extend(self.__redundant_branches(nodes=flow.nodes))
        issues.extend(self.__redundant_waits(nodes=flow.nodes))
        issues.extend(self.__launches(flow=flow, evidence=evidence))

        return Report(issues=tuple(issues))

    def __flatten(self, *, nodes: Tuple[FlowNode, ...]) -> Tuple[FlowNode, ...]:
        """
        Flatten nodes into pre-order, descending into branch bodies.
        """

        flattened: List[FlowNode] = []

        for node in nodes:
            flattened.append(node)

            if isinstance(node, BranchNode):
                flattened.extend(self.__flatten(nodes=node.body))

        return tuple(flattened)

    def __recovery(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Reject nodes derived from a step the evidence marked as recovery.
        """

        issues: List[Issue] = []

        for position, node in indexed:
            for step in node.source_steps:
                recorded = index.get(step)

                if recorded is not None and recorded.guard.condition == EvidenceMarker.RECOVERY:
                    issues.append(
                        Issue(
                            node_index=position,
                            code=IssueCode.RECOVERY_NODE,
                            message=f"Node {position} ({node.kind}) derives from recovery step {step}.",
                        )
                    )

                    break

        return issues

    def __grounded_conditions(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Reject branch guards whose condition is not grounded in the cited conditional step.
        """

        issues: List[Issue] = []

        for position, node in indexed:
            if not isinstance(node, BranchNode):
                continue

            condition = node.guard.condition.strip()
            recorded = index.get(node.guard.source_step)
            candidates = (
                self.__condition_grounding(recorded=recorded) if recorded is not None else ()
            )

            if (
                recorded is None
                or not recorded.guard.conditional
                or not condition
                or not self.__matches_grounding(text=condition, candidates=candidates)
            ):
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.UNGROUNDED_CONDITION,
                        message=(
                            f"Branch at node {position} uses a condition not grounded in "
                            f"step {node.guard.source_step}. Use one of these evidence phrases: "
                            f"{self.__grounding_summary(candidates=candidates)}."
                        ),
                    )
                )

        return issues

    def __provenance(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Require every cited evidence step to exist (the schema already guarantees non-empty).
        """

        issues: List[Issue] = []

        for position, node in indexed:
            if isinstance(node, (BranchNode, LaunchNode)):
                continue

            for step in node.source_steps:
                if step not in index:
                    issues.append(
                        Issue(
                            node_index=position,
                            code=IssueCode.DANGLING_PROVENANCE,
                            message=f"Node {position} cites step {step} absent from the evidence.",
                        )
                    )

        return issues

    def __unguarded_conditional(
        self, *, nodes: Tuple[FlowNode, ...], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Require a top-level node citing a conditional step to sit inside an IF branch.
        """

        conditional = {step for step, recorded in index.items() if recorded.guard.conditional}
        issues: List[Issue] = []

        for position, node in enumerate(nodes):
            if isinstance(node, (BranchNode, LaunchNode)):
                continue

            if any(step in conditional for step in node.source_steps):
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.UNGUARDED_CONDITIONAL,
                        message=(
                            f"Node {position} ({node.kind}) derives from a conditional step but "
                            "is not inside an IF branch."
                        ),
                    )
                )

        return issues

    def __target_content(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Reject tap targets that do not match content recorded on the cited source step.
        """

        issues: List[Issue] = []

        for position, node in indexed:
            if not isinstance(node, (TapNode, MapNode)):
                continue

            if not self.__matches_action_grounding(text=node.selector.text, node=node, index=index):
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.TAP_TARGET_MISMATCH,
                        message=(
                            f"{node.kind} at node {position} targets '{node.selector.text}', "
                            "which is not grounded in the cited action evidence."
                        ),
                    )
                )

        return issues

    def __type_content(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Reject type nodes whose field or typed text differs from the cited source step.
        """

        issues: List[Issue] = []

        for position, node in indexed:
            if not isinstance(node, TypeNode):
                continue

            text_matches = any(
                (recorded := index.get(step)) is not None
                and recorded.text is not None
                and recorded.text == node.text
                for step in node.source_steps
            )
            field_matches = self.__matches_any_target(text=node.field.text, node=node, index=index)

            if not text_matches or not field_matches:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.TYPE_CONTENT_MISMATCH,
                        message=(
                            f"Type at node {position} does not match the recorded field and text "
                            "on its cited source steps."
                        ),
                    )
                )

        return issues

    def __wait_content(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Reject wait subjects that differ from the cited source step's recorded wait subject.
        """

        issues: List[Issue] = []

        for position, node in indexed:
            if not isinstance(node, WaitNode) or node.subject is None:
                continue

            recorded_subjects = [
                recorded.wait.subject
                for step in node.source_steps
                if (recorded := index.get(step)) is not None and recorded.wait.subject
            ]
            if recorded_subjects and node.subject not in recorded_subjects:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.WAIT_SUBJECT_MISMATCH,
                        message=(
                            f"Wait at node {position} uses subject '{node.subject}', which was "
                            "not recorded on its cited source steps."
                        ),
                    )
                )

        return issues

    def __validation_content(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Reject validation subjects that differ from recorded validation target content.
        """

        issues: List[Issue] = []

        for position, node in indexed:
            if not isinstance(node, CheckNode):
                continue

            recorded_targets = tuple(
                target
                for step in node.source_steps
                if (recorded := index.get(step)) is not None and recorded.event == self.__VALIDATION
                for target in self.__validation_grounding(recorded=recorded)
            )

            for check in node.checks:
                if not self.__matches_grounding(text=check.subject, candidates=recorded_targets):
                    issues.append(
                        Issue(
                            node_index=position,
                            code=IssueCode.VALIDATION_SUBJECT_MISMATCH,
                            message=(
                                f"Validation at node {position} checks '{check.subject}', which "
                                "is not grounded in the cited validation evidence. Use one of "
                                "these evidence phrases: "
                                f"{self.__grounding_summary(candidates=recorded_targets)}."
                            ),
                        )
                    )
                    break

        return issues

    def __ungrounded_scrolls(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Require scroll nodes to cite recorded gesture steps and never invent target or direction.
        """

        issues: List[Issue] = []

        for position, node in indexed:
            if not isinstance(node, (ScrollNode, ScrollUntilNode)):
                continue

            gestures = [
                (action, recorded)
                for step in node.source_steps
                if (recorded := index.get(step)) is not None
                and (action := self.__action(value=recorded.action)) in GESTURE_ACTION_TYPES
            ]

            if not gestures:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.UNGROUNDED_SCROLL,
                        message=f"Scroll at node {position} cites no recorded gesture step.",
                    )
                )
                continue

            if not any(
                GESTURE_SCROLL_DIRECTION[action] == node.direction for action, _ in gestures
            ):
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.SCROLL_DIRECTION_MISMATCH,
                        message=(
                            f"Scroll at node {position} uses direction '{node.direction}', which "
                            "does not match the recorded gesture."
                        ),
                    )
                )

            scroll_candidates = tuple(
                phrase
                for _, recorded in gestures
                for phrase in self.__scroll_grounding(recorded=recorded)
            )
            if isinstance(node, ScrollUntilNode) and not any(
                self.__matches_grounding(
                    text=node.target, candidates=self.__scroll_grounding(recorded=recorded)
                )
                for _, recorded in gestures
            ):
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.UNGROUNDED_SCROLL,
                        message=(
                            f"Scroll-until at node {position} targets '{node.target}', which was "
                            "not grounded in its cited gesture steps. Use one of these evidence "
                            f"phrases: {self.__grounding_summary(candidates=scroll_candidates)}."
                        ),
                    )
                )

        return issues

    def __redundant_scrolls(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Reject consecutive scrolls that repeat one episode's page-motion attempt.
        """

        issues: List[Issue] = []
        previous: Optional[Tuple[int, FlowNode]] = None

        for position, node in indexed:
            if not isinstance(node, (ScrollNode, ScrollUntilNode)):
                previous = None
                continue

            if previous is not None and self.__same_scroll_episode(
                previous=previous[1], current=node, index=index
            ):
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.REDUNDANT_SCROLL,
                        message=(
                            f"Scroll at node {position} repeats the previous scroll inside the "
                            "same execution episode; merge them into one replay command."
                        ),
                    )
                )

            previous = (position, node)

        return issues

    @staticmethod
    def __same_scroll_episode(
        *, previous: FlowNode, current: FlowNode, index: Dict[int, EvidenceStep]
    ) -> bool:
        """
        Return whether two scroll nodes are consecutive attempts for the same recorded goal.
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

    def __ungrounded_stores(
        self, *, indexed: List[Tuple[int, FlowNode]], index: Dict[int, EvidenceStep]
    ) -> List[Issue]:
        """
        Require every store to derive from a step the evidence recorded a capture on (no invented stores).
        """

        issues: List[Issue] = []

        for position, node in indexed:
            if not isinstance(node, StoreNode):
                continue

            grounded = any(
                (recorded := index.get(step)) is not None
                and recorded.capture is not None
                and recorded.capture.success
                and recorded.capture.name == node.name
                and recorded.capture.value == node.value
                for step in node.source_steps
            )

            if not grounded:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.UNGROUNDED_STORE,
                        message=(
                            f"Store at node {position} ('{node.value}' as '{node.name}') matches "
                            "no successful recorded capture on its cited steps."
                        ),
                    )
                )

        return issues

    @staticmethod
    def __preferred_targets(*, recorded: EvidenceStep) -> Tuple[str, ...]:
        """
        Return recorded target text in export-name-generalized precedence order.
        """

        if recorded.target.export:
            return (recorded.target.export,)

        if recorded.target.name:
            return (recorded.target.name,)

        if recorded.target.generalized:
            return (recorded.target.generalized,)

        return ()

    def __matches_any_target(
        self, *, text: str, node: FlowNode, index: Dict[int, EvidenceStep]
    ) -> bool:
        """
        Return whether text matches recorded target content on at least one cited source step.
        """

        targets = tuple(
            target
            for step in node.source_steps
            if (recorded := index.get(step)) is not None
            for target in self.__preferred_targets(recorded=recorded)
        )
        return bool(targets) and text in targets

    def __matches_action_grounding(
        self, *, text: str, node: FlowNode, index: Dict[int, EvidenceStep]
    ) -> bool:
        """
        Return whether text is grounded in any cited action evidence.
        """

        return any(
            self.__matches_grounding(
                text=text, candidates=self.__grounding_texts(recorded=recorded)
            )
            for step in node.source_steps
            if (recorded := index.get(step)) is not None
        )

    @classmethod
    def __validation_grounding(cls, *, recorded: EvidenceStep) -> Tuple[str, ...]:
        """
        Return evidence strings that can ground a validation assertion subject.
        """

        if recorded.target.export and not cls.__contains(
            text=recorded.target.export, fragment=recorded.target.name
        ):
            return (recorded.target.export,)

        narrative = tuple(
            text
            for text in (
                recorded.goal.description if recorded.goal is not None else None,
                recorded.rationale,
                recorded.observation,
            )
            if text
        )
        if narrative:
            return narrative

        return cls.__target_grounding(recorded=recorded)

    @classmethod
    def __condition_grounding(cls, *, recorded: EvidenceStep) -> Tuple[str, ...]:
        """
        Return evidence strings that can ground an authored branch condition.
        """

        target_conditions = tuple(
            f"{target} is visible" for target in cls.__target_grounding(recorded=recorded)
        )
        narrative = tuple(
            text
            for text in (
                recorded.rationale,
                recorded.observation,
                recorded.goal.description if recorded.goal is not None else None,
            )
            if text
        )
        if target_conditions or narrative:
            return (*target_conditions, *narrative)

        return cls.__grounding_texts(recorded=recorded)

    @classmethod
    def __matches_grounding(cls, *, text: str, candidates: Tuple[str, ...]) -> bool:
        """
        Return whether text appears in one of the recorded evidence fields.
        """

        normalized = cls.__normalized(text=text)
        return bool(normalized) and any(
            normalized in cls.__normalized(text=candidate) for candidate in candidates
        )

    @classmethod
    def __contains(cls, *, text: Optional[str], fragment: Optional[str]) -> bool:
        """
        Return whether a non-empty fragment is contained in text after normalization.
        """

        if not text or not fragment:
            return False

        return cls.__normalized(text=fragment) in cls.__normalized(text=text)

    @classmethod
    def __grounding_summary(cls, *, candidates: Tuple[str, ...]) -> str:
        """
        Return a concise diagnostics string listing normalized-unique grounding phrases.
        """

        unique: List[str] = []
        seen: List[str] = []
        for candidate in candidates:
            normalized = cls.__normalized(text=candidate)
            if normalized and normalized not in seen:
                seen.append(normalized)
                unique.append(candidate)

        if not unique:
            return "<none>"

        return "; ".join(f"'{candidate}'" for candidate in unique[:3])

    @classmethod
    def __scroll_grounding(cls, *, recorded: EvidenceStep) -> Tuple[str, ...]:
        """
        Return evidence strings that can ground an authored scroll-until target.
        """

        return cls.__grounding_texts(recorded=recorded, include_condition=False)

    @classmethod
    def __grounding_texts(
        cls, *, recorded: EvidenceStep, include_condition: bool = True
    ) -> Tuple[str, ...]:
        """
        Return recorded strings that may ground authored text for the cited step.
        """

        condition = recorded.guard.condition if include_condition else None
        return tuple(
            text
            for text in (
                condition,
                recorded.target.scroll,
                *cls.__target_grounding(recorded=recorded),
                recorded.goal.description if recorded.goal is not None else None,
                recorded.rationale,
                recorded.observation,
            )
            if text
        )

    @staticmethod
    def __normalized(*, text: str) -> str:
        """
        Normalize text for deterministic evidence-containment checks.
        """

        return " ".join(text.casefold().split())

    @staticmethod
    def __target_grounding(*, recorded: EvidenceStep) -> Tuple[str, ...]:
        """
        Return target-specific recorded strings for action grounding.
        """

        return tuple(
            text
            for text in (
                recorded.target.export,
                recorded.target.name,
                recorded.target.generalized,
                recorded.target.element,
            )
            if text
        )

    @staticmethod
    def __action(*, value: str) -> Optional[ActionType]:
        """
        Resolve the recorded action string to an ActionType, or None when unknown.
        """

        try:
            return ActionType(value)
        except ValueError:
            return None

    def __redundant_branches(self, *, nodes: Tuple[FlowNode, ...]) -> List[Issue]:
        """
        Reject consecutive branches sharing one condition; they must be a single IF block.
        """

        issues: List[Issue] = []
        previous: str = ""

        for position, node in enumerate(nodes):
            if not isinstance(node, BranchNode):
                previous = ""
                continue

            condition = node.guard.condition.strip()

            if condition == previous:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.REDUNDANT_BRANCH,
                        message=(
                            f"Branch at node {position} repeats the previous branch condition; "
                            "merge them into one IF block."
                        ),
                    )
                )

            previous = condition

        return issues

    def __redundant_waits(self, *, nodes: Tuple[FlowNode, ...]) -> List[Issue]:
        """
        Reject a wait that duplicates the one immediately before it.
        """

        issues: List[Issue] = []

        for position in range(1, len(nodes)):
            previous, current = nodes[position - 1], nodes[position]

            if not isinstance(previous, WaitNode) or not isinstance(current, WaitNode):
                continue

            if previous.duration == current.duration and previous.subject == current.subject:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.REDUNDANT_WAIT,
                        message=(
                            f"Wait at node {position} duplicates the previous wait; "
                            "merge consecutive identical waits into one."
                        ),
                    )
                )

        return issues

    def __launches(self, *, flow: Flow, evidence: Evidence) -> List[Issue]:
        """
        Enforce that every normalized launch marker appears, in order, as a grounded LaunchNode.
        """

        markers = [step.launch for step in evidence.steps if step.launch is not None]
        launches = [
            (position, node)
            for position, node in enumerate(flow.nodes)
            if isinstance(node, LaunchNode)
        ]
        issues: List[Issue] = []

        if markers and not (flow.nodes and isinstance(flow.nodes[0], LaunchNode)):
            issues.append(
                Issue(code=IssueCode.MISSING_LAUNCH, message="Flow must begin with a launch.")
            )

        if len(launches) != len(markers):
            issues.append(
                Issue(
                    code=IssueCode.LAUNCH_MISMATCH,
                    message=(
                        f"Flow has {len(launches)} launches but the normalized evidence "
                        f"has {len(markers)}."
                    ),
                )
            )

        for order, (position, node) in enumerate(launches):
            if node.package in self.__launchers:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.LAUNCH_MISMATCH,
                        message=f"Launch target '{node.package}' is a launcher package.",
                    )
                )

            if node.provenance is LaunchProvenance.LAUNCHER_TRANSITION and not node.source_steps:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.UNGROUNDED_LAUNCH,
                        message="A launcher-transition launch must cite collapsed launcher steps.",
                    )
                )

            if node.provenance is LaunchProvenance.SYNTHETIC_WARM_START and order != 0:
                issues.append(
                    Issue(
                        node_index=position,
                        code=IssueCode.STRAY_LAUNCH,
                        message="A warm-start launch may appear only as the first launch.",
                    )
                )

            if order < len(markers):
                marker = markers[order]
                if (
                    node.package != marker.package
                    or node.provenance is not marker.provenance
                    or tuple(node.source_steps) != tuple(marker.source_steps)
                ):
                    issues.append(
                        Issue(
                            node_index=position,
                            code=IssueCode.LAUNCH_MISMATCH,
                            message=(
                                f"Launch {order} '{node.package}' ({node.provenance}, "
                                f"steps {tuple(node.source_steps)}) does not match the normalized "
                                f"marker '{marker.package}' ({marker.provenance}, "
                                f"steps {tuple(marker.source_steps)})."
                            ),
                        )
                    )

        return issues

    def __completion(self, *, flow: Flow, evidence: Evidence) -> List[Issue]:
        """
        Enforce goal-grounded completion: a complete run ends in a recorded validation; a partial
        run is flagged partial and never invents one.
        """

        nodes = flow.nodes
        validations = {
            step.index
            for step in evidence.steps
            if step.event == self.__VALIDATION and step.outcome.success
        }
        terminal = nodes[-1] if nodes else None
        grounded = (
            terminal is not None
            and terminal.kind is NodeKind.CHECK
            and bool(set(terminal.source_steps) & validations)
        )

        if evidence.partial:
            return self.__partial_completion(flow=flow, terminal=terminal, grounded=grounded)

        if terminal is None or terminal.kind is not NodeKind.CHECK:
            return [
                Issue(
                    code=IssueCode.MISSING_GOAL_VALIDATION,
                    message="Flow must end with a terminal validation check.",
                )
            ]

        if not grounded:
            return [
                Issue(
                    code=IssueCode.INVENTED_VALIDATION,
                    message="Terminal validation cites no recorded validation step.",
                )
            ]

        return []

    def __partial_completion(
        self, *, flow: Flow, terminal: Optional[FlowNode], grounded: bool
    ) -> List[Issue]:
        """
        A partial run must set flow.partial and must not invent a terminal validation.
        """

        issues: List[Issue] = []

        if not flow.partial:
            issues.append(
                Issue(
                    code=IssueCode.MISSING_PARTIAL,
                    message="Run has no recorded goal validation; flow.partial must be set.",
                )
            )

        if terminal is not None and terminal.kind is NodeKind.CHECK and not grounded:
            issues.append(
                Issue(
                    code=IssueCode.INVENTED_VALIDATION,
                    message="Partial run must not end with an invented validation.",
                )
            )

        return issues
