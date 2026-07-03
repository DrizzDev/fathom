from __future__ import annotations

import unittest
from typing import Dict, FrozenSet
from unittest.mock import Mock

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import (
    ACTION_EXECUTED_TYPES,
    CONTROL_ACTION_TYPES,
    DEVICE_ACTION_TYPES,
    GESTURE_ACTION_TYPES,
    NEXT_PHASE_ACTION_TYPES,
    SPATIAL_ACTION_TYPES,
    ActionType,
)
from fathom.constants.recovery import AUTONOMOUS_RECOVERY_ACTIVE_KINDS
from fathom.core.agent.opener import OpenerSignalPolicy
from fathom.core.agent.reasoner import Reasoner
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.capture.store import CaptureStore
from fathom.core.services.action import ActionExecutor
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.vision import ActionKind, PastActionEntry, action_kind_for


class ExpectedBehaviour(BaseModel):
    """
    The recorded per-command behaviour the capability catalog must reproduce exactly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ActionKind = Field(description="Functional kind from action_kind_for.")
    spatial: bool = Field(description="Membership in SPATIAL_ACTION_TYPES.")
    gesture: bool = Field(description="Membership in GESTURE_ACTION_TYPES.")
    executed: bool = Field(description="Membership in ACTION_EXECUTED_TYPES (legacy dispatched).")
    next_phase: bool = Field(description="Membership in NEXT_PHASE_ACTION_TYPES.")
    control: bool = Field(description="Membership in CONTROL_ACTION_TYPES.")
    device: bool = Field(description="Membership in DEVICE_ACTION_TYPES.")
    recovery_active: bool = Field(description="Kind is an autonomous-recovery active kind.")
    expects_visual_change: bool = Field(description="PastActionEntry.expected_screen_change.")


class CommandBehaviourBaselineTest(unittest.TestCase):
    """
    Golden master pinning today's publicly-derivable per-ActionType behaviour as the contract
    the CommandCatalog must reproduce. Executor non-interactive routing and outer-retry are
    characterized in ExecutorRoutingBaselineTest in this module.
    """

    __BASELINE: Dict[ActionType, ExpectedBehaviour] = {
        ActionType.TAP: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=True,
            gesture=False,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.TYPE: ExpectedBehaviour(
            kind=ActionKind.INPUT,
            spatial=True,
            gesture=False,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.BACK: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.HOME: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.WAIT: ExpectedBehaviour(
            kind=ActionKind.OBSERVATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=False,
            expects_visual_change=False,
        ),
        ActionType.HIDE_KEYBOARD: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.SWIPE: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=True,
            gesture=True,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.SWIPE_UP: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=True,
            gesture=True,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.SWIPE_DOWN: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=True,
            gesture=True,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.SWIPE_LEFT: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=True,
            gesture=True,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.SWIPE_RIGHT: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=True,
            gesture=True,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.SCROLL: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=True,
            gesture=True,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.COMPLETE: ExpectedBehaviour(
            kind=ActionKind.TERMINAL,
            spatial=False,
            gesture=False,
            executed=True,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=False,
            expects_visual_change=False,
        ),
        ActionType.VALIDATE: ExpectedBehaviour(
            kind=ActionKind.VALIDATION,
            spatial=False,
            gesture=False,
            executed=True,
            next_phase=True,
            control=False,
            device=True,
            recovery_active=False,
            expects_visual_change=False,
        ),
        ActionType.LONG_PRESS: ExpectedBehaviour(
            kind=ActionKind.NAVIGATION,
            spatial=True,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=True,
            expects_visual_change=True,
        ),
        ActionType.SAVE_MEMORY: ExpectedBehaviour(
            kind=ActionKind.OBSERVATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=False,
            expects_visual_change=False,
        ),
        ActionType.RETRIEVE_MEMORY: ExpectedBehaviour(
            kind=ActionKind.OBSERVATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=False,
            expects_visual_change=False,
        ),
        ActionType.STORE: ExpectedBehaviour(
            kind=ActionKind.OBSERVATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=False,
            expects_visual_change=False,
        ),
        ActionType.INFER: ExpectedBehaviour(
            kind=ActionKind.OBSERVATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=False,
            expects_visual_change=False,
        ),
        ActionType.UNKNOWN: ExpectedBehaviour(
            kind=ActionKind.UNKNOWN,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=False,
            device=True,
            recovery_active=False,
            expects_visual_change=False,
        ),
        ActionType.ASK_USER: ExpectedBehaviour(
            kind=ActionKind.ESCALATION,
            spatial=False,
            gesture=False,
            executed=False,
            next_phase=False,
            control=True,
            device=False,
            recovery_active=False,
            expects_visual_change=False,
        ),
    }

    def test_baseline_is_exhaustive_over_action_types(self) -> None:
        """
        Every ActionType has a recorded baseline so no command escapes the contract.
        """

        self.assertEqual(set(self.__BASELINE), set(ActionType))

    def test_action_kind_matches_baseline(self) -> None:
        """
        action_kind_for reproduces the recorded kind for every command.
        """

        for action_type, expected in self.__BASELINE.items():
            with self.subTest(action_type=action_type):
                self.assertEqual(action_kind_for(action_type), expected.kind)

    def test_frozenset_membership_matches_baseline(self) -> None:
        """
        The scattered command-classification frozensets match the recorded membership.
        """

        for action_type, expected in self.__BASELINE.items():
            with self.subTest(action_type=action_type):
                self.assertEqual(action_type in SPATIAL_ACTION_TYPES, expected.spatial)
                self.assertEqual(action_type in GESTURE_ACTION_TYPES, expected.gesture)
                self.assertEqual(action_type in ACTION_EXECUTED_TYPES, expected.executed)
                self.assertEqual(action_type in NEXT_PHASE_ACTION_TYPES, expected.next_phase)
                self.assertEqual(action_type in CONTROL_ACTION_TYPES, expected.control)
                self.assertEqual(action_type in DEVICE_ACTION_TYPES, expected.device)

    def test_recovery_active_kind_matches_baseline(self) -> None:
        """
        Whether a command's kind is an autonomous-recovery active kind is preserved.
        """

        for action_type, expected in self.__BASELINE.items():
            with self.subTest(action_type=action_type):
                active = action_kind_for(action_type) in AUTONOMOUS_RECOVERY_ACTIVE_KINDS
                self.assertEqual(active, expected.recovery_active)

    def test_expected_visual_change_matches_baseline(self) -> None:
        """
        PastActionEntry.expected_screen_change is preserved for every command.
        """

        for action_type, expected in self.__BASELINE.items():
            with self.subTest(action_type=action_type):
                entry = PastActionEntry.from_raw(entry={"action": action_type.value})
                self.assertEqual(entry.expected_screen_change, expected.expects_visual_change)


class ExecutorRoutingBaselineTest(unittest.TestCase):
    """
    Independent golden table pinning existing commands' non-interactive routing and outer-retry,
    plus delegation tests proving the executor's private predicates track the injected CommandCatalog
    for every command (so new commands are covered without maintaining a parallel set).
    """

    __LEGACY_NON_INTERACTIVE: FrozenSet[ActionType] = frozenset(
        {
            ActionType.WAIT,
            ActionType.COMPLETE,
            ActionType.VALIDATE,
            ActionType.SAVE_MEMORY,
            ActionType.RETRIEVE_MEMORY,
        }
    )
    __LEGACY_NO_OUTER_RETRY: FrozenSet[ActionType] = frozenset(
        {
            ActionType.SCROLL,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
        }
    )

    __ROUTING_PROBE: str = "_ActionExecutor__is_non_interactive_action"
    __RETRY_PROBE: str = "_ActionExecutor__should_retry_action"

    def __executor(self) -> ActionExecutor:
        """
        Build an executor with stubbed infrastructure for routing characterization only.
        """

        return ActionExecutor(
            device=Mock(),
            telemetry=Mock(),
            path_manager=Mock(),
            max_retries=0,
            catalog=CommandCatalogProvider().build(),
            capture_store=CaptureStore(),
        )

    def __action(self, *, action_type: ActionType) -> Action:
        """
        Build a minimal action of the given type.
        """

        return Action(action_type=action_type, rationale="characterization")

    @staticmethod
    def __existing() -> FrozenSet[ActionType]:
        """
        Return every pre-STORE command; STORE is new behaviour and is excluded from the legacy table.
        """

        return frozenset(action for action in ActionType if action is not ActionType.STORE)

    def test_existing_non_interactive_routing_matches_legacy_baseline(self) -> None:
        """
        Each pre-existing command routes to the non-interactive path exactly as it did before STORE.
        """

        route = getattr(self.__executor(), self.__ROUTING_PROBE)

        for action_type in self.__existing():
            with self.subTest(action_type=action_type):
                routed = route(action=self.__action(action_type=action_type))
                self.assertEqual(routed, action_type in self.__LEGACY_NON_INTERACTIVE)

    def test_existing_outer_retry_matches_legacy_baseline(self) -> None:
        """
        Each pre-existing command's outer-retry eligibility is preserved exactly as before STORE.
        """

        eligible = getattr(self.__executor(), self.__RETRY_PROBE)

        for action_type in self.__existing():
            with self.subTest(action_type=action_type):
                step = Step(
                    action=self.__action(action_type=action_type),
                    step_number=0,
                    screen_hash="x",
                )
                self.assertEqual(
                    eligible(step=step), action_type not in self.__LEGACY_NO_OUTER_RETRY
                )

    def test_non_interactive_routing_delegates_to_catalog(self) -> None:
        """
        The executor's non-interactive predicate tracks the catalog for every command.
        """

        catalog = CommandCatalogProvider().build()
        route = getattr(self.__executor(), self.__ROUTING_PROBE)

        for action_type in ActionType:
            with self.subTest(action_type=action_type):
                routed = route(action=self.__action(action_type=action_type))
                self.assertEqual(routed, catalog.is_non_interactive(action_type=action_type))

    def test_outer_retry_delegates_to_catalog(self) -> None:
        """
        The executor's outer-retry predicate tracks the catalog for every command.
        """

        catalog = CommandCatalogProvider().build()
        eligible = getattr(self.__executor(), self.__RETRY_PROBE)

        for action_type in ActionType:
            with self.subTest(action_type=action_type):
                step = Step(
                    action=self.__action(action_type=action_type),
                    step_number=0,
                    screen_hash="x",
                )
                self.assertEqual(
                    eligible(step=step), catalog.has_outer_retry(action_type=action_type)
                )


class PersistenceEventTypeBaselineTest(unittest.TestCase):
    """
    Golden master for StepResult.to_record event_type: today it is never derived from the action
    type — it passes through Step.event_type or defaults to 'action'.
    """

    def __result(self, *, action_type: ActionType, event_type: object = None) -> StepResult:
        """
        Build a step result for the given action type and optional recorded event type.
        """

        step = Step(
            action=Action(action_type=action_type, rationale="characterization"),
            step_number=0,
            screen_hash="x",
            event_type=event_type,  # type: ignore[arg-type]
        )
        return StepResult(
            step=step,
            success=True,
            duration=0,
            screen_changed=False,
            pre_hash="a",
            post_hash="b",
        )

    def test_unset_event_type_defaults_to_action_for_every_command(self) -> None:
        """
        With no recorded event type, every command persists as an 'action' event.
        """

        for action_type in ActionType:
            with self.subTest(action_type=action_type):
                record = self.__result(action_type=action_type).to_record()
                self.assertEqual(record.event_type, "action")

    def test_explicit_event_type_passes_through(self) -> None:
        """
        A recorded event type is preserved verbatim, independent of the action type.
        """

        record = self.__result(action_type=ActionType.VALIDATE, event_type="validation").to_record()
        self.assertEqual(record.event_type, "validation")


class ReasonerDispatchedBaselineTest(unittest.TestCase):
    """
    Golden master for the reasoner's dispatched semantics: assess_completion marks an action
    dispatched iff its type is in ACTION_EXECUTED_TYPES.
    """

    def __analysis(self, *, action_type: ActionType) -> AnalysisResult:
        """
        Build an analysis result emitting the given action type.
        """

        return AnalysisResult(
            action=Action(action_type=action_type, target="t", rationale="r", confidence=1.0),
            reasoning="r",
            screen_description="s",
            is_sub_goal_complete=False,
            is_goal_complete=False,
            subgoal_completion_reason=None,
            metadata={"tool_args": {}},
        )

    def __sub_goal(self) -> SubGoal:
        """
        Build an action sub-goal fixture.
        """

        return SubGoal(index=0, description="d", kind=SubGoalKind.ACTION, directive=ActionType.TAP)

    def test_dispatched_matches_action_executed_membership(self) -> None:
        """
        The reasoner derives dispatched from ACTION_EXECUTED_TYPES membership for every command.
        """

        reasoner = Reasoner(intent="characterization intent", opener_policy=OpenerSignalPolicy())

        for action_type in ActionType:
            with self.subTest(action_type=action_type):
                evidence = reasoner.assess_completion(
                    analysis=self.__analysis(action_type=action_type),
                    sub_goal=self.__sub_goal(),
                    screen_changed=False,
                    execution_success=True,
                )
                self.assertEqual(evidence.action.dispatched, action_type in ACTION_EXECUTED_TYPES)


class OpenerSignalPolicyBaselineTest(unittest.TestCase):
    """
    Parity: OpenerSignalPolicy.advanced reproduces NEXT_PHASE_ACTION_TYPES exactly.
    """

    def test_advanced_matches_next_phase_membership(self) -> None:
        """
        The policy predicate equals the canonical NEXT_PHASE_ACTION_TYPES membership for every command.
        """

        policy = OpenerSignalPolicy()

        for action_type in ActionType:
            with self.subTest(action_type=action_type):
                self.assertEqual(
                    policy.advanced(action_type=action_type),
                    action_type in NEXT_PHASE_ACTION_TYPES,
                )
