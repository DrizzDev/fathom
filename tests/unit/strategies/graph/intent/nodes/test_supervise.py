from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.command import CommandExecutionMode
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.actions import Action
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.resolution import ResolveStatus
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.schemas.subgoal import ExecutionContract, RequiredActionFamily, ScrollAxis, SubGoal
from fathom.schemas.supervision import VerdictKind
from fathom.strategies.graph.intent.nodes.supervise import SuperviseNode


class SuperviseNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the SUPERVISE node's cancellation and missing-state branches.

    SUPERVISE runs localization, supervision, and bounded healing against
    the planned step. Both pins verify that the node degrades gracefully
    when the prerequisite state is absent: cancellation propagates the
    cancellation reason; missing planned step or capture emits a
    ``SHOULD_RETRY`` signal so the router re-enters GROUND instead of
    letting the silent EXECUTE→OBSERVE→RECORD cascade fail downstream
    with a misleading ``record.missing.step_result`` Sentry alert.
    """

    @staticmethod
    def __provider(*, cancelled: bool = False) -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the cancellation
        check, workflow id, and persistence helper used by the early
        branches. The gate / observer / localizer surfaces stay unmocked
        because they must not be reached on these branches.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=cancelled)
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        return provider

    async def test_cancellation_marks_complete(self) -> None:
        """
        A cancelled run must terminate with :attr:`CompletionReason.CANCELLED`.
        """

        provider = self.__provider(cancelled=True)
        node = SuperviseNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )

    async def test_missing_planned_step_or_capture_signals_should_retry(self) -> None:
        """
        Missing planned step or capture must publish ``SHOULD_RETRY`` so
        :meth:`IntentGraphBuilder.__route_after_supervise` routes back to
        GROUND. Without the signal the router would fall through to
        EXECUTE on a partial state, which is the cascade that surfaced
        on staging as the ``record.missing.step_result`` Sentry alert.
        """

        provider = self.__provider(cancelled=False)
        node = SuperviseNode(provider=provider)

        result: Any = await node(
            state={  # type: ignore[arg-type]
                CommonStateKey.CAPTURE: None,
                IntentStateKey.PLANNED_STEP: None,
            },
        )

        self.assertEqual(result, {IntentStateKey.SHOULD_RETRY: True})
        provider.persistence.persist.assert_called_once_with(
            result={IntentStateKey.SHOULD_RETRY: True},
        )

    async def test_strict_mode_blocks_command_family_drift(self) -> None:
        """
        Strict mode must reject tap/type drift during a scroll sub-goal.
        """

        provider = self.__provider(cancelled=False)
        provider.observer.fallback_observation = AsyncMock(return_value=MagicMock())
        provider.context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        provider.context.package_name = "com.meesho.supply"
        provider.context.agent_state.get_current_sub_goal.return_value = SubGoal(
            index=0,
            description="Scroll vertically until Asha Tiffin is visible",
            execution_contract=ExecutionContract(
                required_action_family=RequiredActionFamily.SCROLL,
                scroll_axis=ScrollAxis.VERTICAL,
            ),
        )
        provider.gate.blocked_execute_result = MagicMock(
            return_value={CommonStateKey.STEP_RESULT: "blocked"}
        )

        node = SuperviseNode(provider=provider)
        planned_step = Step(
            metadata={},
            action=Action(
                action_type=ActionType.TAP,
                target="Search bar",
                rationale="tap search",
                confidence=0.8,
            ),
            step_number=1,
            condition=None,
            event_type="action",
            screen_hash="hash",
            is_conditional=False,
        )
        capture = ScreenCapture(
            width=1080,
            height=2340,
            activity="app",
            image=b"img",
            timestamp=0,
        )

        result: Any = await node(
            state={  # type: ignore[arg-type]
                CommonStateKey.CAPTURE: capture,
                IntentStateKey.PLANNED_STEP: planned_step,
            }
        )

        self.assertTrue(result.get(IntentStateKey.EXECUTION_BLOCKED))
        self.assertEqual(result.get(IntentStateKey.LAST_BLOCK_REASON), "strict_command_mismatch")

    async def test_strict_mode_blocks_surface_drift(self) -> None:
        """
        Strict mode must reject actions whose declared surface drifts from the contract.
        """

        provider = self.__provider(cancelled=False)
        provider.observer.fallback_observation = AsyncMock(return_value=MagicMock())
        provider.context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        provider.context.package_name = "com.meesho.supply"
        provider.context.agent_state.get_current_sub_goal.return_value = SubGoal(
            index=0,
            description="Scroll horizontally below Fast Delivery until Millet Express is visible",
            execution_contract=ExecutionContract(
                required_action_family=RequiredActionFamily.SCROLL,
                scroll_axis=ScrollAxis.HORIZONTAL,
                surface="below Fast Delivery section",
            ),
        )
        provider.gate.blocked_execute_result = MagicMock(
            return_value={CommonStateKey.STEP_RESULT: "blocked"}
        )

        node = SuperviseNode(provider=provider)
        planned_step = Step(
            metadata={},
            action=Action(
                action_type=ActionType.SWIPE_LEFT,
                target="restaurant list",
                surface="below Top Rated section",
                rationale="scroll left",
                confidence=0.8,
            ),
            step_number=1,
            condition=None,
            event_type="action",
            screen_hash="hash",
            is_conditional=False,
        )
        capture = ScreenCapture(
            width=1080,
            height=2340,
            activity="app",
            image=b"img",
            timestamp=0,
        )

        result: Any = await node(
            state={  # type: ignore[arg-type]
                CommonStateKey.CAPTURE: capture,
                IntentStateKey.PLANNED_STEP: planned_step,
            }
        )

        self.assertTrue(result.get(IntentStateKey.EXECUTION_BLOCKED))
        self.assertEqual(result.get(IntentStateKey.LAST_BLOCK_REASON), "strict_command_mismatch")

    async def test_strict_mode_blocks_validate_loop_for_scroll_mission(self) -> None:
        """
        Strict scroll missions must not reopen into validation-only action loops.
        """

        provider = self.__provider(cancelled=False)
        provider.observer.fallback_observation = AsyncMock(return_value=MagicMock())
        provider.context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        provider.context.package_name = "com.meesho.supply"
        provider.context.agent_state.get_current_sub_goal.return_value = SubGoal(
            index=0,
            description="Scroll vertically until Asha Tiffin is visible",
            execution_contract=ExecutionContract(
                required_action_family=RequiredActionFamily.SCROLL,
                scroll_axis=ScrollAxis.VERTICAL,
            ),
        )
        provider.gate.blocked_execute_result = MagicMock(
            return_value={CommonStateKey.STEP_RESULT: "blocked"}
        )

        node = SuperviseNode(provider=provider)
        planned_step = Step(
            metadata={},
            action=Action(
                action_type=ActionType.VALIDATE,
                target="Asha Tiffin card",
                rationale="validate target",
                confidence=0.8,
            ),
            step_number=1,
            condition=None,
            event_type="action",
            screen_hash="hash",
            is_conditional=False,
        )
        capture = ScreenCapture(
            width=1080,
            height=2340,
            activity="app",
            image=b"img",
            timestamp=0,
        )

        result: Any = await node(
            state={  # type: ignore[arg-type]
                CommonStateKey.CAPTURE: capture,
                IntentStateKey.PLANNED_STEP: planned_step,
            }
        )

        self.assertTrue(result.get(IntentStateKey.EXECUTION_BLOCKED))
        self.assertEqual(result.get(IntentStateKey.LAST_BLOCK_REASON), "strict_command_mismatch")

    async def test_strict_mode_allows_terminal_validation_candidate(self) -> None:
        """
        Strict mode must allow a validate step through when the planner is using it
        only as the terminal completion claim for the active mission.
        """

        provider = self.__provider(cancelled=False)
        provider.observer.fallback_observation = AsyncMock(return_value=MagicMock())
        provider.context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        provider.context.package_name = "com.meesho.supply"
        provider.context.agent_state.get_current_sub_goal.return_value = SubGoal(
            index=0,
            description="Scroll vertically until Jars & Containers is visible",
            execution_contract=ExecutionContract(
                required_action_family=RequiredActionFamily.SCROLL,
                scroll_axis=ScrollAxis.VERTICAL,
            ),
        )
        planned_step = Step(
            metadata={"terminal_validation_candidate": True},
            action=Action(
                action_type=ActionType.VALIDATE,
                target="Jars & Containers visibility",
                rationale="validate visible target before completion",
                confidence=1.0,
            ),
            step_number=1,
            condition=None,
            event_type="validation",
            screen_hash="hash",
            is_conditional=False,
        )
        provider.context.resolution.resolve = AsyncMock(
            return_value=MagicMock(
                status=ResolveStatus.UNRESOLVED,
                reason="no label match",
                action=planned_step.action,
            )
        )
        provider.gate.localize = AsyncMock(
            return_value=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                bounds=None,
                source=None,
                confidence=0.0,
            )
        )

        def _apply_localization(*, step: Step, localization: LocalizationResult) -> Step:
            del localization
            return step

        provider.gate.apply_localization.side_effect = _apply_localization
        provider.gate.supervise.return_value = MagicMock(
            kind=VerdictKind.ALLOW,
            message="",
            reason=None,
        )

        node = SuperviseNode(provider=provider)
        capture = ScreenCapture(
            width=1080,
            height=2340,
            activity="app",
            image=b"img",
            timestamp=0,
        )

        result: Any = await node(
            state={  # type: ignore[arg-type]
                CommonStateKey.CAPTURE: capture,
                IntentStateKey.PLANNED_STEP: planned_step,
            }
        )

        self.assertFalse(result.get(IntentStateKey.EXECUTION_BLOCKED, False))
