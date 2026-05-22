from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.screen import ZERO_HASH
from fathom.constants.state import CommonStateKey, IntentStateKey
from fathom.schemas.actions import Action
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus
from fathom.schemas.results import ExecutionResult
from fathom.schemas.screens import ScreenCapture, ScreenDiff, ScreenHashBundle, ScreenState
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.observe import ObserveNode


class ObserveNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the OBSERVE node's two early-exit branches.

    OBSERVE captures the post-action evidence and classifies the outcome.
    When the supervisor blocked the action OBSERVE must skip the
    post-capture entirely — there is no action to observe. When the
    :class:`ExecutionContext` is missing or has no ``execution_result``
    the node returns an empty patch rather than crashing, so the graph
    can resume on the next tick.
    """

    @staticmethod
    def __provider() -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the cancellation
        flag, workflow id, and persistence helper used on the early
        branches. The post-action effects pipeline stays unmocked because
        it must not be reached on these branches.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        return provider

    async def test_execution_blocked_skips_post_capture(self) -> None:
        """
        ``EXECUTION_BLOCKED=True`` was set by SUPERVISE; the action never
        ran so the post-action capture is meaningless. OBSERVE must
        return an empty patch and leave the graph state untouched.
        """

        node = ObserveNode(provider=self.__provider())

        result: Any = await node(
            state={IntentStateKey.EXECUTION_BLOCKED: True},  # type: ignore[arg-type]
        )

        self.assertEqual(result, {})

    async def test_effective_outcome_rescues_failed_device_execution(self) -> None:
        """
        A device-reported failure must still record success when canonical observation proves effect.
        """

        provider = self.__provider()
        provider.context.metrics.record = MagicMock()
        provider.context.agent_state.record_action_effect = MagicMock()
        provider.observer.fallback_observation = AsyncMock(return_value=self.__observation())
        provider.context.outcome_classifier.classify = MagicMock(
            return_value=ActionOutcome(
                action=Action(
                    action_type=ActionType.SWIPE_UP,
                    target="Feed",
                    rationale="scroll",
                    confidence=0.9,
                ),
                status=OutcomeStatus.EFFECTIVE,
                before=self.__observation(),
                after=self.__observation(),
                diff=self.__diff(),
                reason="Visible UI effect.",
            )
        )
        provider.effects.observe = AsyncMock(
            return_value=(
                self.__observation(),
                self.__diff(),
                "bbbbbbbbbbbbbbbb",
                "app",
                None,
            )
        )
        provider.effects.effect_from = MagicMock(return_value=MagicMock())
        provider.effects.log_diff = MagicMock()
        provider.effects.changed = MagicMock(return_value=True)
        provider.persistence.persist = MagicMock()

        node = ObserveNode(provider=provider)
        result: Any = await node(
            state={  # type: ignore[arg-type]
                IntentStateKey.EXECUTION_CONTEXT: self.__execution_context(),
                CommonStateKey.SCREEN_OBSERVATION: self.__observation(),
            },
        )

        step_result = result[CommonStateKey.STEP_RESULT]
        self.assertTrue(step_result.success)
        self.assertTrue(step_result.screen_changed)

    async def test_no_effect_device_action_records_failed_step(self) -> None:
        """
        Device transport success must not override a canonical no-effect outcome.
        """

        provider = self.__provider()
        provider.context.metrics.record = MagicMock()
        provider.context.agent_state.record_action_effect = MagicMock()
        provider.observer.fallback_observation = AsyncMock(return_value=self.__observation())
        provider.context.outcome_classifier.classify = MagicMock(
            return_value=ActionOutcome(
                action=Action(
                    action_type=ActionType.TAP,
                    target="Search",
                    rationale="tap",
                    confidence=0.9,
                ),
                status=OutcomeStatus.NO_EFFECT,
                before=self.__observation(),
                after=self.__observation(),
                diff=self.__diff(),
                reason="No visible UI effect.",
            )
        )
        provider.effects.observe = AsyncMock(
            return_value=(
                self.__observation(),
                self.__diff(),
                "aaaaaaaaaaaaaaaa",
                "app",
                None,
            )
        )
        provider.effects.effect_from = MagicMock(return_value=MagicMock())
        provider.effects.log_diff = MagicMock()
        provider.effects.changed = MagicMock(return_value=False)
        provider.persistence.persist = MagicMock()

        node = ObserveNode(provider=provider)
        result: Any = await node(
            state={  # type: ignore[arg-type]
                IntentStateKey.EXECUTION_CONTEXT: self.__successful_tap_context(),
                CommonStateKey.SCREEN_OBSERVATION: self.__observation(),
            },
        )

        step_result = result[CommonStateKey.STEP_RESULT]
        self.assertFalse(step_result.success)
        self.assertFalse(step_result.screen_changed)

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Minimal observation fixture for canonical effect tests.
        """

        return ScreenObservation(
            activity="app",
            elements=(),
            hashes=ScreenHashBundle(
                visual_hash="a" * 16,
                xml_hash="b" * 16,
                interaction_hash="c" * 16,
            ),
            overlays=(),
            keyboard=KeyboardObservation(visible=False),
        )

    @staticmethod
    def __diff() -> ScreenDiff:
        """
        Minimal effective diff fixture.
        """

        return ScreenDiff(
            phash_distance=24,
            ssim_score=0.4,
            content_pixel_diff_ratio=0.6,
            xml_hash_changed=True,
            interaction_hash_changed=True,
            activity_changed=False,
        )

    @classmethod
    def __execution_context(cls) -> ExecutionContext:
        """
        Failed device execution with enough post-action context for rescue classification.
        """

        return ExecutionContext(
            step=Step(
                action=Action(
                    action_type=ActionType.SWIPE_UP,
                    target="Feed",
                    rationale="scroll",
                    confidence=0.9,
                ),
                screen_hash=ZERO_HASH,
                step_number=0,
            ),
            capture=ScreenCapture(
                width=1080,
                height=2340,
                activity="app",
                image=b"img",
                timestamp=0,
            ),
            pre_screen=ScreenState(
                activity="app",
                timestamp=0,
                activity_hash="0" * 16,
                visual_hash="a" * 16,
                xml_hash="b" * 16,
                interaction_hash="c" * 16,
            ),
            localization=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                confidence=0.0,
            ),
            package="app",
            execution_result=ExecutionResult(
                success=False,
                duration=100,
                error="translation_in_uncertain_band",
            ),
            duration=100,
        )

    @classmethod
    def __successful_tap_context(cls) -> ExecutionContext:
        """
        Successful device dispatch used to prove canonical no-effect still fails the step.
        """

        return ExecutionContext(
            step=Step(
                action=Action(
                    action_type=ActionType.TAP,
                    target="Search",
                    rationale="tap",
                    confidence=0.9,
                ),
                screen_hash=ZERO_HASH,
                step_number=0,
            ),
            capture=ScreenCapture(
                width=1080,
                height=2340,
                activity="app",
                image=b"img",
                timestamp=0,
            ),
            pre_screen=ScreenState(
                activity="app",
                timestamp=0,
                activity_hash="0" * 16,
                visual_hash="a" * 16,
                xml_hash="b" * 16,
                interaction_hash="c" * 16,
            ),
            localization=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                confidence=0.0,
            ),
            package="app",
            execution_result=ExecutionResult(
                success=True,
                duration=100,
                error=None,
            ),
            duration=100,
        )

    async def test_missing_execution_context_returns_empty_state(self) -> None:
        """
        Missing :class:`ExecutionContext` or absent ``execution_result``
        means EXECUTE did not commit. Returning an empty patch instead of
        raising lets the graph resume on the next tick rather than tearing
        down the whole run on a transient upstream bug.
        """

        node = ObserveNode(provider=self.__provider())

        result: Any = await node(
            state={IntentStateKey.EXECUTION_CONTEXT: None},  # type: ignore[arg-type]
        )

        self.assertEqual(result, {})
