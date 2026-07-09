from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.constants.state import CommonStateKey, IntentStateKey
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.resolution import ResolveResult
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.supervise import SuperviseNode


class SuperviseNodeTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers SUPERVISE localization gates before actions reach EXECUTE.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Build the screen capture required by SUPERVISE.
        """

        return ScreenCapture(width=1080, height=2340, activity="app", image=b"", timestamp=1)

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Build a minimal screen observation for localization calls.
        """

        return ScreenObservation(
            activity="app",
            elements=(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            hashes=ScreenHashBundle(
                visual_hash="v",
                xml_hash="x",
                interaction_hash="i",
            ),
        )

    @staticmethod
    def __provider() -> MagicMock:
        """
        Build a provider mock exposing the SUPERVISE dependencies.
        """

        provider = MagicMock()
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.context.workflow_id = "workflow"
        provider.context.package_name = "app"
        provider.context.catalog = CommandCatalogProvider().build()
        provider.persistence.persist = MagicMock()
        provider.observer.fallback_observation = AsyncMock(
            return_value=SuperviseNodeTest.__observation()
        )
        return provider

    async def test_unresolved_tap_does_not_reach_execute_context(self) -> None:
        """
        Unlocalized element actions must route back to GROUND instead of executing raw bbox.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="Search bar",
            confidence=0.9,
            rationale="Tap the search bar",
            bounds=Bounds(
                x=45,
                y=267,
                width=820,
                height=104,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )
        step = Step(step_number=7, screen_hash="v", action=action)
        provider = self.__provider()
        provider.context.resolution.resolve = AsyncMock(
            return_value=ResolveResult.unresolved(
                action=action,
                reason="spatial action emitted without a label_id",
            )
        )
        provider.gate.localize = AsyncMock(
            return_value=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                confidence=0.0,
                reason="No perceived element matched the semantic target.",
            )
        )
        provider.gate.apply_localization = MagicMock(return_value=step)
        node = SuperviseNode(provider=provider)

        result = await node(
            state={
                CommonStateKey.CAPTURE: self.__capture(),
                IntentStateKey.PLANNED_STEP: step,
            }
        )

        self.assertTrue(result[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(result[IntentStateKey.EXECUTION_CONTEXT])
        self.assertIn("refusing to execute", result[CommonStateKey.FAILURE_DIAGNOSTIC])

    async def test_resolved_localization_commits_execution_context(self) -> None:
        """
        Localized element actions remain executable with resolved bounds.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="Search bar",
            confidence=0.9,
            rationale="Tap the search bar",
            bounds=Bounds(
                x=45,
                y=267,
                width=820,
                height=104,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )
        localized_action = action.model_copy(
            update={
                "bounds": Bounds(
                    x=40,
                    y=640,
                    width=1000,
                    height=130,
                    source=CoordinateSource.MODEL,
                    coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                )
            }
        )
        step = Step(step_number=7, screen_hash="v", action=action)
        localized_step = step.model_copy(update={"action": localized_action})
        provider = self.__provider()
        provider.context.resolution.resolve = AsyncMock(
            return_value=ResolveResult.unresolved(
                action=action,
                reason="spatial action emitted without a label_id",
            )
        )
        provider.gate.localize = AsyncMock(
            return_value=LocalizationResult(
                bounds=localized_action.bounds,
                status=LocalizationStatus.RESOLVED,
                confidence=0.95,
            )
        )
        provider.gate.apply_localization = MagicMock(return_value=localized_step)
        node = SuperviseNode(provider=provider)

        result = await node(
            state={
                CommonStateKey.CAPTURE: self.__capture(),
                IntentStateKey.PLANNED_STEP: step,
            }
        )

        self.assertIn(IntentStateKey.EXECUTION_CONTEXT, result)
        self.assertEqual(result[IntentStateKey.PLANNED_STEP].action.bounds.y, 640)
