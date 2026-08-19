from __future__ import annotations

import unittest
from typing import Optional

from fathom.constants import ActionType
from fathom.constants.turn.binding import BindingState
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.services.grounding import GroundingRecorder
from fathom.schemas.actions import Action, Bounds, CoordinateSystem
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import ElementRole, ElementSource, PerceivedElement
from fathom.schemas.steps import Step


class GroundingRecorderTest(unittest.TestCase):
    """
    Cover grounding recording, non-spatial gating, and failure isolation.
    """

    def setUp(self) -> None:
        """
        Build the recorder, catalog, and a bindable button element.
        """

        self.recorder = GroundingRecorder()
        self.catalog = CommandCatalogProvider().build()
        self.bounds = Bounds(
            x=40,
            y=1800,
            width=1000,
            height=160,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )
        self.button = PerceivedElement(
            identifier="login",
            bounds=self.bounds,
            source=ElementSource.XML,
            role=ElementRole.BUTTON,
            confidence=1.0,
            tappable=True,
            interactive=True,
            label_id="7",
        )
        self.localization = LocalizationResult(
            status=LocalizationStatus.RESOLVED,
            bounds=self.bounds,
            source=ElementSource.XML,
            confidence=1.0,
        )

    def test_records_binding_for_spatial_step(self) -> None:
        """
        Produce and log a BOUND result for a snapped tap.
        """

        with self.assertLogs("fathom.core.services.grounding", level="INFO"):
            binding = self.recorder.observe(
                step=self.__step(action_type=ActionType.TAP, label="7", bounds=self.bounds),
                workflow_id="59cd9b0b",
                catalog=self.catalog,
                elements=(self.button,),
                localization=self.localization,
            )

        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding.state, BindingState.BOUND)
        self.assertEqual(binding.anchor, "login")

    def test_skips_non_spatial_step(self) -> None:
        """
        Produce nothing for actions that carry no spatial target.
        """

        binding = self.recorder.observe(
            step=self.__step(action_type=ActionType.VALIDATE, label=None, bounds=None),
            workflow_id="59cd9b0b",
            catalog=self.catalog,
            elements=(self.button,),
            localization=self.localization,
        )

        self.assertIsNone(binding)

    @staticmethod
    def __step(
        *,
        action_type: ActionType,
        label: Optional[str],
        bounds: Optional[Bounds],
    ) -> Step:
        """
        Build a planned step around one action.
        """

        return Step(
            action=Action(
                action_type=action_type,
                rationale="exercise the grounding recorder",
                label_id=label,
                bounds=bounds,
            ),
            screen_hash="hash",
            step_number=4,
        )
