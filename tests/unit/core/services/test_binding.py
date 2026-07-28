from __future__ import annotations

import unittest
from typing import Optional

from fathom.constants import ActionType
from fathom.constants.turn.binding import BindingOrigin, BindingState
from fathom.core.services.binding import Binder
from fathom.schemas.actions import Action, Bounds, CoordinateSystem
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import ElementRole, ElementSource, PerceivedElement


class BinderTest(unittest.TestCase):
    """
    Cover grounding decisions across hierarchy, container, ancestor, and perceptual targets.
    """

    def setUp(self) -> None:
        """
        Build the binder under test.
        """

        self.binder = Binder()

    def test_binds_interactive_match_directly(self) -> None:
        """
        Ground onto the matched element when it is declared clickable.
        """

        button = self.__element(
            identifier="login",
            label="7",
            bounds=self.__bounds(x=40, y=1800, width=1000, height=160),
            interactive=True,
        )

        binding = self.binder.bind(
            action=self.__tap(label="7", bounds=button.bounds),
            elements=(button,),
        )

        self.assertEqual(binding.state, BindingState.BOUND)
        self.assertEqual(binding.origin, BindingOrigin.HIERARCHY)
        self.assertEqual(binding.anchor, "login")
        self.assertEqual(binding.bounds, button.bounds)

    def test_reanchors_container_to_interactive_descendant(self) -> None:
        """
        Move the geometry from a dead row container onto its single clickable switch.
        """

        row = self.__element(
            identifier="row",
            label="3",
            bounds=self.__bounds(x=0, y=1800, width=1080, height=160),
            interactive=False,
        )
        switch = self.__element(
            identifier="switch",
            label="4",
            bounds=self.__bounds(x=900, y=1820, width=160, height=120),
            interactive=True,
        )

        binding = self.binder.bind(
            action=self.__tap(label="3", bounds=row.bounds),
            elements=(row, switch),
        )

        self.assertEqual(binding.state, BindingState.BOUND)
        self.assertEqual(binding.anchor, "switch")
        self.assertEqual(binding.bounds, switch.bounds)

    def test_contests_container_with_competing_descendants(self) -> None:
        """
        Refuse to pick between multiple interactive descendants of one container.
        """

        row = self.__element(
            identifier="row",
            label="3",
            bounds=self.__bounds(x=0, y=1800, width=1080, height=160),
            interactive=False,
        )
        first = self.__element(
            identifier="accept",
            label="4",
            bounds=self.__bounds(x=40, y=1820, width=200, height=120),
            interactive=True,
        )
        second = self.__element(
            identifier="decline",
            label="5",
            bounds=self.__bounds(x=800, y=1820, width=200, height=120),
            interactive=True,
        )

        binding = self.binder.bind(
            action=self.__tap(label="3", bounds=row.bounds),
            elements=(row, first, second),
        )

        self.assertEqual(binding.state, BindingState.CONTESTED)
        self.assertIsNone(binding.anchor)
        self.assertEqual(binding.bounds, row.bounds)

    def test_binds_label_through_interactive_ancestor(self) -> None:
        """
        Keep the label geometry when an enclosing clickable element receives the tap.
        """

        frame = self.__element(
            identifier="frame",
            label="2",
            bounds=self.__bounds(x=0, y=1800, width=1080, height=160),
            interactive=True,
        )
        label = self.__element(
            identifier="caption",
            label="3",
            bounds=self.__bounds(x=40, y=1830, width=260, height=100),
            interactive=False,
        )

        binding = self.binder.bind(
            action=self.__tap(label="3", bounds=label.bounds),
            elements=(frame, label),
        )

        self.assertEqual(binding.state, BindingState.BOUND)
        self.assertEqual(binding.anchor, "frame")
        self.assertEqual(binding.bounds, label.bounds)

    def test_infers_when_no_interactive_relative_exists(self) -> None:
        """
        Degrade to INFERRED when neither the match nor any relative is interactive.
        """

        text = self.__element(
            identifier="banner",
            label="9",
            bounds=self.__bounds(x=100, y=200, width=800, height=90),
            interactive=False,
        )

        binding = self.binder.bind(
            action=self.__tap(label="9", bounds=text.bounds),
            elements=(text,),
        )

        self.assertEqual(binding.state, BindingState.INFERRED)
        self.assertEqual(binding.bounds, text.bounds)

    def test_declared_unclickable_vetoes_role_tappable(self) -> None:
        """
        Trust clickable=false over role-derived tappability and search for a real anchor.
        """

        row = self.__element(
            identifier="row",
            label="3",
            bounds=self.__bounds(x=0, y=1800, width=1080, height=160),
            interactive=False,
            role=ElementRole.BUTTON,
            tappable=True,
        )
        switch = self.__element(
            identifier="switch",
            label="4",
            bounds=self.__bounds(x=900, y=1820, width=160, height=120),
            interactive=True,
        )

        binding = self.binder.bind(
            action=self.__tap(label="3", bounds=row.bounds),
            elements=(row, switch),
        )

        self.assertEqual(binding.anchor, "switch")

    def test_binds_confident_vision_localization(self) -> None:
        """
        Ground an XML-absent target through a confident vision localization.
        """

        bounds = self.__bounds(x=400, y=900, width=280, height=120)

        binding = self.binder.bind(
            action=self.__tap(label=None, bounds=bounds),
            elements=(),
            localization=LocalizationResult(
                status=LocalizationStatus.RESOLVED,
                bounds=bounds,
                source=ElementSource.VISION,
                confidence=0.9,
            ),
        )

        self.assertEqual(binding.state, BindingState.BOUND)
        self.assertEqual(binding.origin, BindingOrigin.VISION)
        self.assertEqual(binding.bounds, bounds)

    def test_infers_weak_vision_localization(self) -> None:
        """
        Degrade a low-confidence vision localization to INFERRED.
        """

        bounds = self.__bounds(x=400, y=900, width=280, height=120)

        binding = self.binder.bind(
            action=self.__tap(label=None, bounds=bounds),
            elements=(),
            localization=LocalizationResult(
                status=LocalizationStatus.RESOLVED,
                bounds=bounds,
                source=ElementSource.VISION,
                confidence=0.3,
            ),
        )

        self.assertEqual(binding.state, BindingState.INFERRED)

    def test_hybrid_origin_when_anchor_source_differs(self) -> None:
        """
        Mark the origin HYBRID when the re-anchor crosses perception channels.
        """

        row = self.__element(
            identifier="row",
            label="3",
            bounds=self.__bounds(x=0, y=1800, width=1080, height=160),
            interactive=False,
        )
        icon = self.__element(
            identifier="icon",
            label="4",
            bounds=self.__bounds(x=900, y=1820, width=120, height=120),
            interactive=True,
            source=ElementSource.ICON,
        )

        binding = self.binder.bind(
            action=self.__tap(label="3", bounds=row.bounds),
            elements=(row, icon),
        )

        self.assertEqual(binding.state, BindingState.BOUND)
        self.assertEqual(binding.origin, BindingOrigin.HYBRID)

    def test_missing_when_action_has_no_bounds(self) -> None:
        """
        Report MISSING for a spatial action that resolved no geometry at all.
        """

        binding = self.binder.bind(
            action=self.__tap(label=None, bounds=None),
            elements=(),
        )

        self.assertEqual(binding.state, BindingState.MISSING)
        self.assertIsNone(binding.bounds)

    @staticmethod
    def __bounds(*, x: int, y: int, width: int, height: int) -> Bounds:
        """
        Build device-pixel bounds.
        """

        return Bounds(
            x=x,
            y=y,
            width=width,
            height=height,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    @staticmethod
    def __tap(*, label: Optional[str], bounds: Optional[Bounds]) -> Action:
        """
        Build a TAP action snapped to the given label and bounds.
        """

        return Action(
            action_type=ActionType.TAP,
            rationale="tap the requested target",
            label_id=label,
            bounds=bounds,
        )

    @staticmethod
    def __element(
        *,
        identifier: str,
        label: str,
        bounds: Bounds,
        interactive: Optional[bool],
        role: ElementRole = ElementRole.CONTAINER,
        tappable: bool = False,
        source: ElementSource = ElementSource.XML,
    ) -> PerceivedElement:
        """
        Build a perceived element with declared clickability.
        """

        resolved_tappable = tappable or interactive is True

        return PerceivedElement(
            identifier=identifier,
            bounds=bounds,
            source=source,
            role=role,
            confidence=1.0,
            tappable=resolved_tappable,
            interactive=interactive,
            label_id=label,
        )
