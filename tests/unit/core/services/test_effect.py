from __future__ import annotations

import unittest

from fathom.core.services.effect import EffectClassifier, EffectRecorder
from fathom.schemas.actions import Bounds, CoordinateSystem
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.screens import ScreenChangeRegion, ScreenDiff


class EffectClassifierTest(unittest.TestCase):
    """
    Cover scoped promotion, foreground regression, and honest degradation on missing inputs.
    """

    def setUp(self) -> None:
        """
        Build the classifier and a date-picker style target region.
        """

        self.classifier = EffectClassifier()
        self.bounds = Bounds(
            x=400,
            y=1200,
            width=200,
            height=120,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def test_promotes_scoped_change_over_flat_global_signals(self) -> None:
        """
        A highlight flip covering the tapped date promotes NO_PROGRESS to PROGRESS.
        """

        reading = self.classifier.classify(
            effect=self.__effect(status=ActionEffectStatus.NO_PROGRESS),
            diff=self.__diff(regions=[ScreenChangeRegion(x=410, y=1210, width=180, height=100)]),
            bounds=self.bounds,
            package="com.app.calendar",
            foreground="com.app.calendar",
        )

        self.assertIs(reading.scoped, True)
        self.assertEqual(reading.trial, ActionEffectStatus.PROGRESS)
        self.assertEqual(reading.live, ActionEffectStatus.NO_PROGRESS)

    def test_ignores_ambient_change_far_from_target(self) -> None:
        """
        Animation noise away from the target neither promotes nor demotes the live status.
        """

        reading = self.classifier.classify(
            effect=self.__effect(status=ActionEffectStatus.NO_PROGRESS),
            diff=self.__diff(regions=[ScreenChangeRegion(x=0, y=0, width=300, height=200)]),
            bounds=self.bounds,
            package="com.app.calendar",
            foreground="com.app.calendar",
        )

        self.assertIs(reading.scoped, False)
        self.assertEqual(reading.overlap, 0.0)
        self.assertEqual(reading.trial, ActionEffectStatus.NO_PROGRESS)

    def test_departed_foreground_overrides_visual_progress(self) -> None:
        """
        The OS-eaten swipe: massive pixel change plus a launcher foreground is REGRESSION.
        """

        reading = self.classifier.classify(
            effect=self.__effect(status=ActionEffectStatus.PROGRESS),
            diff=self.__diff(regions=[]),
            bounds=self.bounds,
            package="com.miniclip.game",
            foreground="com.android.launcher",
        )

        self.assertIs(reading.departed, True)
        self.assertEqual(reading.trial, ActionEffectStatus.REGRESSION)

    def test_degrades_to_none_without_diff(self) -> None:
        """
        Missing diff yields None facets and leaves the live status untouched.
        """

        reading = self.classifier.classify(
            effect=self.__effect(status=ActionEffectStatus.UNCERTAIN),
            diff=None,
            bounds=self.bounds,
            package="com.app.calendar",
            foreground="com.app.calendar",
        )

        self.assertIsNone(reading.scoped)
        self.assertIsNone(reading.overlap)
        self.assertEqual(reading.trial, ActionEffectStatus.UNCERTAIN)

    def test_unknown_foreground_yields_none_departed(self) -> None:
        """
        An empty foreground reading cannot claim departure either way.
        """

        reading = self.classifier.classify(
            effect=self.__effect(status=ActionEffectStatus.PROGRESS),
            diff=self.__diff(regions=[]),
            bounds=self.bounds,
            package="com.app.calendar",
            foreground="",
        )

        self.assertIsNone(reading.departed)
        self.assertEqual(reading.trial, ActionEffectStatus.PROGRESS)

    def test_unbound_target_launch_is_not_regression(self) -> None:
        """
        Run 73efe46d regression: with no requested target (package=None), opening the app
        (foreground changes to it, scoped in-app progress) must NOT be scored a departure.
        """

        reading = self.classifier.classify(
            effect=self.__effect(status=ActionEffectStatus.PROGRESS),
            diff=self.__diff(regions=[ScreenChangeRegion(x=410, y=1210, width=180, height=100)]),
            bounds=self.bounds,
            package=None,
            foreground="com.shopping.supply",
        )

        self.assertIsNone(reading.departed)
        self.assertIsNot(reading.trial, ActionEffectStatus.REGRESSION)
        self.assertEqual(reading.trial, ActionEffectStatus.PROGRESS)

    def test_explicit_target_departure_remains_regression(self) -> None:
        """
        With an explicitly requested target, leaving it for an unexpected package stays REGRESSION.
        """

        reading = self.classifier.classify(
            effect=self.__effect(status=ActionEffectStatus.PROGRESS),
            diff=self.__diff(regions=[]),
            bounds=self.bounds,
            package="com.shopping.supply",
            foreground="com.google.android.gms",
        )

        self.assertIs(reading.departed, True)
        self.assertEqual(reading.trial, ActionEffectStatus.REGRESSION)

    @staticmethod
    def __effect(*, status: ActionEffectStatus) -> ActionEffect:
        """
        Build a minimal live effect with the given status.
        """

        return ActionEffect(status=status, phash_distance=0, visual_progress=0.0)

    @staticmethod
    def __diff(*, regions: list) -> ScreenDiff:
        """
        Build a screen diff carrying only changed regions.
        """

        return ScreenDiff(
            phash_distance=0,
            xml_hash_changed=False,
            interaction_hash_changed=False,
            activity_changed=False,
            changed_regions=regions,
        )


class EffectRecorderTest(unittest.TestCase):
    """
    Cover trial recording and failure isolation.
    """

    def setUp(self) -> None:
        """
        Build the recorder and shared inputs.
        """

        self.effect = ActionEffect(
            status=ActionEffectStatus.NO_PROGRESS,
            phash_distance=0,
            visual_progress=0.0,
        )
        self.bounds = Bounds(
            x=400,
            y=1200,
            width=200,
            height=120,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def test_records_trial_reading(self) -> None:
        """
        Log the comparison and return the typed reading.
        """

        recorder = EffectRecorder()

        with self.assertLogs("fathom.core.services.effect", level="INFO"):
            reading = recorder.observe(
                workflow_id="59cd9b0b",
                effect=self.effect,
                diff=None,
                bounds=self.bounds,
                package="com.app.calendar",
                foreground="com.app.calendar",
            )

        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertEqual(reading.live, ActionEffectStatus.NO_PROGRESS)
