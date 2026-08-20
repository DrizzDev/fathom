from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.constants.flow import LaunchProvenance
from fathom.core.services.generation.classifier import LauncherClassifier
from fathom.core.services.generation.normalizer import RunTraceNormalizer
from fathom.schemas.generation import LaunchMarker, NormalizedEntry
from fathom.schemas.steps import StepRecord


class RunTraceNormalizerTest(unittest.TestCase):
    """
    Cover deterministic launch synthesis: warm start, launcher transitions, and in-flow system steps.
    """

    def setUp(self) -> None:
        """
        Build a shared normalizer.
        """

        self.__normalizer = RunTraceNormalizer(classifier=LauncherClassifier())

    def __launcher(self) -> str:
        """
        Return a representative launcher package from the canonical set.
        """

        return sorted(LAUNCHER_PACKAGES)[0]

    def __fixture_dir(self) -> Path:
        """
        Return the committed launcher run fixture directory.
        """

        return Path("assets/history/2026-06-23/launcher-run")

    def __record(
        self,
        *,
        number: int,
        execution: Optional[str],
        activity: Optional[str] = None,
        action: str = "tap",
    ) -> StepRecord:
        """
        Build a step record with the package context the normalizer reads.
        """

        return StepRecord(
            step_number=number,
            action_type=action,
            target="Element",
            success=True,
            screen_changed=True,
            duration=0,
            activity=activity,
            execution_activity=execution,
        )

    def __launches(self, *, entries: Tuple[NormalizedEntry, ...]) -> List[LaunchMarker]:
        """
        Return the launch markers from a normalized trace.
        """

        return [entry.launch for entry in entries if entry.launch is not None]

    def __kept(self, *, entries: Tuple[NormalizedEntry, ...]) -> Tuple[int, ...]:
        """
        Return the step numbers kept as records.
        """

        numbers: List[int] = []
        for entry in entries:
            record = entry.record
            if record is not None:
                numbers.append(record.step_number)

        return tuple(numbers)

    def test_warm_start_synthesises_leading_launch(self) -> None:
        """
        A run that starts already inside the app gets a synthetic warm-start launch.
        """

        records = (
            self.__record(number=0, execution="com.example.shop"),
            self.__record(number=1, execution="com.example.shop"),
        )

        launches = self.__launches(entries=self.__normalizer.normalize(records=records).entries)

        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0].package, "com.example.shop")
        self.assertEqual(launches[0].provenance, LaunchProvenance.SYNTHETIC_WARM_START)
        self.assertEqual(launches[0].source_steps, ())

    def test_launcher_transition_collapses_and_grounds_launch(self) -> None:
        """
        A launcher-executed step collapses into a grounded transition launch for the entered app.
        """

        records = (
            self.__record(number=0, execution=self.__launcher(), activity="com.example.shop"),
            self.__record(number=1, execution="com.example.shop"),
        )

        entries = self.__normalizer.normalize(records=records).entries
        launches = self.__launches(entries=entries)

        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0].package, "com.example.shop")
        self.assertEqual(launches[0].provenance, LaunchProvenance.LAUNCHER_TRANSITION)
        self.assertEqual(launches[0].source_steps, (0,))
        self.assertEqual(self.__kept(entries=entries), (1,))

    def test_system_package_stays_in_flow(self) -> None:
        """
        A system-package step reached without a launcher hop stays in-flow, not a new launch.
        """

        records = (
            self.__record(number=0, execution="com.example.shop"),
            self.__record(number=1, execution="com.google.android.gms"),
            self.__record(number=2, execution="com.example.shop"),
        )

        entries = self.__normalizer.normalize(records=records).entries

        self.assertEqual(len(self.__launches(entries=entries)), 1)
        self.assertEqual(self.__kept(entries=entries), (0, 1, 2))

    def test_app_triggered_external_surface_stays_in_flow(self) -> None:
        """
        An app-triggered external surface is kept as a step, not a second OPEN_APP.
        """

        records = (
            self.__record(number=0, execution="com.example.health"),
            self.__record(
                number=1,
                execution="com.example.health",
                activity="com.android.chrome",
            ),
            self.__record(number=2, execution="com.android.chrome"),
        )

        entries = self.__normalizer.normalize(records=records).entries
        launches = self.__launches(entries=entries)

        self.assertEqual([marker.package for marker in launches], ["com.example.health"])
        self.assertEqual(self.__kept(entries=entries), (0, 1, 2))

    def test_launcher_opened_browser_remains_a_real_launch(self) -> None:
        """
        An explicit browser app launch is still represented as OPEN_APP.
        """

        records = (
            self.__record(number=0, execution="com.example.health"),
            self.__record(
                number=1,
                execution=self.__launcher(),
                activity="com.android.chrome",
            ),
            self.__record(number=2, execution="com.android.chrome"),
        )

        launches = self.__launches(entries=self.__normalizer.normalize(records=records).entries)

        self.assertEqual(
            [marker.package for marker in launches],
            ["com.example.health", "com.android.chrome"],
        )

    def test_trailing_launcher_step_is_dropped_without_launch(self) -> None:
        """
        A launcher step that leads nowhere is collapsed away and never becomes a launch.
        """

        records = (
            self.__record(number=0, execution="com.example.shop"),
            self.__record(
                number=1, execution=self.__launcher(), activity=self.__launcher(), action="home"
            ),
        )

        entries = self.__normalizer.normalize(records=records).entries

        self.assertEqual(len(self.__launches(entries=entries)), 1)
        self.assertEqual(self.__kept(entries=entries), (0,))

    def test_multiple_launcher_mediated_apps_each_launch(self) -> None:
        """
        Two launcher-mediated app entries produce two grounded launches.
        """

        records = (
            self.__record(
                number=0,
                execution=self.__launcher(),
                activity="com.app.one",
            ),
            self.__record(number=1, execution="com.app.one"),
            self.__record(
                number=2,
                execution=self.__launcher(),
                activity="com.app.two",
            ),
            self.__record(number=3, execution="com.app.two"),
        )

        launches = self.__launches(entries=self.__normalizer.normalize(records=records).entries)

        self.assertEqual([marker.package for marker in launches], ["com.app.one", "com.app.two"])
        self.assertTrue(
            all(marker.provenance == LaunchProvenance.LAUNCHER_TRANSITION for marker in launches)
        )

    def test_every_launcher_package_is_collapsed_into_a_transition(self) -> None:
        """
        Every package in the launcher set is recognised and collapsed into a grounded transition.
        """

        for launcher in sorted(LAUNCHER_PACKAGES):
            records = (
                self.__record(number=0, execution=launcher, activity="com.real.app"),
                self.__record(number=1, execution="com.real.app"),
            )

            launches = self.__launches(entries=self.__normalizer.normalize(records=records).entries)

            self.assertEqual(len(launches), 1, launcher)
            self.assertEqual(launches[0].package, "com.real.app", launcher)
            self.assertEqual(launches[0].provenance, LaunchProvenance.LAUNCHER_TRANSITION, launcher)
            self.assertEqual(launches[0].source_steps, (0,), launcher)

    def test_never_targets_a_launcher_package(self) -> None:
        """
        No synthesised launch targets a launcher package.
        """

        records = (
            self.__record(number=0, execution=self.__launcher(), activity="com.example.shop"),
            self.__record(number=1, execution="com.example.shop"),
        )

        launches = self.__launches(entries=self.__normalizer.normalize(records=records).entries)

        self.assertTrue(all(marker.package not in LAUNCHER_PACKAGES for marker in launches))

    def test_launcher_run_normalizes_to_shopping_launch(self) -> None:
        """
        The recorded launcher-launched Shopping run yields one Shopping launch and keeps system steps.
        """

        if not self.__fixture_dir().exists():
            self.skipTest("launcher run history fixture absent (fixtures are gitignored).")

        records: List[StepRecord] = []
        for path in sorted(self.__fixture_dir().glob("history__com.*.json")):
            payload = json.loads(path.read_text())
            records.extend(StepRecord.model_validate(item) for item in payload.get("history", []))

        entries = self.__normalizer.normalize(records=tuple(records)).entries
        launches = self.__launches(entries=entries)

        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0].package, "com.example.shop")
        self.assertEqual(launches[0].provenance, LaunchProvenance.LAUNCHER_TRANSITION)
        self.assertIn(0, launches[0].source_steps)
        self.assertTrue(all(marker.package not in LAUNCHER_PACKAGES for marker in launches))
