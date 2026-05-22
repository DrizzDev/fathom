from __future__ import annotations

import unittest

from tests.integration.healing._fixtures import FixtureLoader


class HealingFixtureLoaderTest(unittest.TestCase):
    """
    Pins every Phase 12 healing-runtime replay fixture is well-formed and loadable.
    """

    # Pattern-named scenarios — each pins one named regression class. The
    # specific app traces these were derived from live in
    # `tests/fixtures/healing/README.md`.
    __EXPECTED_SCENARIOS = (
        "001",
        "002",
        "003",
        "004",
        "005",
    )

    def test_all_expected_scenarios_present(self) -> None:
        """
        The fixtures directory must expose exactly the five named scenarios.
        """

        self.assertEqual(FixtureLoader.scenarios(), self.__EXPECTED_SCENARIOS)

    def test_every_scenario_loads_into_typed_trace(self) -> None:
        """
        Every scenario parses into a FixtureTrace with at least one step and a valid expectation.
        """

        for name in self.__EXPECTED_SCENARIOS:
            with self.subTest(scenario=name):
                trace = FixtureLoader.load(identifier=name)
                self.assertTrue(trace.intent)
                self.assertGreaterEqual(len(trace.steps), 1)
                self.assertGreater(trace.expected.max_step_count, 0)
                self.assertEqual(trace.expected.raw_llm_coordinates_executed, 0)

    def test_step_frames_and_manifests_exist_on_disk(self) -> None:
        """
        Each step must point at frame and manifest files inside the fixture directory.
        """

        for name in self.__EXPECTED_SCENARIOS:
            with self.subTest(scenario=name):
                trace = FixtureLoader.load(identifier=name)
                for step in trace.steps:
                    frame_path = trace.directory / step.frame
                    manifest_path = trace.directory / step.manifest
                    self.assertTrue(
                        frame_path.is_file(),
                        msg=f"missing frame: {frame_path}",
                    )
                    self.assertTrue(
                        manifest_path.is_file(),
                        msg=f"missing manifest: {manifest_path}",
                    )

    def test_missing_scenario_raises_file_not_found(self) -> None:
        """
        Asking for an identifier that does not exist raises FileNotFoundError with it embedded.
        """

        with self.assertRaises(FileNotFoundError) as caught:
            FixtureLoader.load(identifier="999")
        self.assertIn("999", str(caught.exception))
