from __future__ import annotations

import io
import unittest
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

from rich.console import Console

from fathom.runtime.command.wizard import InteractiveWizard, wizard_argv


def _build_console() -> Console:
    """
    Build a capture-buffer Console the wizard can render into during tests.
    """

    return Console(file=io.StringIO(), force_terminal=True, width=100)


def _prompts(responses: List[str]):
    """
    Build a Prompt.ask side-effect that yields responses in order.
    """

    it = iter(responses)

    def fake_ask(*_args: Any, **_kwargs: Any) -> str:
        return next(it)

    return fake_ask


def _ints(values: List[int]):
    """
    Build an IntPrompt.ask side-effect that yields ints in order.
    """

    it = iter(values)

    def fake_ask(*_args: Any, **_kwargs: Any) -> int:
        return next(it)

    return fake_ask


def _confirms(values: List[bool]):
    """
    Build a Confirm.ask side-effect that yields booleans in order.
    """

    it = iter(values)

    def fake_ask(*_args: Any, **_kwargs: Any) -> bool:
        return next(it)

    return fake_ask


@contextmanager
def _mock_questionary_selects(values: List[str]) -> Iterator[MagicMock]:
    """
    Patch ``questionary.select`` so each invocation returns a Question
    whose ``.ask()`` yields the next item from ``values``.

    The whole wizard's select-menu prompts share one mock, so the
    sequence covers every ``__prompt_select`` call in the order the
    wizard makes them.
    """

    with patch("fathom.runtime.command.wizard.questionary.select") as mock_select:
        mock_select.return_value.ask.side_effect = list(values)
        yield mock_select


class InteractiveWizardHappyPathsTest(unittest.TestCase):
    """
    Cover the two most common flows: run+android and demo+ios.
    """

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_full_flow_run_android(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        # Select menus return "run" then "android". Device prompt falls
        # through to free-text since detection is mocked empty.
        prompt_ask.side_effect = _prompts(
            [
                "ABCDEF123",  # android serial (free text)
                "",  # package (blank → omit)
                "Open the Strava app",  # intent
            ]
        )
        int_prompt_ask.side_effect = _ints([150])
        confirm_ask.side_effect = _confirms([False, True])  # verbose=False, proceed=True

        with (
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_devices", return_value=[]),
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_packages", return_value=[]),
            _mock_questionary_selects(["run", "android"]),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["command"], "run")
        self.assertEqual(result["platform"], "android")
        self.assertEqual(result["serial"], "ABCDEF123")
        self.assertEqual(result["intent"], "Open the Strava app")
        self.assertEqual(result["max_steps"], 150)
        self.assertFalse(result["verbose"])

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_full_flow_demo_ios_with_ios_flags(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        # Select menus: demo → ios → xcuitest. Free-text prompts for
        # device (detection empty), bundle id, and intent.
        prompt_ask.side_effect = _prompts(
            [
                "SIM-UDID-1234",  # device identifier (free text)
                "com.example.swiggy",  # bundle id from package picker (free text)
                "Tap the challenges tab",  # intent
            ]
        )
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([True, True])  # verbose=True, proceed=True

        with (
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_devices", return_value=[]),
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_packages", return_value=[]),
            _mock_questionary_selects(["demo", "ios", "xcuitest"]),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["command"], "demo")
        self.assertEqual(result["platform"], "ios")
        self.assertEqual(result["ios_device_identifier"], "SIM-UDID-1234")
        self.assertEqual(result["ios_bundle_identifier"], "com.example.swiggy")
        self.assertEqual(result["ios_automation_backend"], "xcuitest")
        self.assertEqual(result["intent"], "Tap the challenges tab")
        self.assertTrue(result["verbose"])


class InteractiveWizardControlFlowTest(unittest.TestCase):
    """
    Cover branches where the user aborts or picks `explore`.
    """

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_abort_at_final_confirmation_returns_none(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        # blank serial, blank package, then intent
        prompt_ask.side_effect = _prompts(["", "", "Open app"])
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, False])  # verbose=False, proceed=False

        with (
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_devices", return_value=[]),
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_packages", return_value=[]),
            _mock_questionary_selects(["run", "android"]),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        self.assertIsNone(result)

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_empty_intent_reprompts(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        # Serial blank, package blank, then intent: blank, blank-whitespace,
        # real. Wizard must loop until intent is non-empty.
        prompt_ask.side_effect = _prompts(["", "", "", "   ", "real intent text"])
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        with (
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_devices", return_value=[]),
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_packages", return_value=[]),
            _mock_questionary_selects(["run", "android"]),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["intent"], "real intent text")


class DeviceAutoDetectionTest(unittest.TestCase):
    """
    Cover the best-effort device detection helpers.
    """

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_detection_failure_does_not_crash_wizard(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        # blank serial, blank package, then intent
        prompt_ask.side_effect = _prompts(["", "", "intent"])
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("adb not installed")

        with (
            patch.object(
                InteractiveWizard,
                "_InteractiveWizard__detect_adb_devices",
                side_effect=explode,
            ),
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_packages", return_value=[]),
            _mock_questionary_selects(["run", "android"]),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["command"], "run")

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_detected_devices_render_as_select_menu(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        # Select menu returns command=run, platform=android, then the
        # second detected serial via the device picker.
        prompt_ask.side_effect = _prompts(["", "intent"])  # blank package, then intent
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        with (
            patch.object(
                InteractiveWizard,
                "_InteractiveWizard__detect_devices",
                return_value=["SERIAL-ONE", "SERIAL-TWO"],
            ),
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_packages", return_value=[]),
            _mock_questionary_selects(["run", "android", "SERIAL-TWO"]),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["serial"], "SERIAL-TWO")

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_device_picker_skip_option_omits_serial(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        # Device picker returns the literal "skip (use default)" option
        # → wizard omits the serial key from the result.
        prompt_ask.side_effect = _prompts(["", "intent"])  # blank package, then intent
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        with (
            patch.object(
                InteractiveWizard,
                "_InteractiveWizard__detect_devices",
                return_value=["SERIAL-ONE", "SERIAL-TWO"],
            ),
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_packages", return_value=[]),
            _mock_questionary_selects(["run", "android", "skip (use default)"]),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertNotIn("serial", result)

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_device_picker_manual_entry_branch(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        # Device picker returns "enter manually" — the wizard then
        # prompts for a free-text serial via Prompt.ask.
        # manual serial, then blank package, then intent
        prompt_ask.side_effect = _prompts(["CUSTOM-SERIAL", "", "intent"])
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        with (
            patch.object(
                InteractiveWizard,
                "_InteractiveWizard__detect_devices",
                return_value=["SERIAL-ONE", "SERIAL-TWO"],
            ),
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_packages", return_value=[]),
            _mock_questionary_selects(["run", "android", "enter manually"]),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["serial"], "CUSTOM-SERIAL")


class PackagePickerTest(unittest.TestCase):
    """
    Cover the installed-package picker added after device selection.
    """

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_detected_android_packages_render_in_picker(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        """Selecting a detected package threads it into ``--package``."""

        # device free-text (blank), then intent. Package step uses select.
        prompt_ask.side_effect = _prompts(["", "intent"])
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        with (
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_devices", return_value=[]),
            patch.object(
                InteractiveWizard,
                "_InteractiveWizard__detect_packages",
                return_value=["com.example.alpha", "com.example.beta"],
            ),
            _mock_questionary_selects(
                ["run", "android", "com.example.beta"],
            ),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["package"], "com.example.beta")

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_detected_ios_bundles_pick_into_bundle_identifier_key(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        """iOS package picker writes ``ios_bundle_identifier``, not ``package``."""

        # device free-text (blank), then intent. Package + backend use selects.
        prompt_ask.side_effect = _prompts(["", "intent"])
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        with (
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_devices", return_value=[]),
            patch.object(
                InteractiveWizard,
                "_InteractiveWizard__detect_packages",
                return_value=["com.apple.mobilesafari", "com.example.app"],
            ),
            # device free-text falls through (no detect), so no extra prompt.
            # Selects (in order): command, platform, ios backend, package.
            _mock_questionary_selects(
                ["demo", "ios", "xcuitest", "com.example.app"],
            ),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["ios_bundle_identifier"], "com.example.app")
        self.assertNotIn("package", result)

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_skip_option_omits_package_key(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        """Picking ``skip`` leaves the result without a ``package`` key."""

        # device free-text (blank), then intent.
        prompt_ask.side_effect = _prompts(["", "intent"])
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        with (
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_devices", return_value=[]),
            patch.object(
                InteractiveWizard,
                "_InteractiveWizard__detect_packages",
                return_value=["com.example.alpha"],
            ),
            _mock_questionary_selects(
                ["run", "android", "skip (use default)"],
            ),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertNotIn("package", result)

    @patch("fathom.runtime.command.wizard.Confirm.ask")
    @patch("fathom.runtime.command.wizard.IntPrompt.ask")
    @patch("fathom.runtime.command.wizard.Prompt.ask")
    def test_manual_entry_branch_in_package_picker(
        self,
        prompt_ask: Any,
        int_prompt_ask: Any,
        confirm_ask: Any,
    ) -> None:
        """``enter manually`` collects a free-text package via Prompt.ask."""

        # prompts: device (blank), manual package entry, then intent.
        prompt_ask.side_effect = _prompts(["", "com.custom.pkg", "intent"])
        int_prompt_ask.side_effect = _ints([100])
        confirm_ask.side_effect = _confirms([False, True])

        with (
            patch.object(InteractiveWizard, "_InteractiveWizard__detect_devices", return_value=[]),
            patch.object(
                InteractiveWizard,
                "_InteractiveWizard__detect_packages",
                return_value=["com.example.alpha"],
            ),
            _mock_questionary_selects(
                ["run", "android", "enter manually"],
            ),
        ):
            result = InteractiveWizard(console=_build_console()).run()

        assert result is not None
        self.assertEqual(result["package"], "com.custom.pkg")

    def test_detect_android_packages_parses_pm_list_output(self) -> None:
        """``adb shell pm list packages -3`` output is parsed and sorted."""

        wizard = InteractiveWizard(console=_build_console())
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = (
            "package:com.example.beta\n"
            "package:com.example.alpha\n"
            "junk line\n"
            "package:\n"  # empty after prefix → skipped
        )

        with (
            patch("fathom.runtime.command.wizard.shutil.which", return_value="/usr/bin/adb"),
            patch(
                "fathom.runtime.command.wizard.subprocess.run",
                return_value=fake_result,
            ) as mock_run,
        ):
            packages = wizard._InteractiveWizard__detect_android_packages(  # type: ignore[attr-defined]
                device_id="DEVICE-1",
            )

        self.assertEqual(packages, ["com.example.alpha", "com.example.beta"])
        # Verify the device id was forwarded as `-s DEVICE-1`.
        argv = mock_run.call_args[0][0]
        self.assertIn("-s", argv)
        self.assertIn("DEVICE-1", argv)
        self.assertIn("-3", argv)

    def test_detect_ios_bundle_identifiers_requires_device_id(self) -> None:
        """No simctl call is made when the user skipped device selection."""

        wizard = InteractiveWizard(console=_build_console())
        with patch("fathom.runtime.command.wizard.subprocess.run") as mock_run:
            packages = wizard._InteractiveWizard__detect_packages(  # type: ignore[attr-defined]
                platform="ios",
                device_id=None,
            )
        self.assertEqual(packages, [])
        mock_run.assert_not_called()

    def test_idb_backend_routes_picker_through_idb_list_apps(self) -> None:
        """When the iOS backend is ``idb``, package detection calls
        ``idb list-apps`` rather than ``xcrun simctl listapps``."""

        wizard = InteractiveWizard(console=_build_console())

        captured: List[List[str]] = []

        def fake_run(argv: List[str], **_kwargs: Any) -> MagicMock:
            captured.append(list(argv))
            result = MagicMock()
            result.returncode = 0
            result.stdout = (
                '{"bundle_id": "com.example.alpha"}\n{"bundle_id": "com.example.beta"}\n'
            )
            return result

        with (
            patch(
                "fathom.runtime.command.wizard.shutil.which",
                side_effect=lambda name: f"/usr/local/bin/{name}",
            ),
            patch("fathom.runtime.command.wizard.subprocess.run", side_effect=fake_run),
        ):
            packages = wizard._InteractiveWizard__detect_packages(  # type: ignore[attr-defined]
                platform="ios",
                device_id="UDID-1",
                ios_backend="idb",
            )

        self.assertEqual(packages, ["com.example.alpha", "com.example.beta"])
        # Only one subprocess call, and it was idb (not xcrun).
        self.assertEqual(len(captured), 1)
        self.assertIn("idb", captured[0][0])
        self.assertIn("--udid", captured[0])
        self.assertIn("UDID-1", captured[0])
        self.assertIn("list-apps", captured[0])

    def test_idb_picker_works_without_device_id(self) -> None:
        """``idb`` resolves a default target when ``--udid`` is omitted,
        unlike simctl which requires a UDID."""

        wizard = InteractiveWizard(console=_build_console())

        captured: List[List[str]] = []

        def fake_run(argv: List[str], **_kwargs: Any) -> MagicMock:
            captured.append(list(argv))
            result = MagicMock()
            result.returncode = 0
            result.stdout = '{"bundle_id": "com.example.foo"}\n'
            return result

        with (
            patch(
                "fathom.runtime.command.wizard.shutil.which",
                side_effect=lambda name: f"/usr/local/bin/{name}",
            ),
            patch("fathom.runtime.command.wizard.subprocess.run", side_effect=fake_run),
        ):
            packages = wizard._InteractiveWizard__detect_packages(  # type: ignore[attr-defined]
                platform="ios",
                device_id=None,
                ios_backend="idb",
            )

        self.assertEqual(packages, ["com.example.foo"])
        # No --udid forwarded since the user skipped device selection.
        self.assertNotIn("--udid", captured[0])

    def test_detection_failure_falls_back_to_free_text(self) -> None:
        """Subprocess errors are swallowed; helper returns []."""

        wizard = InteractiveWizard(console=_build_console())
        with patch(
            "fathom.runtime.command.wizard.shutil.which",
            side_effect=RuntimeError("kaboom"),
        ):
            packages = wizard._InteractiveWizard__detect_packages(  # type: ignore[attr-defined]
                platform="android",
                device_id="DEVICE-1",
            )
        self.assertEqual(packages, [])


class WizardArgvTest(unittest.TestCase):
    """
    Cover the conversion from wizard dict back to argparse-compatible argv.
    """

    def test_run_with_all_flags_emits_expected_argv(self) -> None:
        result: Dict[str, Any] = {
            "command": "run",
            "intent": "Open the app",
            "platform": "ios",
            "ios_device_identifier": "UDID-1",
            "ios_bundle_identifier": "com.example",
            "ios_automation_backend": "xcuitest",
            "max_steps": 120,
            "verbose": True,
        }

        argv = wizard_argv(args=result)

        self.assertEqual(argv[0], "run")
        self.assertEqual(argv[1], "Open the app")
        self.assertIn("--platform", argv)
        self.assertIn("ios", argv)
        self.assertIn("--ios-device-identifier", argv)
        self.assertIn("UDID-1", argv)
        self.assertIn("--ios-bundle-identifier", argv)
        self.assertIn("com.example", argv)
        self.assertIn("--ios-automation-backend", argv)
        self.assertIn("xcuitest", argv)
        self.assertIn("--max-steps", argv)
        self.assertIn("120", argv)
        self.assertIn("--verbose", argv)

    def test_blank_optional_flags_are_omitted(self) -> None:
        result = {
            "command": "demo",
            "intent": "hi",
            "platform": "android",
            "max_steps": 100,
            "verbose": False,
            # No serial, no iOS flags, no ios_* keys.
        }

        argv = wizard_argv(args=result)

        self.assertNotIn("--serial", argv)
        self.assertNotIn("--ios-device-identifier", argv)
        self.assertNotIn("--ios-bundle-identifier", argv)

    def test_missing_command_raises(self) -> None:
        with self.assertRaises(ValueError):
            wizard_argv(args={"intent": "x"})

    def test_package_flag_round_trips(self) -> None:
        argv = wizard_argv(
            args={
                "command": "run",
                "intent": "Open Swiggy",
                "platform": "android",
                "package": "in.swiggy.android",
                "max_steps": 100,
                "verbose": False,
            }
        )

        self.assertIn("--package", argv)
        self.assertIn("in.swiggy.android", argv)

    def test_package_argv_parses_and_threads_through_real_parser(self) -> None:
        from fathom.runtime.command.application import CommandApplication

        application = CommandApplication()
        parser = application._CommandApplication__parser  # type: ignore[attr-defined]

        argv = wizard_argv(
            args={
                "command": "run",
                "intent": "Open Swiggy",
                "platform": "android",
                "package": "in.swiggy.android",
                "max_steps": 100,
                "verbose": False,
            }
        )
        namespace = parser.parse_args(argv)

        self.assertEqual(namespace.package, "in.swiggy.android")


class ArgparseIntegrationTest(unittest.TestCase):
    """
    Verify the wizard argv survives parsing through the real parser so
    wizard output is a fully equivalent alternative to typing the CLI.
    """

    def test_run_wizard_argv_parses_through_real_parser(self) -> None:
        from fathom.runtime.command.application import CommandApplication

        application = CommandApplication()
        parser = application._CommandApplication__parser  # type: ignore[attr-defined]

        result = {
            "command": "run",
            "intent": "Open Swiggy",
            "platform": "android",
            "serial": "A1B2",
            "max_steps": 100,
            "verbose": False,
        }

        namespace = parser.parse_args(wizard_argv(args=result))

        self.assertEqual(namespace.command, "run")
        self.assertEqual(namespace.intent, "Open Swiggy")
        self.assertEqual(namespace.platform, "android")
        self.assertEqual(namespace.serial, "A1B2")
        self.assertEqual(namespace.max_steps, 100)

    def test_demo_wizard_argv_parses_through_real_parser(self) -> None:
        from fathom.runtime.command.application import CommandApplication

        application = CommandApplication()
        parser = application._CommandApplication__parser  # type: ignore[attr-defined]

        result = {
            "command": "demo",
            "intent": "Tap challenges",
            "platform": "ios",
            "ios_device_identifier": "UDID-X",
            "ios_automation_backend": "xcuitest",
            "max_steps": 80,
            "verbose": False,
        }

        namespace = parser.parse_args(wizard_argv(args=result))

        self.assertEqual(namespace.command, "demo")
        self.assertEqual(namespace.intent, "Tap challenges")
        self.assertEqual(namespace.platform, "ios")
        self.assertEqual(namespace.ios_device_identifier, "UDID-X")
        self.assertEqual(namespace.ios_automation_backend, "xcuitest")
