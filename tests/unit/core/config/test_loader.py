from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from fathom.core.config.loader import RuntimeConfigLoader
from fathom.schemas.localization import EnsembleMemberName
from fathom.settings.env import FathomSettings


def _settings(**overrides: Any) -> FathomSettings:
    """
    Build a :class:`FathomSettings` instance with explicit overrides.

    The factory hard-isolates settings construction from the developer's
    machine: ``os.environ`` is cleared for the duration of the call and
    ``_env_file`` is set to ``None`` so the local ``.env`` is not
    consulted. Without this, env-source loading (a behavior of
    ``pydantic-settings``) would leak real values like
    ``FATHOM_DOCUMENT_AI_*`` into tests that intend to pin pure defaults.
    """

    with mock.patch.dict("os.environ", {}, clear=True):
        return FathomSettings(_env_file=None, **overrides)


class RuntimeConfigLoaderTest(unittest.TestCase):
    """
    Pins for the settings-driven :class:`RuntimeConfigLoader`.

    The loader is the composition root's source of truth for runtime
    knobs: OCR enable, Document AI credentials, ensemble-localizer
    membership, and journal toggles. Every flag has a strict parsing
    contract — partial Document AI credentials must collapse to ``None``
    instead of producing a half-configured adapter, and unknown ensemble
    members must raise rather than silently being dropped.
    """

    def test_perception_defaults_keep_cv_disabled(self) -> None:
        """
        Production defaults keep CV disabled while OCR, icon, and overlay remain enabled.
        """

        config = RuntimeConfigLoader(settings=_settings()).perception()

        self.assertTrue(config.ocr.enabled)
        self.assertIsNone(config.ocr.document_ai)
        self.assertFalse(config.cv.enabled)
        self.assertTrue(config.icon.enabled)
        self.assertTrue(config.overlay.enabled)
        self.assertFalse(config.keyboard.enabled)
        self.assertFalse(config.journal.local_enabled)

    def test_perception_subsystem_can_be_flipped_off_via_env(self) -> None:
        """
        Operators can disable any individual subsystem by setting its
        env flag.
        """

        config = RuntimeConfigLoader(
            settings=_settings(
                observation_ocr_enabled=False,
                observation_cv_enabled=False,
                observation_icon_enabled=False,
                observation_overlay_enabled=False,
                observation_keyboard_enabled=False,
            ),
        ).perception()

        self.assertFalse(config.ocr.enabled)
        self.assertFalse(config.cv.enabled)
        self.assertFalse(config.icon.enabled)
        self.assertFalse(config.overlay.enabled)
        self.assertFalse(config.keyboard.enabled)

    def test_perception_enables_ocr_only_with_full_credentials(self) -> None:
        """
        OCR=True with full Document AI credentials produces a populated
        :class:`DocumentAiCredentials`.
        """

        config = RuntimeConfigLoader(
            settings=_settings(
                observation_ocr_enabled=True,
                document_ai_project="vision-478905",
                document_ai_location="us",
                document_ai_processor="abc123",
            ),
        ).perception()

        self.assertTrue(config.ocr.enabled)
        self.assertIsNotNone(config.ocr.document_ai)
        assert config.ocr.document_ai is not None
        self.assertEqual(config.ocr.document_ai.project, "vision-478905")
        self.assertEqual(config.ocr.document_ai.location, "us")
        self.assertEqual(config.ocr.document_ai.processor, "abc123")

    def test_perception_drops_partial_document_ai_credentials(self) -> None:
        """
        Partial Document AI credentials yield None and do not crash the loader.
        """

        config = RuntimeConfigLoader(
            settings=_settings(
                observation_ocr_enabled=True,
                document_ai_project="vision-478905",
                document_ai_processor="abc",
            ),
        ).perception()

        self.assertTrue(config.ocr.enabled)
        self.assertIsNone(config.ocr.document_ai)

    def test_perception_threads_inline_google_credentials_dict(self) -> None:
        """
        Inline ``GOOGLE_APPLICATION_CREDENTIALS_JSON`` material rides into :class:`DocumentAiCredentials` so the Document AI adapter authenticates against the same identity Gemini already uses — no reliance on ambient ADC.
        """

        payload = {"type": "service_account", "project_id": "vision-478905"}
        config = RuntimeConfigLoader(
            settings=_settings(
                observation_ocr_enabled=True,
                document_ai_project="vision-478905",
                document_ai_location="us",
                document_ai_processor="abc123",
                google_credentials_json=payload,
            ),
        ).perception()

        assert config.ocr.document_ai is not None
        self.assertEqual(config.ocr.document_ai.credentials, payload)

    def test_perception_threads_google_application_credentials_path(self) -> None:
        """
        When only the credentials *file path* is configured, the loader
        threads that path through. The adapter reads the file at boot.
        """

        config = RuntimeConfigLoader(
            settings=_settings(
                observation_ocr_enabled=True,
                document_ai_project="vision-478905",
                document_ai_location="us",
                document_ai_processor="abc123",
                google_application_credentials="/tmp/key.json",  # nosec - test fixture
            ),
        ).perception()

        assert config.ocr.document_ai is not None
        self.assertEqual(config.ocr.document_ai.credentials, "/tmp/key.json")

    def test_perception_inline_credentials_dict_wins_over_path(self) -> None:
        """
        When both inline JSON and a path are set the inline payload wins
        so server deployments where the secret lives in the env do not
        need an on-disk key file.
        """

        payload = {"type": "service_account", "project_id": "vision-478905"}
        config = RuntimeConfigLoader(
            settings=_settings(
                observation_ocr_enabled=True,
                document_ai_project="vision-478905",
                document_ai_location="us",
                document_ai_processor="abc123",
                google_credentials_json=payload,
                google_application_credentials="/tmp/key.json",  # nosec - test fixture
            ),
        ).perception()

        assert config.ocr.document_ai is not None
        self.assertEqual(config.ocr.document_ai.credentials, payload)

    def test_perception_cv_icon_overlay_keyboard_flags_route_through(self) -> None:
        """
        Each perception subsystem flag round-trips into its nested config.
        """

        config = RuntimeConfigLoader(
            settings=_settings(
                observation_cv_enabled=True,
                observation_icon_enabled=True,
                observation_overlay_enabled=True,
                observation_keyboard_enabled=True,
            ),
        ).perception()

        self.assertTrue(config.cv.enabled)
        self.assertTrue(config.icon.enabled)
        self.assertTrue(config.overlay.enabled)
        self.assertTrue(config.keyboard.enabled)

    def test_journal_local_flag_routes_through(self) -> None:
        """
        Journal local-enabled flag round-trips into the nested config.
        """

        config = RuntimeConfigLoader(
            settings=_settings(journal_local_enabled=True),
        ).perception()

        self.assertTrue(config.journal.local_enabled)

    def test_localization_defaults_enable_full_ensemble(self) -> None:
        """
        Bring-up default: the ensemble vision-localizer is enabled with
        both members (Gemini-vision + DocumentAI-layout) so the supervise
        cascade has a name-based fallback when snap fails.
        """

        config = RuntimeConfigLoader(settings=_settings()).localization()

        self.assertTrue(config.enabled)
        self.assertEqual(
            config.members,
            (EnsembleMemberName.GEMINI_VISION, EnsembleMemberName.DOCUMENT_AI_LAYOUT),
        )

    def test_localization_can_be_flipped_off_via_env(self) -> None:
        """
        Operators can disable the ensemble localizer by setting the env
        flag to false.
        """

        config = RuntimeConfigLoader(
            settings=_settings(ensemble_localizer_enabled=False),
        ).localization()

        self.assertFalse(config.enabled)
        self.assertEqual(config.members, ())

    def test_localization_parses_valid_members(self) -> None:
        """
        Comma-separated valid members are parsed into the EnsembleMemberName tuple.
        """

        config = RuntimeConfigLoader(
            settings=_settings(
                ensemble_localizer_enabled=True,
                ensemble_localizer_members="gemini_vision, document_ai_layout",
            ),
        ).localization()

        self.assertTrue(config.enabled)
        self.assertEqual(
            config.members,
            (EnsembleMemberName.GEMINI_VISION, EnsembleMemberName.DOCUMENT_AI_LAYOUT),
        )

    def test_localization_rejects_unknown_member(self) -> None:
        """
        Any unknown ensemble-member name causes a ValueError naming the supported set.
        """

        with self.assertRaises(ValueError) as caught:
            RuntimeConfigLoader(
                settings=_settings(
                    ensemble_localizer_enabled=True,
                    ensemble_localizer_members="gemini_vision,bogus_member",
                ),
            ).localization()
        self.assertIn("bogus_member", str(caught.exception))

    def test_localization_enabled_flag_with_empty_member_list_returns_empty_tuple(self) -> None:
        """
        Enabling the flag with an explicit empty member list yields an enabled config with an empty tuple — the default member set is only chosen when the operator leaves the env value at its bring-up default.
        """

        config = RuntimeConfigLoader(
            settings=_settings(
                ensemble_localizer_enabled=True,
                ensemble_localizer_members="",
            ),
        ).localization()

        self.assertTrue(config.enabled)
        self.assertEqual(config.members, ())
