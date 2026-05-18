from __future__ import annotations

import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from fathom.adapters.icon.noop import NoopIconDetector
from fathom.adapters.icon.template import TemplateIconDetector
from fathom.adapters.journal.jsonl import JsonRuntimeJournal
from fathom.adapters.journal.noop import NoopRuntimeJournal
from fathom.adapters.localization.document.layout import DocumentAiLayoutLocalizer
from fathom.adapters.localization.gemini.vision import GeminiVisionLocalizer
from fathom.adapters.ocr.document.adapter import DocumentAiOcr
from fathom.adapters.ocr.noop import NoopOcr
from fathom.adapters.perception.overlay.noop import NoopOverlayDetector
from fathom.adapters.perception.overlay.pixel import PixelOverlayDetector
from fathom.core.config.loader import RuntimeConfigLoader
from fathom.interfaces.llm import LLMPort
from fathom.runtime.adapters import AdapterAssembly
from fathom.settings.env import FathomSettings


class _FakeLlm(LLMPort):
    """
    :class:`LLMPort` test double held only as a constructor reference.

    :class:`AdapterAssembly` injects the LLM into vision localizers but
    these tests never trigger a call. ``generate`` therefore raises so a
    regression that quietly starts dispatching during assembly is loud,
    not silent.
    """

    @property
    def model_name(self) -> str:
        """
        Stable identifier surfaced in cache keys and structured logs.
        """

        return "fake-llm"

    async def generate(  # type: ignore[no-untyped-def]
        self,
        *,
        use_cache,
        prompt,
        tools=None,
        system_instruction=None,
        conversation_history=None,
    ):
        """
        Never called from this test suite. A regression that wires the
        LLM into the assembly hot path would surface as ``NotImplementedError``.
        """

        raise NotImplementedError

    async def cleanup(self) -> None:
        """
        Port-required teardown hook. No resources are held by this fake.
        """

        return None


class AdapterAssemblyTest(unittest.TestCase):
    """
    Pins env-driven composition of every adapter the runtime depends on.

    :class:`AdapterAssembly` is the only place env vars cross the
    hexagonal boundary into adapter selection. The tests cover all four
    selection axes: OCR (noop / Document AI / fall-back-on-missing-creds),
    overlay detector (noop / pixel), ensemble localizer (empty / typed
    membership), and journal (noop / JSONL with directory). Document AI
    construction is patched because the real client requires ADC.
    """

    @staticmethod
    def __assemble(
        *,
        journal_directory: Optional[Path] = None,
        **overrides: object,
    ) -> AdapterAssembly:
        """
        Build an :class:`AdapterAssembly` from a synthetic
        :class:`FathomSettings` payload and an optional journal directory.

        ``os.environ`` is cleared and ``_env_file`` set to ``None`` for
        the duration of settings construction so the developer's local
        ``.env`` cannot leak ``FATHOM_*`` values into the test. Without
        this isolation, values like ``FATHOM_ENSEMBLE_LOCALIZER=true``
        would defeat default-case assertions.
        """

        with mock.patch.dict("os.environ", {}, clear=True):
            settings = FathomSettings(_env_file=None, **overrides)
        return AdapterAssembly(
            loader=RuntimeConfigLoader(settings=settings),
            llm=_FakeLlm(),
            workflow_id="run-abc",
            journal_directory=journal_directory,
        )

    def test_ocr_defaults_to_noop_when_disabled(self) -> None:
        """
        OCR disabled yields the noop adapter regardless of Document AI credentials.
        """

        self.assertIsInstance(self.__assemble().ocr(), NoopOcr)

    def test_ocr_returns_document_ai_when_enabled_with_credentials(self) -> None:
        """
        OCR enabled plus full Document AI credentials produces the Document AI adapter.
        """

        assembly = self.__assemble(
            observation_ocr_enabled=True,
            document_ai_project="vision-478905",
            document_ai_location="us",
            document_ai_processor="proc-1",
        )

        with mock.patch(
            "fathom.adapters.ocr.document.adapter.documentai_v1.DocumentProcessorServiceClient",
            return_value=mock.MagicMock(),
        ):
            adapter = assembly.ocr()

        self.assertIsInstance(adapter, DocumentAiOcr)

    def test_ocr_falls_back_to_noop_when_credentials_missing(self) -> None:
        """
        OCR enabled with incomplete credentials falls back to the noop adapter.
        """

        assembly = self.__assemble(
            observation_ocr_enabled=True,
            document_ai_project="vision-478905",
        )

        self.assertIsInstance(assembly.ocr(), NoopOcr)

    def test_overlay_returns_noop_when_disabled(self) -> None:
        """
        Overlay disabled yields the noop detector. Default is on for
        bring-up, so the test must explicitly disable it.
        """

        assembly = self.__assemble(observation_overlay_enabled=False)
        self.assertIsInstance(assembly.overlay(), NoopOverlayDetector)

    def test_overlay_returns_pixel_when_enabled(self) -> None:
        """
        Overlay enabled yields the pixel implementation.
        """

        assembly = self.__assemble(observation_overlay_enabled=True)

        self.assertIsInstance(assembly.overlay(), PixelOverlayDetector)

    def test_icons_returns_noop_when_disabled(self) -> None:
        """
        Icon detector disabled yields the noop adapter. Default is on
        for bring-up, so the test must explicitly disable it.
        """

        assembly = self.__assemble(observation_icon_enabled=False)
        self.assertIsInstance(assembly.icons(), NoopIconDetector)

    def test_icons_returns_template_when_enabled(self) -> None:
        """
        Icon detector enabled binds to the template adapter (registry may be empty).
        """

        detector = self.__assemble(observation_icon_enabled=True).icons()

        self.assertIsInstance(detector, TemplateIconDetector)

    def test_ensemble_empty_when_disabled(self) -> None:
        """
        With the ensemble flag off, the service is constructed with no
        members. Default is on for bring-up, so the test must
        explicitly disable it.
        """

        assembly = self.__assemble(ensemble_localizer_enabled=False)
        self.assertEqual(assembly.ensemble().members, ())

    def test_ensemble_builds_configured_members(self) -> None:
        """
        Enabled ensemble with a member list yields the matching localizer
        instances. ``document_ai_layout`` consumes OCR tokens and is
        therefore only assembled when OCR is fully configured — the
        fixture supplies Document AI credentials so the member is built
        instead of silently dropped.
        """

        assembly = self.__assemble(
            ensemble_localizer_enabled=True,
            ensemble_localizer_members="gemini_vision,document_ai_layout",
            observation_ocr_enabled=True,
            document_ai_project="vision-478905",
            document_ai_location="us",
            document_ai_processor="proc-1",
        )

        members = assembly.ensemble().members

        self.assertEqual(len(members), 2)
        self.assertIsInstance(members[0], GeminiVisionLocalizer)
        self.assertIsInstance(members[1], DocumentAiLayoutLocalizer)

    def test_ensemble_drops_document_ai_layout_when_ocr_disabled(self) -> None:
        """
        ``document_ai_layout`` requires OCR; the assembly must drop it
        when OCR is disabled or unconfigured rather than carrying a
        member that can only ever return ``None``.
        """

        assembly = self.__assemble(
            ensemble_localizer_enabled=True,
            ensemble_localizer_members="gemini_vision,document_ai_layout",
        )

        members = assembly.ensemble().members

        self.assertEqual(len(members), 1)
        self.assertIsInstance(members[0], GeminiVisionLocalizer)

    def test_journal_returns_noop_when_local_disabled(self) -> None:
        """
        Without the local-journal flag, the journal port is the noop adapter.
        """

        self.assertIsInstance(self.__assemble().journal(), NoopRuntimeJournal)

    def test_journal_returns_noop_when_directory_missing(self) -> None:
        """
        Enabling the local journal without a directory still resolves to the noop adapter.
        """

        assembly = self.__assemble(journal_local_enabled=True)

        self.assertIsInstance(assembly.journal(), NoopRuntimeJournal)

    def test_journal_returns_jsonl_when_enabled_with_directory(self) -> None:
        """
        Local-journal + directory configured produces the JSONL adapter.
        """

        assembly = self.__assemble(
            journal_local_enabled=True,
            journal_directory=Path("/tmp"),
        )

        self.assertIsInstance(assembly.journal(), JsonRuntimeJournal)
