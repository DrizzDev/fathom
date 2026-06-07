from __future__ import annotations

import os
import unittest
from typing import List, Tuple

import pytest
from dotenv import load_dotenv

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.constants.abort import DEFAULT_ABORT_DETECTOR_MODEL
from fathom.core.services.abort.factory import AbortDetectorFactory
from fathom.schemas.configuration import LLMConfiguration
from fathom.settings.env import PROJECT_ROOT

_LIVE_ENABLED_FLAG: str = "FATHOM_LIVE_LLM"


class AbortLiveConfigurationFactory:
    """
    Builds the live LLM configuration for the abort-detector live tests.
    """

    @staticmethod
    def enabled() -> bool:
        """
        Return whether the live flag is set so live calls are allowed to proceed.
        """

        load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env.test", override=False)

        return os.getenv(_LIVE_ENABLED_FLAG) == "1"

    @staticmethod
    def build() -> LLMConfiguration:
        """
        Build the abort-detector-specific Gemini configuration from environment variables.
        """

        api_key = os.getenv("GEMINI_API_KEY")

        project_id = (
            os.getenv("GCP_PROJECT")
            or os.getenv("VERTEX_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS_PATH"
        )

        if not api_key and not project_id and not credentials:
            pytest.skip("Live abort tests require GEMINI_API_KEY or Google Cloud credentials.")

        return LLMConfiguration(
            timeout=10.0,
            max_retries=1,
            use_cache=False,
            temperature=0.0,
            api_key=api_key,
            thinking_level="low",
            project_id=project_id,
            credentials=credentials,
            model=DEFAULT_ABORT_DETECTOR_MODEL,
            location=os.getenv("FATHOM_LIVE_LLM_LOCATION", "global"),
        )


class LLMAbortDetectorLiveTest(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end verification of the abort detector against real Gemini.
    """

    __CANONICAL_ABORT_CASES: List[Tuple[str, str]] = [
        ("close_the_execution", "close the execution"),
        ("stop_the_workflow", "stop the workflow"),
        ("cancel_this_run", "cancel this run"),
        ("abort_alone", "abort"),
    ]

    __CANONICAL_NON_ABORT_CASES: List[Tuple[str, str]] = [
        ("tap_on_stop", "tap on stop"),
        ("click_cancel", "click cancel"),
        ("go_to_settings", "go to settings"),
        ("tap_on_terminate", "tap on terminate"),
        ("tap_on_cross_button", "tap on cross button"),
        ("press_the_close_button", "press the close button"),
    ]

    async def asyncSetUp(self) -> None:
        """
        Skip if live opt-in is unset; build the live detector otherwise.
        """

        if not AbortLiveConfigurationFactory.enabled():
            self.skipTest(f"Set {_LIVE_ENABLED_FLAG}=1 to run live abort tests.")

        self.__llm = GeminiLLM(configuration=AbortLiveConfigurationFactory.build())
        self.__detector = AbortDetectorFactory.build(llm=self.__llm)

    async def asyncTearDown(self) -> None:
        """
        Release the live LLM client.
        """

        await self.__llm.cleanup()

    async def test_warmup_does_not_raise(self) -> None:
        """
        Warmup against the real model must complete without raising.
        """

        await self.__detector.warmup()

    async def test_every_canonical_abort_phrase_is_classified_as_aborted(self) -> None:
        """
        Each canonical abort phrase must round-trip to aborted=True.
        """

        for label, response in self.__CANONICAL_ABORT_CASES:
            with self.subTest(case=label):
                decision = await self.__detector.aborted(response=response)
                self.assertTrue(
                    decision.aborted,
                    msg=f"Expected ABORT for {response!r}, got decision={decision}",
                )

    async def test_every_canonical_ui_directive_is_not_classified_as_aborted(self) -> None:
        """
        Each canonical UI-directive phrase must round-trip to aborted=False.
        """

        for label, response in self.__CANONICAL_NON_ABORT_CASES:
            with self.subTest(case=label):
                decision = await self.__detector.aborted(response=response)
                self.assertFalse(
                    decision.aborted,
                    msg=f"Expected NOT-ABORT for {response!r}, got decision={decision}",
                )
