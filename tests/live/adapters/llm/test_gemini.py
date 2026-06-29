from __future__ import annotations

import os
import unittest

import pytest

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.schemas.configuration import LLMConfiguration
from fathom.settings.env import FathomSettings

pytestmark = pytest.mark.release


class LiveGeminiPriorityTest(unittest.IsolatedAsyncioTestCase):
    """
    Live Vertex Gemini checks for priority request routing.
    """

    async def test_priority_request_reports_priority_traffic(self) -> None:
        """
        A live Vertex call must request priority and report priority traffic when available.
        """

        if os.environ.get("FATHOM_LIVE_LLM") != "1":
            pytest.skip("Set FATHOM_LIVE_LLM=1 to run live LLM tests.")

        settings = FathomSettings()
        credentials = settings.google_credentials_dict or settings.google_application_credentials
        configuration = LLMConfiguration(
            api_key=settings.gemini_api_key,
            credentials=credentials,
            project_id=settings.vertex_project_id,
            location=settings.vertex_location,
            model=settings.gemini_model,
            use_cache=False,
            max_retries=0,
            timeout=60.0,
            temperature=0.0,
            thinking_level="low",
        )
        if configuration.api_key:
            pytest.skip("Priority traffic assertion is Vertex-specific for this live test.")

        llm = GeminiLLM(configuration=configuration)
        try:
            result = await llm.generate(
                use_cache=False,
                prompt=['Reply with exactly this JSON object: {"ok": true}'],
            )
        finally:
            await llm.cleanup()

        self.assertEqual(result.metrics["priority_used"], 1.0)
        self.assertEqual(result.metrics["priority_observed"], 1.0)
        self.assertGreater(result.metrics["total_tokens"], 0.0)


if __name__ == "__main__":
    unittest.main()
