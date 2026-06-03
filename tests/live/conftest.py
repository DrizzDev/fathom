from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import pytest
from dotenv import load_dotenv

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import LLMConfiguration

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_LIVE_TESTS_ENV_FLAG: str = "FATHOM_RUN_LIVE_TESTS"
_LIVE_TESTS_DIRECTORY: Path = Path(__file__).resolve().parent


class LiveLlmConfigurationFactory:
    """
    Builds explicit live LLM configuration for gated tests.
    """

    @staticmethod
    def enabled() -> bool:
        """
        Return whether live LLM tests are explicitly enabled.
        """

        LiveLlmConfigurationFactory.load_environment()
        return os.getenv("FATHOM_LIVE_LLM") == "1"

    @staticmethod
    def load_environment() -> None:
        """
        Load local live-test environment files without overriding shell variables.
        """

        load_dotenv(dotenv_path=PROJECT_ROOT / ".env.test", override=False)
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    @staticmethod
    def build() -> LLMConfiguration:
        """
        Build the Gemini configuration from environment variables.
        """

        LiveLlmConfigurationFactory.load_environment()

        api_key = os.getenv("GEMINI_API_KEY")
        project_id = (
            os.getenv("VERTEX_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
        )
        credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS_PATH"
        )

        if not api_key and not project_id and not credentials:
            pytest.skip("Live LLM tests require GEMINI_API_KEY or Google Cloud credentials.")

        return LLMConfiguration(
            api_key=api_key,
            project_id=project_id,
            credentials=credentials,
            model=os.getenv("FATHOM_LIVE_LLM_MODEL", "gemini-3-flash-preview"),
            location=os.getenv("FATHOM_LIVE_LLM_LOCATION", "global"),
            use_cache=False,
            max_retries=int(os.getenv("FATHOM_LIVE_LLM_MAX_RETRIES", "1")),
            timeout=float(os.getenv("FATHOM_LIVE_LLM_TIMEOUT", "60")),
            temperature=float(os.getenv("FATHOM_LIVE_LLM_TEMPERATURE", "0.2")),
            thinking_level="low",
        )


class LiveTestGuard:
    """
    Collection-time guard that keeps live tests out of the default test run.
    """

    @staticmethod
    def is_enabled() -> bool:
        """
        Return True only when the explicit opt-in env flag is set.
        """

        return os.environ.get(_LIVE_TESTS_ENV_FLAG) == "1"

    @staticmethod
    def is_live_item(*, item: pytest.Item) -> bool:
        """
        Return True only for items whose source file lives under tests/live/.

        Conftest hooks fire at session scope, so the items list is global. Without
        this filter the skip would land on unit tests too, producing a fake-green
        run where every test silently skips.
        """

        try:
            item_path = item.path.resolve()
        except (AttributeError, OSError):
            return False

        try:
            item_path.relative_to(_LIVE_TESTS_DIRECTORY)
        except ValueError:
            return False

        return True


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """
    Skip only items under tests/live/ unless the live-tests env flag is set.
    """

    _ = config
    if LiveTestGuard.is_enabled():
        return

    skip_marker = pytest.mark.skip(
        reason=(
            f"Live tests are opt-in. Set {_LIVE_TESTS_ENV_FLAG}=1 to run them; "
            "they call the real Gemini backend and require credentials."
        ),
    )
    for item in items:
        if LiveTestGuard.is_live_item(item=item):
            item.add_marker(skip_marker)


@pytest.fixture
async def live_llm() -> "AsyncIterator[LLMPort]":
    """
    Provide a real LLM adapter for explicitly gated live tests.
    """

    if not LiveLlmConfigurationFactory.enabled():
        pytest.skip("Set FATHOM_LIVE_LLM=1 to run live LLM tests.")

    llm = GeminiLLM(configuration=LiveLlmConfigurationFactory.build())
    try:
        yield llm
    finally:
        await llm.cleanup()
