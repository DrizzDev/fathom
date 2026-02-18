from __future__ import annotations

import logging
from typing import Any, Dict, List

from fathom.interfaces.llm import LLMPort
from fathom.interfaces.summarization import SummarizationPort

logger = logging.getLogger(__name__)


class LLMSummarizer(SummarizationPort):
    """
    Uses an LLM to semantically compress execution traces.
    """

    def __init__(self, llm: LLMPort) -> None:
        """
        Initialize with an LLM provider.
        """

        self.__llm = llm

    async def summarize_trace(self, trace: List[Dict[str, Any]]) -> str:
        """
        Generates a semantic summary of the trace using the LLM.
        """

        if not trace:
            return "No actions performed."

        # Format trace for the model
        trace_text = "\n".join(
            f"- Action: {t.get('action', 'Unknown')}\n  Outcome: {t.get('observation', 'N/A')}"
            for t in trace
        )

        prompt = [
            "TASK: Compress this execution trace into a single, high-density milestone sentence.",
            "REQUIREMENTS:",
            "1. Focus on OUTCOMES and STATE CHANGES (e.g., 'Successfully logged in', 'Failed to find search bar').",
            "2. Discard routine navigation details (e.g., 'scrolled', 'tapped X') unless they failed.",
            "3. capture specific knowledge gained (e.g., 'Found article about X').",
            f"\nTRACE:\n{trace_text}",
        ]

        try:
            result = await self.__llm.generate(
                prompt=prompt,
                system_instruction="You are a state-tracking expert. Synthesize agent logs into semantic milestones.",
            )
            return result.content.strip()
        except Exception as exception:
            logger.error(f"Summarization failed: {exception}")
            return f"Executed {len(trace)} steps (Summarization failed)."
