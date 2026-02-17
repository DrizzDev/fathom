"""
LLM-based summarization adapter.
"""

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
        """Initialize with an LLM provider."""
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
            "TASK: Summarize the following execution trace into a single, concise sentence.",
            "FOCUS: Capture the key actions taken and the resulting state change.",
            "CONSTRAINT: Do not list every step. Synthesize the progress.",
            f"\nTRACE:\n{trace_text}",
        ]

        try:
            # We use a lower temperature for deterministic summarization if possible,
            # but LLMPort abstraction might fix it. The prompt constraints help.
            result = await self.__llm.generate(
                prompt=prompt,
                system_instruction="You are a precise technical summarizer for AI agent logs.",
            )
            return result.content.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Executed {len(trace)} steps (Summarization failed)."
