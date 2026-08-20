from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SummarizationPort(ABC):
    """
    Semantic compression of execution traces into a one-sentence progress summary.
    """

    @abstractmethod
    async def summarize_trace(self, trace: List[Dict[str, Any]]) -> str:
        """
        Compress a trace of OTA (Observation-Thought-Action) entries into one summarizing sentence.
        """

        raise NotImplementedError
