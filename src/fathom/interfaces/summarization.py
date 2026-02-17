"""
Interface for summarization services.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SummarizationPort(ABC):
    """
    Abstract interface for semantic compression of execution traces.
    """

    @abstractmethod
    async def summarize_trace(self, trace: List[Dict[str, Any]]) -> str:
        """
        Compresses a list of trace entries into a concise semantic summary.
        
        Args:
            trace: List of OTA (Observation-Thought-Action) dictionaries.
            
        Returns:
            A single sentence summarizing the progress and state change.
        """
        raise NotImplementedError
