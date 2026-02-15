"""
Adapter to make LLMPort compatible with IVisionProvider interface.

This bridges the new hexagonal architecture (LLMPort) with the old
agent components that expect IVisionProvider.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.interfaces.llm import LLMPort
from fathom.schemas.results import AnalysisResult


class LLMVisionProvider:
    """
    Adapter that makes LLMPort compatible with IVisionProvider.
    
    This allows the old agent components (StepPlanner, GeminiVisionTool)
    to work with the new LLMPort interface.
    """

    def __init__(self, llm: LLMPort) -> None:
        """
        Initialize adapter with LLM port.
        
        Args:
            llm: LLM port to wrap
        """
        self.__llm = llm

    async def analyze(
        self,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Analyze using LLM port.
        
        This method signature matches IVisionProvider.analyze() exactly.
        """
        return await self.__llm.analyze(
            system_instruction=system_instruction,
            user_content=user_content,
            tools=tools,
        )

    async def cleanup(self) -> None:
        """Cleanup LLM resources."""
        await self.__llm.cleanup()
