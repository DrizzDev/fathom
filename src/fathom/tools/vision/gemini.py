from __future__ import annotations

import asyncio
import hashlib
import json
import time
from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.interfaces import IImageStorage, IMemoryProvider, IPromptProvider, IVisionProvider
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.tools.vision.base import VisionTool
from fathom.utils.image import ImageProcessor

logger = getLogger(__name__)


class GeminiVisionTool(VisionTool):
    """
    SOLID Orchestrator for Vision Analysis.
    """

    def __init__(
        self,
        model: IVisionProvider,
        memory: IMemoryProvider,
        prompts: IPromptProvider,
        cloud_storage: IImageStorage,
        local_storage: IImageStorage,
        version: str = "v2_analytical",
    ) -> None:
        self.__model = model
        self.__memory = memory
        self.__prompts = prompts
        self.__cloud_storage = cloud_storage
        self.__local_storage = local_storage
        self.__version = version

    @property
    def version_id(self) -> str:
        """
        Returns current version.
        """
        return self.__version

    async def analyze(
        self,
        intent: str,
        capture: ScreenCapture,
        *,
        use_xml: bool = False,
        context: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Coordinates the analysis flow using injected providers.
        """
        asyncio.create_task(self.__persist(capture.image))

        # Use the stable hash from state or fallback to MD5
        visual_hash = (
            capture.state.visual_hash
            if capture.state
            else hashlib.md5(capture.image, usedforsecurity=False).hexdigest()[:16]
        )

        mem_start = time.time()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash)
        mem_duration = time.time() - mem_start

        instruction = self.__prompts.get_instruction(self.__version)
        tools = {"function_declarations": self.__prompts.get_tools(self.__version)}

        user_content = self.__build_content(intent, knowledge, capture.image)

        analysis_start = time.time()
        analysis = await self.__model.analyze(instruction, user_content, tools)
        analysis_duration = time.time() - analysis_start

        # Populate memory count for decoupled UI reporting
        analysis.memories = len(knowledge.get("previous_actions", []))
        analysis.metrics["memory_retrieval"] = mem_duration
        analysis.metrics["llm_analysis"] = analysis_duration

        await self.__memory.store_observation(
            ScreenState(
                activity=capture.activity,
                activity_hash=hashlib.md5(
                    capture.activity.encode(), usedforsecurity=False
                ).hexdigest()[:8],
                structural_hash="0",
                visual_hash=visual_hash,
                timestamp=int(time.time() * 1000),
            ),
            description=analysis.screen_description,
        )

        return analysis

    async def check_completion(self, intent: str, capture: ScreenCapture) -> bool:
        """
        Check if intent is complete.
        """
        result = await self.analyze(intent, capture)
        return result.is_goal_complete

    def __build_content(self, intent: str, knowledge: Dict[str, Any], screen: bytes) -> List[Any]:
        """
        Assembles request with repetition awareness.
        """
        optimized = ImageProcessor.optimize_for_vision(screen)
        content: List[Any] = [f"Intent: {intent}"]

        if knowledge.get("description"):
            content.append(
                f"State Memory: This screen was identified as '{knowledge['description']}'"
            )

        history = knowledge.get("previous_actions", [])
        if history:
            content.append(f"Historical Experience: {json.dumps(history)}")
            content.append(
                "CRITICAL: Do not repeat previous actions if they failed to change the screen."
            )

        content.append(optimized)
        return content

    async def __persist(self, data: bytes) -> None:
        """
        Background persistence task.
        """
        await asyncio.gather(
            self.__local_storage.save(data), self.__cloud_storage.save(data), return_exceptions=True
        )

    async def cleanup(self) -> None:
        """
        Shut down providers.
        """
        await self.__model.cleanup()
