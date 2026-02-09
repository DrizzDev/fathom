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

logger = getLogger(name=__name__)


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
        version: str = "pro_xml",
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

    @property
    def provider(self) -> IVisionProvider:
        """
        Returns the underlying vision provider.
        """
        return self.__model

    async def analyze(
        self,
        intent: str,
        capture: ScreenCapture,
        *,
        use_xml: bool = False,
        context: Optional[str] = None,
        failures: Optional[List[str]] = None,
        elements: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Coordinates the analysis flow using injected providers.
        """
        asyncio.create_task(coro=self.__persist(data=capture.image))

        # Use the stable hash from state or fallback to MD5
        visual_hash = (
            capture.state.visual_hash
            if capture.state
            else hashlib.md5(string=capture.image, usedforsecurity=False).hexdigest()[:16]
        )

        memory_start_time = time.time()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=visual_hash)
        memory_duration = time.time() - memory_start_time

        instruction_template = self.__prompts.get_instruction(version_id=self.__version)

        # Prepare context strings for the template
        context_string = context if context else "None"
        failure_string = "\n".join(failures) if failures else "None"

        # Format elements manifest for dual-channel grounding (Flash/XML)
        elements_manifest = self.__format_elements(elements=elements)

        instruction = instruction_template.format(
            intent=intent,
            context=context_string,
            failures=failure_string,
            elements=elements_manifest,
        )

        tools = {"function_declarations": self.__prompts.get_tools(version_id=self.__version)}

        user_content = self.__build_content(
            intent=intent,
            context=context,
            failures=failures,
            knowledge=knowledge,
            screen=capture.image,
            elements_manifest=elements_manifest,
        )

        analysis_start_time = time.time()
        analysis = await self.__model.analyze(
            system_instruction=instruction, user_content=user_content, tools=tools
        )
        analysis_duration = time.time() - analysis_start_time

        # Populate memory count for decoupled UI reporting
        analysis.memories = len(knowledge.get("previous_actions", []))
        analysis.metrics["llm_analysis"] = analysis_duration
        analysis.metrics["memory_retrieval"] = memory_duration

        await self.__memory.store_observation(
            screen=ScreenState(
                activity=capture.activity,
                activity_hash=hashlib.md5(
                    string=capture.activity.encode(), usedforsecurity=False
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

        result = await self.analyze(intent=intent, capture=capture)
        return result.is_goal_complete

    def __build_content(
        self,
        intent: str,
        screen: bytes,
        knowledge: Dict[str, Any],
        context: Optional[str] = None,
        elements_manifest: str = "N/A",
        failures: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Assembles request with token-locality.
        Stable intent at top, dynamic image/history at bottom for KV-cache.
        """

        content: List[Any] = [f"Goal: {intent}"]

        # 1. State Memory (Specific to this screen hash)
        if knowledge.get("description"):
            content.append(f"Screen Info: {knowledge['description']}")

        history = knowledge.get("previous_actions", [])
        if history:
            content.append(f"Past actions on this specific screen: {json.dumps(obj=history)}")

        # 2. Dynamic Session Context (Changes every step)
        if context:
            content.append(f"Recent turns (global): {context}")

        if failures:
            content.append(f"Failures on this activity: {', '.join(failures)}")

        # 3. Element Manifest (If present)
        if elements_manifest != "N/A":
            content.append(f"Element Manifest: {elements_manifest}")

        # 4. Image (Most dynamic, must be last)
        optimized = ImageProcessor.optimize_for_vision(image_data=screen)
        content.append(optimized)

        return content

    def __format_elements(self, elements: Optional[Dict[str, Any]]) -> str:
        """
        Converts label map to a high-density textual grounding manifest.
        Format: [ID] Class | Text | Content-Description
        """

        if not elements:
            return "N/A"

        lines = []

        for label_id, information in elements.items():
            if label_id.startswith("__"):  # Skip internal scale factors
                continue

            class_name = str(object=information.get("class", "View")).split(sep=".")[-1]
            text = information.get("text", "").strip()
            description = information.get("content-desc", "").strip()

            value = f"[{label_id}] {class_name}"
            if text:
                value += f" | text: '{text}'"
            if description:
                value += f" | description: '{description}'"
            lines.append(value)

        return "\n".join(lines) if lines else "No interactive elements found."

    async def __persist(self, data: bytes) -> None:
        """
        Background persistence task.
        """

        await asyncio.gather(
            self.__local_storage.save(data=data),
            self.__cloud_storage.save(data=data),
            return_exceptions=True,
        )

    async def cleanup(self) -> None:
        """
        Shut down providers.
        """

        await self.__model.cleanup()
