from __future__ import annotations

import asyncio
import hashlib
import json
import time
from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.interfaces import (
    IImageStorage,
    ILedger,
    IMemoryProvider,
    IVisionProvider,
)
from fathom.prompts.factory import PromptFactory
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.tools.definitions import ToolRegistry
from fathom.tools.vision.base import VisionTool
from fathom.utils.image import ImageProcessor

logger = getLogger(name=__name__)


class GeminiVisionTool(VisionTool):
    """
    SOLID Orchestrator for Vision Analysis using Native Tool Calling.
    """

    def __init__(
        self,
        model: IVisionProvider,
        memory: IMemoryProvider,
        ledger: ILedger,
        local_storage: IImageStorage,
        *,
        version: str = "pro_xml",
        session_id: str = "default",
        package_name: str = "unknown_app",
        gcs_storage: Optional[IImageStorage] = None,
    ) -> None:
        self.__model = model
        self.__version = version
        self.__session_id = session_id
        self.__package_name = package_name

        self.__memory = memory
        self.__ledger = ledger

        self.__gcs_storage = gcs_storage
        self.__local_storage = local_storage

        self.__builder = PromptFactory.get_builder(model_name="gemini")

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
        Coordinates the analysis flow using Native Tool Calling.
        """

        asyncio.create_task(coro=self.__persist(data=capture.image, activity=capture.activity))

        # 1. BRAIN RETRIEVAL
        fingerprint = (
            capture.state.visual_hash
            if capture.state
            else hashlib.md5(string=capture.image, usedforsecurity=False).hexdigest()[:16]
        )

        start = time.time()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=fingerprint)
        retrieval = time.time() - start

        # 2. PROMPT & TOOL SCOPING
        # Static Prompt (Cacheable)
        hints = {"use_xml": use_xml}

        instruction = self.__builder.build(
            intent=intent,
            hints=hints,
        )

        # Dynamic context
        dynamic_context = self.__builder.build_user_context(
            history=context,
            memory=await self.__ledger.get_all(),
        )

        tools = self.__scope_tools(intent=intent)

        # 3. CONTENT ASSEMBLY
        manifest = self.__format_elements(elements=elements)
        payload = self.__build_payload(
            intent=intent,
            screen=capture.image,
            knowledge=knowledge,
            context=dynamic_context,
            manifest=manifest,
            failures=failures,
        )

        # Debug logs
        logger.debug(f"System Instruction:\n{instruction[:200]}...")
        logger.debug(
            f"User Payload (Text Parts):\n{[parts for parts in payload if isinstance(parts, str)]}"
        )

        # 4. EXECUTION
        commence = time.time()
        analysis = await self.__call_api(instruction=instruction, payload=payload, tools=tools)
        duration = time.time() - commence

        # 5. METRICS & BRAIN UPDATE
        analysis.memories = len(knowledge.get("previous_actions", []))
        analysis.metrics["llm_analysis"] = duration
        analysis.metrics["memory_retrieval"] = retrieval

        stats = getattr(self.__model, "cache_stats", {})
        if stats:
            analysis.metrics["cache_hits"] = stats.get("hits", 0)
            analysis.metrics["cache_misses"] = stats.get("misses", 0)

        await self.__memory.store_observation(
            screen=ScreenState(
                activity=capture.activity,
                activity_hash=hashlib.md5(
                    string=capture.activity.encode(), usedforsecurity=False
                ).hexdigest()[:8],
                structural_hash="0",
                visual_hash=fingerprint,
                timestamp=int(time.time() * 1000),
            ),
            description=analysis.screen_description,
        )

        return self.__parse_response(analysis=analysis)

    async def __call_api(
        self, instruction: str, payload: List[Any], tools: Dict[str, Any]
    ) -> AnalysisResult:
        """
        Delegates the actual LLM call to the underlying provider.
        """

        return await self.__model.analyze(
            system_instruction=instruction, user_content=payload, tools=tools
        )

    def __parse_response(self, analysis: AnalysisResult) -> AnalysisResult:
        """
        Maps or enriches the provider's response if needed.
        """

        return analysis

    async def check_completion(self, intent: str, capture: ScreenCapture) -> bool:
        """
        Check if intent is complete.
        """

        result = await self.analyze(intent=intent, capture=capture)
        return result.is_goal_complete

    def __parse_bbox(self, raw: Any) -> Optional[Dict[str, int]]:
        """
        Legacy helper kept for test/backward compatibility.
        Accepts Gemini bbox variants and normalizes to x/y/width/height.
        """

        if isinstance(raw, dict):
            if {"x", "y", "width", "height"}.issubset(raw):
                return {
                    "x": int(raw["x"]),
                    "y": int(raw["y"]),
                    "width": int(raw["width"]),
                    "height": int(raw["height"]),
                }
            if {"ymin", "xmin", "ymax", "xmax"}.issubset(raw):
                xmin = int(raw["xmin"])
                ymin = int(raw["ymin"])
                xmax = int(raw["xmax"])
                ymax = int(raw["ymax"])
                return {
                    "x": xmin,
                    "y": ymin,
                    "width": max(0, xmax - xmin),
                    "height": max(0, ymax - ymin),
                }
            return None

        if isinstance(raw, list) and len(raw) == 4:
            ymin, xmin, ymax, xmax = [int(value) for value in raw]
            return {
                "x": xmin,
                "y": ymin,
                "width": max(0, xmax - xmin),
                "height": max(0, ymax - ymin),
            }

        return None

    def __build_payload(
        self,
        intent: str,
        screen: bytes,
        knowledge: Dict[str, Any],
        context: Optional[str] = None,
        manifest: str = "N/A",
        failures: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Assembles request with token-locality.
        Stable intent at top, dynamic image/history at bottom for KV-cache.
        """

        payload: List[Any] = [f"Goal: {intent}"]

        # 1. State Memory (Specific to this screen hash)
        if knowledge.get("description"):
            payload.append(f"Screen Info: {knowledge['description']}")

        if history := knowledge.get("previous_actions", []):
            payload.append(f"Past actions on this specific screen: {json.dumps(obj=history)}")

        # 2. Dynamic Session Context (Changes every step)
        if context:
            payload.append(f"Recent turns (global): {context}")

        if failures:
            payload.append(f"Failures on this activity: {', '.join(failures)}")

        # 3. Element Manifest (If present)
        if manifest != "N/A":
            payload.append(f"Element Manifest: {manifest}")

        # 4. Image (Most dynamic, must be last)
        optimized = ImageProcessor.optimize_for_vision(image_data=screen)
        payload.append(optimized)

        return payload

    def __format_elements(self, elements: Optional[Dict[str, Any]]) -> str:
        """
        Converts label map to a high-density textual grounding manifest.
        Format: [ID] Class | Text | Content-Description
        """

        if not elements:
            return "N/A"

        lines = []

        for label, information in elements.items():
            if label.startswith("__"):  # Skip internal scale factors
                continue

            kind = str(object=information.get("class", "View")).split(sep=".")[-1]

            value = f"[{label}] {kind}"
            text = information.get("text", "").strip()
            detail = information.get("content-desc", "").strip()

            if text:
                value += f" | text: '{text}'"

            if detail:
                value += f" | description: '{detail}'"

            lines.append(value)

        return "\n".join(lines) if lines else "No interactive elements found."

    async def __persist(self, data: bytes, activity: str) -> None:
        """
        Background persistence task.
        """

        metadata = {
            "activity_name": activity,
            "session_id": self.__session_id,
            "package_name": self.__package_name,
        }

        try:
            tasks = [self.__local_storage.save(data=data, metadata=metadata)]
            if self.__gcs_storage:
                tasks.append(self.__gcs_storage.save(data=data, metadata=metadata))

            await asyncio.gather(*tasks)
        except Exception:
            logger.debug("Screenshot persistence failed", exc_info=True)

    async def cleanup(self) -> None:
        """
        Shut down providers.
        """

        await self.__model.cleanup()

    def __scope_tools(self, intent: str) -> Dict[str, Any]:
        """
        Dynamically selects tools based on the intent context.
        """

        # Base tools always available
        allowed = {"execute_ui", "store_memory", "recall_memory"}

        # Validation tools for verification tasks
        if any(word in intent.lower() for word in ("verify", "check", "confirm", "validate")):
            allowed.update({"validate_state", "verify_goal"})

        definitions = ToolRegistry.get_all_definitions()

        return {
            "function_declarations": [
                definition
                for definition in definitions["function_declarations"]
                if definition["name"] in allowed
            ]
        }
