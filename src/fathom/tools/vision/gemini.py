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
from fathom.prompts.modes import PromptMode
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
    ) -> None:
        self.__model = model
        self.__version = version
        self.__session_id = session_id
        self.__package_name = package_name

        self.__memory = memory
        self.__ledger = ledger

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
        is_stuck: bool = False,
        last_action: Optional[str] = None,
        elements: Optional[Dict[str, Any]] = None,
        mode: Optional[PromptMode] = None,
    ) -> AnalysisResult:
        """
        Coordinates the analysis flow using Native Tool Calling.

        When ``mode`` is provided it overrides the heuristic-based mode
        detection, allowing callers (e.g. the BFS exploration strategy) to
        force a specific prompt mode.
        """

        asyncio.create_task(coro=self.__persist(data=capture.image, activity=capture.activity))

        # 1. BRAIN RETRIEVAL
        fingerprint = (
            capture.state.visual_hash
            if capture.state
            else hashlib.md5(capture.image, usedforsecurity=False).hexdigest()[:16]
        )

        start = time.time()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=fingerprint)
        retrieval = time.time() - start

        # 2. PROMPT & TOOL SCOPING
        resolved_mode = mode if mode is not None else self.__detect_mode(intent=intent)
        hints = {"use_xml": use_xml, "is_stuck": is_stuck, "last_action": last_action}

        instruction = self.__builder.build(
            mode=resolved_mode.value,
            intent="",
            hints=hints,
        )

        task_instructions = self.__builder.build_task_instructions(
            intent=intent,
            hints=hints,
        )

        # Dynamic context
        dynamic_context = self.__builder.build_user_context(
            history=context,
            memory=await self.__ledger.get_all(),
        )

        tools = self.__scope_tools(mode=resolved_mode)

        # 3. CONTENT ASSEMBLY
        manifest = self.__format_elements(elements=elements)
        payload = self.__build_payload(
            instructions=task_instructions,
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
        instructions: str,
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

        payload: List[Any] = [instructions]

        # 1. State Memory (Specific to this screen hash)
        if knowledge.get("description"):
            payload.append(f"Screen Info: {knowledge['description']}")

        if history := knowledge.get("previous_actions", []):
            payload.append(f"Past actions on this specific screen: {json.dumps(obj=history)}")

        # 1b. Navigation Map (Known transitions from this screen)
        if transitions := knowledge.get("transitions", []):
            nav_lines = []
            for t in transitions:
                desc = t.get("destination_description")
                if desc:  # Only include transitions with known destinations
                    target = t.get("action_target") or "unknown"
                    nav_lines.append(f'- {t["action_type"]} "{target}" -> {desc}')
            if nav_lines:
                payload.append("Known navigation from this screen:\n" + "\n".join(nav_lines))

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
            await self.__local_storage.save(data=data, metadata=metadata)
        except Exception:
            logger.debug("Screenshot persistence failed", exc_info=True)

    async def cleanup(self) -> None:
        """
        Shut down providers.
        """

        await self.__model.cleanup()

    def __scope_tools(self, mode: PromptMode) -> Dict[str, Any]:
        """
        Dynamically selects tools based on the intent context.
        """

        # Base tools always available
        allowed = {"execute_ui", "complete_goal", "store_memory"}

        if mode == PromptMode.DEFAULT:
            allowed.update({"recall_memory", "validate_state", "verify_goal"})

        elif mode == PromptMode.INTERACTION:
            allowed.update({"recall_memory"})

        elif mode == PromptMode.VERIFICATION:
            allowed.update({"validate_state", "verify_goal", "recall_memory"})

        elif mode == PromptMode.EXPLORATION:
            # Exploration only needs execute_ui; no goal completion signaling
            allowed = {"execute_ui"}

        # Discovery Mode gets minimal tools (just execute_ui + store)

        definitions = ToolRegistry.get_all_definitions()

        return {
            "function_declarations": [
                definition
                for definition in definitions["function_declarations"]
                if definition["name"] in allowed
            ]
        }

    def __detect_mode(self, intent: str) -> PromptMode:
        """
        Heuristic to detect the mode from the intent.
        """
        intent_lower = intent.lower()
        if any(word in intent_lower for word in ("find", "search", "locate", "where")):
            return PromptMode.DISCOVERY
        if any(word in intent_lower for word in ("verify", "check", "confirm", "validate")):
            return PromptMode.VERIFICATION
        return PromptMode.DEFAULT
