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
        delta_context: Optional[Dict[str, Any]] = None,
        elements: Optional[Dict[str, Any]] = None,
        mode: Optional[PromptMode] = None,
        resolved_fingerprint: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Coordinates the analysis flow using Native Tool Calling.

        When ``mode`` is provided it overrides the heuristic-based mode
        detection, allowing callers (e.g. the BFS exploration strategy) to
        force a specific prompt mode.
        """

        asyncio.create_task(coro=self.__persist(data=capture.image, activity=capture.activity))

        # 1. BRAIN RETRIEVAL
        # Use the caller-provided canonical hash when available so that
        # memory lookups match the canonical hashes used when storing
        # experiences in record_node.  Falls back to the raw capture hash
        # for callers that have not resolved it.
        fingerprint = resolved_fingerprint or (
            capture.state.visual_hash
            if capture.state
            else hashlib.md5(capture.image, usedforsecurity=False).hexdigest()[:16]
        )

        start = time.time()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=fingerprint)
        retrieval = time.time() - start

        # 2. PROMPT & TOOL SCOPING
        resolved_mode = mode if mode is not None else PromptMode.EXPLORATION
        hints = {
            "use_xml": use_xml,
            "is_stuck": is_stuck,
            "last_action": last_action,
            "delta_low_streak": (delta_context or {}).get("low_delta_streak", 0),
            "delta_score": (delta_context or {}).get("last_delta_score"),
        }

        instruction = self.__builder.build(
            mode=resolved_mode.value,
            intent=intent,
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
            delta_context=delta_context,
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

    async def describe_screen(
        self,
        capture: ScreenCapture,
        *,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a rich markdown translation of all visible designs and
        features on the screen via a dedicated VLM call using the
        ``describe_screen`` tool declaration.
        """

        from fathom.tools.definitions import ToolRegistry

        instruction = self.__builder.build(
            mode=PromptMode.SCREEN_TRANSLATION.value,
        )

        image = ImageProcessor.optimize_for_vision(capture.image)
        payload: List[Any] = [image]
        if context:
            payload.append(context)

        tools = ToolRegistry.get_screen_translation_tools()

        result = await self.__model.generate_structured(
            system_instruction=instruction,
            user_content=payload,
            tools=tools,
        )

        return self.__format_translation(result)

    @staticmethod
    def __format_translation(data: Dict[str, Any]) -> str:
        """
        Formats the structured describe_screen tool call args into a
        design-blueprint markdown document.
        """

        activity = data.get("activity_name", "")
        sections = [
            ("Purpose", data.get("screen_purpose", "")),
            ("Layout Blueprint", data.get("layout_blueprint", "")),
            ("Component Inventory", data.get("component_inventory", "")),
            ("Design Tokens", data.get("design_tokens", "")),
        ]

        parts = []
        if activity:
            parts.append(f"**Activity:** `{activity}`")
        for heading, body in sections:
            if body:
                parts.append(f"## {heading}\n{body}")

        return "\n\n".join(parts)

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
        delta_context: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Assembles the dynamic user payload.
        GOAL/intent is in the cached system instruction; only per-step
        hints, context, knowledge, and image go here.
        """

        payload: List[Any] = [instructions] if instructions.strip() else []

        if knowledge.get("description"):
            payload.append(f"SCREEN: {knowledge['description']}")

        # NOTE: Tried actions are provided by build_exploration_context()
        # via the "ALREADY TRIED FROM THIS SCREEN:" section in the context
        # parameter.  The KG edge list is the single source of truth —
        # canonical-hash-resolved, comprehensive, and includes destination
        # descriptions.  The SQLite experience table (previous_actions) is
        # intentionally omitted here to avoid a redundant, potentially
        # inconsistent second tried-action list that dilutes the signal.

        if transitions := knowledge.get("transitions", []):
            nav_lines = []
            for t in transitions:
                desc = t.get("destination_description")
                if desc:
                    target = t.get("action_target") or "unknown"
                    nav_lines.append(f'- {t["action_type"]} "{target}" -> {desc}')
            if nav_lines:
                payload.append("NAV:\n" + "\n".join(nav_lines))

        if context:
            payload.append(f"HISTORY: {context}")

        if failures:
            payload.append(f"FAILURES: {', '.join(failures)}")

        if delta_context:
            payload.append(f"DELTA_CONTEXT: {json.dumps(delta_context)}")

        if manifest != "N/A":
            payload.append(f"ELEMENTS: {manifest}")

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
        Returns the tool definitions for the given mode.
        """

        if mode == PromptMode.EXPLORATION:
            return ToolRegistry.get_exploration_tools()

        if mode == PromptMode.SCREEN_TRANSLATION:
            return ToolRegistry.get_screen_translation_tools()

        # Fallback to exploration tools
        return ToolRegistry.get_exploration_tools()
