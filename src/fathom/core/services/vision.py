from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.core.context.manager import ContextManager
from fathom.core.prompts.factory import PromptFactory
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.audit import AuditService
from fathom.core.services.parsing import ToolResponseParser
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.storage import StoragePort
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.utils.image import ImageProcessor

logger = getLogger(__name__)


class VisionService:
    """
    Service that orchestrates the LLM interaction for UI perception and reasoning.
    """

    def __init__(
        self,
        llm: LLMPort,
        memory: MemoryPort,
        storage: StoragePort,
        *,
        use_cache: bool,
        version: str = "pro",
        session_id: str = "default",
        package_name: str = "unknown_app",
        auditor: Optional[AuditService] = None,
    ) -> None:
        """
        Initialize vision service.
        """

        self.__llm = llm
        self.__memory = memory
        self.__storage = storage

        self.__version = version
        self.__use_cache = use_cache
        self.__session_id = session_id
        self.__package_name = package_name
        self.__auditor = auditor or AuditService()

        # Use the original prompt builder factory
        self.__builder = PromptFactory.get_builder(model_name=self.__llm.model_name)

    async def analyze(
        self,
        intent: str,
        capture: ScreenCapture,
        context_manager: ContextManager,
        screen_width: int,
        screen_height: int,
        *,
        use_xml: bool = False,
        tracking_note: Optional[str] = None,
        failures: Optional[List[str]] = None,
        elements: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Coordinates the analysis flow mirroring GeminiVisionTool strictly.
        """

        # Background persistence (original logic)
        asyncio.create_task(self.__persist(data=capture.image, activity=capture.activity))

        # 1. BRAIN RETRIEVAL
        fingerprint = hashlib.sha256(capture.image).hexdigest()[:16]

        start = time.time()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=fingerprint)

        # Retrieve ALL persistent memory (cross-screen)
        all_memory_raw = await self.__memory.get_all()

        # FILTER OUT system state keys (GCC context dumps)
        # This is a defensive measure to prevent pollution from reaching the prompt
        all_memory = {
            key: value
            for key, value in all_memory_raw.items()
            if not key.startswith(("context:", "ctx_v3:", "ctx_"))
        }

        # Log filtered entries for debugging
        filtered_count = len(all_memory_raw) - len(all_memory)
        if filtered_count > 0:
            filtered_keys = [key for key in all_memory_raw if key not in all_memory]
            logger.info(
                f"[VISION] Filtered system memory | "
                f"filtered_count={filtered_count} | "
                f"filtered_keys={filtered_keys[:5]}"  # Show first 5
            )

        retrieval = time.time() - start

        # Log memory stats
        memory_store = knowledge.get("memory_store", {})
        prev_actions = knowledge.get("previous_actions", [])

        logger.info(
            f"[VISION] Memory Retrieved | "
            f"visual_hash={fingerprint[:6]} | "
            f"screen_memories={len(memory_store)} | "
            f"persistent_memories={len(all_memory)} | "
            f"persistent_keys={list(all_memory.keys())} | "
            f"experiences={len(prev_actions)} | "
            f"duration={retrieval:.3f}s"
        )

        # 2. PROMPT & TOOL SCOPING
        # Dynamic context from ContextManager (GCC-Inspired)
        full_context = context_manager.get_full_context()
        guidance = full_context.get("guidance")

        logger.debug(
            f"[H3] Vision Input Context | guidance={guidance} | trace_len={len(full_context.get('trace', []))}"
        )

        instruction = self.__builder.build(
            intent=intent,
            hints={
                "use_xml": use_xml,
                "screen_width": screen_width,
                "screen_height": screen_height,
            },
        )

        # Pass ALL persistent memory (not just screen-specific)
        dynamic_context = self.__builder.build_user_context(
            memory=all_memory,  # Cross-screen persistent memory
            history=full_context,
            tracking_note=tracking_note,
        )

        logger.debug(
            f"[H3] Dynamic Context Built | "
            f"has_memory={bool(all_memory)} | "
            f"memory_keys={list(all_memory.keys()) if all_memory else []} | "
            f"context_length={len(dynamic_context)}"
        )

        tools = self.__scope_tools(intent=intent)

        # 3. CONTENT ASSEMBLY
        manifest_start = time.time()
        manifest = self.__format_elements(elements=elements)
        manifest_duration = time.time() - manifest_start
        logger.info(f"Formatted manifest length: {len(manifest)} and took {manifest_duration:.3f}s")

        payload_start = time.time()
        payload = self.__build_payload(
            intent=intent,
            manifest=manifest,
            failures=failures,
            knowledge=knowledge,
            screen=capture.image,
            context=dynamic_context,
        )
        payload_duration = time.time() - payload_start

        # Log assembly performance
        logger.info(
            f"[VISION] Assembly | Manifest: {manifest_duration:.3f}s | Payload: {payload_duration:.3f}s"
        )

        # Log prompt context for visibility
        self.__auditor.log_prompt(payload=payload, instruction=instruction)

        # 4. EXECUTION
        commence = time.time()
        response = await self.__llm.generate(
            tools=tools,
            prompt=payload,
            use_cache=self.__use_cache,
            system_instruction=instruction,
        )
        duration = time.time() - commence

        # Log Raw LLM output
        raw_text = response.content[:200].replace("\n", " ") if response.content else "No text"
        logger.info(
            f"[VISION] LLM Response | Duration: {duration:.3f}s | Model: {self.__llm.model_name} | Raw: {raw_text}..."
        )

        # 5. PARSE & ENRICH
        parser = ToolResponseParser()
        analysis = parser.parse(response)

        # Update metrics & metadata
        if response.metrics:
            analysis.metrics.update(response.metrics)

        # Expose payload for debugging/audit
        analysis.metadata["prompt_payload"] = [
            str(item) if not isinstance(item, (dict, bytes)) else "Image(...)" for item in payload
        ]
        analysis.metadata["system_instruction"] = instruction

        analysis.memories = len(knowledge.get("previous_actions", []))
        analysis.metrics["llm_analysis"] = duration
        analysis.metrics["memory_retrieval"] = retrieval

        # 6. BRAIN UPDATE (Store observation)
        await self.__memory.store_observation(
            screen=ScreenState(
                activity=capture.activity,
                visual_hash=fingerprint,
                timestamp=int(time.time() * 1000),
                activity_hash=hashlib.md5(  # nosec
                    capture.activity.encode(), usedforsecurity=False
                ).hexdigest()[:VISUAL_HASH_LENGTH],
                structural_hash="0" * VISUAL_HASH_LENGTH,
            ),
            description=analysis.screen_description,
        )

        return analysis

    async def check_completion(
        self,
        intent: str,
        screen_width: int,
        screen_height: int,
        capture: ScreenCapture,
        context_manager: ContextManager,
        tracking_note: Optional[str] = None,
    ) -> bool:
        """
        Check if intent is complete.
        """

        result = await self.analyze(
            intent=intent,
            capture=capture,
            screen_width=screen_width,
            screen_height=screen_height,
            tracking_note=tracking_note,
            context_manager=context_manager,
        )
        return result.is_goal_complete

    def __build_payload(
        self,
        intent: str,
        screen: bytes,
        knowledge: Dict[str, Any],
        *,
        manifest: str = "N/A",
        context: Optional[str] = None,
        failures: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Assembles request with token-locality (strictly mirrored).
        """

        payload: List[Any] = [f"Goal: {intent}"]

        if knowledge.get("description"):
            payload.append(f"Screen Info: {knowledge['description']}")

        if history := knowledge.get("previous_actions", []):
            payload.append(f"Past actions on this specific screen: {json.dumps(history)}")

        if context:
            payload.append(context)

        if failures:
            payload.append(f"Failures on this activity: {', '.join(failures)}")

        if manifest != "N/A":
            payload.append(f"Element Manifest: {manifest}")

        # Image must be last for KV-cache efficiency
        # Optimization: Reduce resolution (768px) and quality (70) to improve latency
        optimized = ImageProcessor.optimize_for_vision(image_data=screen)
        payload.append(optimized)

        return payload

    def __format_elements(self, elements: Optional[Dict[str, Any]]) -> str:
        """
        Converts label map to grounding manifest (strictly mirrored).
        """

        logger.info(f"Converting {len(elements) if elements else 0} elements into manifest")

        if not elements:
            return "N/A"

        lines = []

        for label, info in elements.items():
            if label.startswith("__"):
                continue

            kind = str(info.get("class", "View")).split(".")[-1]
            value = f"[{label}] {kind}"

            # Inject Ground Truth Bounds
            if bounds_str := str(info.get("bounds", "")):
                with contextlib.suppress(Exception):
                    # Parse [x1,y1][x2,y2]
                    parts = (
                        bounds_str.replace("][", ",").replace("[", "").replace("]", "").split(",")
                    )
                    if len(parts) == 4:
                        x1, y1, x2, y2 = map(int, parts)
                        w, h = x2 - x1, y2 - y1
                        value += f" ({x1},{y1},{w},{h})"

            text = str(info.get("text", "")).strip()
            detail = str(info.get("content-desc", "")).strip()

            # if str(info.get("enabled", "true")).lower() == "false":
            #     value += " [DISABLED]"

            # if str(info.get("clickable", "false")).lower() == "true":
            #     value += " [CLICKABLE]"

            if text:
                value += f" | text: '{text}'"

            if detail:
                value += f" | description: '{detail}'"

            lines.append(value)

        return "\n".join(lines) if lines else "No interactive elements found."

    def __scope_tools(self, intent: str) -> Dict[str, Any]:
        """
        Dynamically selects tools (strictly mirrored).
        """

        allowed = {"execute_ui", "store_memory", "recall_memory"}
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

    async def __persist(self, data: bytes, activity: str) -> None:
        """
        Background persistence.
        """

        package = activity if activity and activity != "unknown" else self.__package_name

        with contextlib.suppress(Exception):
            await self.__storage.save(
                data=data,
                metadata={
                    "type": "screenshots",
                    "package_name": package,
                    "activity_name": activity,
                    "session_id": self.__session_id,
                },
            )
