from __future__ import annotations

import asyncio  # noqa: TC003 — used at runtime for Task types
import contextlib
import hashlib
import json
import time
from logging import getLogger
from typing import Any, Dict, List, Optional, TypedDict

from fathom.constants.events import FathomEvent
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.core.context.manager import ContextManager
from fathom.core.exceptions import ToolValidationError, VisionError
from fathom.core.prompts.factory import PromptFactory
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.audit import AuditService
from fathom.core.services.parsing import ToolResponseParser
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.prompt import PromptUserContext, SubGoalFocus
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.conversation import ConversationTurn, TurnPart
from fathom.schemas.results import AnalysisResult, GenerateResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.utils.image import ImageProcessor

logger = getLogger(__name__)


class SubGoalContext(TypedDict):
    """
    Minimal sub-goal context for vision prompts (no AgentState dependency).
    """

    index: int
    total: int
    description: str


class VisionService:
    """
    Service that orchestrates the LLM interaction for UI perception and reasoning.
    """

    def __init__(
        self,
        llm: LLMPort,
        memory: MemoryPort,
        telemetry: TelemetryPort,
        *,
        use_cache: bool,
        session_id: str = "",
        auditor: Optional[AuditService] = None,
    ) -> None:
        """
        Initialize vision service.
        """

        self.__llm = llm
        self.__memory = memory
        self.__telemetry = telemetry
        self.__auditor = auditor or AuditService()

        self.__use_cache = use_cache
        self.__session_id = session_id
        self.__parser = ToolResponseParser()
        self.__background_tasks: set[asyncio.Task[Any]] = set()

        # Use the original prompt builder factory
        self.__builder = PromptFactory.get_builder(model_name=self.__llm.model_name)
        self.__tool_definitions = ToolRegistry.get_all_definitions()

    async def prewarm(self) -> None:
        """
        Prewarm the planner prompt cache for the known planner tool variants.

        Called concurrently with intent decomposition to reduce first-call latency.
        Each variant creates a separate cached content entry in the provider.
        """

        instruction = self.__builder.build()

        start = time.time()
        for tools in self.__planner_tool_variants():
            await self.__llm.prewarm(tools=tools, system_instruction=instruction)

        duration = time.time() - start

        await self.__telemetry.debug(
            "Latency phase completed",
            phase="planner_prewarm",
            duration=duration,
            type=FathomEvent.LATENCY_PHASE,
        )

    def __planner_tool_variants(self) -> List[Dict[str, Any]]:
        """
        Return the small fixed set of planner tool variants worth prewarming.
        """

        try:
            definitions = self.__tool_definitions["function_declarations"]
            by_name = {definition["name"]: definition for definition in definitions}

            return [
                {"function_declarations": [by_name["execute_ui"]]},
                {"function_declarations": [by_name["execute_ui"], by_name["verify_goal"]]},
            ]
        except KeyError as exception:
            logger.warning("[VisionService] Tool definition missing for prewarm: %s", exception)
            return []

    async def analyze(
        self,
        intent: str,
        capture: ScreenCapture,
        context_manager: ContextManager,
        *,
        visual_hash: str,
        screen_width: int,
        screen_height: int,
        use_xml: bool = False,
        is_stuck: bool = False,
        last_action: Optional[str] = None,
        tracking_note: Optional[str] = None,
        failures: Optional[List[str]] = None,
        elements: Optional[Dict[str, Any]] = None,
        sub_goal_info: Optional[SubGoalContext] = None,
        delta_context: Optional[Dict[str, object]] = None,
        prior_rejection_history: Optional[List[ConversationTurn]] = None,
    ) -> AnalysisResult:
        """
        Coordinates the analysis flow mirroring GeminiVisionTool strictly.
        """

        analyze_start = time.time()

        # Note: Screenshot persistence is now handled upstream by PerceptionService.
        # No background persistence needed here.

        # 1. BRAIN RETRIEVAL
        fingerprint = self.__resolve_capture_fingerprint(capture=capture, visual_hash=visual_hash)
        prompt_image = self.__resolve_prompt_image(capture=capture)

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

        await self.__telemetry.debug(
            "Compiled execution context",
            type=FathomEvent.CONTEXT_CAPTURED,
            session_id=self.__session_id,
            trace_count=len(full_context.get("trace", [])),
            milestone_count=len(full_context.get("milestones", [])),
            guidance_count=len(context_manager.get_user_guidance()),
        )

        logger.debug(
            f"[H3] Vision Input Context | guidance={guidance} | trace_len={len(full_context.get('trace', []))}"
        )

        instruction = self.__builder.build()

        # Sub-goal context (if provided by caller) - SINGLE FOCUS MODE
        # Only current sub-goal is passed to Gemini to prevent skip-ahead behavior.
        if sub_goal_info:
            logger.debug(
                f"[Vision] Single sub-goal focus mode: step [{sub_goal_info['index'] + 1}/{sub_goal_info['total']}] | "
                f"Task: {sub_goal_info['description'][:60]}"
            )

        # Build the typed contract consumed by PromptBuilder.build_user_context.
        sub_goal_focus: Optional[SubGoalFocus] = None
        if sub_goal_info:
            sub_goal_focus = SubGoalFocus(
                index=int(sub_goal_info["index"]),
                total=int(sub_goal_info["total"]),
                description=str(sub_goal_info["description"]),
            )

        prompt_context = PromptUserContext(
            intent=intent,
            memory=all_memory,
            trace=tuple(full_context.get("trace", [])),
            milestones=tuple(full_context.get("milestones", [])),
            guidance=tuple(full_context.get("guidance", []) or ()),
            sub_goal_info=sub_goal_focus,
            screen_width=screen_width,
            screen_height=screen_height,
            use_xml=use_xml,
            current_screen_hash=fingerprint[:8],
            tracking_note=tracking_note,
        )

        dynamic_context = self.__builder.build_user_context(prompt_context)

        if is_stuck:
            dynamic_context += (
                "\n\n<SYSTEM_ALERT>\n"
                "Loop risk detected; avoid repeating the same ineffective action."
                "\n</SYSTEM_ALERT>"
            )

        if failures:
            failed_actions = "; ".join(failures)
            dynamic_context += (
                "\n\n<SYSTEM_ALERT>\n"
                "CRITICAL: The following actions have FAILED or been repeated without progress "
                "on this screen. You MUST choose a DIFFERENT action or approach to achieve "
                f"the same goal.\nFailed: {failed_actions}\n"
                "</SYSTEM_ALERT>"
            )

        if last_action:
            dynamic_context += (
                f"\n\n<LAST_ACTION>\nMost recent action: {last_action}\n</LAST_ACTION>"
            )

        if delta_context:
            dynamic_context += (
                f"\n\n<DELTA_CONTEXT>\n{json.dumps(delta_context, default=str)}\n</DELTA_CONTEXT>"
            )

        logger.debug(
            f"[H3] Dynamic Context Built | "
            f"has_memory={bool(all_memory)} | "
            f"memory_keys={list(all_memory.keys()) if all_memory else []} | "
            f"context_length={len(dynamic_context)}"
        )

        tool_scope_start = time.time()
        tools = self.__scope_tools(intent=intent)
        tool_scope_duration = time.time() - tool_scope_start

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
            screen=prompt_image,
            context=dynamic_context,
        )
        payload_duration = time.time() - payload_start

        # Log assembly performance
        logger.info(
            f"[VISION] Assembly | Manifest: {manifest_duration:.3f}s | Payload: {payload_duration:.3f}s"
        )

        await self.__telemetry.debug(
            "Built vision prompt",
            type=FathomEvent.PROMPT_BUILT,
            session_id=self.__session_id,
            instruction=instruction,
            payload=self.__sanitize_recursive(data=payload),
        )

        # Log prompt context for console visibility
        self.__auditor.log_prompt(payload=payload, instruction=instruction)

        # 4. EXECUTION WITH MULTI-TURN FEEDBACK LOOP
        #
        # Instead of appending error text to a flat prompt (stateless retry), we use
        # Gemini's native multi-turn conversation: on rejection, the model sees its own
        # rejected tool call as a prior model turn, followed by a user turn explaining
        # exactly why it was rejected. This creates a genuine feedback loop where the
        # model can reason about its mistake and correct course.
        max_validation_retries = 2
        analysis: Optional[AnalysisResult] = None
        response = None
        duration = 0.0
        parse_duration = 0.0

        # Conversation history accumulates across retry attempts for multi-turn feedback.
        # Seed with prior rejection history from previous graph iterations (outer loop).
        conversation_history: List[ConversationTurn] = list(prior_rejection_history or [])
        # Preserve original payload so multi-turn history always references the full prompt.
        original_payload = list(payload)

        # Escalate thinking progressively on each validation retry so the
        # model reasons more carefully about the schema constraints it violated.
        thinking_levels = ["low", "medium", "high"]

        for attempt in range(max_validation_retries + 1):
            commence = time.time()
            retry_thinking = (
                thinking_levels[attempt] if attempt < len(thinking_levels) else thinking_levels[-1]
            )
            response = await self.__llm.generate(
                tools=tools,
                prompt=payload,
                use_cache=self.__use_cache,
                system_instruction=instruction,
                conversation_history=conversation_history if conversation_history else None,
                thinking_level=retry_thinking,
            )
            duration = time.time() - commence

            # Log Raw LLM output
            raw_text = response.content[:200].replace("\n", " ") if response.content else "No text"
            logger.info(
                f"[VISION] LLM Response | Duration: {duration:.3f}s | "
                f"Model: {self.__llm.model_name} | Raw: {raw_text}..."
            )

            # 5. PARSE & ENRICH
            parse_start = time.time()
            try:
                analysis = self.__parser.parse(response)
                parse_duration = time.time() - parse_start
                break
            except Exception as exception:
                if isinstance(exception, ToolValidationError) and attempt < max_validation_retries:
                    feedback = getattr(exception, "feedback", None)
                    message = getattr(feedback, "message", str(exception))
                    logger.warning(
                        "[VISION] Tool validation failed (attempt %s/%s): %s",
                        attempt + 1,
                        max_validation_retries + 1,
                        message,
                    )

                    # Build multi-turn feedback: the model sees its own rejected output
                    # as a prior turn, then a correction turn explaining why it was wrong.
                    # This is structurally stronger than appending text to a flat prompt.
                    # Extend (not replace) to preserve any prior_rejection_history seeded
                    # from previous graph iterations.
                    conversation_history.extend(
                        self.__build_rejection_history(
                            original_payload=original_payload,
                            rejected_response=response,
                        )
                    )

                    # On retry with conversation history, payload becomes just the
                    # correction instruction (the history already contains the original prompt).
                    payload = [
                        f"Your previous tool call was rejected: {message}\n"
                        "You MUST call the tool again with a DIFFERENT action. "
                        "Choose an alternative approach to achieve the same goal."
                    ]
                    continue

                # Non-validation errors or exhausted retries propagate as before.
                raise

        if analysis is None or response is None:
            raise VisionError("Vision analysis did not produce a valid result.", retryable=False)

        # Update metrics & metadata
        if response.metrics:
            analysis.metrics.update(response.metrics)

        # Expose original payload for debugging/audit (not the mutated correction text
        # that replaces payload on validation retry).
        analysis.metadata["prompt_payload"] = [
            str(item) if not isinstance(item, (dict, bytes)) else "Image(...)"
            for item in original_payload
        ]
        analysis.metadata["system_instruction"] = instruction

        analysis.memories = len(knowledge.get("previous_actions", []))
        analysis.metrics["llm_analysis"] = duration
        analysis.metrics["memory_retrieval"] = retrieval
        analysis.metrics["tool_scope_ms"] = tool_scope_duration * 1000
        analysis.metrics["manifest_ms"] = manifest_duration * 1000
        analysis.metrics["payload_ms"] = payload_duration * 1000
        analysis.metrics["parse_ms"] = parse_duration * 1000
        analysis.metrics["analyze_ms"] = (time.time() - analyze_start) * 1000
        analysis.metrics["llm_analysis_ms"] = duration * 1000
        analysis.metrics["memory_retrieval_ms"] = retrieval * 1000

        # 6. BRAIN UPDATE (Store observation)
        await self.__memory.store_observation(
            screen=ScreenState(
                activity=capture.activity,
                visual_hash=fingerprint,
                timestamp=int(time.time() * 1000),
                activity_hash=hashlib.md5(  # nosec
                    capture.activity.encode(), usedforsecurity=False
                ).hexdigest()[:VISUAL_HASH_LENGTH],
            ),
            description=analysis.screen_description,
        )

        return analysis

    def build_rejection_history_from_analysis(
        self,
        *,
        analysis: AnalysisResult,
        rejection_reason: str,
    ) -> List[ConversationTurn]:
        """
        Build multi-turn rejection history from an AnalysisResult for cross-iteration
        feedback. The planner calls this when rejecting a repeated action so the next
        vision.analyze() cycle can pass it as conversation_history.

        Returns provider-neutral ConversationTurn objects representing the model's
        rejected tool call and the rejection reason.
        """

        # Model turn: reconstruct the rejected tool call
        model_parts: List[TurnPart] = []
        if analysis.action:
            model_parts.append(
                TurnPart.from_function_call(
                    name="execute_ui",
                    args=analysis.metadata.get("tool_args", {}),
                )
            )
        if not model_parts:
            model_parts.append(TurnPart.from_text(text=analysis.reasoning or "(empty)"))

        model_turn = ConversationTurn(role="model", parts=model_parts)

        # User turn: rejection feedback
        user_turn = ConversationTurn(
            role="user",
            parts=[TurnPart.from_text(text=rejection_reason)],
        )

        return [model_turn, user_turn]

    @staticmethod
    def __build_rejection_history(
        *,
        original_payload: List[Any],
        rejected_response: GenerateResult,
    ) -> List[ConversationTurn]:
        """
        Build a multi-turn conversation history from a rejected LLM response.

        Returns a list of ConversationTurn objects representing:
          1. Original user turn (prompt + image)
          2. Model's rejected response (its own tool call)
          3. (The next user turn with correction will be sent as the current prompt)

        This gives the LLM full visibility into what it proposed and why it was wrong,
        enabling genuine self-correction rather than stateless retry.
        """

        # Turn 1: Original user prompt
        user_parts: List[TurnPart] = []
        for item in original_payload:
            if isinstance(item, bytes):
                mime = "image/png"
                if item[:3] == b"\xff\xd8\xff":
                    mime = "image/jpeg"
                user_parts.append(TurnPart.from_image(data=item, mime_type=mime))
            elif isinstance(item, str):
                user_parts.append(TurnPart.from_text(text=item))

        user_turn = ConversationTurn(role="user", parts=user_parts)

        # Turn 2: Model's rejected response (reconstruct from GenerateResult)
        model_parts: List[TurnPart] = []
        if rejected_response.content:
            model_parts.append(TurnPart.from_text(text=rejected_response.content))
        if rejected_response.tool_calls:
            for tc in rejected_response.tool_calls:
                model_parts.append(
                    TurnPart.from_function_call(
                        name=getattr(tc, "name", "execute_ui"),
                        args=dict(getattr(tc, "args", {})),
                    )
                )

        if not model_parts:
            model_parts.append(TurnPart.from_text(text="(empty response)"))

        model_turn = ConversationTurn(role="model", parts=model_parts)

        return [user_turn, model_turn]

    def __resolve_capture_fingerprint(self, *, visual_hash: str, capture: ScreenCapture) -> str:
        """
        Resolve the stable visual fingerprint used for memory lookup.
        """

        if visual_hash:
            return visual_hash[:VISUAL_HASH_LENGTH]

        if capture.state is not None and capture.state.visual_hash:
            return capture.state.visual_hash[:VISUAL_HASH_LENGTH]

        raise ValueError("Vision analysis requires a prepared visual_hash")

    def __resolve_prompt_image(self, *, capture: ScreenCapture) -> bytes:
        """
        Resolve the image payload to send to the vision model.
        """

        if capture.annotated_image:
            return capture.annotated_image

        return capture.image

    def __sanitize_recursive(self, data: Any) -> Any:
        """
        Replace binary prompt content with stable descriptors.
        """

        if isinstance(data, bytes):
            return f"<binary:{len(data)}>"

        if isinstance(data, dict):
            return {key: self.__sanitize_recursive(data=value) for key, value in data.items()}

        if isinstance(data, (list, tuple)):
            return [self.__sanitize_recursive(item) for item in data]

        return data

    async def check_completion(
        self,
        intent: str,
        visual_hash: str,
        screen_width: int,
        screen_height: int,
        capture: ScreenCapture,
        context_manager: ContextManager,
        *,
        tracking_note: Optional[str] = None,
    ) -> bool:
        """
        Check if intent is complete.
        """

        result = await self.analyze(
            intent=intent,
            capture=capture,
            visual_hash=visual_hash,
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

        payload: List[Any] = []

        if not context or "goal:" not in context.lower():
            payload.append(f"GOAL: {intent}")

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
