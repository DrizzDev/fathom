from __future__ import annotations

import asyncio  # noqa: TC003 — used at runtime for Task types
import contextlib
import hashlib
import json
import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, TypedDict

from fathom.constants.events import FathomEvent
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.core.agent.tools import ToolScope
from fathom.core.context.manager import ContextManager
from fathom.core.exceptions import ToolValidationError, VisionError
from fathom.core.prompts.factory import PromptFactory
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.audit import AuditService
from fathom.core.services.parsing import ToolResponseParser
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.capabilities import RuntimeCapabilities
from fathom.schemas.conversation import ConversationTurn, TurnPart
from fathom.schemas.observation import LoopObservation, ScreenObservation
from fathom.schemas.results import AnalysisResult, GenerateResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.tools import AllowedTools
from fathom.schemas.vision import PastActionEntry
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
        capabilities: RuntimeCapabilities,
        tool_scope: Optional[ToolScope] = None,
        session_id: str = "",
        auditor: Optional[AuditService] = None,
    ) -> None:
        """Initialize vision service with the live runtime capabilities."""

        self.__llm = llm
        self.__memory = memory
        self.__telemetry = telemetry
        self.__auditor = auditor or AuditService()

        self.__use_cache = use_cache
        self.__session_id = session_id
        self.__parser = ToolResponseParser()
        self.__background_tasks: Set[asyncio.Task[Any]] = set()

        self.__capabilities = capabilities
        self.__tool_scope = tool_scope or ToolScope()

        self.__builder = PromptFactory.get_builder(model_name=self.__llm.model_name)

    async def prewarm(self) -> None:
        """Prewarm the LLM cache with the canonical tool variants for the live runtime."""

        start = time.time()
        for variant in self.__prewarm_variants():
            instruction = self.__builder.build(tools=variant)
            payload = ToolRegistry.definitions(names=variant.names)
            await self.__llm.prewarm(tools=payload, system_instruction=instruction)

        await self.__telemetry.debug(
            "Latency phase completed",
            duration=time.time() - start,
            phase="planner.prewarm",
            type=FathomEvent.LATENCY_PHASE,
        )

    def __prewarm_variants(self) -> List[AllowedTools]:
        """Return canonical tool variants matched to the runtime capabilities."""

        return [
            self.__tool_scope.compute(intent=intent, capabilities=self.__capabilities)
            for intent in ("interact", "verify")
        ]

    async def analyze(
        self,
        intent: str,
        tools: AllowedTools,
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
        loop_observation: Optional[LoopObservation] = None,
        screen_observation: Optional[ScreenObservation] = None,
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

        instruction = self.__builder.build(tools=tools)

        # Sub-goal context (if provided by caller) - SINGLE FOCUS MODE
        # Only current sub-goal is passed to Gemini to prevent skip-ahead behavior.
        if sub_goal_info:
            logger.debug(
                f"[Vision] Single sub-goal focus mode: step [{sub_goal_info['index'] + 1}/{sub_goal_info['total']}] | "
                f"Task: {sub_goal_info['description'][:60]}"
            )

        # Pass ALL persistent memory (not just screen-specific)
        dynamic_context = self.__builder.build_user_context(
            memory=all_memory,  # Cross-screen persistent memory
            history=full_context,
            tracking_note=tracking_note,
            intent=intent,
            hints={
                "use_xml": use_xml,
                "screen_width": screen_width,
                "screen_height": screen_height,
            },
            sub_goal_info=sub_goal_info,
            # Thread current screen hash so the trace can annotate stale observations
            current_screen_hash=fingerprint[:8],
        )

        if loop_observation is not None:
            dynamic_context += self.__render_loop_observation(observation=loop_observation)
        elif is_stuck:
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

        if screen_observation is not None:
            dynamic_context += self.__render_screen_observation(observation=screen_observation)

        logger.debug(
            f"[H3] Dynamic Context Built | "
            f"has_memory={bool(all_memory)} | "
            f"memory_keys={list(all_memory.keys()) if all_memory else []} | "
            f"context_length={len(dynamic_context)}"
        )

        tool_scope_start = time.time()
        allowed_tools = ToolRegistry.definitions(names=tools.names)
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
        max_validation_retries = 1
        analysis: Optional[AnalysisResult] = None
        response = None
        duration = 0.0
        parse_duration = 0.0

        # Conversation history accumulates across retry attempts for multi-turn feedback.
        # Seed with prior rejection history from previous graph iterations (outer loop).
        conversation_history: List[ConversationTurn] = list(prior_rejection_history or [])
        # Preserve original payload so multi-turn history always references the full prompt.
        original_payload = list(payload)

        for attempt in range(max_validation_retries + 1):
            commence = time.time()
            response = await self.__llm.generate(
                prompt=payload,
                tools=allowed_tools,
                use_cache=self.__use_cache,
                system_instruction=instruction,
                conversation_history=conversation_history if conversation_history else None,
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
                        "You MUST call the tool again with corrected fields that satisfy "
                        "the schema. Keep the same intended UI action when it is still "
                        "the right next step; choose a different action only if the "
                        "current screen or validation feedback proves the original "
                        "intent was wrong."
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
        analysis.metadata["current_workflow_screen_actions"] = (
            self.__current_workflow_screen_actions(
                trace=full_context.get("trace", []),
                current_screen_hash=fingerprint[:8],
            )
        )

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

    @staticmethod
    def __current_workflow_screen_actions(
        *,
        trace: Any,
        current_screen_hash: str,
    ) -> List[Dict[str, Any]]:
        """
        Return successful actions from this workflow on the current screen.
        """

        if not isinstance(trace, list):
            return []

        actions: List[Dict[str, Any]] = []
        for entry in trace:
            if not isinstance(entry, dict):
                continue
            observation = str(entry.get("observation") or "")
            if not observation.startswith("Screen: "):
                continue
            parts = observation.split(" ")
            entry_hash = parts[1][:8] if len(parts) > 1 else ""
            if entry_hash != current_screen_hash:
                continue
            action = entry.get("action")
            if not isinstance(action, dict):
                continue
            target = (
                action.get("natural_language_target")
                or action.get("target")
                or action.get("label_id")
            )
            if not target:
                continue
            actions.append(
                {
                    "success": True,
                    "action": action.get("action_type"),
                    "target": str(target),
                }
            )

        return actions

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
            annotated = [
                PastActionEntry.from_raw(entry=entry).model_dump(mode="json")
                for entry in history
                if isinstance(entry, dict)
            ]
            payload.append(f"Past actions on this specific screen: {json.dumps(annotated)}")

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

    def __render_loop_observation(self, *, observation: LoopObservation) -> str:
        """
        Render a :class:`LoopObservation` as a structured block in the
        ANALYZE prompt.

        The block is deliberately framed as an observation, not as an
        instruction. The agent reads it, may consult it when choosing
        the next action, and is free to disagree if it has reason. The
        runtime enforces nothing here; loop-breaking remains a planner
        concern, not a prompt contract.
        """

        progress = (
            ", ".join(f"{score:.3f}" for score in observation.progress_scores)
            if observation.progress_scores
            else "(none recorded)"
        )

        alternatives = (
            "\n  - " + "\n  - ".join(observation.suggested_alternatives)
            if observation.suggested_alternatives
            else " (none)"
        )

        note_line = f"\nnote: {observation.note}" if observation.note else ""

        return (
            "\n\n<SYSTEM_OBSERVATION>\n"
            f"repeated_action: {observation.repeated_action}\n"
            f"count: {observation.count}\n"
            f"screen_relation: {observation.screen_relation.value}\n"
            f"progress_scores (oldest first): {progress}\n"
            f"alternatives:{alternatives}{note_line}\n"
            "This is an observation from the runtime, not an instruction. "
            "Decide the next action yourself; consider whether the current "
            "approach is working or whether to ask the user.\n"
            "</SYSTEM_OBSERVATION>"
        )

    def __render_screen_observation(self, *, observation: ScreenObservation) -> str:
        """
        Render compact runtime perception facts into the ANALYZE prompt.
        """

        calls_to_action = []
        for element in observation.calls_to_action[:5]:
            label = element.text or element.identifier
            calls_to_action.append(
                f"  - id={element.identifier} text={label} source={element.source.value}"
            )

        scroll_regions = []
        for region in observation.scroll[:3]:
            if region.manifest_label_id:
                scroll_regions.append(
                    f"  - manifest_label={region.manifest_label_id} x={region.bounds.x} y={region.bounds.y} "
                    f"w={region.bounds.width} h={region.bounds.height} axis={region.axis}"
                )
                continue

            hint = region.observation_region_id or region.identifier or "observation_scroll_region"
            scroll_regions.append(
                f"  - observation_hint={hint} x={region.bounds.x} y={region.bounds.y} "
                f"w={region.bounds.width} h={region.bounds.height} axis={region.axis} "
                "(hint only; NOT a manifest label_id)"
            )

        overlay_lines = []
        for index, overlay in enumerate(observation.overlays[:3], start=1):
            overlay_lines.append(
                f"  - overlay_{index}: candidates={len(overlay.candidates)} "
                f"x={overlay.bounds.x} y={overlay.bounds.y} "
                f"w={overlay.bounds.width} h={overlay.bounds.height}"
            )

        return (
            "\n\n<SCREEN_OBSERVATION>\n"
            f"keyboard_visibility: {observation.keyboard.visibility.value.lower()}\n"
            f"overlay_count: {len(observation.overlays)}\n"
            "visible_calls_to_action:\n"
            f"{chr(10).join(calls_to_action) if calls_to_action else '  - none'}\n"
            "scroll_regions:\n"
            f"{chr(10).join(scroll_regions) if scroll_regions else '  - none'}\n"
            "overlays:\n"
            f"{chr(10).join(overlay_lines) if overlay_lines else '  - none'}\n"
            "</SCREEN_OBSERVATION>"
        )

    def __format_elements(self, elements: Optional[Dict[str, Any]]) -> str:
        """
        Convert the drawer label map into a grounding manifest the
        planner can bind ``label_id`` references against.

        Reads attribute keys from both platform vocabularies — Android
        (``class``, ``text``, ``content-desc``) and iOS / XCUITest
        (``type``, ``name``, ``label``, ``value``) — so an element
        coming out of :class:`IOSParser` (which keeps its raw XML
        attribute names) renders with its real semantic label instead
        of an anonymous ``View``.
        """

        logger.info(f"Converting {len(elements) if elements else 0} elements into manifest")

        if not elements:
            return "N/A"

        lines = []

        for label, info in elements.items():
            if label.startswith("__"):
                continue

            kind = self.__manifest_kind(info=info)
            value = f"[{label}] {kind}"

            if bounds_str := str(info.get("bounds", "")):
                with contextlib.suppress(Exception):
                    parts = (
                        bounds_str.replace("][", ",").replace("[", "").replace("]", "").split(",")
                    )
                    if len(parts) == 4:
                        x1, y1, x2, y2 = map(int, parts)
                        w, h = x2 - x1, y2 - y1
                        value += f" ({x1},{y1},{w},{h})"

            text = self.__manifest_text(info=info)
            detail = self.__manifest_detail(info=info)

            if text:
                value += f" | text: '{text}'"

            if detail:
                value += f" | description: '{detail}'"

            lines.append(value)

        return "\n".join(lines) if lines else "No interactive elements found."

    @staticmethod
    def __manifest_kind(*, info: Dict[str, Any]) -> str:
        """
        Resolve the element kind label, falling back across Android and
        iOS attribute names. The iOS ``XCUIElementType`` prefix is
        stripped so the manifest reads as ``Button`` / ``Icon`` etc.
        """

        raw = str(info.get("class") or info.get("type") or "View")
        last = raw.split(".")[-1]
        return last.replace("XCUIElementType", "") or last

    @staticmethod
    def __manifest_text(*, info: Dict[str, Any]) -> str:
        """
        Resolve the primary text label across both platforms' attribute
        names. iOS uses ``label`` / ``name``; Android uses ``text``.
        """

        for key in ("text", "label", "name"):
            if (raw := info.get(key)) is not None and (stripped := str(raw).strip()):
                return stripped
        return ""

    @staticmethod
    def __manifest_detail(*, info: Dict[str, Any]) -> str:
        """
        Resolve the secondary descriptor. iOS uses ``value``; Android
        uses ``content-desc``.
        """

        for key in ("content-desc", "value"):
            if (raw := info.get(key)) is not None and (stripped := str(raw).strip()):
                return stripped
        return ""
