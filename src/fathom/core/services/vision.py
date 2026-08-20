from __future__ import annotations

import contextlib
import hashlib
import json
import time
from logging import getLogger
from typing import Any, Dict, List, Optional

from typing_extensions import NotRequired, TypedDict

from fathom.constants.events import FathomEvent
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.perception import VISION_IMAGE_MAX_DIMENSION, VISION_IMAGE_QUALITY
from fathom.constants.tools import TurnMode
from fathom.core.agent.tools import DEFAULT_TOOL_POLICIES, ToolScope
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
from fathom.schemas.planner import PlannerMetrics
from fathom.schemas.results import AnalysisResult, GenerateResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.tools import AllowedTools, ToolPolicyContext
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
    durable: NotRequired[bool]
    assertion: NotRequired[str]


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
        """
        Initialize vision service with the live runtime capabilities.
        """

        self.__llm = llm
        self.__memory = memory
        self.__telemetry = telemetry
        self.__auditor = auditor or AuditService()

        self.__use_cache = use_cache
        self.__session_id = session_id
        self.__parser = ToolResponseParser()

        self.__capabilities = capabilities
        self.__tool_scope = tool_scope or ToolScope(policies=DEFAULT_TOOL_POLICIES)

        self.__builder = PromptFactory.get_builder(model_name=self.__llm.model_name)

    async def prewarm(self) -> None:
        """
        Prewarm the LLM cache with the canonical tool variants for the live runtime.
        """

        start = time.time()

        for variant in self.__prewarm_variants():
            instruction = self.__builder.build(tools=variant)
            payload = ToolRegistry.definitions(names=variant.names)
            await self.__llm.prewarm(tools=payload, system_instruction=instruction)

        await self.__telemetry.debug(
            "Latency phase completed",
            phase="planner.prewarm",
            duration=time.time() - start,
            type=FathomEvent.PHASE_HEARTBEAT,
        )

    def __prewarm_variants(self) -> List[AllowedTools]:
        """
        Return canonical tool variants matched to the runtime capabilities.
        """

        return [
            self.__tool_scope.compute(
                context=ToolPolicyContext(capabilities=self.__capabilities, modes=modes),
            )
            for modes in (frozenset(), frozenset({TurnMode.VERIFY}))
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
        Run one planner turn: assemble the ANALYZE prompt from memory and context, generate with
        multi-turn schema-repair, and parse the tool call into an AnalysisResult.
        """

        analyze_start = time.time()

        # Screenshot persistence is handled upstream by PerceptionService; none is needed here.

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

        filtered_count = len(all_memory_raw) - len(all_memory)
        if filtered_count > 0:
            filtered_keys = [key for key in all_memory_raw if key not in all_memory]
            logger.info(
                f"[VISION] Filtered system memory | "
                f"filtered_count={filtered_count} | "
                f"filtered_keys={filtered_keys[:5]}"  # Show first 5
            )

        retrieval = time.time() - start

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

        # Dynamic context from ContextManager (GCC-Inspired)
        full_context = context_manager.get_full_context()
        guidance = full_context.get("guidance")

        await self.__telemetry.debug(
            "Compiled execution context",
            type=FathomEvent.CONTEXT_CAPTURED,
            metadata={"session_id": self.__session_id},
            trace_count=len(full_context.get("trace", [])),
            milestone_count=len(full_context.get("milestones", [])),
            guidance_count=len(context_manager.get_user_guidance()),
        )

        logger.info(
            "Vision input context prepared",
            extra={
                "component": "core.services.vision",
                "event": "vision.input_context.prepared",
                "guidance.present": guidance is not None,
                "trace.count": len(full_context.get("trace", [])),
            },
        )

        instruction = self.__builder.build(tools=tools)

        # Sub-goal context (if provided by caller) - SINGLE FOCUS MODE
        # Only current sub-goal is passed to Gemini to prevent skip-ahead behavior.
        if sub_goal_info:
            logger.info(
                "Vision single-sub-goal focus mode enabled",
                extra={
                    "component": "core.services.vision",
                    "event": "vision.sub_goal_focus.enabled",
                    "sub_goal.index": sub_goal_info["index"],
                    "sub_goal.total": sub_goal_info["total"],
                    "sub_goal.description.length": len(sub_goal_info["description"]),
                },
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

        logger.info(
            "Vision dynamic context built",
            extra={
                "component": "core.services.vision",
                "event": "vision.dynamic_context.built",
                "memory.present": bool(all_memory),
                "context.length": len(dynamic_context),
                "memory.count": len(all_memory) if all_memory else 0,
            },
        )

        tool_scope_start = time.time()
        allowed_tools = ToolRegistry.definitions(names=tools.names)
        tool_scope_duration = time.time() - tool_scope_start

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

        logger.info(
            f"[VISION] Assembly | Manifest: {manifest_duration:.3f}s | Payload: {payload_duration:.3f}s"
        )

        await self.__telemetry.debug(
            "Built vision prompt",
            instruction=instruction,
            type=FathomEvent.PROMPT_BUILT,
            metadata={"session_id": self.__session_id},
            payload=self.__sanitize_recursive(data=payload),
        )

        self.__auditor.log_prompt(payload=payload, instruction=instruction)

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

            raw_text = response.content[:200].replace("\n", " ") if response.content else "No text"
            logger.info(
                f"[VISION] LLM Response | Duration: {duration:.3f}s | "
                f"Model: {self.__llm.model_name} | Raw: {raw_text}..."
            )

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

                if isinstance(exception, ToolValidationError):
                    feedback = getattr(exception, "feedback", None)
                    message = getattr(feedback, "message", str(exception))
                    logger.error(
                        "Tool validation retries exhausted",
                        extra={
                            "component": "core.services.vision",
                            "event": "planner.schema_retry.exhausted",
                            "tool.name": getattr(feedback, "tool_name", "unknown"),
                            "tool.error.message": message,
                            "attempt.count": max_validation_retries + 1,
                        },
                    )

                # Non-validation errors or exhausted retries propagate to the graph boundary.
                raise

        if analysis is None or response is None:
            raise VisionError("Vision analysis did not produce a valid result.", retryable=False)

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
        analysis.planner = PlannerMetrics(latency=duration, calls=attempt + 1)
        analysis.metrics["memory_retrieval"] = retrieval
        analysis.metrics["tool_scope_ms"] = tool_scope_duration * 1000
        analysis.metrics["manifest_ms"] = manifest_duration * 1000
        analysis.metrics["payload_ms"] = payload_duration * 1000
        analysis.metrics["parse_ms"] = parse_duration * 1000
        analysis.metrics["analyze_ms"] = (time.time() - analyze_start) * 1000
        analysis.metrics["llm_analysis_ms"] = duration * 1000
        analysis.metrics["memory_retrieval_ms"] = retrieval * 1000

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
        Return actions dispatched by this workflow on the current screen; dispatch is not outcome proof.
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
                    "target": str(target),
                    "action": action.get("action_type"),
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
        Build multi-turn rejection history from an AnalysisResult for cross-iteration feedback.

        The planner calls this when rejecting a repeated action so the next ``vision.analyze()``
        cycle can pass it as ``conversation_history``; returns the model's rejected tool call and
        the rejection reason as provider-neutral ConversationTurns.
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

        Returns two ConversationTurns: the original user turn (prompt + image) and the model's
        rejected tool call; the correction is sent separately as the next user prompt.
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
        Assemble the ordered prompt parts, placing the screenshot last for KV-cache locality.
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

        # Image must be last for KV-cache efficiency. Cap the longest edge high enough to preserve
        # the pixel position that disambiguates near-identical controls; the downscale saves no tokens.
        optimized = ImageProcessor.optimize_for_vision(
            image_data=screen,
            quality=VISION_IMAGE_QUALITY,
            max_dimension=VISION_IMAGE_MAX_DIMENSION,
        )
        payload.append(optimized)

        return payload

    def __render_loop_observation(self, *, observation: LoopObservation) -> str:
        """
        Render a :class:`LoopObservation` as a structured block in the ANALYZE prompt.

        The block is framed as an observation, not an instruction: the runtime enforces nothing
        and loop-breaking stays a planner decision.
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
        Convert the drawer label map into a grounding manifest the planner can bind ``label_id``
        references against.

        Reads both platform vocabularies — Android (``class``, ``text``, ``content-desc``) and
        iOS / XCUITest (``type``, ``name``, ``label``, ``value``) — so an :class:`IOSParser` element,
        which keeps its raw XML attribute names, renders with its real semantic label instead of an
        anonymous ``View``.
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
