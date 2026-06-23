"""
Parsing of exploration tool-call responses (explore_ui + describe_screen).
"""

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.constants.exploration import ExpectedOutcome, FocusRelevance
from fathom.core.exceptions import VisionError
from fathom.core.prompts.exploration import ExplorationPromptBuilder
from fathom.core.prompts.tools import ToolRegistry
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult, GenerateResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.translation import ScreenTranslation

logger = getLogger(__name__)

REGIONS = frozenset({"top_bar", "bottom_nav", "content", "modal", "overlay", "fab", "footer"})
ELEMENT_CATEGORIES = frozenset(
    {
        "global_navigation",
        "primary_action",
        "content_item",
        "filter_or_category",
        "secondary_control",
        "overlay_dismiss",
    }
)


class ExplorationResponseParser:
    """
    Parses explore_ui and an optional describe_screen tool call into an AnalysisResult.
    """

    def parse(self, response: GenerateResult) -> AnalysisResult:
        """
        Parses the explore_ui action and any inline screen translation.
        """

        try:
            tool_calls = response.tool_calls
            if not tool_calls:
                return self.__fallback(message=response.content)

            primary = None
            describe = None
            for call in tool_calls:
                name = getattr(call, "name", "")
                if name == "explore_ui":
                    primary = call
                elif name == "describe_screen":
                    describe = call
                else:
                    logger.warning("Unknown exploration tool call ignored: %s", name)

            if primary is None:
                return self.__fallback(message="No explore_ui tool call in response")

            result = self.__parse_exploration(arguments=dict(primary.args or {}))
            result.metadata["tool_name"] = "explore_ui"

            if describe is not None:
                self.__attach_translation(result=result, arguments=dict(describe.args or {}))

            return result

        except VisionError:
            raise
        except Exception as exception:
            logger.exception("Failed to parse exploration tool response")
            raise VisionError(f"Exploration response parsing failed: {exception}") from exception

    def __parse_exploration(self, *, arguments: Dict[str, Any]) -> AnalysisResult:
        """
        Builds an AnalysisResult from explore_ui arguments.
        """

        assistant_message = str(arguments.get("assistant_message", ""))
        action_data = arguments.get("action")
        if not isinstance(action_data, dict):
            return self.__fallback(message=assistant_message)

        metadata: Dict[str, Any] = {}
        for key in ("element_category", "expected_outcome", "region"):
            if action_data.get(key):
                metadata[key] = action_data[key]

        return AnalysisResult(
            action=self.__build_action(data=action_data),
            reasoning=assistant_message,
            is_goal_complete=False,
            content_exhausted=bool(arguments.get("content_exhausted", False)),
            screen_description=str(arguments.get("screen_description") or "Exploration step"),
            focus_relevance=self.__parse_relevance(arguments.get("focus_relevance")),
            metadata=metadata,
        )

    @staticmethod
    def __build_action(*, data: Dict[str, Any]) -> Action:
        """
        Builds an Action from an explore_ui action object.
        """

        bounds: Optional[Bounds] = None
        tap_target = data.get("tap_target")
        if isinstance(tap_target, dict):
            bounds = Bounds(
                x=int(tap_target.get("x", 0)),
                y=int(tap_target.get("y", 0)),
                width=0,
                height=0,
            )

        try:
            action_type = ActionType(str(data.get("action_type", "tap")).lower())
        except ValueError:
            action_type = ActionType.TAP

        raw_region = data.get("region")
        raw_category = data.get("element_category")
        raw_text = data.get("text")

        return Action(
            bounds=bounds,
            action_type=action_type,
            natural_language_target=data.get("target_name") or "UI Element",
            rationale=str(data.get("rationale", "")),
            confidence=ExplorationResponseParser.__safe_float(data.get("confidence", 0.9)),
            overlay_detected=bool(data.get("overlay_detected", False)),
            region=raw_region if raw_region in REGIONS else None,
            element_category=raw_category if raw_category in ELEMENT_CATEGORIES else None,
            expected_outcome=ExplorationResponseParser.__parse_expected(
                data.get("expected_outcome")
            ),
            text=str(raw_text) if raw_text else None,
        )

    @staticmethod
    def __parse_expected(value: Any) -> Optional[ExpectedOutcome]:
        """
        Coerces a raw expected-outcome string into the enum, or None when invalid.
        """

        try:
            return ExpectedOutcome(value)
        except ValueError:
            return None

    @staticmethod
    def __parse_relevance(value: Any) -> Optional[FocusRelevance]:
        """
        Coerces a raw focus-relevance string into the enum, or None when invalid.
        """

        try:
            return FocusRelevance(value)
        except ValueError:
            return None

    @staticmethod
    def __attach_translation(*, result: AnalysisResult, arguments: Dict[str, Any]) -> None:
        """
        Renders a describe_screen call into the rich-description metadata and the
        screen category.
        """

        try:
            translation = ScreenTranslation.model_validate(arguments)
        except Exception:
            logger.warning("Failed to parse describe_screen args", exc_info=True)
            return

        result.metadata["rich_description"] = translation.to_markdown()
        result.category = translation.category

    @staticmethod
    def __safe_float(value: Any, default: float = 0.9) -> float:
        """
        Coerces a value to float, falling back to a default.
        """

        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def __fallback(*, message: str) -> AnalysisResult:
        """
        Returns a safe WAIT analysis when parsing fails or no action is found.
        """

        return AnalysisResult(
            action=Action(
                action_type=ActionType.WAIT,
                rationale=message,
                target="No valid action",
                confidence=0.0,
            ),
            reasoning=message,
            is_goal_complete=False,
            screen_description="Fallback state",
        )


class ExplorationVisionService:
    """
    Drives the exploration scan: builds the system prompt, calls the model with
    the exploration tools, and parses the response into an AnalysisResult.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        use_cache: bool = True,
        guarded: bool = True,
        builder: Optional[ExplorationPromptBuilder] = None,
        parser: Optional[ExplorationResponseParser] = None,
    ) -> None:
        self.__llm = llm
        self.__use_cache = use_cache
        self.__guarded = guarded
        self.__builder = builder or ExplorationPromptBuilder()
        self.__parser = parser or ExplorationResponseParser()

    async def prewarm(self, *, intent: str = "") -> None:
        """
        Pre-creates the provider-side prompt cache for the exploration instruction.
        """

        await self.__llm.prewarm(
            system_instruction=self.__builder.build_system_prompt(
                intent=intent, guarded=self.__guarded
            ),
            tools=ToolRegistry.get_exploration_definitions(),
        )

    async def scan(
        self,
        *,
        capture: ScreenCapture,
        knowledge_context: str,
        intent: str = "",
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Asks the model for the next exploration action given the screen and graph context.

        When ``failures`` is supplied -- e.g. the dedup or sampling guard rejected a
        prior proposal -- the reasons are prepended as a corrective directive so the
        model picks a different element on the re-prompt.
        """

        prompt: List[PromptPart] = []
        feedback = self.__format_failures(failures=failures)
        if feedback is not None:
            prompt.append(feedback)
        prompt.extend([knowledge_context, capture.image])

        result = await self.__llm.generate(
            use_cache=self.__use_cache,
            prompt=prompt,
            tools=ToolRegistry.get_exploration_definitions(),
            system_instruction=self.__builder.build_system_prompt(
                intent=intent, guarded=self.__guarded
            ),
        )
        return self.__parser.parse(result)

    @staticmethod
    def __format_failures(*, failures: Optional[List[str]]) -> Optional[str]:
        """
        Renders rejection reasons from the dedup/sampling guards into a directive.
        """

        reasons = [reason.strip() for reason in (failures or []) if reason and reason.strip()]
        if not reasons:
            return None

        bullets = "\n".join(f"- {reason}" for reason in reasons)
        return (
            "PREVIOUS PROPOSAL REJECTED - do not repeat it:\n"
            f"{bullets}\n"
            "Pick a DIFFERENT untried element that satisfies these constraints."
        )
