from __future__ import annotations

import logging
from typing import Any, Dict, List

from fathom.core.prompts.summarization import (
    SUMMARIZATION_SYSTEM,
    SUMMARIZATION_TOOL_DEFINITION,
    build_summarization_user_prompt,
    format_milestone,
)
from fathom.core.prompts.trace import extract_action_fields
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.summarization import SummarizationPort

logger = logging.getLogger(__name__)


class LLMSummarizer(SummarizationPort):
    """
    Uses an LLM with structured tool calling to semantically compress execution traces.

    Produces structured milestones that capture:
    - What was accomplished (outcomes)
    - How it was done (key actions)
    - What challenges were faced (failures/retries)

    Uses tool calling for speed and structured output.
    """

    def __init__(self, llm: LLMPort) -> None:
        """
        Initialize with an LLM provider.
        """

        self.__llm = llm

    async def summarize_trace(self, trace: List[Dict[str, Any]]) -> str:
        """
        Generates a structured semantic summary using tool calling.

        Returns a formatted milestone string that includes:
        - Main accomplishment
        - Key actions taken
        - Challenges faced (if any)
        """

        if not trace:
            return "No actions performed."

        # Extract structured information from trace
        failures = []
        actions_taken = []
        screens_visited = set()

        for entry in trace:
            action = entry.get("action", {})
            thought = entry.get("thought", "")
            observation = entry.get("observation", "")

            # Pull canonical (action_type, target) via the shared helper
            # so this matches every other place that walks the trace.
            action_type_str, target = extract_action_fields(entry)

            # Success flag is summarization-specific (only this caller
            # cares); read it directly without rerunning the shape check.
            if isinstance(action, dict):
                success = action.get("success", True)
            else:
                success = bool(getattr(action, "success", True))

            actions_taken.append(f"{action_type_str}:{target}")

            if not success or "fail" in thought.lower():
                failures.append(f"{action_type_str} on {target}")

            # Track screens
            if observation:
                screen_hash = (
                    observation.split(":")[1].strip() if ":" in observation else observation
                )
                screens_visited.add(screen_hash[:8])

        # Build compact trace representation
        failures_list = failures if failures else ["None"]
        sample_actions_list = actions_taken[:50] + (["..."] if len(actions_taken) > 50 else [])

        prompt = build_summarization_user_prompt(
            total_steps=len(trace),
            unique_screens=len(screens_visited),
            sample_actions=sample_actions_list,
            failures=failures_list,
        )

        try:
            result = await self.__llm.generate(
                prompt=prompt,
                use_cache=False,
                tools=SUMMARIZATION_TOOL_DEFINITION,
                system_instruction=SUMMARIZATION_SYSTEM,
            )

            # Extract structured response from tool call
            if result.tool_calls and len(result.tool_calls) > 0:
                tool_call = result.tool_calls[0]
                # Handle both dict and object types for tool_call
                if isinstance(tool_call, dict):
                    args = tool_call.get("args", {})
                else:
                    args = getattr(tool_call, "args", {})

                return format_milestone(
                    accomplishment=args.get("accomplishment", ""),
                    key_actions=args.get("key_actions", []),
                    challenges=args.get("challenges", "None"),
                )

            # Fallback to content if no tool call
            if result.content:
                return result.content.strip()

            # Last resort fallback
            return f"Completed {len(trace)} steps across {len(screens_visited)} screens"

        except Exception as exception:
            logger.error(f"Summarization failed: {exception}")
            return f"Executed {len(trace)} steps ({len(failures)} failures) across {len(screens_visited)} screens"
