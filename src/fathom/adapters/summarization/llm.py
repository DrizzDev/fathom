from __future__ import annotations

import logging
from typing import Any, Dict, List

from fathom.core.prompts.templates import SUMMARIZATION_SYSTEM
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

            # Extract action details
            if isinstance(action, dict):
                success = action.get("success", True)
                target = action.get("target", "unknown")
                action_type = action.get("action_type", "unknown")
            else:
                success = True
                target = getattr(action, "target", "unknown")
                action_type = getattr(action, "action_type", "unknown")

            # Convert enum to string if needed
            action_type_str = (
                action_type.value
                if hasattr(action_type, "value") and not isinstance(action_type, str)
                else str(action_type)
            )

            # Build action description
            actions_taken.append(f"{action_type_str}:{str(target)}")

            # Track failures
            if not success or "fail" in thought.lower():
                failures.append(f"{str(action_type)} on {str(target)}")

            # Track screens
            if observation:
                screen_hash = (
                    observation.split(":")[1].strip() if ":" in observation else observation
                )
                screens_visited.add(screen_hash[:8])

        # Use tool calling for structured summarization
        tool_definition = {
            "function_declarations": [
                {
                    "name": "create_milestone",
                    "description": "Create a structured milestone summary of the execution segment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "accomplishment": {
                                "type": "string",
                                "description": "Main outcome or state change achieved (e.g., 'Successfully configured custom schedule', 'Navigated to payment screen')",
                            },
                            "key_actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "2-3 most important actions that led to the accomplishment (e.g., 'Selected Monday, Tuesday, Friday', 'Entered date range')",
                            },
                            "challenges": {
                                "type": "string",
                                "description": "Any failures or retries encountered, or 'None' if smooth execution",
                            },
                        },
                        "required": ["accomplishment", "key_actions", "challenges"],
                    },
                }
            ]
        }

        # Build compact trace representation
        failures_list = failures if failures else ["None"]
        sample_actions_list = actions_taken[:50] + (["..."] if len(actions_taken) > 50 else [])

        trace_summary = {
            "total_steps": len(trace),
            "failures": failures_list,
            "sample_actions": sample_actions_list,
            "unique_screens": len(screens_visited),
            "action_types": list({action.split(":")[0] for action in actions_taken}),
        }

        actions_str = ", ".join(str(action) for action in sample_actions_list)
        failures_str = ", ".join(str(failure) for failure in failures_list)

        prompt = [
            "Analyze this execution segment and create a milestone summary:\n",
            f"Steps: {trace_summary['total_steps']}",
            f"Screens visited: {trace_summary['unique_screens']}",
            f"Actions: {actions_str}",
            f"Failures: {failures_str}",
            "\nCreate a milestone that captures what was accomplished, how, and any challenges.",
        ]

        try:
            result = await self.__llm.generate(
                prompt=prompt,
                use_cache=False,
                tools=tool_definition,
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

                key_actions = args.get("key_actions", [])
                challenges = args.get("challenges", "None")
                accomplishment = args.get("accomplishment", "")

                # Format as structured milestone
                milestone_parts = [accomplishment]

                if key_actions:
                    milestone_parts.append(f"via {', '.join(key_actions)}")

                if challenges and challenges.lower() != "none":
                    milestone_parts.append(f"(faced: {challenges})")

                return ". ".join(milestone_parts)

            # Fallback to content if no tool call
            if result.content:
                return result.content.strip()

            # Last resort fallback
            return f"Completed {len(trace)} steps across {len(screens_visited)} screens"

        except Exception as exception:
            logger.error(f"Summarization failed: {exception}")
            return f"Executed {len(trace)} steps ({len(failures)} failures) across {len(screens_visited)} screens"
