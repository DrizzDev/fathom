from __future__ import annotations

import json
from logging import getLogger
from typing import Any, List

from fathom.interfaces import IVisionProvider

logger = getLogger(__name__)


class IntentDecomposer:
    """
    Service responsible for breaking down complex user intents into atomic steps.
    """

    def __init__(self, model: IVisionProvider) -> None:
        self.__model = model

    async def decompose(self, intent: str) -> List[str]:
        """
        Decomposes a complex intent into a list of atomic strings.
        """

        instruction = """
You are an AI planner for a Mobile UI Agent.
Task: Break down the complex user intent into a sequence of atomic, independently executable sub-intents.

Requirements:
1. ATOMIC: Each step must be a single, clear goal.
2. SEQUENTIAL: Steps must be in the correct logical order.
3. EXPLICIT VALUES: Include specific values (names, numbers) from the original intent.
4. FORMAT: Return ONLY a valid JSON list of strings.

Example: "Search for Noodles and add it to cart, then search for Milk and add to cart"
Output: ["Search for Noodles", "Add Noodles to cart", "Search for Milk", "Add Milk to cart"]
"""
        user_content: List[Any] = [f"Complex Intent: {intent}"]

        try:
            # We use the analysis method of the provider directly
            result = await self.__model.analyze(
                system_instruction=instruction, user_content=user_content, tools=None
            )

            # For now, let's try to parse the reasoning or message if it contains JSON
            text = result.reasoning
            if "[" in text and "]" in text:
                start_index = text.find("[")
                end_index = text.rfind("]") + 1
                data = json.loads(text[start_index:end_index])

                if isinstance(data, list):
                    return [str(item) for item in data]

            return [intent]  # Fallback to original

        except Exception as exception:
            logger.error(f"Decomposition failed: {exception}")
            return [intent]
