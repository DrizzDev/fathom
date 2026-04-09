"""Provider-neutral intent classification prompt policy.

Owns the system instruction, the ``classify_intent`` tool schema, and
the user-prompt builder used by any LLM provider when deciding whether
a user intent should be decomposed into multiple sub-goals or executed
as a single cohesive sub-goal by the planner.

Mirrors the shape of ``fathom.core.prompts.summarization``: a service
(``fathom.core.services.intent_classifier.IntentClassifier``) calls
``LLMPort.generate`` with ``CLASSIFICATION_TOOL_DEFINITION`` as the
``tools`` argument and extracts the structured boolean from the
resulting ``tool_calls[0].args``.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "CLASSIFICATION_SYSTEM",
    "CLASSIFICATION_TOOL_DEFINITION",
    "CLASSIFICATION_TOOL_NAME",
    "build_classification_user_prompt",
]


CLASSIFICATION_TOOL_NAME = "classify_intent"


CLASSIFICATION_SYSTEM = (
    "You are an intent classifier for a mobile UI automation agent.\n"
    "Given a user intent, decide whether it should be decomposed into\n"
    "multiple sequential sub-goals or executed as a single cohesive\n"
    "sub-goal by the planner.\n"
    "\n"
    "Return your decision by calling the classify_intent tool with a\n"
    "boolean should_decompose and a brief reason. Do not reply with\n"
    "free-form text; always use the tool."
)


CLASSIFICATION_TOOL_DEFINITION: Dict[str, Any] = {
    "function_declarations": [
        {
            "name": CLASSIFICATION_TOOL_NAME,
            "description": (
                "Classify whether a user intent should be decomposed "
                "into multiple sub-goals (True) or executed as a "
                "single cohesive sub-goal (False)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "should_decompose": {
                        "type": "boolean",
                        "description": (
                            "True if the intent spans multiple distinct "
                            "goals, requires decision-making between "
                            "branches based on what the screen shows, "
                            "or crosses multiple apps. False if the "
                            "intent describes a single self-contained "
                            "workflow in one app with one terminal "
                            "state, even if several mechanical actions "
                            "are needed to complete it."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "One-sentence justification for the "
                            "decision. Name the criterion that drove "
                            "it (e.g. 'single login workflow', "
                            "'requires comparing options', 'crosses "
                            "two apps')."
                        ),
                    },
                },
                "required": ["should_decompose", "reason"],
            },
        }
    ]
}


def build_classification_user_prompt(*, intent: str) -> List[str]:
    """Render the provider-neutral user prompt for intent classification.

    Returned as a list of string parts so the adapter can pass it
    directly to ``LLMPort.generate`` without extra joining — same shape
    as ``fathom.core.prompts.summarization.build_summarization_user_prompt``.
    """

    return [
        "Classify the following user intent.\n",
        f"INTENT: {intent}\n",
        "\n",
        "Return should_decompose = FALSE (single sub-goal) when ALL of:\n"
        "  (a) There is exactly ONE terminal state that visibly\n"
        '      confirms completion (e.g. "logged in", "results\n'
        "      page for 'iphone' is visible\", \"banana is in the\n"
        '      cart").\n'
        "  (b) Every step is a mechanical action toward that state.\n"
        "  (c) The whole workflow takes place in ONE app / ONE flow.\n"
        "  (d) The intent does NOT contain open-ended decision verbs\n"
        "      that require judgment based on what the screen shows\n"
        '      (e.g. "find a good", "pick the best", "choose",\n'
        '      "compare", "pick the cheapest"). Deterministic verbs\n'
        '      like "open", "go to", "tap", "type", "add",\n'
        '      "select [named item]", "scroll to", "enter",\n'
        '      "log in", "search for [literal]" are all fine.\n'
        "\n"
        "Users naturally describe simple workflows with waypoints\n"
        '("open X, go to Y, do Z"). Waypoint phrasing is NOT a\n'
        "signal to decompose \u2014 focus on the semantic structure.\n"
        "\n"
        "Return should_decompose = TRUE (multiple sub-goals) when ANY\n"
        "of (a)-(d) is violated.\n"
        "\n"
        "EXAMPLES \u2014 should_decompose = FALSE (simple):\n"
        '  - "Tap the login button"\n'
        "  - \"Enter password 'test123'\"\n"
        '  - "Open Settings app"\n'
        '  - "Scroll to the bottom of the page"\n'
        "  - \"Log in with email 'foo@bar.com' and password 'test123'\"\n"
        "  - \"Search for 'iphone' on Amazon\"\n"
        "  - \"Open YouTube and play 'lofi radio'\"\n"
        '  - "Add milk to the shopping list"\n'
        '  - "Open Instacart, go to the Aldi store page, and add\n'
        '    banana to cart"\n'
        "\n"
        "EXAMPLES \u2014 should_decompose = TRUE (complex):\n"
        '  - "Search for iphone, pick the cheapest one under $800,\n'
        '    and add it to cart"  (decision verb)\n'
        '  - "Open Instacart and find a good deal on bananas at\n'
        '    Aldi"  (decision verb)\n'
        '  - "Enable dark mode in Settings, then check any new\n'
        '    notifications in the Profile tab"  (two independent\n'
        "    terminal states)\n"
        '  - "Order 3 items from my last Amazon order"  (requires\n'
        "    reasoning over multiple products)\n"
        "\n"
        "Call classify_intent now with your decision and a one-sentence reason.\n",
    ]
