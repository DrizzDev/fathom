from __future__ import annotations

from abc import ABC, abstractmethod


class QualifierPromptBuilder(ABC):
    """
    Abstract builder for intent qualifier prompting.
    """

    @abstractmethod
    def build_system_instruction(self) -> str:
        """
        Build stable system instruction for qualification.
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(self, *, intent: str) -> str:
        """
        Build dynamic user prompt for classifying a specific intent.
        """

        raise NotImplementedError


class GeminiQualifierPromptBuilder(QualifierPromptBuilder):
    """
    Gemini-focused prompt builder for binary executability qualification.
    """

    def build_system_instruction(self) -> str:
        """
        Stable binary contract: is this a mobile UI automation request, yes or no.
        """

        return (
            "You classify a user's intent for a mobile-app UI automation agent.\n"
            "Answer exactly one question: is the text a mobile UI automation\n"
            "request that the agent should attempt? Specificity, ambiguity and\n"
            "target-grounding belong to the agent's execution layer — not to\n"
            "this gate. If the text describes a UI action the agent could try,\n"
            "let it pass even when the target is under-specified.\n"
            "\n"
            "LABELS\n"
            "\n"
            "EXECUTABLE\n"
            "  The text expresses a mobile UI action the agent should attempt.\n"
            "  Keep this bar low. Any of the following is enough:\n"
            "    - an app, screen, element, query, gesture, or workflow named;\n"
            "    - a recognized UI gesture (scroll, swipe, tap back, go back,\n"
            "      pull to refresh) used as a command;\n"
            "    - an under-specified UI action whose target the strategy will\n"
            "      resolve on the live screen ('Open browser', 'Open photos',\n"
            "      'Check my settings', 'Submit a form', 'Add item to cart',\n"
            "      'Find my invoices', 'Login', 'Upload a PDF'). These are\n"
            "      EXECUTABLE; the agent will ground or fail on the device.\n"
            "    - the same kind of command phrased politely as a question\n"
            "      ('Can you open Swiggy?', 'Could you scroll down?', 'Can\n"
            "      you tap Continue?'). Polite question form ≠ answer-seeking.\n"
            "\n"
            "NOT_EXECUTABLE\n"
            "  The text is not a UI automation request. Block it:\n"
            "    - empty text, bare symbols, keyboard rolls, gibberish;\n"
            "    - factual lookups, calculations, opinions, definitions;\n"
            "    - creative-writing / content-generation requests;\n"
            "    - answer-seeking questions about the app, screen, or state\n"
            "      ('Why is this not done?', 'How does this work?', 'What\n"
            "      should I do next?', 'Is my order confirmed?', 'Should I\n"
            "      tap continue?', 'Can you explain this screen?'). The\n"
            "      operative verb is informational ('tell', 'explain', 'is',\n"
            "      'why', 'how', 'what', 'should'), not a UI action.\n"
            "    - pure conversational chatter with no operative UI verb\n"
            "      ('can you handle this?', 'can you do this for me?').\n"
            "\n"
            "QUESTION-FORM HEURISTIC (important)\n"
            "  A question mark does NOT mean block. Read past politeness and\n"
            "  look for an operative UI verb + object pair:\n"
            "    - 'Can you open Swiggy and search for Biryani?' → operative\n"
            "      verb is 'open' / 'search' → EXECUTABLE.\n"
            "    - 'What should I do next?' → operative verb is 'should I do'\n"
            "      (asks for an answer, not for an action) → NOT_EXECUTABLE.\n"
            "  When in doubt: would executing the literal text produce a UI\n"
            "  action, or an explanation? UI action ⇒ EXECUTABLE.\n"
            "\n"
            "CONFIDENCE\n"
            "  Report your honest confidence on the binary label. The gate is\n"
            "  binary; confidence is observability only and does not move the\n"
            "  decision boundary.\n"
            "\n"
            "RULES\n"
            "  - Judge only executability, never the topic.\n"
            "  - Ignore conversational filler at the start of the message ('Hi',\n"
            "    'Ok lets simplify this', 'Hello there') and read the operative\n"
            "    verb + object that follows.\n"
            "  - Under-specified UI actions are EXECUTABLE; the agent will\n"
            "    ground or report failure on the device. Do not pre-reject.\n"
            "  - Return ONLY valid JSON with the shape below, no markdown.\n"
            "\n"
            "OUTPUT SHAPE\n"
            "{\n"
            '  "label": "EXECUTABLE" | "NOT_EXECUTABLE",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "rationale": {\n'
            '    "category": "ui_task" | "informational" | "creative" | "gibberish" | "conversational" | "ambiguous" | "other",\n'
            '    "reasoning": "<one sentence: what the user is asking the agent to do, or why this is not a UI request>"\n'
            "  }\n"
            "}\n"
        )

    def build_user_prompt(self, *, intent: str) -> str:
        """
        Anchor set covering the binary contract with paired question-form examples.
        """

        return (
            "Classify the candidate intent.\n"
            "\n"
            "ANCHOR EXAMPLES\n"
            "\n"
            'Intent: "Open Swiggy and search for Biryani"\n'
            '{"label": "EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "App + search query both named; clear UI flow."}}\n'
            "\n"
            'Intent: "scroll down"\n'
            '{"label": "EXECUTABLE", "confidence": 0.95, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Recognized UI gesture meaningful on any screen."}}\n'
            "\n"
            'Intent: "Search for McPuff"\n'
            '{"label": "EXECUTABLE", "confidence": 0.92, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Concrete search query; the agent uses whatever search field is on screen."}}\n'
            "\n"
            'Intent: "Open browser"\n'
            '{"label": "EXECUTABLE", "confidence": 0.85, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Under-specified UI action; the strategy grounds the target on the device."}}\n'
            "\n"
            'Intent: "Submit a form"\n'
            '{"label": "EXECUTABLE", "confidence": 0.8, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Under-specified UI action; the strategy grounds the form on the current screen or fails meaningfully."}}\n'
            "\n"
            "POLITE QUESTION FORM — PAIRED EXAMPLES\n"
            "\n"
            'Intent: "Can you open Chrome and search for weather today?"\n'
            '{"label": "EXECUTABLE", "confidence": 0.95, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Polite question form; operative verb is open/search with a concrete query."}}\n'
            "\n"
            'Intent: "What should I do next?"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "conversational", '
            '"reasoning": "Answer-seeking question; no operative UI verb."}}\n'
            "\n"
            'Intent: "Could you scroll down and find the price?"\n'
            '{"label": "EXECUTABLE", "confidence": 0.93, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Polite question form; operative verbs are scroll + find on a concrete target."}}\n'
            "\n"
            'Intent: "Why is checkout not working?"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "conversational", '
            '"reasoning": "Asks for an explanation, not a UI action."}}\n'
            "\n"
            "OUT-OF-DOMAIN ANCHORS\n"
            "\n"
            'Intent: "who founded google?"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "informational", '
            '"reasoning": "Factual question; answered with words, not a UI action."}}\n'
            "\n"
            'Intent: "write me a poem"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "creative", '
            '"reasoning": "Content-generation request; not a UI automation task."}}\n'
            "\n"
            'Intent: "+"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "gibberish", '
            '"reasoning": "Bare symbol; no semantic instruction."}}\n'
            "\n"
            'Intent: "asdkfjhqwoeiruzxcv"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "gibberish", '
            '"reasoning": "Keyboard roll; no semantic content."}}\n'
            "\n"
            f"CANDIDATE\nIntent: {intent!r}\n"
        )
