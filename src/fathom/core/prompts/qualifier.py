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
    Gemini-focused prompt builder for executability qualification.
    """

    def build_system_instruction(self) -> str:
        """
        Stable system instruction enforcing the executability-only judgement.
        """

        return (
            "You are an executability qualifier for a mobile-app UI automation agent named Fathom.\n"
            "\n"
            "Your one and only task is to answer this question:\n"
            "  Can this request reasonably be translated into actions that Fathom can perform\n"
            "  inside a mobile application?\n"
            "\n"
            "Rules you MUST follow:\n"
            "1. Judge ONLY executability, never the topic. Do not reject because the topic is math,\n"
            "   coding, history, or science. Reject only when no UI surface could perform the request.\n"
            "2. Tasks like searching, scrolling to find an item, opening a page, checking settings,\n"
            "   reviewing a list, or navigating somewhere are all EXECUTABLE even if they do not\n"
            "   modify state. Read-only UI tasks count.\n"
            "3. When uncertain, bias toward EXECUTABLE. Only return NOT_EXECUTABLE when you can\n"
            "   articulate which concrete UI action would be impossible.\n"
            "4. Short, sloppy, poorly written, or compound intents are still EXECUTABLE if you can\n"
            "   imagine any plausible UI interpretation.\n"
            "5. NOT_EXECUTABLE is reserved for pure information requests, conversational chatter,\n"
            "   creative-writing prompts, calculations the user wants answered, opinions, and\n"
            "   gibberish with no plausible UI mapping.\n"
            "\n"
            "Return ONLY valid JSON with this exact shape, no markdown, no commentary:\n"
            "{\n"
            '  "label": "EXECUTABLE" | "PROBABLY_EXECUTABLE" | "PROBABLY_NOT_EXECUTABLE" | "NOT_EXECUTABLE",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "rationale": {\n'
            '    "category": "ui_task" | "informational" | "creative" | "gibberish" | "ambiguous" | "other",\n'
            '    "reasoning": "<one sentence: which UI action the request maps to, or why no UI surface fits>"\n'
            "  }\n"
            "}\n"
        )

    def build_user_prompt(self, *, intent: str) -> str:
        """
        Few-shot examples plus the candidate intent.
        """

        return (
            "Classify the following intent. Examples first, then the candidate.\n"
            "\n"
            "EXAMPLES\n"
            'Intent: "Search for McPuff"\n'
            '{"label": "EXECUTABLE", "confidence": 0.95, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Performs a search action in the current app."}}\n'
            "\n"
            'Intent: "Scroll vertically until you find Asha Tiffin on the screen"\n'
            '{"label": "EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "A find-via-scroll navigation task with a clear target string."}}\n'
            "\n"
            'Intent: "open the contact app add new contact to it"\n'
            '{"label": "EXECUTABLE", "confidence": 0.9, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Poorly written but maps to opening Contacts and creating an entry."}}\n'
            "\n"
            'Intent: "Find my invoices"\n'
            '{"label": "EXECUTABLE", "confidence": 0.9, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Read-only navigation to an invoices view; valid UI task."}}\n'
            "\n"
            'Intent: "what is 2 + 2?"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "informational", '
            '"reasoning": "User wants an answer computed; no UI surface performs this."}}\n'
            "\n"
            'Intent: "who founded google?"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.95, '
            '"rationale": {"category": "informational", '
            '"reasoning": "A factual question with no UI action; user expects an answer."}}\n'
            "\n"
            'Intent: "write me a poem about the moon"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.97, '
            '"rationale": {"category": "creative", '
            '"reasoning": "A content-generation request, not a UI interaction."}}\n'
            "\n"
            'Intent: "asdkfjhqwoeiruzxcv"\n'
            '{"label": "NOT_EXECUTABLE", "confidence": 0.95, '
            '"rationale": {"category": "gibberish", '
            '"reasoning": "No semantic content that maps to any UI action."}}\n'
            "\n"
            'Intent: "check my settings"\n'
            '{"label": "EXECUTABLE", "confidence": 0.92, '
            '"rationale": {"category": "ui_task", '
            '"reasoning": "Navigation to a settings screen is a valid UI task."}}\n'
            "\n"
            f"CANDIDATE\nIntent: {intent!r}\n"
        )
