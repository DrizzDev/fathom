from __future__ import annotations

from abc import ABC, abstractmethod


class DecompositionPromptBuilder(ABC):
    """
    Abstract builder for intent decomposition prompting.
    """

    @abstractmethod
    def build_system_instruction(self) -> str:
        """
        Build stable system instruction for decomposition generation.
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(self, *, intent: str) -> str:
        """
        Build dynamic user prompt for decomposing an intent.
        """

        raise NotImplementedError


class GeminiDecompositionPromptBuilder(DecompositionPromptBuilder):
    """
    Gemini-focused prompt builder for sequential intent decomposition.
    Produces sub-goals carrying an ``objective`` and one typed success ``proposal``
    (observed, command, or capture) that the boundary translates to canonical success.
    """

    def build_system_instruction(self) -> str:
        """
        Stable system instruction kept short to maximize cache hits.
        """

        return (
            "You are an expert task planner. Always preserve the user's exact wording.\n"
            "Return ONLY valid JSON (no markdown, no extra text)."
        )

    def build_user_prompt(self, *, intent: str) -> str:
        """
        Compose the decomposition user prompt from named sections.
        """

        sections = [
            self.__header(intent=intent),
            self.__rules(),
            self.__compound_action_rules(),
            self.__proposal_vocabulary(),
            self.__examples(),
            self.__output_schema(),
        ]
        return "\n\n".join(sections)

    @staticmethod
    def __header(*, intent: str) -> str:
        """
        Opening section with the user's intent verbatim.
        """

        return (
            "You are an expert at breaking down user intents into executable micro-tasks.\n\n"
            f"INTENT: {intent}"
        )

    @staticmethod
    def __rules() -> str:
        """
        Core decomposition rules — atomic, ordered, preserve wording.
        """

        return (
            "INSTRUCTIONS:\n"
            "1. Break down the intent into sequential, non-skippable steps\n"
            "2. Each step must be atomic and testable\n"
            "3. Steps must be in execution order (no parallelization)\n"
            "4. Each step should be 1-2 sentences, action-oriented\n"
            "5. CRITICAL: Do not skip any steps required to achieve the intent\n"
            "6. CRITICAL: You MUST use the user's exact wording and terminology wherever possible\n"
            "   - Do NOT paraphrase, generalize, or rephrase their specific requests\n"
            "   - Preserve specific app names, button names, field names, and action verbs\n"
            "   - Keep technical terms and product names exactly as stated"
        )

    @staticmethod
    def __compound_action_rules() -> str:
        """
        Guidance on when to keep multi-verb steps together vs split them.
        """

        return (
            "IMPORTANT - COMPOUND ACTIONS:\n"
            "Keep these patterns as SINGLE steps (do NOT split them):\n"
            '- "Scroll to {{destination}} and select/tap/click {{target}}" -> ONE step '
            "(scroll is just navigation to reach the target)\n"
            '- "Navigate to {{destination}} and do {{action}}" -> ONE step '
            "(navigate is means to reach the action)\n"
            '- "Find {{target}} and tap/click {{target}}" -> ONE step '
            "(find is means to reach the target)\n"
            '- "Go to {{destination}} and verify/check {{state}}" -> ONE step '
            "(navigation + verification together)\n"
            "\n"
            "Split these patterns into SEPARATE steps:\n"
            '- Any "store/capture {{value_or_subject}} as {{variable_name}}" clause is its own step, '
            "even when followed by another action.\n"
            "- Any conditional capture must separate the condition proof, the capture, and the "
            "follow-up action. Shape: first validate/check {{condition}}, then store/capture "
            "{{value_or_subject}} as {{variable_name}}, then emit the next requested action "
            "as a separate step."
        )

    @staticmethod
    def __proposal_vocabulary() -> str:
        """
        Describe the three success proposal kinds and the command requirement operations.
        """

        return (
            "PROPOSAL FIELD (MANDATORY):\n"
            "Each sub-goal carries an 'objective' (imperative text, user's exact wording) and one\n"
            "typed 'proposal' declaring how the sub-goal's success is defined. Pick exactly one kind:\n"
            "\n"
            "  observed -> success is an observable screen state. Use this for reaching a screen,\n"
            "              revealing content, or verifying/confirming a condition.\n"
            '              Shape: {"kind": "OBSERVED", "assertion": "<observable state>"}\n'
            "\n"
            "  command  -> the user explicitly requested a specific device primitive, cited verbatim.\n"
            '              Shape: {"kind": "COMMAND", "requirement": {<operation + typed params>},\n'
            '                      "quote": "<exact intent words naming the operation>",\n'
            '                      "postcondition": {"assertion": "<optional observable result>"}}\n'
            "              The 'quote' MUST be an exact substring of the intent. Use these operations:\n"
            '                TAP / LONG_PRESS -> {"operation": "tap", "target": "<element>"}\n'
            '                TYPE             -> {"operation": "type", "target": "<field>", '
            '"text": "<text>"}\n'
            '                SCROLL           -> {"operation": "scroll", '
            '"direction": "UP|DOWN|LEFT|RIGHT", "target": "<optional surface>"}\n'
            '                SWIPE            -> {"operation": "swipe", '
            '"direction": "UP|DOWN|LEFT|RIGHT", "target": "<optional surface>"}\n'
            '                WAIT             -> {"operation": "wait", "condition": "<awaited state>", '
            '"bound": <seconds>}\n'
            '                BACK / HOME / HIDE_KEYBOARD -> {"operation": "back"}\n'
            "\n"
            "  capture  -> a 'store {value} as {name}' clause: save a visible value into a variable.\n"
            '              Shape: {"kind": "CAPTURE", "subject": "<what to capture>", '
            '"name": "<variable name>", "provenance": "USER|MODEL"}\n'
            "\n"
            "Rules:\n"
            "- Prefer 'observed' when success is defined by an outcome or state rather than a\n"
            "  specific gesture; an observed goal may take several device actions to satisfy.\n"
            "- Use 'command' only when the user named an exact primitive and you can cite the quote.\n"
            "- A store/capture clause is always its own 'capture' sub-goal; it never merges into the\n"
            "  follow-up action. If the user checks a precondition before storing, emit that\n"
            "  precondition as a separate 'observed' sub-goal immediately before the capture.\n"
            "- Capture 'provenance': USER when the intent explicitly named the variable (copy that\n"
            "  name verbatim); MODEL when the user gave no name (propose a valid snake_case\n"
            "  identifier). Never invent a USER name."
        )

    @staticmethod
    def __examples() -> str:
        """
        Worked examples covering the three proposal kinds and required splits.
        """

        return (
            "EXAMPLES (every sub-goal carries 'objective' and one 'proposal'):\n"
            'GOOD: User says "Tap the login button"\n'
            '      -> {"objective": "Tap the login button",\n'
            '          "proposal": {"kind": "COMMAND", "requirement": {"operation": "tap", '
            '"target": "login button"}, "quote": "Tap the login button"}}\n'
            "\n"
            'GOOD: User says "Open Settings app"\n'
            '      -> {"objective": "Open Settings app",\n'
            '          "proposal": {"kind": "OBSERVED", "assertion": "the Settings app is open"}}\n'
            "\n"
            'GOOD: User says "Scroll to labs section and select any category"\n'
            '      -> {"objective": "Scroll to labs section and select any category",\n'
            '          "proposal": {"kind": "OBSERVED", "assertion": "a labs category is selected"}}\n'
            "\n"
            'GOOD: User says "Go to cart and verify total amount"\n'
            '      -> {"objective": "Go to cart and verify total amount",\n'
            '          "proposal": {"kind": "OBSERVED", "assertion": "the cart total is displayed"}}\n'
            "\n"
            'GOOD: User says "Swipe up to reveal more results"\n'
            '      -> {"objective": "Swipe up to reveal more results",\n'
            '          "proposal": {"kind": "COMMAND", "requirement": {"operation": "swipe", '
            '"direction": "UP"}, "quote": "Swipe up"}}\n'
            "\n"
            'GOOD: User says "Capture the verification code as otp_code"\n'
            '      -> {"objective": "Capture the verification code as otp_code",\n'
            '          "proposal": {"kind": "CAPTURE", "subject": "the verification code", '
            '"name": "otp_code", "provenance": "USER"}}\n'
            "\n"
            'GOOD: User says "Verify the balance is visible; if it is, store the balance as '
            'account_balance and open transactions"\n'
            '      -> {"objective": "Verify the balance is visible",\n'
            '          "proposal": {"kind": "OBSERVED", "assertion": "the balance is visible"}}\n'
            '      -> {"objective": "Store the balance as account_balance",\n'
            '          "proposal": {"kind": "CAPTURE", "subject": "the balance", '
            '"name": "account_balance", "provenance": "USER"}}\n'
            '      -> {"objective": "Open transactions",\n'
            '          "proposal": {"kind": "OBSERVED", "assertion": "the transactions screen is '
            'open"}}\n'
            "\n"
            'BAD: User says "Capture the verification code as otp_code and continue"\n'
            '     -> {"objective": "Capture the verification code as otp_code and continue",\n'
            '         "proposal": {"kind": "OBSERVED", "assertion": "..."}}\n'
            "     Reason: the capture clause was merged into the follow-up action.\n"
            "\n"
            "BAD: paraphrasing the user's wording in the objective is prohibited.\n"
            "BAD: omitting the proposal field is prohibited.\n"
            "BAD: a 'command' quote that is not an exact substring of the intent is prohibited."
        )

    @staticmethod
    def __output_schema() -> str:
        """
        Strict JSON shape the LLM must return.
        """

        return (
            "Return ONLY a valid JSON with this structure:\n"
            "{\n"
            '  "sub_goals": [\n'
            "    {\n"
            '      "objective": "<imperative step text using user\'s exact wording>",\n'
            '      "proposal":  {\n'
            '        "kind": "<one of: OBSERVED | COMMAND | CAPTURE>",\n'
            '        "...":  "<the fields required by that kind, per the vocabulary above>"\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "confidence": 0.9\n'
            "}"
        )
