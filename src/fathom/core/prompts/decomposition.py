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
    Produces sub-goals carrying a structured ``directive`` (ActionType enum) so the
    completion gate can compare planner output against the decomposer's contract.
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
            self.__directive_vocabulary(),
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
            '- "Scroll to X and select/tap/click Y" -> ONE step (scroll is just navigation to reach Y)\n'
            '- "Navigate to X and do Y" -> ONE step (navigate is means to reach Y)\n'
            '- "Find X and tap/click Y" -> ONE step (find is means to reach Y)\n'
            '- "Go to X and verify/check Y" -> ONE step (navigation + verification together)'
        )

    @staticmethod
    def __directive_vocabulary() -> str:
        """
        Map natural-language verbs to the structured ``directive`` ActionType.

        This is the load-bearing section that drives the completion gate's
        type-match check, so the mapping is exhaustive and unambiguous.
        """

        return (
            "DIRECTIVE FIELD (MANDATORY):\n"
            "Each sub-goal must carry a 'directive' field naming the action type the\n"
            "planner is expected to emit. Use these exact tokens (lower-case, no spaces):\n"
            "\n"
            "  tap          -> 'Tap X', 'Click X', 'Press X', 'Select X' (when X is a UI element),\n"
            "                  'Open Y app' (tapping an app icon)\n"
            "  type         -> 'Enter X', 'Type X', 'Input X', 'Fill X with Y'\n"
            "  validate     -> 'Validate X', 'Verify X', 'Check X', 'Confirm X',\n"
            "                  'Ensure X is displayed', 'Assert X is visible'\n"
            "  swipe_up     -> 'Swipe up', 'Scroll up' (when directional and the\n"
            "                  surface is a vertical list/feed)\n"
            "  swipe_down   -> 'Swipe down', 'Scroll down'\n"
            "  swipe_left   -> 'Swipe left', 'Scroll left' (carousels)\n"
            "  swipe_right  -> 'Swipe right'\n"
            "  scroll       -> Generic 'Scroll to X' when no direction is specified\n"
            "  wait         -> 'Wait', 'Wait for X', 'Wait N seconds'\n"
            "  long_press   -> 'Long press X', 'Hold X', 'Press and hold X'\n"
            "  back         -> 'Go back', 'Press back', 'Navigate back'\n"
            "  home         -> 'Go to home', 'Press home'\n"
            "  enter        -> 'Press enter', 'Submit', 'Confirm via keyboard'\n"
            "  hide_keyboard-> 'Dismiss keyboard', 'Hide keyboard'\n"
            "  ask_user     -> Explicit 'Ask user for X' steps\n"
            "\n"
            "Rules:\n"
            "- Choose the SINGLE most-specific directive for each step.\n"
            "- Prefer 'swipe_up' / 'swipe_down' / 'swipe_left' / 'swipe_right' over the\n"
            "  generic 'scroll' whenever a direction is implied.\n"
            "- A compound step that ends in tapping (e.g. 'Scroll to X and tap Y') takes\n"
            "  the directive of the FINAL action -> 'tap'.\n"
            "- A compound step that ends in validation (e.g. 'Go to cart and verify total')\n"
            "  takes 'validate'."
        )

    @staticmethod
    def __examples() -> str:
        """
        Worked examples covering canonical happy-path mappings.
        """

        return (
            "EXAMPLES:\n"
            'GOOD: User says "Tap the login button"\n'
            '      -> {"description": "Tap the login button", "directive": "tap"}\n'
            "\n"
            "GOOD: User says \"Enter password 'test123'\"\n"
            '      -> {"description": "Enter password \'test123\'", "directive": "type"}\n'
            "\n"
            'GOOD: User says "Open Settings app"\n'
            '      -> {"description": "Open Settings app", "directive": "tap"}\n'
            "\n"
            'GOOD: User says "Scroll to labs section and select any category"\n'
            '      -> {"description": "Scroll to labs section and select any category",\n'
            '          "directive": "tap"}   (compound: final action wins)\n'
            "\n"
            'GOOD: User says "Go to cart and verify total amount"\n'
            '      -> {"description": "Go to cart and verify total amount",\n'
            '          "directive": "validate"}\n'
            "\n"
            'GOOD: User says "Swipe up to reveal more results"\n'
            '      -> {"description": "Swipe up to reveal more results",\n'
            '          "directive": "swipe_up"}\n'
            "\n"
            'GOOD: User says "Validate srp page is loaded"\n'
            '      -> {"description": "Validate srp page is loaded",\n'
            '          "directive": "validate"}\n'
            "\n"
            "BAD: paraphrasing the user's wording is prohibited.\n"
            "BAD: omitting the directive field is prohibited.\n"
            "BAD: inventing tokens outside the directive vocabulary above is prohibited."
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
            '      "description": "<imperative step text using user\'s exact wording>",\n'
            '      "criterion":   "<observable screen state when the step is complete>",\n'
            '      "directive":   "<one of: tap | type | validate | swipe_up | swipe_down |\n'
            "                                 swipe_left | swipe_right | scroll | wait |\n"
            "                                 long_press | back | home | enter |\n"
            '                                 hide_keyboard | ask_user>"\n'
            "    }\n"
            "  ],\n"
            '  "confidence": 0.9\n'
            "}"
        )
