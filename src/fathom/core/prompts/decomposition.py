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
            "  tap          -> 'Tap {{target}}', 'Click {{target}}', 'Press {{target}}',\n"
            "                  'Select {{target}}' (when the target is a UI element),\n"
            "                  'Open {{app_name}} app' (tapping an app icon)\n"
            "  type         -> 'Enter {{text}}', 'Type {{text}}', 'Input {{text}}',\n"
            "                  'Fill {{field}} with {{text}}'\n"
            "  validate     -> 'Validate {{state}}', 'Verify {{state}}', 'Check {{condition}}',\n"
            "                  'Confirm {{state}}', 'Ensure {{state}} is displayed'\n"
            "  swipe_up     -> 'Swipe up', 'Scroll up' (when directional and the\n"
            "                  surface is a vertical list/feed)\n"
            "  swipe_down   -> 'Swipe down', 'Scroll down'\n"
            "  swipe_left   -> 'Swipe left', 'Scroll left' (carousels)\n"
            "  swipe_right  -> 'Swipe right'\n"
            "  scroll       -> Generic 'Scroll to {{destination}}' when no direction is specified\n"
            "  wait         -> 'Wait', 'Wait for {{state}}', 'Wait N seconds'\n"
            "  store        -> 'Store {{value_or_subject}} as {{variable_name}}',\n"
            "                  'Capture {{value_or_subject}} as {{variable_name}}'\n"
            "                  (save a visible/context value into a script variable)\n"
            "  long_press   -> 'Long press {{target}}', 'Hold {{target}}', "
            "'Press and hold {{target}}'\n"
            "  back         -> 'Go back', 'Press back', 'Navigate back'\n"
            "  home         -> 'Go to home', 'Press home'\n"
            "  hide_keyboard-> 'Dismiss keyboard', 'Hide keyboard'\n"
            "  ask_user     -> Explicit 'Ask user for X' steps\n"
            "\n"
            "Rules:\n"
            "- Choose the SINGLE most-specific directive for each step.\n"
            "- Prefer 'swipe_up' / 'swipe_down' / 'swipe_left' / 'swipe_right' over the\n"
            "  generic 'scroll' whenever a direction is implied.\n"
            "- A store/capture clause never loses to the final action of a compound step;\n"
            "  split it into its own 'store' sub-goal before the follow-up action.\n"
            "- A store/capture sub-goal must describe the capture itself. If the user asks to\n"
            "  check, verify, validate, or confirm a precondition before storing, emit that\n"
            "  precondition as a separate 'validate' sub-goal immediately before the store.\n"
            "- A compound step that ends in tapping (e.g. 'Scroll to X and tap Y') takes\n"
            "  the directive of the FINAL action -> 'tap', unless it contains a store/capture\n"
            "  clause that must be split out first.\n"
            "- A compound step that ends in validation (e.g. 'Go to cart and verify total')\n"
            "  takes 'validate'.\n"
            "\n"
            "PROOF FIELD (MANDATORY):\n"
            "Each sub-goal must also carry a 'proof' field with exactly one of:\n"
            "  DURABLE    -> the step mutates persistent state whose outcome must be\n"
            "                observed before moving on (add to cart, pay, save, submit,\n"
            "                delete, send, place order, toggle a persisted setting).\n"
            "  TRANSIENT  -> the step only navigates or reveals state (open, tap through,\n"
            "                scroll, read, focus a field).\n"
            "- When uncertain, choose DURABLE: a wrong DURABLE costs one extra check;\n"
            "  a wrong TRANSIENT can silently skip outcome verification."
        )

    @staticmethod
    def __examples() -> str:
        """
        Worked examples covering canonical happy-path mappings.
        """

        return (
            "EXAMPLES:\n"
            'GOOD: User says "Tap the login button"\n'
            '      -> {"description": "Tap the login button", "directive": "tap",\n'
            '          "proof": "TRANSIENT"}\n'
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
            'GOOD: User says "Capture the verification code as otp_code"\n'
            '      -> {"description": "Capture the verification code as otp_code",\n'
            '          "directive": "store"}\n'
            "\n"
            'GOOD: User says "If upload completed, capture the confirmation ID as confirmation_id '
            'and close the dialog"\n'
            '      -> {"description": "Check whether upload completed",\n'
            '          "directive": "validate"}\n'
            '      -> {"description": "If upload completed, capture the confirmation ID as '
            'confirmation_id",\n'
            '          "directive": "store"}\n'
            '      -> {"description": "Close the dialog", "directive": "tap"}\n'
            "\n"
            'GOOD: User says "Verify the balance is visible; if it is, store the balance as '
            'account_balance and open transactions"\n'
            '      -> {"description": "Verify the balance is visible",\n'
            '          "directive": "validate"}\n'
            '      -> {"description": "If the balance is visible, store the balance as '
            'account_balance",\n'
            '          "directive": "store"}\n'
            '      -> {"description": "Open transactions", "directive": "tap"}\n'
            "\n"
            'BAD: User says "Capture the verification code as otp_code and continue"\n'
            '     -> {"description": "Capture the verification code as otp_code and continue",\n'
            '         "directive": "tap"}\n'
            "     Reason: the store clause was merged into the follow-up action.\n"
            "\n"
            'BAD: User says "Verify the balance is visible; if it is, store the balance as '
            'account_balance"\n'
            '     -> {"description": "Verify the balance is visible if it is, store the balance '
            'as account_balance",\n'
            '         "directive": "store"}\n'
            "     Reason: the prerequisite validation was merged into the store command.\n"
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
            "                                 store | long_press | back | home |\n"
            '                                 hide_keyboard | ask_user>"\n'
            "    }\n"
            "  ],\n"
            '  "confidence": 0.9\n'
            "}"
        )
