from enum import Enum


class PromptMode(Enum):
    """
    Defines the operational mode for the prompt and tool selection.
    """

    DEFAULT = "default"
    DISCOVERY = "discovery"  # Navigation, finding elements
    INTERACTION = "interaction"  # Typing, tapping, forms
    VERIFICATION = "verification"  # Assertions, checks
    EXPLORATION = "exploration"  # BFS app mapping, screen discovery
    SCREEN_TRANSLATION = "screen_translation"  # Rich screen description
