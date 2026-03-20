from __future__ import annotations

import re

ORDINAL_MAP = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "ninth",
    "10th": "tenth",
}

NUMERIC_ORDINAL_RE = re.compile(pattern=r"\b(\d+)(?:st|nd|rd|th)\b", flags=re.IGNORECASE)
GENERIC_TARGETS = frozenset(
    {
        "element",
        "ui element",
        "none",
        "label",
        "unknown",
        "a visible item",
    }
)
SWIPE_ACTIONS = frozenset({"swipe_up", "swipe_down", "swipe_left", "swipe_right", "scroll"})

SCREEN_RE = re.compile(
    pattern=r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(screen|page)\b",
    flags=re.IGNORECASE,
)
LABEL_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "any",
        "some",
        "no",
        "or",
        "and",
        "this",
        "that",
        "on",
        "in",
        "at",
        "to",
        "of",
        "is",
        "it",
        "my",
        "its",
    }
)

SCROLL_VERB_RE = re.compile(
    pattern=r"(?:find|look(?:ing)?\s+for|search(?:ing)?\s+for)\s+(.+?)(?:\.|,\s|;|$)",
    flags=re.IGNORECASE,
)
PROPER_PHRASE_RE = re.compile(pattern=r"\b([A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)+)")
DYNAMIC_TARGET_PREFIXES = (
    "add to cart button for ",
    "increase quantity button for ",
    "decrease quantity button for ",
    "remove item button for ",
)
STORE_NAME_PATTERN = re.compile(
    pattern=(
        r"\b(?:"
        r"walmart|costco|target|kroger|safeway|publix|aldi|instacart|"
        r"whole\s+foods|trader\s+joe'?s|amazon\s+fresh|tesco"
        r")\b\s+"
        r"(?=(?:"
        r"continue\s+shopping|cart|button|item|entry|row|store|aisle|checkout|basket"
        r")\b)"
    ),
    flags=re.IGNORECASE,
)
