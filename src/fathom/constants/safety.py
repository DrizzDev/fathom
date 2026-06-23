from __future__ import annotations

from enum import StrEnum
from typing import Dict, Final, FrozenSet

# Tokens that flag an action as potentially destructive when present in its
# target description or rationale. The supervisor blocks any action whose
# textual context overlaps this set so the planner is forced to escalate.
UNSAFE_ACTION_KEYWORDS: Final[FrozenSet[str]] = frozenset(
    {
        "wipe",
        "purge",
        "erase",
        "format",
        "delete account",
        "factory reset",
        "remove account",
    }
)


class SensitiveCategory(StrEnum):
    """
    Area a broad-coverage crawl should describe but never act into.

    PAYMENT     - Checkout, billing, and payment-method entry.
    AUTH        - Sign-in, sign-up, OTP, and sign-out.
    DESTRUCTIVE - Irreversible data or account changes.
    """

    PAYMENT = "payment"
    AUTH = "auth"
    DESTRUCTIVE = "destructive"


# Keywords (matched on word boundaries against an action's target and rationale)
# that mark it as entering a sensitive area, grouped by the area they guard. The
# traversal guard vetoes a matching action so a broad-coverage crawl never pays,
# authenticates, or destroys data. Tokens are chosen to be specific: bare "pay"
# is excluded so it cannot fire on "payment options" the crawl only describes.
DEFAULT_SENSITIVE_KEYWORDS: Final[Dict[SensitiveCategory, FrozenSet[str]]] = {
    SensitiveCategory.PAYMENT: frozenset(
        {
            "checkout",
            "place order",
            "pay now",
            "proceed to pay",
            "make payment",
            "add card",
            "card number",
            "cvv",
            "upi",
            "net banking",
        }
    ),
    SensitiveCategory.AUTH: frozenset(
        {
            "login",
            "log in",
            "sign in",
            "sign up",
            "signup",
            "otp",
            "password",
            "logout",
            "log out",
        }
    ),
    SensitiveCategory.DESTRUCTIVE: frozenset(
        {
            "delete account",
            "remove account",
            "factory reset",
            "clear data",
        }
    ),
}
