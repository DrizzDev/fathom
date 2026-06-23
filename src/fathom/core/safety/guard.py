from __future__ import annotations

import re
from typing import FrozenSet, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.safety import DEFAULT_SENSITIVE_KEYWORDS, SensitiveCategory
from fathom.constants.screen import ScreenCategory

# Screen categories the crawl treats as sensitive: it describes them but does not
# act into them, so a broad-coverage run never authenticates or pays.
_SENSITIVE_SCREENS: FrozenSet[ScreenCategory] = frozenset(
    {ScreenCategory.AUTH, ScreenCategory.PAYMENT}
)


class TraversalVerdict(BaseModel):
    """
    Outcome of a traversal-guard check on a proposed exploration action.
    """

    model_config = ConfigDict(frozen=True)

    allowed: bool = Field(description="Whether the action may be executed during traversal.")
    category: Optional[SensitiveCategory] = Field(
        default=None, description="Sensitive area the action would enter, when blocked."
    )
    matched_keyword: Optional[str] = Field(
        default=None, description="Keyword that triggered the veto, when blocked."
    )

    @property
    def reason(self) -> Optional[str]:
        """
        Veto message for the scan re-prompt, or None when the action is allowed.
        """

        if self.allowed or self.category is None:
            return None
        return (
            f'Avoid the {self.category.value} area (matched "{self.matched_keyword}"): '
            "describe this screen but pick a DIFFERENT element that does not "
            "authenticate, pay, or destroy data."
        )


class TraversalGuard:
    """
    Vetoes exploration actions that would enter sensitive areas (payment, auth, destructive).
    """

    def __init__(
        self,
        *,
        denylist: Optional[Mapping[SensitiveCategory, FrozenSet[str]]] = None,
    ) -> None:
        self.__denylist = denylist if denylist is not None else DEFAULT_SENSITIVE_KEYWORDS

    def inspect_action(self, *, target: str, rationale: str = "") -> TraversalVerdict:
        """
        Returns a blocking verdict when an action's text enters a sensitive area.

        Matches keywords on word boundaries against the action's target and
        rationale, mirroring the intent-level safety classifier, so a single
        token like "otp" matches but does not fire inside an unrelated word.
        """

        haystack = f"{target} {rationale}".lower()
        for category, keywords in self.__denylist.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", haystack):
                    return TraversalVerdict(
                        allowed=False, category=category, matched_keyword=keyword
                    )
        return TraversalVerdict(allowed=True)

    @staticmethod
    def is_sensitive_screen(*, category: ScreenCategory) -> bool:
        """
        Whether a screen's category marks it as a sensitive area to avoid acting into.
        """

        return category in _SENSITIVE_SCREENS
