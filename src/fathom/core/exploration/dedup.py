"""
Deduplication and sampling policy for proposed exploration actions.
"""

from __future__ import annotations

from typing import AbstractSet, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import ActionType
from fathom.core.exploration.config import DedupConfig, SamplingConfig
from fathom.schemas.actions import Action

REPEATABLE_ACTION_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.BACK,
        ActionType.SCROLL,
        ActionType.SWIPE,
        ActionType.SWIPE_UP,
        ActionType.SWIPE_DOWN,
        ActionType.SWIPE_LEFT,
        ActionType.SWIPE_RIGHT,
    }
)


class ActionKey(BaseModel):
    """
    Stable identity of an action on a screen, resilient to label drift.
    """

    model_config = ConfigDict(frozen=True)

    kind: str = Field(description="Lower-cased action type")
    label: str = Field(description="Lower-cased coordinate bucket or element label")


class DedupPolicy:
    """
    Decides whether a proposed action is novel, repeatable, or over-sampled.
    """

    def __init__(self, *, dedup: DedupConfig, sampling: SamplingConfig) -> None:
        self.__retries = dedup.retries
        self.__limits = dict(sampling.limits)

    @property
    def retries(self) -> int:
        """
        Number of re-prompts allowed before exhaustion is forced.
        """

        return self.__retries

    @staticmethod
    def key_for(action: Action) -> ActionKey:
        """
        Build the dedup key from an action's coordinate bucket or label.
        """

        bucket = action.bounds.coord_bucket() if action.bounds is not None else None
        label = bucket or action.natural_language_target or action.target or ""
        return ActionKey(kind=action.action_type.value.lower(), label=label.lower())

    @staticmethod
    def is_repeatable(action: Action) -> bool:
        """
        Whether the action may legitimately be issued again on the same screen.
        """

        return action.action_type in REPEATABLE_ACTION_TYPES

    def limit_for(self, category: Optional[str]) -> Optional[int]:
        """
        Sampling cap for an element category, or None when uncapped.
        """

        if category is None:
            return None
        return self.__limits.get(category)

    def is_over_sampled(self, *, category: Optional[str], sampled: int) -> bool:
        """
        Whether the category has reached its per-screen sampling cap.
        """

        limit = self.limit_for(category)
        return limit is not None and sampled >= limit

    def is_novel(self, *, action: Action, tried: AbstractSet[ActionKey]) -> bool:
        """
        Whether the action has not yet been tried on the current screen.
        """

        return self.key_for(action) not in tried
