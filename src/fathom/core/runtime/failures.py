from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from fathom.schemas.actions import Action
from fathom.schemas.supervision import BlockReason


class FailureRecord:
    """
    Immutable record of one blocked or failed action.
    """

    def __init__(self, *, key: str, reason: BlockReason, detail: str) -> None:
        """
        Initialize a failure record.
        """

        self.__key = key
        self.__reason = reason
        self.__detail = detail

    @property
    def key(self) -> str:
        """
        Return the stable action key.
        """

        return self.__key

    @property
    def reason(self) -> BlockReason:
        """
        Return the block reason.
        """

        return self.__reason

    @property
    def detail(self) -> str:
        """
        Return the failure detail.
        """

        return self.__detail


class FailureMemory:
    """
    Maintains bounded blocked-action memory for supervision.
    """

    def __init__(self, *, window: int = 20) -> None:
        """
        Initialize failure memory.
        """

        self.__records: Deque[FailureRecord] = deque(maxlen=window)

    def record(self, *, action: Action, reason: BlockReason, detail: str) -> None:
        """
        Record a failed or blocked action.
        """

        self.__records.append(
            FailureRecord(
                reason=reason,
                detail=detail,
                key=self.key_for(action=action),
            )
        )

    def is_blocked(self, *, action: Action, reason: Optional[BlockReason] = None) -> bool:
        """
        Return whether an action is blocked for the optional reason.
        """

        key = self.key_for(action=action)
        return any(
            record.key == key and (reason is None or record.reason == reason)
            for record in self.__records
        )

    def records(self) -> List[FailureRecord]:
        """
        Return recent failure records oldest first.
        """

        return list(self.__records)

    @staticmethod
    def key_for(*, action: Action) -> str:
        """
        Build a stable action key for repeat detection.
        """

        target = action.natural_language_target or action.target or "element"

        return f"{action.action_type.value}:{target.strip().lower()}"
