from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from governance.constants import RuleId
from governance.schemas.finding import Violation
from governance.schemas.module import ParsedModule


class Rule(ABC):
    """
    An architecture-fitness rule that inspects one parsed module for violations.
    """

    @property
    @abstractmethod
    def identifier(self) -> RuleId:
        """
        Stable identifier for this rule.
        """

    @property
    @abstractmethod
    def waivable(self) -> bool:
        """
        Whether a debt record may accept this rule's violations during migration.
        A non-waivable rule's violations always block; a record may only track remediation.
        """

    @abstractmethod
    def check(self, *, module: ParsedModule) -> List[Violation]:
        """
        Return the violations this rule finds in the given module.
        """
