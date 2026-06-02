from __future__ import annotations

from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.schemas.qualification import QualificationVerdict, Rationale


class PermissiveIntentQualifier(IntentQualifierPort):
    """
    Qualifier that always returns EXECUTABLE with maximum confidence.
    """

    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Always classify the intent as executable so no run is blocked.
        """

        _ = intent

        return QualificationVerdict(
            message=None,
            confidence=1.0,
            label=QualificationLabel.EXECUTABLE,
            rationale=Rationale(
                category=RationaleCategory.PERMISSIVE,
                reasoning="Permissive qualifier accepts every intent without inspection.",
            ),
        )
