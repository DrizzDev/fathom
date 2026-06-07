from __future__ import annotations

import unittest
from typing import List

from fathom.core.services.abort.composite import CompositeAbortDetector
from fathom.interfaces.abort import AbortDetectorPort
from fathom.schemas.abort import AbortDecision


class _ScriptedDetector(AbortDetectorPort):
    """
    Detector double returning a scripted decision and tracking warmup calls.
    """

    def __init__(self, *, decision: AbortDecision) -> None:
        """
        Pre-seed the canned decision delivered by every call to :meth:`aborted`.
        """

        self.__decision = decision
        self.calls: List[str] = []
        self.warmup_calls: int = 0

    async def aborted(self, *, response: str) -> AbortDecision:
        """
        Record the response and return the canned decision.
        """

        self.calls.append(response)
        return self.__decision

    async def warmup(self) -> None:
        """
        Track each warmup invocation.
        """

        self.warmup_calls += 1


class CompositeAbortDetectorRoutingTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the composite's primary-vs-fallback routing policy.
    """

    async def test_primary_decision_is_returned_when_not_fallback(self) -> None:
        """
        A decisive primary verdict short-circuits the composite without consulting fallback.
        """

        primary = _ScriptedDetector(
            decision=AbortDecision(aborted=True, confidence=0.92, fallback=False)
        )
        fallback = _ScriptedDetector(
            decision=AbortDecision(aborted=False, confidence=1.0, fallback=True)
        )
        composite = CompositeAbortDetector(primary=primary, fallback=fallback)

        decision = await composite.aborted(response="close the execution")

        self.assertTrue(decision.aborted)
        self.assertFalse(decision.fallback)

        self.assertEqual(fallback.calls, [])
        self.assertEqual(primary.calls, ["close the execution"])

    async def test_fallback_is_consulted_when_primary_abstains(self) -> None:
        """
        When primary returns ``fallback=True`` the composite delegates to the fallback.
        """

        primary = _ScriptedDetector(
            decision=AbortDecision(aborted=False, confidence=0.0, fallback=True)
        )
        fallback = _ScriptedDetector(
            decision=AbortDecision(aborted=True, confidence=0.9, fallback=True)
        )
        composite = CompositeAbortDetector(primary=primary, fallback=fallback)

        decision = await composite.aborted(response="please cancel")

        self.assertTrue(decision.aborted)
        self.assertTrue(decision.fallback)
        self.assertEqual(primary.calls, ["please cancel"])
        self.assertEqual(fallback.calls, ["please cancel"])


class CompositeAbortDetectorWarmupTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins composite warmup fans out to both detectors.
    """

    async def test_warmup_warms_both_detectors(self) -> None:
        """
        Composite warmup must call ``warmup`` on each component detector once.
        """

        primary = _ScriptedDetector(
            decision=AbortDecision(aborted=False, confidence=0.0, fallback=False)
        )
        fallback = _ScriptedDetector(
            decision=AbortDecision(aborted=False, confidence=0.0, fallback=True)
        )
        composite = CompositeAbortDetector(primary=primary, fallback=fallback)

        await composite.warmup()

        self.assertEqual(primary.warmup_calls, 1)
        self.assertEqual(fallback.warmup_calls, 1)
