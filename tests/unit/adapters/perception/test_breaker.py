from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from fathom.adapters.perception.breaker import HierarchyBreaker
from fathom.constants.screen import HierarchyProvenance


class HierarchyBreakerTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the bounded-cooldown policy and the attempt/skip provenance distinction.
    """

    async def test_healthy_dump_keeps_breaker_closed(self) -> None:
        """
        A dump that yields a hierarchy never opens the breaker and carries no fallback provenance.
        """

        breaker = HierarchyBreaker(cooldown=3)
        dump = AsyncMock(return_value=(b"img", "<hierarchy/>"))
        screenshot = AsyncMock(return_value=b"shot")

        snapshot = await breaker.snapshot(dump=dump, screenshot=screenshot)

        self.assertEqual(snapshot.image, b"img")
        self.assertEqual(snapshot.hierarchy, "<hierarchy/>")
        self.assertIsNone(snapshot.provenance)
        self.assertFalse(breaker.open)
        screenshot.assert_not_called()

    async def test_failed_attempt_is_distinct_from_open_circuit(self) -> None:
        """
        The tripping dump is provenance ATTEMPT_FAILED; the subsequent skipped captures are CIRCUIT_OPEN, so telemetry can count real attempts.
        """

        breaker = HierarchyBreaker(cooldown=2)
        dump = AsyncMock(return_value=(b"img", None))
        screenshot = AsyncMock(return_value=b"shot")

        first = await breaker.snapshot(dump=dump, screenshot=screenshot)
        self.assertEqual(first.provenance, HierarchyProvenance.ATTEMPT_FAILED)
        self.assertIsNone(first.hierarchy)
        self.assertTrue(breaker.open)
        self.assertEqual(dump.await_count, 1)

        for _ in range(2):
            skipped = await breaker.snapshot(dump=dump, screenshot=screenshot)
            self.assertEqual(skipped.image, b"shot")
            self.assertEqual(skipped.provenance, HierarchyProvenance.CIRCUIT_OPEN)

        # Exactly one real (expensive) dump attempt occurred; the rest were skipped.
        self.assertEqual(dump.await_count, 1)
        self.assertEqual(screenshot.await_count, 2)
        self.assertFalse(breaker.open)

    async def test_breaker_reattempts_dump_after_cooldown_drains(self) -> None:
        """
        Once the cooldown drains the breaker reattempts the dump and recovers on success.
        """

        breaker = HierarchyBreaker(cooldown=1)
        healthy = AsyncMock(return_value=(b"img", "<hierarchy/>"))
        failing = AsyncMock(return_value=(b"img", None))
        screenshot = AsyncMock(return_value=b"shot")

        await breaker.snapshot(dump=failing, screenshot=screenshot)  # trips
        await breaker.snapshot(dump=failing, screenshot=screenshot)  # drains cooldown
        self.assertFalse(breaker.open)

        recovered = await breaker.snapshot(dump=healthy, screenshot=screenshot)

        self.assertEqual(recovered.hierarchy, "<hierarchy/>")
        self.assertIsNone(recovered.provenance)
        healthy.assert_awaited_once()
