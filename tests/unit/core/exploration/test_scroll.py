from __future__ import annotations

import unittest

from fathom.constants.exploration import SCROLL_PROBE_MAX_PROBES
from fathom.core.exploration.config import ScrollProbeConfig
from fathom.core.exploration.scroll import ScrollProbePolicy


class TestScrollProbePolicy(unittest.TestCase):
    """The gate forces a first probe, then probes only while content keeps appearing, up to a cap."""

    def setUp(self) -> None:
        self.__policy = ScrollProbePolicy(config=ScrollProbeConfig(maximum=4))

    def test_first_probe_always_fires(self) -> None:
        self.assertTrue(self.__policy.should_probe(probes=0, advanced=False))

    def test_continues_while_probes_advance(self) -> None:
        self.assertTrue(self.__policy.should_probe(probes=1, advanced=True))
        self.assertTrue(self.__policy.should_probe(probes=3, advanced=True))

    def test_stops_when_a_probe_reveals_nothing(self) -> None:
        self.assertFalse(self.__policy.should_probe(probes=1, advanced=False))

    def test_stops_at_the_cap_even_if_still_advancing(self) -> None:
        self.assertFalse(self.__policy.should_probe(probes=4, advanced=True))

    def test_maximum_property(self) -> None:
        self.assertEqual(self.__policy.maximum, 4)

    def test_zero_maximum_disables_the_gate(self) -> None:
        disabled = ScrollProbePolicy(config=ScrollProbeConfig(maximum=0))
        self.assertFalse(disabled.should_probe(probes=0, advanced=True))

    def test_default_cap_matches_the_constant(self) -> None:
        self.assertEqual(
            ScrollProbePolicy(config=ScrollProbeConfig()).maximum, SCROLL_PROBE_MAX_PROBES
        )
