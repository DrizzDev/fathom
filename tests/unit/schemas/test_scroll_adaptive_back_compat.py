"""
Pin the backwards-compatibility shim for the deleted adaptive-scroll subsystem.

Older hosts (enricher's healing bridge) still construct
``ScrollInteractionPolicy.AdaptivePolicy(...)``. The shim must accept the
legacy kwargs without raising so those hosts continue to boot while they
migrate.
"""

from __future__ import annotations

import logging
import unittest

from fathom.schemas.configuration import ScrollInteractionPolicy


class TestAdaptivePolicyBackwardCompatibility(unittest.TestCase):
    """
    Pins that the deprecated AdaptivePolicy shim remains importable and inert.
    """

    def test_enricher_legacy_construction_succeeds(self) -> None:
        """
        Exact legacy invocation from the enricher healing bridge must not raise.
        """

        policy = ScrollInteractionPolicy(
            adaptive=ScrollInteractionPolicy.AdaptivePolicy(
                enabled=True,
                verify=False,
                budget=200000,
                maximum_attempts=2,
            ),
        )

        self.assertIsNotNone(policy.adaptive)
        self.assertEqual(policy.edge_margin_ratio, 0.15)

    def test_unknown_legacy_kwargs_are_ignored(self) -> None:
        """
        Extra legacy kwargs on the shim must be accepted silently.
        """

        shim = ScrollInteractionPolicy.AdaptivePolicy(unknown_legacy_field="ignored")
        self.assertFalse(shim.enabled)

    def test_construction_emits_deprecation_warning(self) -> None:
        """
        Each shim construction emits a structured deprecation warning.
        """

        with self.assertLogs("fathom.schemas.configuration", level=logging.WARNING) as captured:
            ScrollInteractionPolicy.AdaptivePolicy(enabled=True)
        self.assertTrue(
            any("deprecated" in record.getMessage() for record in captured.records),
            "Expected a deprecation warning when constructing the shim.",
        )
