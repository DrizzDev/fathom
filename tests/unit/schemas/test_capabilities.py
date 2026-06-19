from __future__ import annotations

import unittest

import pytest
from pydantic import ValidationError

from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities


class HITLCapabilityTest(unittest.TestCase):
    """Pins the HITLCapability schema contract."""

    def test_enabled_true_is_readable(self) -> None:
        """enabled is set and readable."""

        capability = HITLCapability(enabled=True)

        self.assertTrue(capability.enabled)

    def test_enabled_false_is_readable(self) -> None:
        """Autonomous runtime maps to enabled=False."""

        capability = HITLCapability(enabled=False)

        self.assertFalse(capability.enabled)

    def test_is_immutable(self) -> None:
        """Frozen model prevents post-construction mutation."""

        capability = HITLCapability(enabled=False)

        with pytest.raises(ValidationError):
            capability.enabled = True  # type: ignore[misc]

    def test_requires_enabled(self) -> None:
        """enabled has no default and must be supplied explicitly."""

        with pytest.raises(ValidationError):
            HITLCapability()  # type: ignore[call-arg]


class RuntimeCapabilitiesTest(unittest.TestCase):
    """Pins the RuntimeCapabilities schema contract."""

    def test_exposes_nested_hitl_flag(self) -> None:
        """Hitl capability is reachable via the nested field."""

        capabilities = RuntimeCapabilities(hitl=HITLCapability(enabled=True))

        self.assertTrue(capabilities.hitl.enabled)

    def test_is_immutable(self) -> None:
        """Frozen model prevents field reassignment."""

        capabilities = RuntimeCapabilities(hitl=HITLCapability(enabled=False))

        with pytest.raises(ValidationError):
            capabilities.hitl = HITLCapability(enabled=True)  # type: ignore[misc]

    def test_requires_hitl(self) -> None:
        """hitl has no default and must be supplied explicitly."""

        with pytest.raises(ValidationError):
            RuntimeCapabilities()  # type: ignore[call-arg]

    def test_equality_by_value(self) -> None:
        """Two capability records with the same nested flag compare equal."""

        self.assertEqual(
            RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
            RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        self.assertNotEqual(
            RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
            RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
