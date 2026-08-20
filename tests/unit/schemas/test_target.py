from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.target import TargetAuthority


class TargetAuthorityTest(unittest.TestCase):
    """
    Pins the run's target-authority value object separating requested target from foreground.
    """

    def test_requested_binds_to_package(self) -> None:
        """
        An explicitly requested package produces bound authority.
        """

        authority = TargetAuthority.requested(package="com.example.shop")

        self.assertTrue(authority.bound)
        self.assertEqual(authority.package, "com.example.shop")

    def test_unbound_has_no_package(self) -> None:
        """
        Unbound authority carries no package and reports unbound.
        """

        authority = TargetAuthority.unbound()

        self.assertFalse(authority.bound)
        self.assertIsNone(authority.package)

    def test_is_immutable(self) -> None:
        """
        Authority is frozen and rejects mutation.
        """

        authority = TargetAuthority.unbound()

        with self.assertRaises(ValidationError):
            authority.package = "com.example.shop"

    def test_rejects_unknown_field(self) -> None:
        """
        An unexpected field is rejected by the forbid-extra contract.
        """

        with self.assertRaises(ValidationError):
            TargetAuthority.model_validate({"package": None, "origin": "REQUESTED"})

    def test_equality_by_value(self) -> None:
        """
        Two authorities with equal packages compare equal.
        """

        self.assertEqual(
            TargetAuthority.requested(package="com.app"),
            TargetAuthority.requested(package="com.app"),
        )
