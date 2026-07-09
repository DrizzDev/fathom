from __future__ import annotations

import unittest

from fathom.core.capture.store import CaptureStore
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.capture import Capture


class CaptureStoreTest(unittest.TestCase):
    """
    Covers the run-owned capture registry: write, read, overwrite, missing, clear, and failure.
    """

    def test_write_then_read_returns_capture(self) -> None:
        """
        A written capture is retrievable by name.
        """

        store = CaptureStore()
        store.write(capture=Capture.succeeded(name="otp", value="123456", step=3))

        self.assertEqual(store.read(name="otp").value, "123456")

    def test_write_overwrites_prior_capture(self) -> None:
        """
        Re-capturing a name keeps the latest value.
        """

        store = CaptureStore()
        store.write(capture=Capture.succeeded(name="otp", value="111111", step=2))
        store.write(capture=Capture.succeeded(name="otp", value="222222", step=5))

        self.assertEqual(store.read(name="otp").value, "222222")

    def test_exists_reflects_membership(self) -> None:
        """
        exists is True only after a capture is written under that name.
        """

        store = CaptureStore()
        self.assertFalse(store.exists(name="otp"))

        store.write(capture=Capture.succeeded(name="otp", value="123456", step=1))
        self.assertTrue(store.exists(name="otp"))

    def test_read_missing_fails_fast(self) -> None:
        """
        Reading an absent capture raises rather than returning a silent empty value.
        """

        with self.assertRaises(InvariantViolation):
            CaptureStore().read(name="otp")

    def test_clear_empties_the_store(self) -> None:
        """
        clear drops every captured value.
        """

        store = CaptureStore()
        store.write(capture=Capture.succeeded(name="otp", value="123456", step=1))
        store.clear()

        self.assertFalse(store.exists(name="otp"))

    def test_failed_capture_is_stored_and_explicit(self) -> None:
        """
        A failed capture is recorded with its reason and no value, never as a fake success.
        """

        store = CaptureStore()
        store.write(capture=Capture.failed(name="otp", reason="no OTP field on screen", step=4))

        stored = store.read(name="otp")
        self.assertTrue(store.exists(name="otp"))
        self.assertFalse(stored.success)
        self.assertIsNone(stored.value)
        self.assertEqual(stored.reason, "no OTP field on screen")
