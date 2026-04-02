from __future__ import annotations

import unittest

from fathom.core.exceptions import DeviceConnectionClosedError, DeviceError, FathomError


class FathomErrorTest(unittest.TestCase):
    """
    Cover generic client-message resolution for exceptions.
    """

    def test_device_error_defaults_to_retryable(self) -> None:
        """
        Preserve retryability for generic device errors.
        """

        self.assertTrue(DeviceError("transient device issue").retryable)

    def test_closed_device_connection_uses_stable_display_message(self) -> None:
        """
        Use the exception type to resolve a stable display message.
        """

        message = DeviceConnectionClosedError("internal detail").display(
            fallback="Failed to capture the current app screen. Please retry.",
        )

        self.assertEqual(message, "Lost the device connection. Please retry the run.")

    def test_uses_fallback_for_non_fathom_errors(self) -> None:
        """
        Fall back to the caller-provided message for unknown exception types.
        """

        exception = RuntimeError("internal failure")
        message = (
            exception.display(fallback="Failed to capture the current app screen. Please retry.")
            if isinstance(exception, FathomError)
            else "Failed to capture the current app screen. Please retry."
        )

        self.assertEqual(message, "Failed to capture the current app screen. Please retry.")

    def test_plain_fathom_error_uses_fallback(self) -> None:
        """
        Fall back when the exception type does not override client messaging.
        """

        message = DeviceError("internal detail").display(
            fallback="Failed to save execution details for the current step.",
        )

        self.assertEqual(message, "Failed to save execution details for the current step.")
