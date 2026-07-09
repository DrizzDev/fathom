from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.capture import Capture, CaptureRequest


class CaptureModelTest(unittest.TestCase):
    """
    Pins the Capture model's success/failure invariant and its construction helpers.
    """

    def test_succeeded_carries_value_and_no_reason(self) -> None:
        """
        A successful capture exposes its value and no failure reason.
        """

        capture = Capture.succeeded(name=" otp ", value=" 123456 ", step=2)

        self.assertTrue(capture.success)
        self.assertEqual(capture.name, "otp")
        self.assertEqual(capture.value, "123456")
        self.assertIsNone(capture.reason)

    def test_failed_carries_reason_and_no_value(self) -> None:
        """
        A failed capture exposes its reason and no value.
        """

        capture = Capture.failed(name=" otp ", reason=" field not found ", step=2)

        self.assertFalse(capture.success)
        self.assertEqual(capture.name, "otp")
        self.assertIsNone(capture.value)
        self.assertEqual(capture.reason, "field not found")

    def test_success_without_value_is_rejected(self) -> None:
        """
        Declaring success without a captured value is invalid.
        """

        with self.assertRaises(ValidationError):
            Capture(name="otp", step=1, success=True)

    def test_failure_without_reason_is_rejected(self) -> None:
        """
        Declaring failure without a reason is invalid.
        """

        with self.assertRaises(ValidationError):
            Capture(name="otp", step=1, success=False)

    def test_success_with_reason_is_rejected(self) -> None:
        """
        A success must not also carry a failure reason.
        """

        with self.assertRaises(ValidationError):
            Capture(name="otp", step=1, success=True, value="x", reason="oops")

    def test_empty_name_is_rejected(self) -> None:
        """
        A capture must carry a non-empty variable name.
        """

        with self.assertRaises(ValidationError):
            Capture.succeeded(name="", value="123456", step=1)

    def test_success_with_empty_value_is_rejected(self) -> None:
        """
        A successful capture must carry a non-empty value.
        """

        with self.assertRaises(ValidationError):
            Capture.succeeded(name="otp", value="", step=1)

        with self.assertRaises(ValidationError):
            Capture.succeeded(name="otp", value="   ", step=1)

    def test_failure_with_empty_reason_is_rejected(self) -> None:
        """
        A failed capture must carry a non-empty reason.
        """

        with self.assertRaises(ValidationError):
            Capture.failed(name="otp", reason="", step=1)

        with self.assertRaises(ValidationError):
            Capture.failed(name="otp", reason="   ", step=1)


class CaptureRequestModelTest(unittest.TestCase):
    """
    Pins the intent-derived CaptureRequest payload carried by a STORE action.
    """

    def test_value_request_is_accepted(self) -> None:
        """
        A STORE request carries the captured value alongside the subject.
        """

        request = CaptureRequest(name=" abc ", subject=" price of soap ", value=" ₹86 ")

        self.assertEqual(request.name, "abc")
        self.assertEqual(request.subject, "price of soap")
        self.assertEqual(request.value, "₹86")

    def test_empty_name_is_rejected(self) -> None:
        """
        A request must carry a non-empty variable name.
        """

        with self.assertRaises(ValidationError):
            CaptureRequest(name="", subject="xyz", value="₹86")

    def test_empty_subject_is_rejected(self) -> None:
        """
        A request must carry a non-empty subject.
        """

        with self.assertRaises(ValidationError):
            CaptureRequest(name="abc", subject="", value="₹86")

    def test_empty_value_is_rejected(self) -> None:
        """
        A request must carry a non-empty captured value.
        """

        with self.assertRaises(ValidationError):
            CaptureRequest(name="abc", subject="price", value="")

        with self.assertRaises(ValidationError):
            CaptureRequest(name="abc", subject="price", value="   ")
