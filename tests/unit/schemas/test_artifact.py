from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.schemas.actions import Action
from fathom.schemas.artifact import (
    ArtifactKind,
    ArtifactReceipt,
    ArtifactRecord,
    LocalArtifactPolicy,
    OcrRawPayload,
    PerceptionPayload,
    PipelineConfiguration,
    ScreenshotPayload,
    TracePayload,
)
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle


class ArtifactRecordValidationTest(unittest.TestCase):
    """
    Pins :class:`ArtifactRecord` schema validation and discriminator routing.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Deterministic :class:`ScreenCapture` fixture; payloads only need bytes.
        """

        return ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=b"PNG",
            timestamp=0,
        )

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Minimal :class:`ScreenObservation` placeholder for perception payloads.
        """

        return ScreenObservation(
            activity="app",
            elements=(),
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="a" * 16,
                interaction_hash="b" * 16,
            ),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

    @staticmethod
    def __action() -> Action:
        """
        Minimal :class:`Action` fixture for trace payloads.
        """

        return Action(
            action_type=ActionType.TAP,
            target="x",
            rationale="t",
            confidence=1.0,
        )

    def test_perception_payload_carries_capture_and_observation(self) -> None:
        """
        :class:`PerceptionPayload` requires both capture and observation.
        """

        payload = PerceptionPayload(
            capture=self.__capture(),
            observation=self.__observation(),
        )

        self.assertEqual(payload.kind, ArtifactKind.PERCEPTION)

    def test_screenshot_payload_kind_locked_to_screenshot(self) -> None:
        """
        :class:`ScreenshotPayload`'s discriminator cannot be overridden.
        """

        payload = ScreenshotPayload(capture=self.__capture())

        self.assertEqual(payload.kind, ArtifactKind.SCREENSHOT)

    def test_ocr_raw_payload_kind_locked_to_ocr_raw(self) -> None:
        """
        Raw OCR JSON payloads route through the OCR raw artifact kind.
        """

        payload = OcrRawPayload(content='{"text": "Swiggy"}')

        self.assertEqual(payload.kind, ArtifactKind.OCR_RAW)

    def test_record_round_trips_through_json(self) -> None:
        """
        A record serialises and re-validates without loss; the sidecar
        replay path depends on this round-trip.
        """

        original = ArtifactRecord(
            session_id="run-1",
            package_name="app",
            step_number=3,
            created=1_700_000_000_000,
            payload=TracePayload(
                capture=self.__capture(),
                coords=(10, 20),
                action=self.__action(),
            ),
        )

        revived = ArtifactRecord.model_validate_json(original.model_dump_json())

        self.assertEqual(revived.payload.kind, ArtifactKind.TRACE)
        self.assertEqual(revived.step_number, 3)

    def test_record_rejects_negative_step_number(self) -> None:
        """
        Schema constraint pins ``step_number >= 0``.
        """

        with self.assertRaises(ValidationError):
            ArtifactRecord(
                session_id="run-1",
                package_name="app",
                step_number=-1,
                created=0,
                payload=ScreenshotPayload(capture=self.__capture()),
            )

    def test_record_rejects_blank_session_identifier(self) -> None:
        """
        Schema constraint pins ``session_id`` non-empty.
        """

        with self.assertRaises(ValidationError):
            ArtifactRecord(
                session_id="",
                package_name="app",
                step_number=0,
                created=0,
                payload=ScreenshotPayload(capture=self.__capture()),
            )

    def test_discriminator_routes_payload_correctly_on_decode(self) -> None:
        """
        Discriminated-union decoding selects the right payload class
        based on ``kind``.
        """

        envelope = {
            "session_id": "run-1",
            "package_name": "app",
            "step_number": 0,
            "created": 1,
            "payload": {
                "kind": ArtifactKind.PERCEPTION.value,
                "capture": self.__capture().model_dump(mode="json"),
                "observation": self.__observation().model_dump(mode="json"),
            },
        }

        record = ArtifactRecord.model_validate(envelope)

        self.assertIsInstance(record.payload, PerceptionPayload)


class ArtifactReceiptTest(unittest.TestCase):
    """
    Pins the receipt shape used to communicate cleanup intent back to the pipeline.
    """

    def test_receipt_demands_identifier_and_cleanup_flag(self) -> None:
        """
        Both fields are required; the schema is strict-extra.
        """

        receipt = ArtifactReceipt(identifier="cloud://x", local_cleanup=True)

        self.assertEqual(receipt.identifier, "cloud://x")
        self.assertTrue(receipt.local_cleanup)

    def test_receipt_rejects_extra_fields(self) -> None:
        """
        ``extra='forbid'`` keeps the receipt boundary clean.
        """

        with self.assertRaises(ValidationError):
            ArtifactReceipt(identifier="x", local_cleanup=False, surprise=True)  # type: ignore[call-arg]


class LocalArtifactPolicyTest(unittest.TestCase):
    """
    Pins the EFS retention policy schema and its embedding into PipelineConfiguration.
    """

    def test_default_cleanup_is_enabled(self) -> None:
        """
        Crawler and other consumers without explicit overrides must keep today's cleanup behaviour.
        """

        self.assertTrue(LocalArtifactPolicy().cleanup)
        self.assertTrue(PipelineConfiguration().local.cleanup)

    def test_cleanup_can_be_disabled_for_host_managed_retention(self) -> None:
        """
        Hosts that own a fallback sweep disable the policy explicitly.
        """

        policy = LocalArtifactPolicy(cleanup=False)

        self.assertFalse(policy.cleanup)
        self.assertFalse(PipelineConfiguration(local=policy).local.cleanup)

    def test_policy_rejects_extra_fields(self) -> None:
        """
        ``extra='forbid'`` keeps the boundary tight.
        """

        with self.assertRaises(ValidationError):
            LocalArtifactPolicy(cleanup=True, surprise=1)  # type: ignore[call-arg]
