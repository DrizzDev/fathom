import json
import unittest

from pydantic import ValidationError

from fathom.constants.storage import StorageBackend
from fathom.schemas.artifacts import (
    ScreenArtifact,
    ScreenArtifactBundle,
    StepArtifacts,
)


class TestScreenArtifact(unittest.TestCase):
    """
    Unit tests for the `ScreenArtifact` reference model.
    """

    def test_uri_and_image_are_both_optional(self) -> None:
        """
        Either or both of ``uri`` and ``image`` may be set; remaining
        metadata fields default safely.
        """

        artifact = ScreenArtifact(uri="/tmp/screens/before.png")

        self.assertEqual(artifact.uri, "/tmp/screens/before.png")
        self.assertIsNone(artifact.image)
        self.assertEqual(artifact.storage_backend, StorageBackend.LOCAL)
        self.assertEqual(artifact.mime_type, "image/png")
        self.assertIsNone(artifact.captured_at)
        self.assertIsNone(artifact.visual_hash)
        self.assertIsNone(artifact.width)
        self.assertIsNone(artifact.height)

    def test_bytes_only_construction_is_supported(self) -> None:
        """
        An artifact may carry bytes without any storage URI when the
        producer hands them through directly.
        """

        artifact = ScreenArtifact(image=b"\x89PNG\r\n")

        self.assertIsNone(artifact.uri)
        self.assertEqual(artifact.image, b"\x89PNG\r\n")

    def test_full_construction_preserves_all_fields(self) -> None:
        """
        All metadata fields round-trip through construction and serialization.
        """

        artifact = ScreenArtifact(
            uri="gs://drizz/screens/before.png",
            image=b"\x89PNG\r\n",
            storage_backend=StorageBackend.CLOUD,
            captured_at=1714200000000,
            visual_hash="0123456789abcdef",
            width=1080,
            height=2400,
            mime_type="image/png",
        )

        payload = artifact.model_dump(mode="json")

        self.assertEqual(payload["uri"], "gs://drizz/screens/before.png")
        self.assertEqual(payload["storage_backend"], StorageBackend.CLOUD.value)
        self.assertEqual(payload["captured_at"], 1714200000000)
        self.assertEqual(payload["visual_hash"], "0123456789abcdef")
        self.assertEqual(payload["width"], 1080)
        self.assertEqual(payload["height"], 2400)
        self.assertEqual(payload["mime_type"], "image/png")

    def test_image_bytes_are_excluded_from_json_dump(self) -> None:
        """
        Bytes must stay on the in-process instance but never inflate the
        wire payload; both ``model_dump`` and ``model_dump_json`` strip
        the field so telemetry stays small.
        """

        artifact = ScreenArtifact(uri="gs://drizz/a.png", image=b"\x89PNG\r\n")

        payload = artifact.model_dump(mode="json")
        json_payload = json.loads(artifact.model_dump_json())

        self.assertNotIn("image", payload)
        self.assertNotIn("image", json_payload)
        self.assertEqual(artifact.image, b"\x89PNG\r\n")

    def test_storage_backend_rejects_unknown_value(self) -> None:
        """
        Unknown storage backends fail validation, preventing string drift
        between configuration, factories, and artifact references.
        """

        with self.assertRaises(ValidationError):
            ScreenArtifact(uri="/tmp/before.png", storage_backend="unknown_backend")  # type: ignore[arg-type]

    def test_artifact_is_frozen(self) -> None:
        """
        `ScreenArtifact` is immutable so consumers can rely on stable references.
        """

        artifact = ScreenArtifact(uri="/tmp/before.png")

        with self.assertRaises(ValidationError):
            artifact.uri = "/tmp/other.png"  # type: ignore[misc]


class TestScreenArtifactBundle(unittest.TestCase):
    """
    Unit tests for the before/after screen artifact bundle.
    """

    def test_both_sides_optional_default_to_none(self) -> None:
        """
        A bundle can be constructed empty when no captures are persisted.
        """

        bundle = ScreenArtifactBundle()

        self.assertIsNone(bundle.before)
        self.assertIsNone(bundle.after)

    def test_only_before_is_supported(self) -> None:
        """
        After-side may be missing when post-action persistence fails.
        """

        bundle = ScreenArtifactBundle(
            before=ScreenArtifact(uri="/tmp/before.png"),
        )

        self.assertIsNotNone(bundle.before)
        self.assertIsNone(bundle.after)

    def test_only_after_is_supported(self) -> None:
        """
        Before-side may be missing for synthetic or recovery steps.
        """

        bundle = ScreenArtifactBundle(
            after=ScreenArtifact(uri="/tmp/after.png"),
        )

        self.assertIsNone(bundle.before)
        self.assertIsNotNone(bundle.after)

    def test_full_bundle_round_trips(self) -> None:
        """
        A populated bundle serializes to nested before/after JSON keys.
        """

        bundle = ScreenArtifactBundle(
            before=ScreenArtifact(uri="/tmp/before.png", visual_hash="aaaa"),
            after=ScreenArtifact(uri="/tmp/after.png", visual_hash="bbbb"),
        )

        payload = bundle.model_dump(mode="json")

        self.assertEqual(payload["before"]["uri"], "/tmp/before.png")
        self.assertEqual(payload["before"]["visual_hash"], "aaaa")
        self.assertEqual(payload["after"]["uri"], "/tmp/after.png")
        self.assertEqual(payload["after"]["visual_hash"], "bbbb")


class TestStepArtifacts(unittest.TestCase):
    """
    Unit tests for the namespaced step artifacts envelope.
    """

    def test_default_construction_is_empty(self) -> None:
        """
        Steps without artifacts produce a fully optional envelope.
        """

        artifacts = StepArtifacts()

        self.assertIsNone(artifacts.screen)

    def test_screen_namespace_is_nested(self) -> None:
        """
        Screen artifacts live under the `screen` namespace, not flattened.
        """

        artifacts = StepArtifacts(
            screen=ScreenArtifactBundle(
                before=ScreenArtifact(uri="/tmp/before.png"),
                after=ScreenArtifact(uri="/tmp/after.png"),
            ),
        )

        payload = artifacts.model_dump(mode="json")

        self.assertIn("screen", payload)
        self.assertIn("before", payload["screen"])
        self.assertIn("after", payload["screen"])
        self.assertEqual(payload["screen"]["before"]["uri"], "/tmp/before.png")
        self.assertEqual(payload["screen"]["after"]["uri"], "/tmp/after.png")

    def test_step_artifacts_is_frozen(self) -> None:
        """
        `StepArtifacts` cannot be mutated after construction.
        """

        artifacts = StepArtifacts(
            screen=ScreenArtifactBundle(before=ScreenArtifact(uri="/tmp/before.png")),
        )

        with self.assertRaises(ValidationError):
            artifacts.screen = None  # type: ignore[misc]

    def test_serialization_omits_unset_artifact_namespaces(self) -> None:
        """
        Future artifact namespaces must be additive without leaking nulls in
        canonical (non-default) JSON exports for existing consumers.
        """

        artifacts = StepArtifacts(
            screen=ScreenArtifactBundle(before=ScreenArtifact(uri="/tmp/before.png")),
        )

        payload = artifacts.model_dump(mode="json", exclude_none=True)

        self.assertEqual(set(payload.keys()), {"screen"})
        self.assertNotIn("after", payload["screen"])


class TestScreenArtifactBundleAnnotatedSlot(unittest.TestCase):
    """
    Pin the additive annotated slot on the artifact bundle.
    """

    def test_bundle_defaults_annotated_slot_to_none(self) -> None:
        """
        Backward compatibility: existing bundles without annotated work unchanged.
        """

        bundle = ScreenArtifactBundle()

        self.assertIsNone(bundle.annotated)

    def test_bundle_round_trips_all_three_slots(self) -> None:
        """
        before / after / annotated all survive a model_dump → model_validate
        round trip, matching the JSON shape downstream consumers read.
        """

        bundle = ScreenArtifactBundle(
            before=ScreenArtifact(uri="cdn://b"),
            after=ScreenArtifact(uri="cdn://a"),
            annotated=ScreenArtifact(uri="cdn://n"),
        )
        payload = bundle.model_dump(mode="json")

        self.assertEqual(payload["before"]["uri"], "cdn://b")
        self.assertEqual(payload["after"]["uri"], "cdn://a")
        self.assertEqual(payload["annotated"]["uri"], "cdn://n")
        self.assertEqual(ScreenArtifactBundle.model_validate(payload), bundle)

    def test_step_artifacts_round_trip_preserves_annotated(self) -> None:
        """
        Downstream readers consume ``artifacts.screen.annotated.uri`` from
        the dumped envelope, so the dump path must surface the new slot.
        """

        envelope = StepArtifacts(
            screen=ScreenArtifactBundle(
                before=ScreenArtifact(uri="cdn://b"),
                after=ScreenArtifact(uri="cdn://a"),
                annotated=ScreenArtifact(uri="cdn://n"),
            ),
        )
        payload = envelope.model_dump(mode="json")

        self.assertEqual(payload["screen"]["annotated"]["uri"], "cdn://n")


class TestScreenArtifactBundleTracesSlot(unittest.TestCase):
    """
    Pin the additive traces slot on the artifact bundle; None means no trace path ran, () means it ran empty.
    """

    def test_bundle_defaults_traces_slot_to_none(self) -> None:
        """
        Backward compatibility: existing bundles without traces work unchanged and report absent path.
        """

        bundle = ScreenArtifactBundle()

        self.assertIsNone(bundle.traces)

    def test_bundle_round_trips_traces_tuple(self) -> None:
        """
        A populated traces tuple survives the dump and validate round-trip with stable order.
        """

        traces = (
            ScreenArtifact(uri="cdn://trace-0"),
            ScreenArtifact(uri="cdn://trace-1"),
        )
        bundle = ScreenArtifactBundle(
            before=ScreenArtifact(uri="cdn://b"),
            after=ScreenArtifact(uri="cdn://a"),
            annotated=ScreenArtifact(uri="cdn://n"),
            traces=traces,
        )
        payload = bundle.model_dump(mode="json")

        self.assertEqual(
            [trace["uri"] for trace in payload["traces"]], ["cdn://trace-0", "cdn://trace-1"]
        )
        self.assertEqual(ScreenArtifactBundle.model_validate(payload), bundle)

    def test_step_artifacts_round_trip_preserves_traces(self) -> None:
        """
        Caller surfaces gesture traces via ``artifacts.screen.traces[*].uri`` on the dumped envelope.
        """

        envelope = StepArtifacts(
            screen=ScreenArtifactBundle(
                traces=(ScreenArtifact(uri="cdn://t"),),
            ),
        )
        payload = envelope.model_dump(mode="json")

        self.assertEqual(payload["screen"]["traces"][0]["uri"], "cdn://t")

    def test_empty_traces_tuple_is_distinguishable_from_absent(self) -> None:
        """
        Empty tuple means the trace path ran and produced no artifacts; None means the path never ran.
        """

        ran_empty = ScreenArtifactBundle(traces=())
        never_ran = ScreenArtifactBundle()

        self.assertEqual(ran_empty.traces, ())
        self.assertIsNone(never_ran.traces)
