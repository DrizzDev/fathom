from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.authoring.adapters.artifacts import FileAuthoringArtifactProvider
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import AuthoringKind
from fathom.schemas.artifacts import ScreenArtifact, ScreenArtifactBundle, StepArtifacts
from fathom.schemas.authoring import AuthoringArtifactConfiguration, AuthoringTask
from fathom.schemas.flow import Evidence, EvidenceStep, StepCapture


class FileAuthoringArtifactProviderTest(unittest.TestCase):
    """
    Cover local artifact attachment for model authoring requests.
    """

    def test_run_authoring_attaches_configured_image_and_manifest_payloads(self) -> None:
        """
        Run authoring must attach optional screenshots and manifests when configured.
        """

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "screen.png"
            manifest = root / "tree.xml"
            image.write_bytes(b"image-bytes")
            manifest.write_text("<hierarchy />")

            artifacts = StepArtifacts(
                screen=ScreenArtifactBundle(
                    after=ScreenArtifact(uri=str(image), mime_type="image/png"),
                    traces=(ScreenArtifact(uri=str(manifest), mime_type="text/xml"),),
                )
            )
            evidence = Evidence(
                intent="find product",
                goal="find product",
                package="com.example",
                artifacts=(str(manifest),),
                steps=(
                    EvidenceStep(
                        action="tap",
                        event="action",
                        index=1,
                        artifacts=artifacts,
                    ),
                ),
            )
            task = AuthoringTask(
                kind=AuthoringKind.RUN,
                intent="find product",
                step_number=1,
                execution_id="execution-1",
                evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
            )
            provider = FileAuthoringArtifactProvider(
                configuration=AuthoringArtifactConfiguration(
                    max_images=2,
                    include_images=True,
                    include_manifests=True,
                )
            )

            parts = provider.build(task=task)

        self.assertIn(b"image-bytes", parts)
        self.assertTrue(
            any(
                isinstance(part, dict) and "<hierarchy />" in str(part.get("text", ""))
                for part in parts
            )
        )

    def test_image_attachment_respects_configuration(self) -> None:
        """
        Image artifacts must be skipped when image attachment is disabled.
        """

        with TemporaryDirectory() as temporary:
            image = Path(temporary) / "screen.png"
            image.write_bytes(b"image-bytes")
            evidence = Evidence(
                intent="tap",
                goal="tap",
                package="com.example",
                steps=(
                    EvidenceStep(
                        action="tap",
                        event="action",
                        index=1,
                        artifacts=StepArtifacts(
                            screen=ScreenArtifactBundle(
                                after=ScreenArtifact(uri=str(image), mime_type="image/png")
                            )
                        ),
                    ),
                ),
            )
            task = AuthoringTask(
                kind=AuthoringKind.RUN,
                intent="tap",
                step_number=1,
                execution_id="execution-1",
                evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
            )
            provider = FileAuthoringArtifactProvider(
                configuration=AuthoringArtifactConfiguration(include_images=False)
            )

            parts = provider.build(task=task)

        self.assertNotIn(b"image-bytes", parts)

    def test_run_authoring_prefers_capture_screen_when_image_budget_is_bounded(self) -> None:
        """
        Run authoring spends a small image budget on screens that explain important commands.
        """

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            generic = root / "generic.png"
            capture = root / "capture.png"
            generic.write_bytes(b"generic-image")
            capture.write_bytes(b"capture-image")
            evidence = Evidence(
                intent="store price",
                goal="store price",
                package="com.example",
                steps=(
                    EvidenceStep(
                        action="tap",
                        event="action",
                        index=1,
                        artifacts=StepArtifacts(
                            screen=ScreenArtifactBundle(
                                after=ScreenArtifact(uri=str(generic), mime_type="image/png")
                            )
                        ),
                    ),
                    EvidenceStep(
                        action="store",
                        event="action",
                        index=2,
                        capture=StepCapture(
                            name="price",
                            subject="product price",
                            value="87",
                            success=True,
                        ),
                        artifacts=StepArtifacts(
                            screen=ScreenArtifactBundle(
                                after=ScreenArtifact(uri=str(capture), mime_type="image/png")
                            )
                        ),
                    ),
                ),
            )
            task = AuthoringTask(
                kind=AuthoringKind.RUN,
                intent="store price",
                step_number=2,
                execution_id="execution-1",
                evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
            )
            provider = FileAuthoringArtifactProvider(
                configuration=AuthoringArtifactConfiguration(
                    max_images=1,
                    include_images=True,
                )
            )

            parts = provider.build(task=task)

        self.assertIn(b"capture-image", parts)
        self.assertNotIn(b"generic-image", parts)
