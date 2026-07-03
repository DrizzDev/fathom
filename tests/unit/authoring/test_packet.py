from __future__ import annotations

import unittest

from fathom.authoring.agent.packet import AuthoringPacketBuilder
from fathom.authoring.agent.reference import AuthoringReferenceProvider
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import AuthoringKind
from fathom.constants.dialect import DialectName
from fathom.schemas.authoring import AuthoringTask
from fathom.schemas.flow import Evidence


class AuthoringPacketBuilderTest(unittest.TestCase):
    """
    Cover typed authoring packet construction.
    """

    def test_builds_packet_without_copying_evidence(self) -> None:
        """
        Packet construction must reuse the supplied task evidence.
        """

        evidence = Evidence(intent="open app", goal="home visible", package="com.example")
        task = AuthoringTask(
            kind=AuthoringKind.RUN,
            intent="open app",
            step_number=2,
            workflow_id="workflow-1",
            evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
        )
        reference = AuthoringReferenceProvider().reference(dialect=DialectName.DRIZZ)

        packet = AuthoringPacketBuilder().build(task=task, dialect=reference)

        self.assertEqual(packet.task, task)
        assert packet.task.evidence.run is not None
        self.assertIs(packet.task.evidence.run.source, evidence)
        self.assertEqual(packet.dialect.name, DialectName.DRIZZ)
