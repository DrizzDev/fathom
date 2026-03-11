import unittest

from fathom.constants import ContextScope, DeviceConnectionType, DevicePlatform, ExecutionMode
from fathom.constants.run import SignalAdapterType, TargetKind
from fathom.schemas.run import (
    ExplorationObjectiveConfiguration,
    ExplorationRunRequest,
    IntentObjectiveConfiguration,
    IntentRunRequest,
    InteractionConfiguration,
    MemoryConfiguration,
    ModelSelectionConfiguration,
    RealignmentPolicy,
    ResourceConfiguration,
    RunMetadata,
    RuntimeConfiguration,
    TargetConfiguration,
)


class TestRunRequest(unittest.TestCase):
    """
    Unit tests for the canonical run request contract.
    """

    def test_intent_run_request_uses_canonical_nested_sections(self) -> None:
        """
        Validate the canonical host-agnostic run request shape.
        """

        request = IntentRunRequest(
            objective=IntentObjectiveConfiguration(
                intent="Open Airbnb",
                package_name="com.airbnb.android",
                max_steps=25,
                use_xml=True,
            ),
            runtime=RuntimeConfiguration(
                session_id="session-123",
                execution_id="execution-123",
                interactive=True,
                signal_type=SignalAdapterType.INTERACTIVE,
            ),
            memory=MemoryConfiguration(
                conversation_id="conversation-123",
                context_scope=ContextScope.CONVERSATION,
            ),
            resources=ResourceConfiguration(
                targets=[
                    TargetConfiguration(
                        name="primary",
                        kind=TargetKind.DEVICE,
                        device_configuration={
                            "type": DeviceConnectionType.REMOTE,
                            "platform": DevicePlatform.IOS,
                            "remote": {
                                "session_id": "session-123",
                                "execution_id": "execution-123",
                                "provider_url": "https://core.drizz.io/v1",
                            },
                        },
                    )
                ],
                language_model_configuration=ModelSelectionConfiguration(
                    planner_configuration={"model": "gemini-test"}
                ),
            ),
            interaction=InteractionConfiguration(
                realignment=RealignmentPolicy(budget=10),
                intent_configuration={"max_steps": 25, "use_xml_grounding": True},
                execution_configuration={"max_retries": 3, "stability_wait": 0.5},
                exploration_configuration={"max_steps": 25},
            ),
            metadata=RunMetadata(provider_name="GENYMOTION", device_name="iPhone 16"),
        )

        payload = request.model_dump(mode="json")

        self.assertIn("objective", payload)
        self.assertIn("runtime", payload)
        self.assertIn("memory", payload)
        self.assertIn("resources", payload)
        self.assertIn("interaction", payload)
        self.assertIn("telemetry", payload)
        self.assertIn("metadata", payload)
        self.assertEqual(payload["objective"]["mode"], ExecutionMode.INTENT)
        self.assertEqual(payload["runtime"]["signal_type"], SignalAdapterType.INTERACTIVE)
        self.assertEqual(payload["resources"]["targets"][0]["kind"], TargetKind.DEVICE)
        self.assertEqual(payload["interaction"]["realignment"]["budget"], 10)

    def test_run_request_requires_at_least_one_target(self) -> None:
        """
        Validate that canonical requests reject empty target lists.
        """

        with self.assertRaises(ValueError):
            IntentRunRequest(
                objective=IntentObjectiveConfiguration(intent="Open Airbnb"),
                resources=ResourceConfiguration(targets=[]),
            )

    def test_exploration_run_request_defaults_to_exploration_mode(self) -> None:
        """
        Validate exploration defaults remain deterministic.
        """

        request = ExplorationRunRequest(
            objective=ExplorationObjectiveConfiguration(max_steps=30),
            resources=ResourceConfiguration(
                targets=[
                    TargetConfiguration(
                        device_configuration={
                            "type": "REMOTE",
                            "platform": "ANDROID",
                            "remote": {
                                "session_id": "session-123",
                                "execution_id": "execution-123",
                                "provider_url": "https://core.drizz.io/v1",
                            },
                        }
                    )
                ]
            ),
        )

        self.assertEqual(request.objective.mode, ExecutionMode.EXPLORATION)
        self.assertFalse(request.objective.use_xml)
        self.assertEqual(request.objective.intent, "Explore application structure")
