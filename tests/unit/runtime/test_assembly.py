from __future__ import annotations

import unittest

from fathom.constants.llm import InferencePriorityMode
from fathom.constants.qualification import DEFAULT_QUALIFIER_MODEL
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.schemas.configuration import QualifierConfiguration
from fathom.schemas.run import IntentObjectiveConfiguration, RunRequest, TargetConfiguration
from fathom.settings.env import FathomSettings


class RunAssemblyBuilderQualifierLLMConfigurationTest(unittest.TestCase):
    """
    Assembly must derive the qualifier LLM configuration from its bound settings,
    not from environment defaults or a fresh FathomSettings(). This is the regression
    check for the staging bug where a builder-internal FathomSettings() saw no credentials.
    """

    def test_qualifier_llm_inherits_credentials_from_bound_settings(self) -> None:
        """
        Credentials must come from the settings the caller bound, not from env.
        """

        bound_settings = FathomSettings(
            gemini_api_key="test-api-key",
            vertex_location="bound-location",
            vertex_project_id="bound-project",
            google_application_credentials="/fake/credentials.json",
        )

        assembly = RunAssemblyBuilder(settings=bound_settings)
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration()
        )

        self.assertEqual(configuration.api_key, "test-api-key")
        self.assertEqual(configuration.location, "bound-location")
        self.assertEqual(configuration.project_id, "bound-project")
        self.assertEqual(configuration.credentials, "/fake/credentials.json")

    def test_qualifier_knobs_flow_into_llm_configuration(self) -> None:
        """Inference knobs on QualifierConfiguration must reach the LLMConfiguration so the dedicated qualifier LLM behaves deterministically."""

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration.evolve(thinking_level="minimal"),
        )
        self.assertEqual(configuration.thinking_level, "minimal")

    def test_qualifier_model_defaults_to_constant(self) -> None:
        """The qualifier owns its model selection — it must not silently inherit the planner's GEMINI_MODEL setting (a known prod regression: preview model leaking through)."""

        assembly = RunAssemblyBuilder(
            settings=FathomSettings(
                gemini_api_key="x",
                gemini_model="gemini-3-flash-preview",  # planner model — must NOT leak
            )
        )
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration()
        )
        self.assertEqual(configuration.model, DEFAULT_QUALIFIER_MODEL)

    def test_qualifier_model_can_be_overridden_via_evolve(self) -> None:
        """
        Caller can choose a different qualifier model via evolve() without
        changing the planner's GEMINI_MODEL env var and without restating
        every other inference field.
        """

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration.evolve(model="gemini-2.5-flash"),
        )
        self.assertEqual(configuration.model, "gemini-2.5-flash")

    def test_qualifier_timeout_and_retries_flow_into_llm_configuration(self) -> None:
        """
        Per-attempt timeout and retry budget must reach the LLMConfiguration so
        the adapter applies them. Adapter owns the retry loop; no nested retries.
        """

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration.evolve(timeout=3.0, max_retries=4),
        )
        self.assertEqual(configuration.timeout, 3.0)
        self.assertEqual(configuration.max_retries, 4)

    def test_planner_priority_policy_flows_from_settings(self) -> None:
        """
        Planner LLM configuration must inherit priority settings from the bound settings object.
        """

        assembly = RunAssemblyBuilder(
            settings=FathomSettings(
                gemini_api_key="x",
                capacity_enabled=True,
                capacity_mode=InferencePriorityMode.ADAPTIVE.value,
                capacity_window=4,
                capacity_failures=2,
                capacity_slows=3,
                capacity_latency=6.5,
                capacity_recovery=2,
            ),
        )
        configuration = assembly.build_planner_model_configuration(
            request=RunRequest(
                objective=IntentObjectiveConfiguration(intent="Open settings"),
                resources={"targets": [TargetConfiguration()]},
            ),
        )

        self.assertTrue(configuration.priority.enabled)
        self.assertEqual(configuration.priority.mode, InferencePriorityMode.ADAPTIVE)
        self.assertEqual(configuration.priority.adaptive.window, 4)
        self.assertEqual(configuration.priority.adaptive.threshold.failures, 2)
        self.assertEqual(configuration.priority.adaptive.threshold.slows, 3)
        self.assertEqual(configuration.priority.adaptive.threshold.latency, 6.5)
        self.assertEqual(configuration.priority.adaptive.threshold.recovery, 2)

    def test_request_priority_configuration_overrides_settings(self) -> None:
        """
        Run requests may explicitly override planner priority policy without changing environment settings.
        """

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        request = RunRequest(
            objective=IntentObjectiveConfiguration(intent="Open settings"),
            resources={
                "targets": [TargetConfiguration()],
                "language_model_configuration": {
                    "planner_configuration": {
                        "priority": {
                            "enabled": False,
                            "mode": InferencePriorityMode.ALWAYS.value,
                        },
                    },
                },
            },
        )

        configuration = assembly.build_planner_model_configuration(request=request)

        self.assertFalse(configuration.priority.enabled)

    def test_qualifier_priority_policy_flows_from_settings(self) -> None:
        """
        Dedicated qualifier LLM configuration must use the same priority policy as planner.
        """

        assembly = RunAssemblyBuilder(
            settings=FathomSettings(
                gemini_api_key="x",
                capacity_mode=InferencePriorityMode.ADAPTIVE.value,
                capacity_recovery=3,
            ),
        )
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration(),
        )

        self.assertEqual(configuration.priority.mode, InferencePriorityMode.ADAPTIVE)
        self.assertEqual(configuration.priority.adaptive.threshold.recovery, 3)


if __name__ == "__main__":
    unittest.main()
