from __future__ import annotations

import unittest

from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.schemas.configuration import QualifierConfiguration
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
        """
        Temperature, use_cache and thinking_level on QualifierConfiguration must reach
        the LLMConfiguration so the dedicated qualifier LLM behaves deterministically.
        """

        assembly = RunAssemblyBuilder(settings=FathomSettings(gemini_api_key="x"))
        configuration = assembly.build_qualifier_model_configuration(
            configuration=QualifierConfiguration(
                temperature=0.0,
                use_cache=False,
                thinking_level="minimal",
            )
        )
        self.assertFalse(configuration.use_cache)
        self.assertEqual(configuration.temperature, 0.0)
        self.assertEqual(configuration.thinking_level, "minimal")


if __name__ == "__main__":
    unittest.main()
