from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.qualification import (
    DEFAULT_QUALIFIER_MAX_RETRIES,
    DEFAULT_QUALIFIER_MODEL,
    DEFAULT_QUALIFIER_TEMPERATURE,
    DEFAULT_QUALIFIER_THINKING_LEVEL,
    DEFAULT_QUALIFIER_TIMEOUT,
    DEFAULT_QUALIFIER_USE_CACHE,
)
from fathom.schemas.configuration import (
    InferenceConfiguration,
    QualifierConfiguration,
)


class InferenceConfigurationContractTest(unittest.TestCase):
    """
    InferenceConfiguration is the generic shape. By design it has NO defaults;
    every field must be supplied by the consumer's parent configuration via
    `default_factory`. This guarantees that one consumer's tuning cannot leak
    into another's defaults by accident.
    """

    def test_construction_requires_every_field(self) -> None:
        """
        Building InferenceConfiguration with any field missing must raise.
        Regression: if defaults are reintroduced, consumers can silently
        inherit values that were never explicitly chosen for them.
        """

        with self.assertRaises(ValidationError):
            InferenceConfiguration()  # type: ignore[call-arg]

        with self.assertRaises(ValidationError):
            InferenceConfiguration(model="gemini-2.5-flash-lite")  # type: ignore[call-arg]


class QualifierConfigurationDefaultsTest(unittest.TestCase):
    """
    QualifierConfiguration owns every qualifier-specific default via its
    default_factory. The values must come from the constants module so the
    eval-validated choices remain the single source of truth.
    """

    def test_default_inference_matches_constants(self) -> None:
        """
        Every default lookup must match the constants module exactly; if any
        drifts, this fails fast and tells the reader where to fix it.
        """

        inference = QualifierConfiguration().inference

        self.assertEqual(inference.model, DEFAULT_QUALIFIER_MODEL)
        self.assertEqual(inference.temperature, DEFAULT_QUALIFIER_TEMPERATURE)
        self.assertEqual(inference.use_cache, DEFAULT_QUALIFIER_USE_CACHE)
        self.assertEqual(inference.thinking_level, DEFAULT_QUALIFIER_THINKING_LEVEL)
        self.assertEqual(inference.timeout, DEFAULT_QUALIFIER_TIMEOUT)
        self.assertEqual(inference.max_retries, DEFAULT_QUALIFIER_MAX_RETRIES)


class QualifierConfigurationEvolveTest(unittest.TestCase):
    """
    evolve() lets callers override individual inference fields while keeping
    the qualifier-tuned defaults for everything else — without the boilerplate
    of respecifying every field that the strict-no-defaults InferenceConfiguration
    would otherwise force.
    """

    def test_evolve_overrides_only_named_fields(self) -> None:
        """
        Naming one field must override only that field; everything else stays
        at the eval-validated qualifier default.
        """

        configuration = QualifierConfiguration.evolve(model="gemini-3.5-flash")

        self.assertEqual(configuration.inference.model, "gemini-3.5-flash")
        self.assertEqual(configuration.inference.temperature, DEFAULT_QUALIFIER_TEMPERATURE)
        self.assertEqual(configuration.inference.use_cache, DEFAULT_QUALIFIER_USE_CACHE)
        self.assertEqual(configuration.inference.thinking_level, DEFAULT_QUALIFIER_THINKING_LEVEL)
        self.assertEqual(configuration.inference.timeout, DEFAULT_QUALIFIER_TIMEOUT)
        self.assertEqual(configuration.inference.max_retries, DEFAULT_QUALIFIER_MAX_RETRIES)

    def test_evolve_overrides_multiple_fields(self) -> None:
        """
        Naming several fields must override exactly those; others stay default.
        """

        configuration = QualifierConfiguration.evolve(
            model="gemini-2.5-flash",
            timeout=8.0,
            max_retries=4,
        )

        self.assertEqual(configuration.inference.model, "gemini-2.5-flash")
        self.assertEqual(configuration.inference.timeout, 8.0)
        self.assertEqual(configuration.inference.max_retries, 4)
        self.assertEqual(configuration.inference.temperature, DEFAULT_QUALIFIER_TEMPERATURE)
        self.assertEqual(configuration.inference.use_cache, DEFAULT_QUALIFIER_USE_CACHE)
        self.assertEqual(configuration.inference.thinking_level, DEFAULT_QUALIFIER_THINKING_LEVEL)

    def test_evolve_with_no_overrides_matches_default_construction(self) -> None:
        """
        Calling evolve() with no kwargs must produce the same configuration as
        the no-arg constructor. Degenerate case but valuable as a contract pin.
        """

        evolved = QualifierConfiguration.evolve()
        default = QualifierConfiguration()

        self.assertEqual(evolved, default)

    def test_evolve_does_not_mutate_the_default(self) -> None:
        """
        evolve() must return a new instance; subsequent calls must still get
        the original qualifier-tuned defaults from default_factory.
        """

        QualifierConfiguration.evolve(model="gemini-3.5-flash", timeout=30.0)

        fresh_default = QualifierConfiguration().inference
        self.assertEqual(fresh_default.model, DEFAULT_QUALIFIER_MODEL)
        self.assertEqual(fresh_default.timeout, DEFAULT_QUALIFIER_TIMEOUT)

    def test_evolve_rejects_unknown_inference_fields(self) -> None:
        """
        Pydantic's extra=forbid on InferenceConfiguration must surface typos in
        evolve() kwargs as ValidationError, not silently drop them.
        """

        with self.assertRaises(ValidationError):
            QualifierConfiguration.evolve(modle="gemini-3.5-flash")  # type: ignore[call-arg]

    def test_evolve_keeps_enabled_true(self) -> None:
        """
        evolve() touches only the inference block; the qualifier's enabled flag
        must keep its default. Catches an accidental signature broadening that
        leaks parent-level fields into the inference override path.
        """

        configuration = QualifierConfiguration.evolve(model="gemini-3.5-flash")
        self.assertTrue(configuration.enabled)


if __name__ == "__main__":
    unittest.main()
