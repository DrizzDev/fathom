from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.authoring import AuthoringMode
from fathom.constants.llm import (
    DEFAULT_PRIORITY_FAILURE_THRESHOLD,
    DEFAULT_PRIORITY_LATENCY_THRESHOLD,
    DEFAULT_PRIORITY_RECOVERY_SUCCESSES,
    DEFAULT_PRIORITY_SLOW_THRESHOLD,
    DEFAULT_PRIORITY_WINDOW,
    InferencePriorityMode,
)
from fathom.constants.qualification import (
    DEFAULT_QUALIFIER_MAX_RETRIES,
    DEFAULT_QUALIFIER_MODEL,
    DEFAULT_QUALIFIER_TEMPERATURE,
    DEFAULT_QUALIFIER_THINKING_LEVEL,
    DEFAULT_QUALIFIER_TIMEOUT,
    DEFAULT_QUALIFIER_USE_CACHE,
)
from fathom.constants.storage import StorageBackend
from fathom.schemas.authoring import (
    AuthoringConfiguration,
    RunConfiguration,
    StepAuthoringConfiguration,
)
from fathom.schemas.configuration import (
    InferenceConfiguration,
    LLMConfiguration,
    PostgresInteractionConfiguration,
    PriorityInferenceConfiguration,
    QualifierConfiguration,
    StorageConfiguration,
)


class InferenceConfigurationContractTest(unittest.TestCase):
    """
    InferenceConfiguration has no defaults; parent configs own defaults.
    """

    def test_construction_requires_every_field(self) -> None:
        """
        Missing inference fields must raise validation errors.
        """

        with self.assertRaises(ValidationError):
            InferenceConfiguration()  # type: ignore[call-arg]

        with self.assertRaises(ValidationError):
            InferenceConfiguration(model="gemini-2.5-flash-lite")  # type: ignore[call-arg]


class LLMConfigurationDefaultsTest(unittest.TestCase):
    """
    LLMConfiguration owns the shared runtime model defaults.
    """

    def test_default_timeout_is_thirty_seconds(self) -> None:
        """
        The shared Gemini per-attempt timeout must stay bounded for script and planner calls.
        """

        configuration = LLMConfiguration()

        self.assertEqual(configuration.timeout, 30.0)


class AuthoringConfigurationTest(unittest.TestCase):
    """
    Authoring configuration keeps step and run switches nested and typed.
    """

    def test_defaults_enable_run_authoring_and_disable_step_authoring(self) -> None:
        """
        Fathom should attempt final run authoring while keeping rich per-step authoring opt-in.
        """

        configuration = AuthoringConfiguration()

        self.assertTrue(configuration.run.enabled)
        self.assertIs(configuration.step.mode, AuthoringMode.DISABLED)

    def test_nested_overrides_are_preserved(self) -> None:
        """
        Callers can configure run and step authoring independently.
        """

        configuration = AuthoringConfiguration(
            run=RunConfiguration(enabled=False),
            step=StepAuthoringConfiguration(mode=AuthoringMode.ASYNC),
        )

        self.assertFalse(configuration.run.enabled)
        self.assertIs(configuration.step.mode, AuthoringMode.ASYNC)


class QualifierConfigurationDefaultsTest(unittest.TestCase):
    """
    QualifierConfiguration owns qualifier-specific defaults.
    """

    def test_default_inference_matches_constants(self) -> None:
        """
        Default qualifier inference values must match the constants module.
        """

        inference = QualifierConfiguration().inference

        self.assertEqual(inference.model, DEFAULT_QUALIFIER_MODEL)
        self.assertEqual(inference.temperature, DEFAULT_QUALIFIER_TEMPERATURE)
        self.assertEqual(inference.use_cache, DEFAULT_QUALIFIER_USE_CACHE)
        self.assertEqual(inference.thinking_level, DEFAULT_QUALIFIER_THINKING_LEVEL)
        self.assertEqual(inference.timeout, DEFAULT_QUALIFIER_TIMEOUT)
        self.assertEqual(inference.max_retries, DEFAULT_QUALIFIER_MAX_RETRIES)


class PriorityInferenceConfigurationDefaultsTest(unittest.TestCase):
    """
    Priority inference defaults are intentionally enabled and always-on for current rollout.
    """

    def test_default_priority_configuration_is_always_enabled(self) -> None:
        """
        Default LLM configuration must request priority for every call.
        """

        configuration = PriorityInferenceConfiguration()

        self.assertTrue(configuration.enabled)
        self.assertEqual(configuration.mode, InferencePriorityMode.ALWAYS)

    def test_default_adaptive_thresholds_match_constants(self) -> None:
        """
        Adaptive defaults must come from constants to keep rollout knobs discoverable.
        """

        adaptive = PriorityInferenceConfiguration().adaptive

        self.assertEqual(adaptive.window, DEFAULT_PRIORITY_WINDOW)
        self.assertEqual(adaptive.threshold.failures, DEFAULT_PRIORITY_FAILURE_THRESHOLD)
        self.assertEqual(adaptive.threshold.slows, DEFAULT_PRIORITY_SLOW_THRESHOLD)
        self.assertEqual(adaptive.threshold.latency, DEFAULT_PRIORITY_LATENCY_THRESHOLD)
        self.assertEqual(adaptive.threshold.recovery, DEFAULT_PRIORITY_RECOVERY_SUCCESSES)


class QualifierConfigurationEvolveTest(unittest.TestCase):
    """
    evolve() overrides selected inference fields while preserving defaults.
    """

    def test_evolve_overrides_only_named_fields(self) -> None:
        """
        One override must not change unrelated inference defaults.
        """

        configuration = QualifierConfiguration.evolve(model="gemini-3.5-flash")

        self.assertEqual(configuration.inference.model, "gemini-3.5-flash")
        self.assertEqual(configuration.inference.timeout, DEFAULT_QUALIFIER_TIMEOUT)
        self.assertEqual(configuration.inference.use_cache, DEFAULT_QUALIFIER_USE_CACHE)
        self.assertEqual(configuration.inference.temperature, DEFAULT_QUALIFIER_TEMPERATURE)
        self.assertEqual(configuration.inference.max_retries, DEFAULT_QUALIFIER_MAX_RETRIES)
        self.assertEqual(configuration.inference.thinking_level, DEFAULT_QUALIFIER_THINKING_LEVEL)

    def test_evolve_overrides_multiple_fields(self) -> None:
        """
        Naming several fields must override exactly those; others stay default.
        """

        configuration = QualifierConfiguration.evolve(
            timeout=8.0,
            max_retries=4,
            model="gemini-2.5-flash",
        )

        self.assertEqual(configuration.inference.model, "gemini-2.5-flash")
        self.assertEqual(configuration.inference.timeout, 8.0)
        self.assertEqual(configuration.inference.max_retries, 4)
        self.assertEqual(configuration.inference.use_cache, DEFAULT_QUALIFIER_USE_CACHE)
        self.assertEqual(configuration.inference.temperature, DEFAULT_QUALIFIER_TEMPERATURE)
        self.assertEqual(configuration.inference.thinking_level, DEFAULT_QUALIFIER_THINKING_LEVEL)

    def test_evolve_with_no_overrides_matches_default_construction(self) -> None:
        """
        Calling evolve() with no kwargs must produce the same configuration as
        the no-arg constructor. Degenerate case but valuable as a contract pin.
        """

        default = QualifierConfiguration()
        evolved = QualifierConfiguration.evolve()

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
        Unknown evolve fields must raise instead of being silently ignored.
        """

        with self.assertRaises(ValidationError):
            QualifierConfiguration.evolve(modle="gemini-3.5-flash")


class StorageConfigurationDefaultBackendsTest(unittest.TestCase):
    """
    Storage defaults should remain local-only unless explicitly changed.
    """

    def test_default_backends_are_local_only(self) -> None:
        """
        Default storage configuration must not attempt cloud upload.
        """

        configuration = StorageConfiguration()

        self.assertEqual(configuration.backends, {StorageBackend.LOCAL})

    def test_explicit_cloud_only_backends_respected(self) -> None:
        """
        Explicit cloud-only storage must remain cloud-only.
        """

        configuration = StorageConfiguration(
            backends={StorageBackend.CLOUD},
            storage_bucket="example-bucket",
        )

        self.assertEqual(configuration.backends, {StorageBackend.CLOUD})


class TestPostgresInteractionConfiguration(unittest.TestCase):
    """
    Unit tests for Postgres component-field configuration.
    """

    def __minimal_kwargs(self) -> dict:
        """
        Required-field kwargs for the configuration; defaults applied to
        host/user/password/database leave the test focused on the
        validator under test.
        """

        return {
            "host": "localhost",
            "user": "fathom",
            "password": "secret",
            "database": "fathom",
        }

    def test_arbitrary_password_is_accepted_unchanged(self) -> None:
        """
        Passwords are passed straight to asyncpg with no URL encoding,
        so reserved URL characters survive intact.
        """

        kwargs = self.__minimal_kwargs()
        kwargs["password"] = "o]sDs$X|)cDisP"

        config = PostgresInteractionConfiguration(**kwargs)

        self.assertEqual("o]sDs$X|)cDisP", config.password)

    def test_missing_host_is_rejected(self) -> None:
        """
        Host is mandatory; an empty value must fail validation.
        """

        kwargs = self.__minimal_kwargs()
        kwargs["host"] = ""

        with self.assertRaises(ValidationError):
            PostgresInteractionConfiguration(**kwargs)

    def test_pool_max_must_be_at_least_min(self) -> None:
        """
        The validator must reject a pool whose maximum is below its
        minimum.
        """

        kwargs = self.__minimal_kwargs()
        kwargs["pool_min_size"] = 5
        kwargs["pool_max_size"] = 2

        with self.assertRaises(ValidationError):
            PostgresInteractionConfiguration(**kwargs)

    def test_dsn_only_construction_is_accepted(self) -> None:
        """
        When a DSN is supplied the discrete host/user/password become optional.
        """

        config = PostgresInteractionConfiguration(
            dsn="postgresql://fathom:secret@db.local:5432/fathom?sslmode=require",
        )

        self.assertEqual(
            "postgresql://fathom:secret@db.local:5432/fathom?sslmode=require",
            config.dsn,
        )
        self.assertIsNone(config.host)
        self.assertIsNone(config.user)
        self.assertIsNone(config.password)

    def test_dsn_supersedes_discrete_fields_when_both_supplied(self) -> None:
        """
        Both modes may be provided; the validator does not reject the combo
        and pool creation uses the DSN. The discrete fields stay queryable on the model for diagnostics.
        """

        config = PostgresInteractionConfiguration(
            user="discrete",
            host="discrete.local",
            password="discrete-secret",
            dsn="postgresql://override:override@override.local:6543/override",
        )

        self.assertEqual("discrete.local", config.host)
        self.assertEqual(
            "postgresql://override:override@override.local:6543/override",
            config.dsn,
        )

    def test_neither_dsn_nor_discrete_credentials_is_rejected(self) -> None:
        """
        Constructing with no DSN and no host/user/password must fail loudly.
        """

        with self.assertRaises(ValidationError):
            PostgresInteractionConfiguration()

    def test_partial_discrete_credentials_without_dsn_is_rejected(self) -> None:
        """
        Discrete mode requires host AND user AND password; any one missing
        without a DSN fails the validator.
        """

        with self.assertRaises(ValidationError):
            PostgresInteractionConfiguration(host="db.local", user="fathom")

    def test_empty_dsn_is_rejected(self) -> None:
        """
        DSN is min_length=1 so an explicit empty string fails the field
        validator before the connection-mode validator runs.
        """

        with self.assertRaises(ValidationError):
            PostgresInteractionConfiguration(dsn="")
