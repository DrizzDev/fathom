"""Verify model-role manifest construction and serialization."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, Optional, Sequence
from unittest.mock import MagicMock

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.adapters.summarization.llm import LLMSummarizer
from fathom.constants.llm import ModelProvider
from fathom.constants.manifest import (
    MANIFEST_VERSION,
    SHARED_ARCHITECT_NOTE,
    UNDECLARED_BINDING_NOTE,
    UNREAD_CONFIGURATION_NOTE,
    ModelRole,
    RoleBinding,
)
from fathom.core.services.qualifier import LLMIntentQualifier, PermissiveIntentQualifier
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.interfaces.summarization import SummarizationPort
from fathom.runtime.manifest import ManifestResolver
from fathom.schemas.abort import AbortDetectorConfiguration
from fathom.schemas.configuration import (
    FathomConfiguration,
    LLMConfiguration,
    OracleConfiguration,
    QualifierConfiguration,
)
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import InferenceSettings, ModelIdentity, StructuredOutput
from fathom.schemas.localization import EnsembleMemberName
from fathom.schemas.manifest import RuntimeManifest
from fathom.schemas.perception import LocalizationEnsembleConfiguration
from fathom.schemas.results import GenerateResult


class _StubLLM(LLMPort):
    """
    Provider-declaring LLM double whose derive() returns a genuinely distinct sibling.
    """

    def __init__(
        self,
        *,
        model: str = "stub-1",
        temperature: float = 1.0,
        cache: bool = True,
        provider: ModelProvider = ModelProvider.OPENAI,
        client: Optional[object] = None,
    ) -> None:
        """
        Record the identity this double declares and the client it exposes, if any.
        """

        self.__model = model
        self.__cache = cache
        self.__client = client
        self.__provider = provider
        self.__temperature = temperature

    @property
    def model_name(self) -> str:
        """
        Model identifier this double reports.
        """

        return self.__model

    @property
    def client(self) -> Optional[object]:
        """
        Embedding-capable client this double exposes.
        """

        return self.__client

    @property
    def identity(self) -> ModelIdentity:
        """
        Declared provider, model, and settings.
        """

        return ModelIdentity(
            provider=self.__provider,
            model=self.__model,
            settings=InferenceSettings(cache=self.__cache, temperature=self.__temperature),
        )

    def derive(self, *, overrides: LLMConfiguration) -> LLMPort:
        """
        Return a distinct sibling carrying the overridden temperature and cache setting.
        """

        return _StubLLM(
            model=self.__model,
            provider=self.__provider,
            cache=overrides.use_cache,
            temperature=overrides.temperature,
        )

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[PromptPart],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
        structured_output: Optional[StructuredOutput] = None,
    ) -> GenerateResult:
        """
        Unused by manifest resolution.
        """

        raise NotImplementedError

    async def cleanup(self) -> None:
        """
        Unused by manifest resolution.
        """


class _SelfDerivingLLM(_StubLLM):
    """
    LLM double that keeps the port default of returning itself from derive().
    """

    def derive(self, *, overrides: LLMConfiguration) -> LLMPort:
        """
        Return the same instance, as an adapter that has not implemented derive() does.
        """

        return self


class _UndeclaredSummarizer(SummarizationPort):
    """
    Summarizer double that does not expose the adapter it was composed with.
    """

    async def summarize_trace(self, trace: list) -> str:
        """
        Unused by manifest resolution.
        """

        return ""


class _ResolverFixture:
    """
    Builds a ManifestResolver over chosen ports without standing up a full runner.
    """

    @staticmethod
    def build(
        *,
        planner: LLMPort,
        architect: Optional[LLMPort] = None,
        qualifier: Optional[object] = None,
        summarizer: Optional[object] = None,
        configuration: Optional[FathomConfiguration] = None,
    ) -> ManifestResolver:
        """
        Compose a resolver, defaulting each role to the wiring FathomBuilder produces.
        """

        settings = configuration or FathomConfiguration()

        return ManifestResolver(
            planner=planner,
            architect=architect
            or planner.derive(overrides=LLMConfiguration(temperature=0.0, use_cache=False)),
            qualifier=qualifier
            or LLMIntentQualifier(llm=planner, configuration=settings.qualifier),
            summarizer=summarizer or LLMSummarizer(llm=planner),
            configuration=settings,
        )

    @staticmethod
    def resolve(
        *,
        resolver: ManifestResolver,
        steps: int = 50,
        localization: Optional[LocalizationEnsembleConfiguration] = None,
    ) -> RuntimeManifest:
        """
        Resolve a manifest, defaulting the ensemble to disabled.
        """

        return resolver.resolve(
            steps=steps,
            localization=localization
            or LocalizationEnsembleConfiguration(enabled=False, members=()),
        )


class ManifestPlannerRoleTest(unittest.TestCase):
    """
    The planner owns its adapter and its declared identity reaches the manifest verbatim.
    """

    def test_planner_records_declared_provider_model_and_settings(self) -> None:
        """
        The planner entry carries the provider, model, and settings the adapter declares.
        """

        planner = _StubLLM(model="gpt-x", provider=ModelProvider.OPENAI, temperature=0.7)
        manifest = _ResolverFixture.resolve(resolver=_ResolverFixture.build(planner=planner))

        entry = manifest.entry(role=ModelRole.PLANNER)

        assert entry is not None and entry.resolved is not None
        self.assertEqual(entry.binding, RoleBinding.INDEPENDENT)
        self.assertEqual(entry.resolved.provider, ModelProvider.OPENAI)
        self.assertEqual(entry.resolved.model, "gpt-x")
        self.assertEqual(entry.resolved.settings.temperature, 0.7)

    def test_undeclared_adapter_reports_unknown_provider_not_a_class_name_guess(self) -> None:
        """
        An adapter that declares nothing yields UNKNOWN, never a name derived from its class.
        """

        planner = MagicMock(spec=LLMPort)
        planner.model_name = "whatever"
        manifest = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=planner, architect=planner)
        )

        entry = manifest.entry(role=ModelRole.PLANNER)

        assert entry is not None
        self.assertIsNone(entry.resolved)


class ManifestArchitectDerivationTest(unittest.TestCase):
    """
    Regression: the manifest must separate a real deterministic sibling from a self-returning
    derive(), because the second silently runs decomposition at the planner's temperature.
    """

    def test_distinct_sibling_is_derived_and_carries_the_deterministic_settings(self) -> None:
        """
        A derive() that returns a new adapter yields DERIVED with temperature 0.0 and no cache.
        """

        planner = _StubLLM(temperature=1.0, cache=True)
        manifest = _ResolverFixture.resolve(resolver=_ResolverFixture.build(planner=planner))

        entry = manifest.entry(role=ModelRole.ARCHITECT)

        assert entry is not None and entry.resolved is not None
        self.assertEqual(entry.binding, RoleBinding.DERIVED)
        self.assertEqual(entry.resolved.settings.temperature, 0.0)
        self.assertFalse(entry.resolved.settings.cache)
        self.assertIsNone(entry.note)

    def test_self_returning_derive_is_shared_and_keeps_the_planner_settings(self) -> None:
        """
        A derive() that returns the planner yields SHARED, temperature 1.0, and an explicit note.
        """

        planner = _SelfDerivingLLM(temperature=1.0, cache=True)
        manifest = _ResolverFixture.resolve(resolver=_ResolverFixture.build(planner=planner))

        entry = manifest.entry(role=ModelRole.ARCHITECT)

        assert entry is not None and entry.resolved is not None
        self.assertEqual(entry.binding, RoleBinding.SHARED)
        self.assertEqual(entry.source, ModelRole.PLANNER)
        self.assertEqual(entry.resolved.settings.temperature, 1.0)
        self.assertTrue(entry.resolved.settings.cache)
        self.assertEqual(entry.note, SHARED_ARCHITECT_NOTE)

    def test_the_two_derive_behaviours_produce_different_manifests(self) -> None:
        """
        The serialized architect entries differ, so the difference cannot be read past.
        """

        derived = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=_StubLLM())
        ).entry(role=ModelRole.ARCHITECT)
        shared = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=_SelfDerivingLLM())
        ).entry(role=ModelRole.ARCHITECT)

        assert derived is not None and shared is not None
        self.assertNotEqual(derived.model_dump(mode="json"), shared.model_dump(mode="json"))


class ManifestQualifierRoleTest(unittest.TestCase):
    """
    The qualifier is only independent when composition gave it a dedicated adapter.
    """

    def test_qualifier_reusing_the_planner_is_shared_and_says_so(self) -> None:
        """
        A qualifier composed on the planner adapter reports SHARED sourced from PLANNER.
        """

        planner = _StubLLM()
        manifest = _ResolverFixture.resolve(resolver=_ResolverFixture.build(planner=planner))

        entry = manifest.entry(role=ModelRole.QUALIFIER)

        assert entry is not None and entry.requested is not None
        self.assertEqual(entry.binding, RoleBinding.SHARED)
        self.assertEqual(entry.source, ModelRole.PLANNER)
        self.assertEqual(entry.requested.model, QualifierConfiguration().inference.model)
        self.assertIsNotNone(entry.note)

    def test_dedicated_qualifier_adapter_is_independent(self) -> None:
        """
        A qualifier composed on its own adapter reports INDEPENDENT with that adapter's model.
        """

        planner = _StubLLM(model="planner-1")
        dedicated = _StubLLM(model="qualifier-lite", temperature=0.0)
        manifest = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(
                planner=planner,
                qualifier=LLMIntentQualifier(llm=dedicated),
            )
        )

        entry = manifest.entry(role=ModelRole.QUALIFIER)

        assert entry is not None and entry.resolved is not None
        self.assertEqual(entry.binding, RoleBinding.INDEPENDENT)
        self.assertIsNone(entry.source)
        self.assertEqual(entry.resolved.model, "qualifier-lite")
        self.assertEqual(entry.resolved.settings.temperature, 0.0)

    def test_disabled_qualifier_is_reported_disabled_with_no_adapter(self) -> None:
        """
        With qualifier.enabled=False the role is NONE, has no resolved adapter, and keeps the request.
        """

        configuration = FathomConfiguration(qualifier=QualifierConfiguration(enabled=False))
        manifest = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(
                planner=_StubLLM(),
                qualifier=PermissiveIntentQualifier(),
                configuration=configuration,
            )
        )

        entry = manifest.entry(role=ModelRole.QUALIFIER)

        assert entry is not None and entry.requested is not None
        self.assertFalse(entry.enabled)
        self.assertEqual(entry.binding, RoleBinding.NONE)
        self.assertIsNone(entry.resolved)
        self.assertEqual(entry.requested.model, QualifierConfiguration().inference.model)


class ManifestUndeclaredBindingTest(unittest.TestCase):
    """
    A port that hides its adapter is reported as undeclared, never assumed independent.
    """

    def test_summarizer_without_the_capability_reports_unknown_binding(self) -> None:
        """
        A third-party summarizer yields UNKNOWN rather than a false claim of independence.
        """

        manifest = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(
                planner=_StubLLM(),
                summarizer=_UndeclaredSummarizer(),
            )
        )

        entry = manifest.entry(role=ModelRole.SUMMARIZER)

        assert entry is not None
        self.assertEqual(entry.binding, RoleBinding.UNKNOWN)
        self.assertEqual(entry.note, UNDECLARED_BINDING_NOTE)
        self.assertIsNone(entry.resolved)

    def test_default_summarizer_shares_the_planner_adapter(self) -> None:
        """
        The summarizer FathomBuilder installs runs on the planner adapter.
        """

        planner = _StubLLM(model="planner-1")
        manifest = _ResolverFixture.resolve(resolver=_ResolverFixture.build(planner=planner))

        entry = manifest.entry(role=ModelRole.SUMMARIZER)

        assert entry is not None and entry.resolved is not None
        self.assertEqual(entry.binding, RoleBinding.SHARED)
        self.assertEqual(entry.source, ModelRole.PLANNER)
        self.assertEqual(entry.resolved.model, "planner-1")


class ManifestOptionalRoleTest(unittest.TestCase):
    """
    Abort, oracle, localizer, and embedder each report the enablement composition gives them.
    """

    def test_abort_detector_records_the_configured_model_it_never_uses(self) -> None:
        """
        The abort role resolves to the planner adapter while requesting a different model.
        """

        planner = _StubLLM(model="planner-1")
        manifest = _ResolverFixture.resolve(resolver=_ResolverFixture.build(planner=planner))

        entry = manifest.entry(role=ModelRole.ABORT)

        assert entry is not None and entry.resolved is not None and entry.requested is not None
        self.assertTrue(entry.enabled)
        self.assertEqual(entry.binding, RoleBinding.SHARED)
        self.assertEqual(entry.resolved.model, "planner-1")
        self.assertEqual(entry.requested.model, AbortDetectorConfiguration().inference.model)
        self.assertNotEqual(entry.requested.model, entry.resolved.model)
        self.assertEqual(entry.note, UNREAD_CONFIGURATION_NOTE)

    def test_oracle_follows_its_configuration_flag(self) -> None:
        """
        The oracle is SHARED when configuration enables it and NONE when it does not.
        """

        planner = _StubLLM()
        enabled = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(
                planner=planner,
                configuration=FathomConfiguration(oracle=OracleConfiguration(enabled=True)),
            )
        ).entry(role=ModelRole.ORACLE)
        disabled = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(
                planner=planner,
                configuration=FathomConfiguration(oracle=OracleConfiguration(enabled=False)),
            )
        ).entry(role=ModelRole.ORACLE)

        assert enabled is not None and disabled is not None
        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.binding, RoleBinding.SHARED)
        self.assertFalse(disabled.enabled)
        self.assertEqual(disabled.binding, RoleBinding.NONE)
        self.assertIsNone(disabled.resolved)

    def test_localizer_enabled_only_when_its_ensemble_member_is_configured(self) -> None:
        """
        The vision localizer needs both the ensemble enabled and its member present.
        """

        resolver = _ResolverFixture.build(planner=_StubLLM())

        active = _ResolverFixture.resolve(
            resolver=resolver,
            localization=LocalizationEnsembleConfiguration(
                enabled=True,
                members=(EnsembleMemberName.GEMINI_VISION,),
            ),
        ).entry(role=ModelRole.LOCALIZER)
        without_member = _ResolverFixture.resolve(
            resolver=resolver,
            localization=LocalizationEnsembleConfiguration(
                enabled=True,
                members=(EnsembleMemberName.DOCUMENT_AI_LAYOUT,),
            ),
        ).entry(role=ModelRole.LOCALIZER)

        assert active is not None and without_member is not None
        self.assertTrue(active.enabled)
        self.assertEqual(active.binding, RoleBinding.SHARED)
        self.assertFalse(without_member.enabled)
        self.assertEqual(without_member.binding, RoleBinding.NONE)

    def test_provider_neutral_localizer_member_is_supported(self) -> None:
        """The provider-neutral member name enables the bound planner localizer."""

        entry = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=_StubLLM()),
            localization=LocalizationEnsembleConfiguration(
                enabled=True,
                members=(EnsembleMemberName.VISION_LLM,),
            ),
        ).entry(role=ModelRole.LOCALIZER)

        assert entry is not None
        self.assertTrue(entry.enabled)
        self.assertEqual(entry.binding, RoleBinding.SHARED)

    def test_embedder_enabled_only_when_the_planner_exposes_a_client(self) -> None:
        """
        The embedder role tracks whether the planner adapter can back a sibling embedding adapter.
        """

        with_client = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=_StubLLM(client=object()))
        ).entry(role=ModelRole.EMBEDDER)
        without_client = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=_StubLLM(client=None))
        ).entry(role=ModelRole.EMBEDDER)

        assert with_client is not None and without_client is not None
        self.assertTrue(with_client.enabled)
        self.assertEqual(with_client.binding, RoleBinding.SHARED)
        self.assertFalse(without_client.enabled)
        self.assertEqual(without_client.binding, RoleBinding.NONE)


class ManifestSerializationTest(unittest.TestCase):
    """
    The manifest is JSON-serializable, stable across resolutions, and carries no secrets.
    """

    @staticmethod
    def __credentialed_planner() -> _StubLLM:
        """
        Planner whose model string embeds values a leak would surface.
        """

        return _StubLLM(model="planner-1", provider=ModelProvider.GEMINI)

    def test_manifest_serializes_to_json_with_every_role_and_the_step_budget(self) -> None:
        """
        Serialization yields the version, the effective step budget, and all eight roles.
        """

        manifest = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=self.__credentialed_planner()),
            steps=37,
        )
        payload = json.loads(json.dumps(manifest.model_dump(mode="json")))

        self.assertEqual(payload["version"], MANIFEST_VERSION)
        self.assertEqual(payload["steps"], 37)
        self.assertEqual(
            [entry["role"] for entry in payload["roles"]],
            [role.value for role in ModelRole],
        )

    def test_repeated_resolution_produces_an_identical_payload(self) -> None:
        """
        Two resolutions of the same composition serialize identically.
        """

        resolver = _ResolverFixture.build(planner=self.__credentialed_planner())

        first = _ResolverFixture.resolve(resolver=resolver, steps=12)
        second = _ResolverFixture.resolve(resolver=resolver, steps=12)

        self.assertEqual(
            json.dumps(first.model_dump(mode="json"), sort_keys=True),
            json.dumps(second.model_dump(mode="json"), sort_keys=True),
        )

    @staticmethod
    def __credentialed_gemini() -> GeminiLLM:
        """
        Bind a credentialed configuration to a Gemini adapter without building an SDK client.

        Client construction validates service-account material against the Google SDK and is
        irrelevant to what ``identity`` exposes, which is the surface under test here.
        """

        gemini = object.__new__(GeminiLLM)
        gemini._GeminiLLM__client = None  # type: ignore[attr-defined]
        gemini._GeminiLLM__configuration = LLMConfiguration(  # type: ignore[attr-defined]
            api_key="super-secret-key",
            model="gemini-3.5-flash",
            project_id="secret-project",
            location="secret-location",
            credentials={"type": "service_account", "private_key": "-----BEGIN PRIVATE KEY"},
        )
        return gemini

    def test_serialized_manifest_carries_no_credentials(self) -> None:
        """
        A planner bound to a credentialed configuration leaks no key, project, or credential.
        """

        planner = self.__credentialed_gemini()
        manifest = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=planner, architect=planner)
        )
        payload = json.dumps(manifest.model_dump(mode="json"))

        self.assertNotIn("super-secret-key", payload)
        self.assertNotIn("secret-project", payload)
        self.assertNotIn("secret-location", payload)
        self.assertNotIn("BEGIN PRIVATE KEY", payload)
        self.assertNotIn("service_account", payload)
        self.assertIn("gemini-3.5-flash", payload)

    def test_manifest_exposes_only_its_declared_fields(self) -> None:
        """
        No prompt, screenshot, or user-entered value can ride along: the shape is closed.

        Structural guard — a future field carrying run content fails here before it ships.
        """

        manifest = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(planner=self.__credentialed_planner())
        )
        payload = manifest.model_dump(mode="json")

        self.assertEqual(set(payload), {"version", "steps", "roles"})
        self.assertEqual(
            set(payload["roles"][0]),
            {"role", "enabled", "binding", "source", "requested", "resolved", "note"},
        )
        self.assertEqual(
            set(payload["roles"][0]["resolved"]),
            {"provider", "model", "settings"},
        )
        self.assertEqual(
            set(payload["roles"][0]["resolved"]["settings"]),
            {"cache", "thinking", "temperature"},
        )

    def test_providers_reports_each_distinct_resolved_provider(self) -> None:
        """
        A composition mixing two providers reports both, once each.
        """

        planner = _StubLLM(model="planner-1", provider=ModelProvider.OPENAI)
        dedicated = _StubLLM(model="qualifier-1", provider=ModelProvider.ANTHROPIC)
        manifest = _ResolverFixture.resolve(
            resolver=_ResolverFixture.build(
                planner=planner,
                qualifier=LLMIntentQualifier(llm=dedicated),
            )
        )

        self.assertEqual(
            manifest.providers(),
            (ModelProvider.OPENAI, ModelProvider.ANTHROPIC),
        )
