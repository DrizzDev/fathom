from __future__ import annotations

from enum import StrEnum


class InferenceTier(StrEnum):
    """
    Provider-neutral inference service tier requested for a single LLM call.
    """

    STANDARD = "standard"
    PRIORITY = "priority"


class InferencePriorityMode(StrEnum):
    """
    Provider-neutral policy mode for selecting elevated inference capacity.
    """

    ALWAYS = "always"
    ADAPTIVE = "adaptive"


class InferencePriorityTransitionReason(StrEnum):
    """
    Provider-neutral reasons for adaptive priority tier changes.
    """

    SLOW = "slow_responses"
    TRANSIENT = "transient_failures"
    RECOVERY = "healthy_recovery"


DEFAULT_PRIORITY_WINDOW = 10
DEFAULT_PRIORITY_SLOW_THRESHOLD = 3
DEFAULT_PRIORITY_FAILURE_THRESHOLD = 2
DEFAULT_PRIORITY_RECOVERY_SUCCESSES = 5
DEFAULT_PRIORITY_LATENCY_THRESHOLD = 15.0


class StructuredOutputMediaType(StrEnum):
    """
    Output media types adapters bind for constrained-decoding LLM calls.
    """

    JSON = "application/json"


# Gemini provider HTTP-style status codes
GEMINI_CANCELLED_STATUS_CODE = 499


class ReasoningEffort(StrEnum):
    """
    How much internal reasoning a model is asked to spend before answering.

    Provider-neutral by design. Every frontier provider now exposes some version of
    this dial under its own name, and the alternative to naming it once here is a
    per-model conditional in each adapter - which is how a harness ends up encoding a
    vendor's release schedule in its control flow.

    ``NONE`` is a real level, not an absence. Some models accept function tools only
    when reasoning is explicitly switched off, so "off" has to be expressible; a run
    that wants the provider's own default omits the setting entirely instead.
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


GEMINI_RATE_LIMIT_STATUS_CODE = 429
GEMINI_STALE_CACHE_STATUS_CODE = 400
GEMINI_GATEWAY_TIMEOUT_STATUS_CODE = 504
GEMINI_PROVIDER_OVERLOADED_STATUS_CODE = 529

GEMINI_PRIORITY_SERVICE_TIER_VALUE = "priority"

GEMINI_VERTEX_REQUEST_TYPE_HEADER = "X-Vertex-AI-LLM-Request-Type"
GEMINI_VERTEX_PRIORITY_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"

GEMINI_VERTEX_SHARED_REQUEST_TYPE = "shared"
GEMINI_VERTEX_PRIORITY_REQUEST_TYPE = "priority"
GEMINI_PRIORITY_TRAFFIC_TYPE = "ON_DEMAND_PRIORITY"

GEMINI_TRANSIENT_STATUS_CODES = (
    GEMINI_RATE_LIMIT_STATUS_CODE,
    GEMINI_GATEWAY_TIMEOUT_STATUS_CODE,
    GEMINI_PROVIDER_OVERLOADED_STATUS_CODE,
)

# Gemini transient retry policy
GEMINI_RETRY_AFTER_JITTER_SECONDS = 0.5
GEMINI_GENERIC_RETRY_JITTER_SECONDS = 0.5
GEMINI_TRANSIENT_RETRY_JITTER_SECONDS = 1.0
GEMINI_MAX_TRANSIENT_RETRY_DELAY_SECONDS = 60.0

# Gemini error markers
GEMINI_RATE_LIMIT_ERROR_MARKERS = ("resource_exhausted",)
GEMINI_PROVIDER_OVERLOAD_ERROR_MARKERS = (
    "overloaded",
    "temporarily unavailable",
    "upstream service unavailable",
)
GEMINI_CANCELLED_ERROR_MARKERS = ("cancelled",)
GEMINI_STALE_CACHE_NAME_MARKERS = (
    "cachedcontent",
    "cache content",
    "cached content",
    "cached_content",
)
GEMINI_STALE_CACHE_STATE_MARKERS = (
    "invalid",
    "not found",
    "resource state",
    "invalid argument",
    "invalid_argument",
    "invalid resource state",
)


class ModelProvider(StrEnum):
    """
    Provider identity an LLM adapter declares for itself, never inferred from its class name.

    Values mirror the ``LLMConfiguration.provider`` vocabulary and are persisted on recorded
    runs, so they keep that literal protocol form rather than the internal upper-case style.
    """

    GEMINI = "gemini"
    VERTEX_AI = "vertex_ai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    UNKNOWN = "unknown"

    @classmethod
    def declared(cls, *, value: str) -> "ModelProvider":
        """
        Map a configuration-declared provider value onto its typed identity.
        """

        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN
