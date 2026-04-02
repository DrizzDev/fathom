from __future__ import annotations

# Gemini provider HTTP-style status codes
GEMINI_CANCELLED_STATUS_CODE = 499
GEMINI_RATE_LIMIT_STATUS_CODE = 429
GEMINI_STALE_CACHE_STATUS_CODE = 400
GEMINI_PROVIDER_OVERLOADED_STATUS_CODE = 529

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
