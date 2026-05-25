"""
Constants for Temporal workflow sandbox.
"""

# External libraries that should be passed through the Temporal sandbox
# to avoid warnings and errors during workflow execution.
__EXTERNAL_LIBRARIES = {
    "sentry_sdk",
    "gevent",
    "greenlet",
    "eventlet",
    "threading",
}

# Fathom internal modules that should be passed through
__FATHOM_MODULES = {
    "fathom",
    "fathom.state.enums",
    "fathom.schemas.steps",
    "fathom.domain.state",
    "fathom.domain.enums",
    "fathom.state.context",
    "fathom.domain.request",
    "fathom.schemas.results",
    "fathom.domain.configuration",
}

WORKFLOW_PASSTHROUGH_MODULES = {
    *__EXTERNAL_LIBRARIES,
    *__FATHOM_MODULES,
}
