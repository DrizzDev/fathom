from __future__ import annotations

from enum import StrEnum

# Visual hashing
VISUAL_HASH_LENGTH = 16

# Launcher packages - actions on these should never persist/export during execution
LAUNCHER_PACKAGES: frozenset[str] = frozenset(
    {
        # Android
        "com.google.android.apps.nexuslauncher",  # Google Pixel default
        "com.android.launcher",
        "com.android.launcher3",
        "com.sec.android.app.launchers",  # Samsung default
        "com.miui.home",  # MIUI default
        "com.oppo.launcher",  # OPPO default
        # iOS
        "com.apple.springboard",  # iOS home screen
    }
)

# Swipe and scroll distances (pixels)
DEFAULT_SWIPE_DISTANCE = 300
DEFAULT_SCROLL_DISTANCE = 300
BOUNDS_SWIPE_DISTANCE = 100

# Timing (milliseconds)
DEFAULT_SWIPE_DURATION = 350  # Swipe gesture duration; 350ms reliable on modern devices
DEFAULT_STABILITY_WAIT = 500  # Wait after action for screen to stabilize
CAPTURE_OVERHEAD_MS = 150.0  # Estimated screenshot I/O time subtracted from stability wait
# Hard upper-bound for "screen stability" waits.
# Stored as milliseconds to keep all timing constants consistent.
MAX_STABILITY_WAIT_MS = 1500
# Hard upper-bound for LLM-requested action waits.
# Stored as milliseconds to keep all timing constants consistent.
MAX_ACTION_WAIT_MS = 10_000

# Retry configuration
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 500  # Base delay for exponential backoff (milliseconds)

# Number of verification rejections on the SAME sub-goal after which
# the agent calls the decomposer LLM to rebuild the remaining plan.
# Set high enough that transient verifier noise doesn't force an
# expensive redecomposition, but low enough that genuinely-wrong plans
# are replaced before the action budget runs out.
#
# Cheap "action-loop" recovery is handled separately: the planner fires
# a rejection-history retry when `is_action_repeating_on_screen` detects
# the same tap/type emitted 3+ times on the same screen
# (see fathom.core.agent.planner.Planner.plan).
REDECOMPOSE_VERIFY_FAILURE_THRESHOLD = 5

# Number of cheap planner rejection-history retries on the SAME
# sub-goal after which the agent escalates to the expensive
# decomposer replan. The cheap retry fires on every detection of
# is_action_repeating_on_screen; this threshold bounds how many
# fresh LLM-picked actions the planner gets to try before we give
# up on local recovery and ask the decomposer for a new plan shape.
#
# Threshold 3 means: retry fires twice (counter 1, 2) and on the
# third detection (counter=3) the ANALYZE node escalates instead
# of injecting guidance again. At 3 the vision LLM has already
# seen two previous rejected actions in its rejection history, so
# if a third distinct action ALSO loops, the plan shape is likely
# wrong.
#
# This path is independent of REDECOMPOSE_VERIFY_FAILURE_THRESHOLD
# above: the two counters fire on different signals
# (is_action_repeating_on_screen vs. record_verify_failure) and
# track mutually-exclusive failure modes.
PLANNER_RETRY_ESCALATION_THRESHOLD = 3

# Active-thread count threshold above which the GCC context manager
# branches a new conversation. Chosen to keep prompt context bounded.
GCC_BRANCHING_THRESHOLD = 15

# Decomposition results below this confidence are discarded in favor
# of the heuristic fallback.
MINIMUM_DECOMPOSITION_CONFIDENCE = 0.6

# User-facing failure messages for intent graph nodes
GROUNDING_FAILURE_MESSAGE = "Failed to capture the current app screen. Please retry."
RECORDING_FAILURE_MESSAGE = "Failed to save execution details for the current step."

# Signal adapter heartbeat interval (seconds)
SIGNAL_HEARTBEAT_INTERVAL = 5.0
RECOMMENDED_WORKFLOW_TASK_TIMEOUT_SECONDS = 60

# Remote device request timeout (seconds)
REMOTE_DEVICE_REQUEST_TIMEOUT_SECONDS = 60.0

# Maximum time (seconds) to wait for background tasks during shutdown
DRAIN_TIMEOUT = 30.0


class SignalType(StrEnum):
    """
    HITL control signals.
    """

    ASK = "ASK"
    STOP = "STOP"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    INJECT = "INJECT"
    CONTINUE = "CONTINUE"
    CANCELLED = "CANCELLED"


class ExecutionPhase(StrEnum):
    """
    Execution DAG phases.
    """

    ACT = "ACT"
    LEARN = "LEARN"
    REASON = "REASON"
    PERCEIVE = "PERCEIVE"
    EVALUATE = "EVALUATE"
    CHECKPOINT = "CHECKPOINT"
    SIGNAL_CHECK = "SIGNAL_CHECK"
