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
DEFAULT_RETRY_DELAY = 500  # Base delay for exponential backoff

# Sub-goal action budget
# Maximum number of actions the agent may emit against a single sub-goal
# before the replanning path is triggered. Chosen empirically to catch
# loops without cutting short legitimate multi-step sub-goals.
MAX_ACTIONS_PER_SUBGOAL = 15

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
