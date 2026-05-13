from __future__ import annotations

from fathom.constants.execution import VISUAL_HASH_LENGTH

ZERO_HASH: str = "0" * VISUAL_HASH_LENGTH

MAX_VISUAL_HASH_DISTANCE: int = 64
DEFAULT_SAME_SCREEN_THRESHOLD: int = 8

SIGNATURE_TEXT_PREVIEW_LENGTH: int = 32
SIGNATURE_VALUE_PREVIEW_LENGTH: int = 16

BOUNDS_DIGEST_LENGTH: int = 6
INTERACTION_TEXT_PREVIEW_LENGTH: int = 30

ACTION_EFFECT_SSIM_THRESHOLD: float = 0.98
ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD: int = 4
ACTION_EFFECT_SCROLL_DISTANCE_THRESHOLD_PX: float = 5.0
ACTION_EFFECT_CONTENT_DIFF_RATIO_THRESHOLD: float = 0.005

ACTIVITY_CHANGED_SIGNAL_WEIGHT: int = 2
XML_HASH_CHANGED_SIGNAL_WEIGHT: int = 2
INTERACTION_HASH_CHANGED_SIGNAL_WEIGHT: int = 1

SSIM_CHANGED_SIGNAL_WEIGHT: int = 1
PHASH_CHANGED_SIGNAL_WEIGHT: int = 1

CONTENT_DIFF_SIGNAL_WEIGHT: int = 1
SCROLL_CHANGED_SIGNAL_WEIGHT: int = 1

MEANINGFUL_STATE_SSIM_THRESHOLD: float = 0.95
MEANINGFUL_STATE_SIGNAL_WEIGHT_THRESHOLD: int = 2
MEANINGFUL_STATE_PHASH_DISTANCE_THRESHOLD: int = 8
MEANINGFUL_STATE_CONTENT_DIFF_RATIO_THRESHOLD: float = 0.05
MEANINGFUL_STATE_SCROLL_DISTANCE_THRESHOLD_PX: float = 30.0

LOOP_OSCILLATION_AB_WINDOW: int = 4
LOOP_OSCILLATION_ABC_WINDOW: int = 6
LOOP_SCROLL_STALL_MIN_STREAK: int = 7
LOOP_SCROLL_STALL_DISTANCE_THRESHOLD: int = 4
LOOP_ACTION_VELOCITY_INTERVAL_THRESHOLD_SECONDS: float = 1.5

# Tight pHash hamming threshold for the visual-only near-duplicate loop detector.
# Catches cases where DOM micro-changes (animation frames, transient overlays,
# map redraws) flip xml/interaction hashes but the screen is visually identical.
# Independent from DEFAULT_SAME_SCREEN_THRESHOLD which is the broader equality bar.
LOOP_NEAR_DUPLICATE_HAMMING_THRESHOLD: int = 4

# Hamming threshold the loop detector uses to decide that the screen
# genuinely advanced versus only changed cosmetically. Deliberately much
# higher than ``LOOP_NEAR_DUPLICATE_HAMMING_THRESHOLD`` so micro-changes
# (status bar tick, suggestion-count increment, anti-aliasing noise) do
# not trip the "advance the buffer" path that clears accumulating
# evidence. This is the threshold consulted by
# :meth:`LoopDetector.observe_screen` — distinct from the near-duplicate
# threshold used by the visual-only stuck detector.
#
# Placeholder: pin against 3.txt scroll-loop pHashes + yVKnb coachmark
# pHashes (see ``tests/unit/schemas/test_loop_detector_fixtures.py``).
SCREEN_PROGRESS_HAMMING_THRESHOLD: int = 16

# Maximum hamming distance for two hashes to be considered members of
# the same "near-duplicate cluster" when evaluating whether a sequence
# of scroll-like actions is producing real progress. Looser than
# ``LOOP_NEAR_DUPLICATE_HAMMING_THRESHOLD`` so detector handles the pHash
# jitter natural to long lists.
LOOP_HASH_CLUSTER_HAMMING_THRESHOLD: int = 6

# Minimum window occurrences required before the loop detector classifies a
# pattern as stuck. Applied to all five detectors (repetition, near-duplicate,
# oscillation, scroll-stall, action-velocity).
LOOP_REPETITION_THRESHOLD: int = 3

# Size of the sliding window the loop detector inspects for pattern analysis.
LOOP_DETECTOR_WINDOW_SIZE: int = 15

# Maximum autonomous recovery attempts (BACK / SCROLL / HOME) before the loop
# detector declines further recovery and the agent terminates as STUCK.
LOOP_MAX_AUTONOMOUS_RECOVERIES: int = 3

# Action-effect classifier thresholds. Map raw ScreenDiff metrics onto a
# coarse ``ActionEffectStatus`` (progress | no_progress | uncertain).
#
# IMPORTANT: these starting values are placeholders. Phase 1B fixture
# tests (see ``tests/unit/schemas/test_action_effect.py``) replay pHash /
# SSIM sequences from the 3.txt scroll loop and the yVKnb 41-step
# coachmark loop and pin the thresholds against those traces. Do not
# treat these defaults as load-bearing without verifying the fixtures.
#
# ``progress`` requires either a noticeable pHash jump, low SSIM, large
# content_diff, OR a meaningful scroll translation.
ACTION_EFFECT_PROGRESS_SSIM_BELOW: float = 0.90
ACTION_EFFECT_PROGRESS_PHASH_ABOVE: int = 8
ACTION_EFFECT_PROGRESS_CONTENT_DIFF_ABOVE: float = 0.05
ACTION_EFFECT_PROGRESS_SCROLL_DISTANCE_PX_ABOVE: float = 30.0

# ``no_progress`` requires ALL signals to be below their respective floors
# (no significant pHash, ssim very close to 1, content_diff near zero,
# scroll translation negligible). Anything in between is ``uncertain``.
ACTION_EFFECT_NO_PROGRESS_SSIM_ABOVE: float = 0.99
ACTION_EFFECT_NO_PROGRESS_PHASH_BELOW_OR_EQ: int = 2
ACTION_EFFECT_NO_PROGRESS_CONTENT_DIFF_BELOW_OR_EQ: float = 0.01
ACTION_EFFECT_NO_PROGRESS_SCROLL_DISTANCE_PX_BELOW_OR_EQ: float = 5.0

# Window of recent action effects rendered into the ANALYZE prompt for the
# agent's own trajectory observation.
ACTION_EFFECT_TRAJECTORY_WINDOW: int = 5

# Number of trailing NO_PROGRESS classifications that constitutes a
# "no progress" stuck signal worth escalating via the recovery
# coordinator. Kept conservative so the agent gets one or two free
# tries on a tough screen before recovery dispatches.
NO_PROGRESS_RECOVERY_THRESHOLD: int = 3

STATUS_BAR_HEIGHT_PX: int = 80
NAVIGATION_BAR_HEIGHT_PX: int = 60

PIXEL_CHANGE_THRESHOLD: int = 15
MIN_CHANGED_REGION_AREA_PX: int = 100

DILATION_ITERATIONS: int = 2
DILATION_KERNEL_SIZE: int = 10

SSIM_K1: float = 0.01
SSIM_K2: float = 0.03
SSIM_GAUSSIAN_SIGMA: float = 1.5
SSIM_GAUSSIAN_KERNEL_SIZE: int = 11
