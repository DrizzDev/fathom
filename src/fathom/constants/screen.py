from __future__ import annotations

from fathom.constants.execution import VISUAL_HASH_LENGTH

ZERO_HASH: str = "0" * VISUAL_HASH_LENGTH

MAX_VISUAL_HASH_DISTANCE: int = 64
DEFAULT_SAME_SCREEN_THRESHOLD: int = 8

SIGNATURE_TEXT_PREVIEW_LENGTH: int = 32
SIGNATURE_VALUE_PREVIEW_LENGTH: int = 16

BOUNDS_DIGEST_LENGTH: int = 6
INTERACTION_TEXT_PREVIEW_LENGTH: int = 30

# Repeated-decorative-text suppression: when more than this many
# elements share an identical lowercase text label after stripping,
# keep only the first occurrence. Targets noisy XML patterns where
# the same StaticText ("•", "Adyar", rating numbers) is emitted once
# per card and balloons the manifest without adding semantic info.
REPEATED_TEXT_SUPPRESSION_THRESHOLD: int = 2

ACTION_EFFECT_SSIM_THRESHOLD: float = 0.98
ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD: int = 4
ACTION_EFFECT_SCROLL_DISTANCE_THRESHOLD_PX: float = 5.0
ACTION_EFFECT_CONTENT_DIFF_RATIO_THRESHOLD: float = 0.005

# When two frames register as effectively identical (very high SSIM
# combined with negligible content-pixel diff), :func:`cv2.phaseCorrelate`
# can still return a small non-zero shift from DC-noise. We gate the
# scroll computation behind these thresholds so the comparator returns
# a clean zero translation instead of bogus scroll evidence.
SCROLL_IDENTICAL_FRAME_SSIM_THRESHOLD: float = 0.999
SCROLL_IDENTICAL_FRAME_CONTENT_DIFF_RATIO_THRESHOLD: float = 0.001

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
# Chosen from scroll-loop and coachmark-loop fixture coverage; keep this
# threshold high enough that cosmetic pHash jitter does not clear loop evidence.
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

# Maximum native loop-ladder attempts (BACK / SCROLL / HOME) before the
# detector declines further actions and the agent terminates as STUCK.
LOOP_MAX_AUTONOMOUS_RECOVERIES: int = 3

# Action-effect classifier thresholds. Map raw ScreenDiff metrics onto a
# coarse ``ActionEffectStatus`` (progress | no_progress | uncertain).
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

# Minimum number of repeated action occurrences in the recent window
# before a :class:`LoopObservation` is worth surfacing to the agent.
# A single repetition is not yet a loop; two is the smallest evidence
# threshold that justifies a structured observation in the prompt.
MIN_LOOP_OBSERVATION_REPETITIONS: int = 2

# Number of trailing NO_PROGRESS classifications that crosses from
# "ambiguous outcome" into "worth flagging in the prompt".
MIN_NO_PROGRESS_FOR_OBSERVATION: int = 2

# Minimum number of recent screens required before the screen-relation
# classifier (used by :meth:`AgentState.build_loop_observation`) can
# decide whether the last two captures are near-duplicates. One screen
# is insufficient evidence to classify a relation.
MIN_SCREENS_FOR_NEAR_DUPLICATE: int = 2

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
