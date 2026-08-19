from __future__ import annotations

# Minimum SequenceMatcher ratio for goal-aligned context.
RATIONALE_CONTEXT_RELEVANCE_THRESHOLD: float = 0.60

# Ratio threshold for keyword-based completion match.
RATIONALE_KEYWORD_MATCH_THRESHOLD: float = 0.72

# Floor below which similarity is too low for keyword-based rationale verification;
# admits paraphrases of the target while rejecting rationales that barely overlap it.
RATIONALE_MIN_SIMILARITY_FLOOR: float = 0.35

# Below this rationale-to-sub-goal similarity an asserted claim is flagged as lateral credit; observe-only.
LATERAL_CREDIT_SIMILARITY_THRESHOLD: float = 0.45

# Minimum decomposer confidence; a plan below this floor is a validation finding
# that triggers the single bounded repair, and fails closed if still below.
MINIMUM_DECOMPOSITION_CONFIDENCE: float = 0.6

# Upper bound on decomposed sub-goals; a larger plan is rejected as malformed.
MAXIMUM_DECOMPOSITION_SUB_GOALS: int = 50

# Delta score below which a step counts as low-progress for streak tracking.
LOW_DELTA_PROGRESS_THRESHOLD: float = 0.3

# Number of ANALYZE turns a human instruction may remain active after
# injection. This keeps HITL guidance from disappearing after one ignored
# model turn while still preventing stale instructions from becoming a
# permanent imperative on later screens.
USER_GUIDANCE_ANALYZE_TTL: int = 3

# Consecutive ``validate`` + ``flagged_complete`` emits against a non-validate
# directive before the completion gate accepts the divergence as an implicit
# completion. At 2, the first validate is rejected and the second is accepted.
IMPLICIT_COMPLETION_THRESHOLD: int = 2

# Action confidence floor; below this the action is rejected outright.
ACTION_MIN_CONFIDENCE: float = 0.40

# Confidence floor required when retrying after a previous failure of the same action.
ACTION_MIN_CONFIDENCE_AFTER_FAILURE: float = 0.80

# Synthetic confidence assigned when a next-phase action is detected.
ACTION_NEXT_PHASE_CONFIDENCE: float = 0.85

# Words in reasoning that signal a sub-goal or intent has been completed.
COMPLETION_KEYWORDS: frozenset[str] = frozenset(
    {
        "done",
        "verified",
        "finished",
        "achieved",
        "complete",
        "completed",
        "confirmed",
        "satisfied",
        "successful",
        "accomplished",
    }
)

# Sub-goal description words that identify an opener/launcher goal.
OPENER_GOAL_WORDS: frozenset[str] = frozenset({"open", "launch", "navigate", "go to", "start"})

# Reasoning words indicating a next-phase task is in progress.
NEXT_PHASE_KEYWORDS: frozenset[str] = frozenset(
    {
        "go",
        "set",
        "tap",
        "type",
        "pick",
        "sign",
        "fill",
        "check",
        "swipe",
        "click",
        "close",
        "enter",
        "login",
        "accept",
        "verify",
        "scroll",
        "choose",
        "select",
        "confirm",
        "dismiss",
        "navigate",
    }
)
