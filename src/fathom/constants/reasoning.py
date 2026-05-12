from __future__ import annotations

# Minimum SequenceMatcher ratio for goal-aligned context.
RATIONALE_CONTEXT_RELEVANCE_THRESHOLD: float = 0.60

# Ratio threshold for keyword-based completion match.
RATIONALE_KEYWORD_MATCH_THRESHOLD: float = 0.72

# Floor below which similarity is too low for keyword-based rationale verification.
RATIONALE_MIN_SIMILARITY_FLOOR: float = 0.10

# Magnitude below which a post-action screen change is treated as noise.
MEANINGFUL_SCREEN_DELTA_FLOOR: float = 0.05

# Minimum decomposer confidence below which the fallback single-goal path is used.
MINIMUM_DECOMPOSITION_CONFIDENCE: float = 0.6

# Delta score below which a step counts as low-progress for streak tracking.
LOW_DELTA_PROGRESS_THRESHOLD: float = 0.3

# Action confidence floor; below this the action is rejected outright.
ACTION_MIN_CONFIDENCE: float = 0.40

# Confidence floor required when retrying after a previous failure of the same action.
ACTION_MIN_CONFIDENCE_AFTER_FAILURE: float = 0.80

# Synthetic confidence assigned when a next-phase action is detected.
ACTION_NEXT_PHASE_CONFIDENCE: float = 0.85

# Sub-goal description tokens that classify the step as a validation
# (observation-only) goal where the screen is not expected to change.
VALIDATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "verify",
        "validate",
        "confirm",
        "check if",
        "check that",
    }
)

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
