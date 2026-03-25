from __future__ import annotations

# Minimum SequenceMatcher ratio for a context to be considered semantically
# aligned with the current goal. Below this, context alignment is not counted.
RATIONALE_CONTEXT_RELEVANCE_THRESHOLD: float = 0.60

# Minimum ratio for keyword-based completion to register as a strong match.
RATIONALE_KEYWORD_MATCH_THRESHOLD: float = 0.72

# Absolute minimum semantic similarity that must be present before keyword-based
# rationale verification is accepted. Prevents near-zero similarity accidental
# keyword hits (e.g. similarity=0.04) from triggering completion on term overlap alone.
RATIONALE_MIN_SIMILARITY_FLOOR: float = 0.10

# Confidence floor below which an action is rejected outright.
ACTION_MIN_CONFIDENCE: float = 0.40

# Confidence floor required after a previous failure for the same action.
ACTION_MIN_CONFIDENCE_AFTER_FAILURE: float = 0.80

# Synthetic confidence assigned when a next-phase action transition is detected,
# indicating high certainty that the current opener sub-goal is already complete.
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

# Words in a sub-goal description that identify it as an opener/launcher goal.
# Used to detect when the agent has moved past the opening phase.
OPENER_GOAL_WORDS: frozenset[str] = frozenset({"open", "launch", "navigate", "go to", "start"})

# Words in LLM reasoning that indicate the agent is performing a next-phase task,
# implying the current opener sub-goal is already satisfied.
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
