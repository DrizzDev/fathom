from __future__ import annotations

# Minimum SequenceMatcher ratio for goal-aligned context.
RATIONALE_CONTEXT_RELEVANCE_THRESHOLD: float = 0.60

# Ratio threshold for keyword-based completion match.
RATIONALE_KEYWORD_MATCH_THRESHOLD: float = 0.72

# Floor below which similarity is too low for keyword-based rationale verification.
# Raised from 0.10 to 0.35: the old floor accepted any rationale that mentioned
# a generic completion keyword (``done``, ``complete``, ``finished``…) with as
# little as 10% string overlap against the target. That admitted unrelated
# narrations that happened to use a completion word, causing false rationale
# verifications. 0.35 still admits paraphrases (good) but rejects rationales
# that share virtually nothing with the target text (the regression mode).
RATIONALE_MIN_SIMILARITY_FLOOR: float = 0.35

# Below this rationale-to-sub-goal similarity an asserted claim is flagged as lateral credit; observe-only.
LATERAL_CREDIT_SIMILARITY_THRESHOLD: float = 0.45

# Minimum decomposer confidence below which the fallback single-goal path is used.
MINIMUM_DECOMPOSITION_CONFIDENCE: float = 0.6

# Delta score below which a step counts as low-progress for streak tracking.
LOW_DELTA_PROGRESS_THRESHOLD: float = 0.3

# Number of ANALYZE turns a human instruction may remain active after
# injection. This keeps HITL guidance from disappearing after one ignored
# model turn while still preventing stale instructions from becoming a
# permanent imperative on later screens.
USER_GUIDANCE_ANALYZE_TTL: int = 3

# Number of consecutive ``validate`` + ``flagged_complete`` emits the planner
# may produce against a non-validate directive before the completion gate
# accepts the divergence as an implicit-completion claim. Bridges the gap
# between (a) the original validate-shortcut bug (one stray validate must NOT
# bypass a tap directive) and (b) genuine app-flow variance where the app
# skips an intermediate screen and the named action is no longer reachable.
# Threshold of 2 means: first validate is rejected (keeps the LLM honest);
# second consecutive validate is accepted (lets stale sub-goals resolve).
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
