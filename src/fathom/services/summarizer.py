"""
Progressive step summarizer for long-running workflows.

Compresses older step records into structured phase summaries, preserving
key decisions, milestones, and failure patterns while freeing token budget
for recent raw steps.  Rule-based (no LLM call): zero latency overhead.

Architecture:
    Tier 0  Milestones  — key navigation decisions, never pruned
    Tier 1  Phases      — compressed summaries of N-step batches
    Tier 2  Raw steps   — recent items in ActionHistory deque (unchanged)

Phases auto-merge when exceeding the budget, implementing *progressive*
compression: a 100-step workflow produces at most ``max_phases`` phase
summaries regardless of total step count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, List

logger = getLogger(__name__)


@dataclass(frozen=True)
class PhaseSummary:
    """A compressed summary of a contiguous group of steps."""

    start_step: int
    end_step: int
    summary: str
    screens_visited: List[str]
    key_actions: List[str]
    failures: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "summary": self.summary,
            "screens_visited": list(self.screens_visited),
            "key_actions": list(self.key_actions),
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseSummary":
        """Restore from checkpoint data."""
        return cls(
            start_step=int(data["start_step"]),
            end_step=int(data["end_step"]),
            summary=str(data["summary"]),
            screens_visited=list(data.get("screens_visited", [])),
            key_actions=list(data.get("key_actions", [])),
            failures=list(data.get("failures", [])),
        )


@dataclass
class SummarizedHistory:
    """
    Tiered history container: milestones + phase summaries.

    Raw recent steps are NOT stored here — they remain in the
    ``ActionHistory`` deque and are appended at format time by
    ``ActionHistory.get_summarized_context()``.
    """

    milestones: List[str] = field(default_factory=list)
    phase_summaries: List[PhaseSummary] = field(default_factory=list)
    _max_phases: int = field(default=5, repr=False)
    _max_milestones: int = field(default=8, repr=False)

    def add_milestone(self, description: str) -> None:
        """Record a key milestone (successful screen-changing navigation)."""

        if description not in self.milestones:
            self.milestones.append(description)

        # Keep milestones bounded — drop oldest when over budget
        if len(self.milestones) > self._max_milestones:
            self.milestones = self.milestones[-self._max_milestones :]

    def add_phase(self, phase: PhaseSummary) -> None:
        """Add a phase summary; merge oldest phases if over budget."""

        self.phase_summaries.append(phase)

        if len(self.phase_summaries) > self._max_phases:
            self._merge_oldest_phases()

    def _merge_oldest_phases(self) -> None:
        """Merge the two oldest phases into one super-summary."""

        if len(self.phase_summaries) < 2:
            return

        old_a = self.phase_summaries[0]
        old_b = self.phase_summaries[1]

        merged = PhaseSummary(
            start_step=old_a.start_step,
            end_step=old_b.end_step,
            summary=f"{old_a.summary} Then: {old_b.summary}",
            screens_visited=list(dict.fromkeys(old_a.screens_visited + old_b.screens_visited)),
            key_actions=old_a.key_actions[-2:] + old_b.key_actions[-2:],
            failures=old_a.failures[-1:] + old_b.failures[-1:],
        )

        self.phase_summaries = [merged] + self.phase_summaries[2:]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "milestones": list(self.milestones),
            "phase_summaries": [p.to_dict() for p in self.phase_summaries],
            "max_phases": self._max_phases,
            "max_milestones": self._max_milestones,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SummarizedHistory":
        """Restore from checkpoint data."""
        history = cls(
            _max_phases=int(data.get("max_phases", 5)),
            _max_milestones=int(data.get("max_milestones", 8)),
        )
        history.milestones = list(data.get("milestones", []))
        history.phase_summaries = [
            PhaseSummary.from_dict(p) for p in data.get("phase_summaries", [])
        ]
        return history

    def format_context(self, max_chars: int = 2000) -> str:
        """
        Render tiered history for the LLM context.

        Uses ``=== SECTION ===`` markers consistent with
        ``AgentState.get_smart_context()`` style.  Phase summaries go in
        the middle-of-context (acceptable lower attention), milestones
        are brief enough to also land here.
        """

        parts: List[str] = []

        if self.phase_summaries:
            parts.append("=== EARLIER STEPS (Summarized) ===")
            for phase in self.phase_summaries:
                line = f"[Steps {phase.start_step}-{phase.end_step}] {phase.summary}"
                if phase.failures:
                    line += f" | Failures: {', '.join(phase.failures[-2:])}"
                parts.append(line)

        result = "\n".join(parts)

        if len(result) > max_chars:
            result = result[:max_chars] + "…"

        return result


class StepSummarizer:
    """
    Converts batches of raw step records into compressed PhaseSummaries.

    Call ``ingest()`` each time a step record is evicted from the
    ActionHistory deque.  Every ``phase_size`` ingestions, the buffer is
    automatically flushed into a new ``PhaseSummary``.

    Args:
        phase_size: Number of steps per phase before compression.
        max_phases: Maximum phase summaries before recursive merging.
        max_milestones: Maximum milestones to retain.
    """

    def __init__(
        self,
        phase_size: int = 10,
        max_phases: int = 5,
        max_milestones: int = 8,
    ) -> None:
        self._phase_size = phase_size
        self._buffer: List[Dict[str, Any]] = []
        self._history = SummarizedHistory(
            _max_phases=max_phases,
            _max_milestones=max_milestones,
        )
        self._step_offset = 0

    @property
    def history(self) -> SummarizedHistory:
        """Access the accumulated summarized history."""
        return self._history

    def ingest(self, step_record: Dict[str, Any]) -> None:
        """
        Accept a step record evicted from the ActionHistory deque.

        When the internal buffer reaches ``phase_size``, it is
        automatically compressed into a ``PhaseSummary``.
        """

        self._buffer.append(step_record)

        # Detect milestones: screen changes after successful navigation
        if step_record.get("screen_changed") and step_record.get("success"):
            activity = step_record.get("activity", "")
            target = step_record.get("target", "")
            if activity and target:
                self._history.add_milestone(
                    f"{step_record.get('type', '?')} '{target}' → {activity}"
                )

        if len(self._buffer) >= self._phase_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Compress buffered steps into a PhaseSummary."""

        if not self._buffer:
            return

        start = self._step_offset + 1
        end = self._step_offset + len(self._buffer)

        # Unique screens visited (preserving order)
        screens = list(dict.fromkeys(record.get("activity", "unknown") for record in self._buffer))

        # Key actions: successful screen-changing actions
        key_actions = [
            f"{record['type']}:{record['target']}"
            for record in self._buffer
            if record.get("success") and record.get("screen_changed")
        ]

        # Failures
        failures = [
            f"{record['type']}:{record['target']}"
            for record in self._buffer
            if not record.get("success")
        ]

        # Build narrative
        success_count = sum(1 for record in self._buffer if record.get("success"))
        fail_count = len(self._buffer) - success_count

        summary_parts = []
        if screens:
            summary_parts.append(f"Visited {', '.join(screens[:4])}")

        summary_parts.append(f"{success_count}✓, {fail_count}✗")

        if key_actions:
            summary_parts.append(f"Key: {', '.join(key_actions[:4])}")

        phase = PhaseSummary(
            start_step=start,
            end_step=end,
            summary=". ".join(summary_parts),
            screens_visited=screens,
            key_actions=key_actions[:5],
            failures=failures[:3],
        )

        self._history.add_phase(phase)
        self._step_offset = end
        self._buffer.clear()

        logger.debug(
            f"StepSummarizer: flushed steps {start}-{end} into phase summary "
            f"(total phases: {len(self._history.phase_summaries)})"
        )

    def flush(self) -> None:
        """Force-flush any remaining buffered steps."""
        if self._buffer:
            self._flush_buffer()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the full summarizer state for checkpointing.

        Includes the accumulated history, the unflushed buffer, and
        configuration so that restoration produces an identical state.
        """
        return {
            "phase_size": self._phase_size,
            "step_offset": self._step_offset,
            "buffer": list(self._buffer),
            "history": self._history.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepSummarizer":
        """
        Restore a ``StepSummarizer`` from checkpoint data.

        Reconstructs the history, buffer, and step offset so that
        subsequent ``ingest()`` calls continue seamlessly.
        """
        history_data = data.get("history", {})
        summarizer = cls(
            phase_size=int(data.get("phase_size", 10)),
            max_phases=int(history_data.get("max_phases", 5)),
            max_milestones=int(history_data.get("max_milestones", 8)),
        )
        summarizer._history = SummarizedHistory.from_dict(history_data)
        summarizer._buffer = list(data.get("buffer", []))
        summarizer._step_offset = int(data.get("step_offset", 0))
        return summarizer
