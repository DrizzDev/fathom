from __future__ import annotations

from typing import List, Tuple

from fathom.schemas.authoring.prompt import PromptEvidence
from fathom.schemas.authoring.reference import DRIZZ_COMMANDS
from fathom.schemas.flow import Evidence, Issue


class FlowPromptBuilder:
    """
    Builds the Layer-2 system instruction and user prompt for evidence-grounded flow generation.
    """

    __ROLE = (
        "You are the final synthesis layer of a two-layer pipeline. Layer one already enriched "
        "every recorded step with the planner's reasoning, target, condition, and generalization "
        "signals; you receive those steps grouped into execution episodes together with the "
        "user's intent. Using "
        "your own judgment, write the run as a Drizz automation script, expressed as the "
        "structured Flow below; it is rendered to Drizz deterministically, so never write script "
        "syntax yourself."
    )
    __EVIDENCE_GUIDE = (
        "The evidence is grouped into episodes. Each episode represents one contiguous execution "
        "segment, usually one sub-goal. Read the episode goal first, then read its steps:\n"
        "- goal.description names the user-level purpose for the episode; use it to understand "
        "why the steps happened.\n"
        "- goal.directive is the command family expected for the episode, when available.\n"
        "- target.export is the recorded script target. target.name and target.generalized are "
        "fallback target phrases when no export target was recorded. target.element is the recorded "
        "UI role, such as button or field. Use the episode goal, action, rationale, observation, "
        "and element role to choose a complete replay target; do not copy placeholder or helper "
        "text when the evidence identifies the control's role.\n"
        "- Use exact labels for stable controls or fixed visible targets. Use relative or dynamic "
        "phrases when a target was selected by query, order, filter, rating, price, or other "
        "runtime context. Combine exact and dynamic context when either alone is ambiguous.\n"
        "- on a scroll step, target.scroll names the content the step recorded. Prefer one clean "
        "scroll command for the episode. Use Scroll <dir> until <target> only when the target is a "
        "concrete replay objective grounded in target.scroll, the episode goal, rationale, or "
        "observation; otherwise use a plain Scroll <dir>. Use Scroll <dir> inside <target.export> "
        "when the gesture was confined to a container. The "
        "recorded action names the finger gesture, not the page motion: a swipe up reveals content "
        "below, so it is Scroll down; a swipe down is Scroll up.\n"
        "- a step with a successful capture field stored a value: emit a StoreNode whose value is "
        "capture.value and name is capture.name, citing that step. Never emit a Store for a step "
        "that has no successful capture value.\n"
        "- guard.conditional being true means the step ran under a condition: place its node inside "
        "an IF branch.\n"
        "- guard.condition is evidence for the branch condition, not a sentence you must copy. "
        "Author a concise condition grounded in the guard, target, rationale, or observation. It "
        "must describe the specific visible state, not a generic category.\n"
        "- a step with event 'launch' is a pre-decided app launch: emit a LaunchNode copying its "
        "launch.package, launch.provenance, and launch.source_steps exactly. Emit every launch "
        "step, in order, and add no launches of your own.\n"
        "- recovery and loop-escape steps are already removed before you see the evidence; where "
        "what remains still shows the same screen cycle repeating, fold it to the single intended "
        "pass using your judgment.\n"
        "- Repeated actions inside one episode are usually attempts toward one user-level purpose. "
        "Collapse them into the clean replay command unless each action has a distinct semantic "
        "purpose.\n"
        "- Validation asserts the intended state for the episode or run. A visible control may be "
        "evidence for that state, but do not use an incidental anchor as the validation subject "
        "when the episode goal names the real state.\n"
        "- the top-level goal field is the run's destination state; the final validation must "
        "assert that goal, never an input, search, or intermediate screen passed through.\n"
        "- action, observation, and rationale carry the planner's reasoning, for context."
    )
    __JUDGMENT = (
        "Reason over BOTH the user's intent and the recorded evidence together; never rely on "
        "either alone. Decide each target's specificity yourself: keep it exact when the user "
        "meant a specific thing, and generalize it only when it was genuinely incidental or "
        "variable. The launch already opens the app from its launcher, so do not also tap the "
        "app's own launcher icon. Fold retries, loops, and recovery noise into the clean sequence "
        "the user actually intended. These are judgment calls, not mechanical rules."
    )
    __LANGUAGE_HEADER = "DRIZZ LANGUAGE REFERENCE (pure syntax and usage):"
    __REFERENCE_HEADER = "Commands (Flow node -> Drizz form (e.g. example): purpose):"
    __LANGUAGE_IDIOMS = (
        "Usage:\n"
        "- One command per line; you emit the structured Flow and it is rendered to these lines, "
        "so think in commands, never in raw text.\n"
        "- Quoting and the delimiter choice are handled by the renderer; put the bare target text "
        "in the Flow and never add quotes, colons, or braces yourself.\n"
        "- A tap names the replay-stable target: exact for stable controls, dynamic/relative for "
        "runtime-selected list results, and combined when that best captures the user's intent.\n"
        "- An IF block carries a concise evidence-grounded condition and contains only the branch "
        "that actually executed (there is no ELSE).\n"
        "- A validation asserts a destination state is visible or present.\n"
        "- Emit exactly one LaunchNode per launch evidence step, in order; the first is the entry "
        "point and any later launches are app switches the run actually made."
    )
    __LANGUAGE_AVOID = (
        "Never in the language:\n"
        "- Emit a command not listed above, or hand-write Drizz syntax, braces, or quotes.\n"
        "- Use a whole sentence or the entire intent as one target."
    )
    __POLICY_HEADER = "GENERATION POLICY (grounding and completion):"
    __POLICY_AVOID = (
        "Never in policy:\n"
        "- Re-tap the app's launcher icon after OPEN_APP.\n"
        "- Invent a terminal validation when the run did not actually reach its goal.\n"
        "- Keep consecutive duplicate commands the recording only repeated by accident."
    )
    __INVARIANTS = (
        "The Flow must satisfy these invariants (they are checked deterministically):\n"
        "- It begins with a launch, and there is exactly one LaunchNode per launch evidence step, "
        "in the same order, each copying that step's package, provenance, and source_steps; never "
        "launch a launcher package.\n"
        "- Every node except launches cites the evidence step numbers it derives from in "
        "source_steps; nothing is invented.\n"
        "- No node derives from a step whose recorded condition is 'recovery'.\n"
        "- A branch condition must be grounded in the cited conditional step's guard, target, "
        "rationale, or observation, with only the executed branch present.\n"
        "- When a recorded step ran under a condition (its guard is conditional), place that "
        "step's node inside an IF branch using an evidence-grounded condition; never emit it "
        "unguarded.\n"
        "- Group every step that ran under one condition into a single IF branch; never emit "
        "consecutive IF branches that share the same condition.\n"
        "- Collapse repeated attempts inside one episode, such as consecutive waits, repeated "
        "scrolling toward the same goal, or retry taps on the same target.\n"
        "- A scroll node must cite a recorded scroll gesture, and any scroll-until target must be "
        "grounded in the cited step's target, episode goal, rationale, or observation.\n"
        "- A Store node derives only from a step whose evidence recorded a successful capture; "
        "use capture.value and capture.name.\n"
        "- A complete run ends with a validation that cites a recorded validation step. If the "
        "evidence is marked partial (no recorded validation step), set the flow's partial flag "
        "true and do not invent a terminal validation."
    )
    __EVIDENCE_LABEL = "Recorded run evidence (JSON):"
    __FEEDBACK_INTRO = (
        "Repair the previous Flow before reading the evidence again. The next Flow must fix "
        "every issue below. "
        "When an issue lists evidence phrases, choose an authored phrase grounded in those phrases; "
        "do not repeat the rejected node text. If a rejected field is unchanged, the Flow will fail "
        "again."
    )

    def system_instruction(self) -> str:
        """
        Return the role, the Drizz language reference, and the generation policy as one block.
        """

        return "\n\n".join((self.__ROLE, self.__language_reference(), self.__generation_policy()))

    def __language_reference(self) -> str:
        """
        Build the pure Drizz language section: commands, usage idioms, and language anti-patterns.
        """

        reference = "\n".join(
            f"- {doc.name} -> {doc.syntax}  (e.g. {doc.example}): {doc.purpose}"
            for doc in DRIZZ_COMMANDS
        )
        return "\n\n".join(
            (
                self.__LANGUAGE_HEADER,
                f"{self.__REFERENCE_HEADER}\n{reference}",
                self.__LANGUAGE_IDIOMS,
                self.__LANGUAGE_AVOID,
            )
        )

    def __generation_policy(self) -> str:
        """
        Build the policy section: how to read evidence, judgment, invariants, and policy avoids.
        """

        return "\n\n".join(
            (
                self.__POLICY_HEADER,
                self.__EVIDENCE_GUIDE,
                self.__JUDGMENT,
                self.__INVARIANTS,
                self.__POLICY_AVOID,
            )
        )

    def user_prompt(self, *, evidence: Evidence, feedback: Tuple[Issue, ...] = ()) -> str:
        """
        Build the user prompt from the evidence and any prior gate feedback.
        """

        packet = PromptEvidence.from_evidence(evidence=evidence)
        sections: List[str] = []

        if feedback:
            sections.append(self.__FEEDBACK_INTRO)
            sections.extend(f"- {issue.code}: {issue.message}" for issue in feedback)
            sections.append("")

        sections.extend(
            [
                self.__EVIDENCE_LABEL,
                packet.model_dump_json(exclude_none=True),
            ]
        )

        return "\n".join(sections)
