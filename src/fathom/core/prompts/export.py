"""Provider-neutral script export prompt policy.

Owns the rules, grammar, and templates used by any LLM provider when
composing deterministic automation scripts from execution traces. Adapter
layers (e.g. ``adapters/prompts/gemini_export.py``) are thin shims that
satisfy the ``ExportPromptBuilder`` port and delegate here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from fathom.interfaces.prompt import ExportPromptBuilder

__all__ = [
    "ExportPromptBuilder",
    "EXPORT_SYSTEM_INSTRUCTION",
    "build_export_user_prompt",
]


EXPORT_SYSTEM_INSTRUCTION = (
    "You convert mobile UI execution traces into deterministic automation scripts.\n"
    "Output only script lines; no markdown, no commentary.\n"
    "Grammar:\n"
    "- OPEN_APP <package>\n"
    "- IF <condition> (then newline)\n"
    "- { (opening brace alone on the next line)\n"
    "- }\n"
    "- Inside IF blocks: indent action/validation lines (Tap on..., Type '...' into..., "
    "Wait..., Scroll..., Swipe..., Validate..., Press...).\n"
    "Rules:\n"
    "1) Preserve chronological order and user intent.\n"
    "2) Group consecutive condition-bound actions into the same IF block when they share the "
    "same intent guard.\n"
    "3) For intents like 'if cart is not empty clear cart then add item', include ALL "
    "cart-clearing actions inside IF cart-not-empty block, and keep remaining actions outside "
    "that block.\n"
    "4) Never invent new actions, screens, or targets. Use only the provided trace data.\n"
    "5) For repeatability, replace product-specific dynamic targets with generic references "
    "(e.g., 'the first search result') if the user intent does not explicitly name that product. "
    "Strip any store or brand names (like Costco, Walmart) from targets.\n"
    "6) Do not include store-brand names in action/condition targets; use generic functional "
    "targets instead.\n"
    "7) Keep OPEN_APP and final validations when supported by trace.\n"
    "8) The final non-structural line is final_validation: a single concise UI state assertion "
    "(what screen, page, or primary content is visible or displayed), inferred from intent and "
    "trace. Align with the user's goal semantically but abstract procedural wording into an "
    "end-state phrase—not a generic placeholder and not a verbatim copy of a long imperative "
    "intent.\n"
    "9) Emit script via tool using schema-compliant plain text only (no markdown fences).\n"
    "10) Executable actions must be copied exactly from allowed step-derived action lines. "
    "Do not invent or paraphrase action text like 'Clear all items'.\n"
    "11) When a package is provided, the first executable line MUST be exactly OPEN_APP "
    "<package>.\n"
    "12) If user intent includes conditional language (e.g., 'if', 'when', 'if cart is not "
    "empty'), you MUST represent that branch using IF block syntax.\n"
    "13) If the intent requests multiple checks (validate/verify/assert/check/confirm), you "
    "MUST distribute them across action-anchored validations in action_validations{}. NEVER "
    "collapse multiple checks into a single final validation line. Example: intent='Validate X, "
    "Validate Y, Validate Z' → action_validations must map at least 2 checks to different "
    "action IDs, preserving 1 for final_validation.\n"
    "14) Return structured tool args: conditional_blocks[].action_ids, remaining_action_ids[], "
    "action_validations{}, final_validation.\n"
    "15) action_validations keys must be action IDs from the catalog. CRITICAL: Every "
    "validation statement MUST explicitly start with 'Validate that ' (or 'Validate ') and end "
    "with a proper full stop.\n"
    "16) Use only action IDs from the provided action catalog; do not emit raw executable "
    "action text.\n"
    "17) CRITICAL: If the execution trace has intermediate points where user validations "
    "should occur (between actions), anchor each validation to the nearest preceding action ID "
    "in action_validations.\n"
    "18) conditional_blocks must handle sequential logic atomically. When a condition guards a "
    "multi-step interaction (e.g., dismissing a dropdown, then scrolling and selecting within "
    "it), group ALL consecutive actions belonging to that interaction inside the block's "
    "action_ids.\n"
    "19) final_validation must not restate imperative steps already covered by catalog "
    "actions—avoid click/tap/type/select/navigate/search-for phrasing and chained 'and then' "
    "procedures in that line.\n"
    "20) Use action_validations for state checks tied to specific earlier actions (e.g. list "
    "visible after search); use final_validation only for the terminal UI state after the last "
    "catalog action.\n"
    "21) final_validation and action_validations MUST be short factual assertions ONLY. Do NOT "
    "append explanations, justifications, or phrases like 'fulfilling the requirement to...', "
    "'as requested by...', 'which confirms that...'. State WHAT is visible, nothing more.\n"
    "    GOOD: 'Validate that the Recommended options in Fine Dining section is visible'\n"
    "    BAD: 'Validate that the Recommended options in Fine Dining section is visible, "
    "fulfilling the requirement to scroll until it is found'\n"
    "22) NEVER use the filler word 'element' in any validation line, final_validation, or "
    "action_validations entry. 'element' is meaningless in a script — always name the actual "
    "visible thing (a button, label, card, page, section, toast, banner, result, icon, field, "
    "etc.).\n"
    "    GOOD: 'Validate that Popular Chains section is visible'\n"
    "    BAD: 'Validate element' or 'Validate that the element is visible'\n"
    "23) CANONICAL SUBJECT FIELDS — each trace step carries the authoritative display name "
    "for its action in a kind-specific field alongside the generic ``target``:\n"
    "    • validate step → read ``validation_subject`` (e.g. 'cart is empty', "
    "'Settings screen open')\n"
    "    • wait step     → read ``wait_subject`` (e.g. 'search results to appear')\n"
    "    • swipe/scroll  → read ``scroll_target`` (e.g. 'Vitamins and supplements')\n"
    "    • everything else → ``target`` (and ``export_target`` if present)\n"
    "    When composing final_validation or action_validations, ALWAYS prefer the "
    "kind-specific field over the generic ``target``. ``target`` is a best-effort display "
    "fallback and may equal the literal string 'unknown' for steps that store their subject "
    "in one of the canonical fields. REASON: the parser routes every action kind to its "
    "canonical subject at construction time, so the subject field is the one place the "
    "runtime guarantees a non-placeholder value.\n"
    "    If a step's canonical subject field is missing, empty, or 'unknown', reuse the "
    "preceding catalog action's target (e.g. 'Validate that the Wow! Momo restaurant card is "
    "visible') — never fall back to the word 'element' or 'unknown'."
)


def build_export_user_prompt(
    *,
    intent: str,
    goal_state: str,
    package_name: str,
    trace_payload: Sequence[Dict[str, Any]],
    action_catalog_lines: Sequence[str],
) -> str:
    """Render the provider-neutral export user prompt."""

    catalog_formatted = "\n".join(f"- {line}" for line in list(action_catalog_lines)) or "- (none)"

    return (
        f"User intent: {intent or goal_state or 'N/A'}\n"  # nosec B608
        f"Goal state: {goal_state or intent or 'N/A'}\n"
        f"Package: {package_name or 'N/A'}\n\n"
        "Opening-line constraint:\n"
        "- If package is provided, first executable line must be exactly: "
        f"OPEN_APP {package_name or '<package>'}\n\n"
        "Action catalog (STRICT, use IDs):\n"
        f"{catalog_formatted}\n\n"
        "Catalog tag legend:\n"
        "- Lines marked '[IF <type> | <condition>]' are AUTHORITATIVE conditional "
        "actions captured at runtime. You MUST place them inside a conditional_block "
        "with that condition text and conditional_type.\n"
        "- 'type' is one of: blocker, transient, error, optional.\n"
        "- Untagged lines are unconditional and belong in remaining_action_ids "
        "(unless the user intent itself wraps them in an if/when guard).\n\n"
        "Action constraints:\n"
        "- Select action IDs only from the action catalog.\n"
        "- Do not rewrite, summarize, or paraphrase executable actions.\n"
        "- Preserve chronological order from the trace when grouping into IF blocks.\n\n"
        "Conditional-block constraint:\n"
        "- Every '[IF ...]'-tagged catalog entry MUST appear inside a conditional_block "
        "whose condition matches the tag's condition text and conditional_type matches "
        "the tag's type. Group consecutive same-tag entries into a single block.\n"
        "- If the intent has an 'if/when' guard, include at least one IF block for it "
        "even when no catalog entry is pre-tagged.\n\n"
        "Tool output format constraint:\n"
        "- Return structured tool args with keys: conditional_blocks, remaining_action_ids, "
        "action_validations, final_validation.\n"
        "- In conditional_blocks, use action_ids (not action text).\n"
        "- In action_validations, map 1+ action IDs to intermediate validation lines (must "
        "start with 'Validate'). CRITICAL: populate this field whenever intent has multiple "
        "validation requirements (e.g., 'Validate X', 'Validate Y', 'Verify Z'). Map each to a "
        "different action ID.\n"
        "- Do not return a free-form script string.\n\n"
        "Final-goal requirement:\n"
        "- final_validation must be one short line starting with 'Validate' that names the "
        "destination UI state as visible or displayed (e.g. experience details page, search "
        "results, cart), derived from what the last catalog actions achieved.\n"
        "- If the user intent is procedural, abstract to the resulting screen or primary "
        "object—do not repeat tap/click/type steps that already appear as catalog actions.\n"
        "- Example: after tapping a specific list item, prefer 'Validate that the experience "
        "details page is visible' over restating list-and-click instructions.\n"
        "- Avoid generic endings like 'Validate Goal State is visible'.\n\n"
        "Trace schema hints (each step in the JSON below carries these keys):\n"
        "- action_type: the action kind (tap, type, validate, wait, swipe_up, scroll, ...)\n"
        "- target: best-effort generic display name (may equal 'unknown' for validate/wait/scroll)\n"
        "- validation_subject: AUTHORITATIVE display name for validate steps — prefer this\n"
        "- wait_subject: AUTHORITATIVE display name for wait steps — prefer this\n"
        "- scroll_target: AUTHORITATIVE display name for swipe/scroll steps — prefer this\n"
        "- rationale: the agent's reasoning for the step; do NOT copy into script lines\n"
        "- condition / is_conditional / conditional_type: IF-block guard data\n"
        "When composing action_validations or final_validation, read the canonical subject "
        "field (validation_subject / wait_subject / scroll_target) instead of falling back "
        "to 'target' — 'target' is a display convenience and may be 'unknown'.\n\n"
        f"Execution trace JSON:\n{json.dumps(list(trace_payload), ensure_ascii=True, indent=2)}\n\n"
        "Generate the final script now."
    )
