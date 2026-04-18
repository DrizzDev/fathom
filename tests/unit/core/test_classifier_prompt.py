"""Tests for the classification prompt policy constants and builder."""

from __future__ import annotations

from fathom.core.prompts.classification import (
    CLASSIFICATION_SYSTEM,
    CLASSIFICATION_TOOL_DEFINITION,
    CLASSIFICATION_TOOL_NAME,
    build_classification_user_prompt,
)


def test_tool_definition_is_well_formed() -> None:
    """The tool definition must match the Gemini function_declarations shape."""

    assert "function_declarations" in CLASSIFICATION_TOOL_DEFINITION
    declarations = CLASSIFICATION_TOOL_DEFINITION["function_declarations"]
    assert len(declarations) == 1

    declaration = declarations[0]
    assert declaration["name"] == CLASSIFICATION_TOOL_NAME

    params = declaration["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"should_decompose", "reason"}

    properties = params["properties"]
    assert properties["should_decompose"]["type"] == "boolean"
    assert properties["reason"]["type"] == "string"


def test_system_instruction_mentions_tool_name() -> None:
    """The system instruction must reference the tool the model is expected
    to call, so the model knows what to emit."""

    assert CLASSIFICATION_TOOL_NAME in CLASSIFICATION_SYSTEM


def test_user_prompt_embeds_intent() -> None:
    """The rendered prompt must embed the verbatim intent string."""

    intent = "Open Instacart, go to the Aldi store page, and add banana to cart"
    prompt_parts = build_classification_user_prompt(intent=intent)
    prompt_text = "".join(prompt_parts)

    assert intent in prompt_text


def test_user_prompt_contains_decision_verb_guidance() -> None:
    """The prompt must list the disqualifying decision verbs so the LLM
    knows what phrases force decomposition."""

    prompt_parts = build_classification_user_prompt(intent="any intent")
    prompt_text = "".join(prompt_parts)

    assert "pick the cheapest" in prompt_text
    assert "find a good" in prompt_text
    assert "compare" in prompt_text


def test_user_prompt_includes_positive_and_negative_examples() -> None:
    """Both example sections must be present for anchoring the LLM."""

    prompt_parts = build_classification_user_prompt(intent="any intent")
    prompt_text = "".join(prompt_parts)

    assert "should_decompose = FALSE" in prompt_text
    assert "should_decompose = TRUE" in prompt_text
    assert "Log in with email" in prompt_text
    assert "Open Instacart" in prompt_text
