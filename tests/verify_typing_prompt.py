import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from fathom.prompts.gemini import GeminiPromptBuilder


def test_typing_prompt():
    builder = GeminiPromptBuilder()
    intent = "Type 'hello world' in the search bar"
    prompt = builder.build(intent=intent)

    print("--- PROMPT START ---")
    print(prompt)
    print("--- PROMPT END ---")

    # Check for the rule in COMMON_RULES part
    has_common_rule = "CRITICAL - TAP BEFORE TYPE" in prompt
    # Check for the rule in CONTEXTUAL_RULES part
    has_contextual_rule = "CRITICAL SEQ: Use 'tap' to gain focus" in prompt

    print(f"\nCommon Rule present: {has_common_rule}")
    print(f"Contextual Rule present: {has_contextual_rule}")

    assert has_common_rule, "Common rule missing!"
    assert has_contextual_rule, "Contextual rule missing!"
    print("\nSUCCESS: Verification script passed.")


if __name__ == "__main__":
    test_typing_prompt()
