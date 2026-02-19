import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from fathom.prompts.gemini import GeminiPromptBuilder


def test_typing_prompt():
    builder = GeminiPromptBuilder()
    intent = "Type 'hello world' in the search bar"

    system_prompt = builder.build(intent=intent)
    task_instructions = builder.build_task_instructions(
        intent=intent, hints={"typing_text": "hello world"}
    )

    print("--- SYSTEM PROMPT ---")
    print(system_prompt)
    print("--- TASK INSTRUCTIONS ---")
    print(task_instructions)
    print("--- END ---")

    has_common_rule = "CRITICAL - TAP BEFORE TYPE" in system_prompt
    has_contextual_rule = "CRITICAL SEQ: Use 'tap' to gain focus" in task_instructions

    print(f"\nCommon Rule present (system): {has_common_rule}")
    print(f"Contextual Rule present (task): {has_contextual_rule}")

    assert has_common_rule, "Common rule missing from system prompt!"
    assert has_contextual_rule, "Contextual rule missing from task instructions!"
    print("\nSUCCESS: Verification script passed.")


if __name__ == "__main__":
    test_typing_prompt()
