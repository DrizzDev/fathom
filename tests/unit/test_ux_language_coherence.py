from rich.console import Console

from fathom.services.ux import UXService


def test_render_tool_call_normalizes_reasoning_and_evidence_text() -> None:
    service = UXService()
    service._UXService__console = Console(record=True, width=160)

    service.render_tool_call(
        "execute_ui",
        {
            "assistant_message": "i  found   lemon item  with price .",
            "action": {
                "action_type": "validate",
                "is_valid": True,
                "validation_reason": "lemon item   has  $0.40 .",
            },
        },
        duration=0.2,
    )

    rendered = service._UXService__console.export_text()
    assert "I found lemon item with price." in rendered
    assert "Lemon item has $0.40." in rendered
