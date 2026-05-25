from __future__ import annotations

import unittest

from fathom.core.services.exporter.trace_payload import build_export_payload


class TracePayloadTest(unittest.TestCase):
    def test_dict_step_records_are_serialized_without_str_keyword_error(self) -> None:
        payload = build_export_payload(
            step_results=[
                {
                    "activity": "com.example.app",
                    "action_type": "tap",
                    "event_type": "action",
                    "export_target": "Continue",
                    "rationale": "Move to the next screen",
                    "screen_changed": True,
                }
            ]
        )

        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["rationale"], "Move to the next screen")
        self.assertEqual(payload[0]["target"], "Continue")
