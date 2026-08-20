from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.conversation.wire import (
    WireAnswerBody,
    WireObservation,
    WireProgressBody,
    WireQuestionBody,
    WireRequestBody,
    WireResultBody,
)


class TestWireProgressBody(unittest.TestCase):
    """
    Pin the compact progress-body wire shape rendered under mode=user.
    """

    def test_summary_alone_omits_observation_after_dump(self) -> None:
        """
        A progress body with only summary must serialize to just the summary key.
        """

        body = WireProgressBody(summary="tapping search")
        dumped = body.model_dump(mode="json", exclude_none=True)

        self.assertEqual({"summary": "tapping search"}, dumped)

    def test_observation_evidence_appears_when_supplied(self) -> None:
        """
        Nested observation.evidence must appear when it carries a string.
        """

        body = WireProgressBody(
            summary="tapping search",
            observation=WireObservation(evidence="home screen with search bar"),
        )
        dumped = body.model_dump(mode="json", exclude_none=True)

        self.assertEqual(
            {
                "summary": "tapping search",
                "observation": {"evidence": "home screen with search bar"},
            },
            dumped,
        )

    def test_missing_summary_rejected(self) -> None:
        """
        Summary is required; construction without it must raise ValidationError.
        """

        with self.assertRaises(ValidationError):
            WireProgressBody()  # type: ignore[call-arg]

    def test_extra_fields_rejected(self) -> None:
        """
        Extra fields (e.g. leftover audit-shape keys) must be rejected at boundary.
        """

        with self.assertRaises(ValidationError):
            WireProgressBody(
                step=1,  # type: ignore[call-arg]
                summary="tapping search",
            )


class TestWireResultBody(unittest.TestCase):
    """
    Pin the compact result-body wire shape for terminal run messages.
    """

    def test_success_case_shape(self) -> None:
        """
        Successful result body carries summary + success + reason.
        """

        body = WireResultBody(summary="cart updated", success=True, reason="verified")
        dumped = body.model_dump(mode="json", exclude_none=True)

        self.assertEqual(
            {"summary": "cart updated", "success": True, "reason": "verified"},
            dumped,
        )

    def test_reason_optional_when_absent(self) -> None:
        """
        Reason is optional and must be omitted when None.
        """

        body = WireResultBody(summary="done", success=True)
        dumped = body.model_dump(mode="json", exclude_none=True)

        self.assertEqual({"summary": "done", "success": True}, dumped)

    def test_failure_case_preserves_success_flag(self) -> None:
        """
        Failure result must carry success=False so the client can style red.
        """

        body = WireResultBody(summary="aborted", success=False, reason="user cancel")
        dumped = body.model_dump(mode="json", exclude_none=True)

        self.assertFalse(dumped["success"])
        self.assertEqual("aborted", dumped["summary"])

    def test_extra_fields_rejected(self) -> None:
        """
        Audit fields like `error` / `steps` / `detail` / `status` must not slip through.
        """

        with self.assertRaises(ValidationError):
            WireResultBody(
                steps=7,  # type: ignore[call-arg]
                success=True,
                summary="done",
            )


class TestWireRequestBody(unittest.TestCase):
    """
    Pin the compact request-body wire shape for user intent messages.
    """

    def test_intent_only(self) -> None:
        """
        Request body carries only the intent string.
        """

        body = WireRequestBody(intent="search for burgers")
        dumped = body.model_dump(mode="json", exclude_none=True)

        self.assertEqual({"intent": "search for burgers"}, dumped)

    def test_missing_intent_rejected(self) -> None:
        """
        Intent is required.
        """

        with self.assertRaises(ValidationError):
            WireRequestBody()  # type: ignore[call-arg]

    def test_extra_package_fields_rejected(self) -> None:
        """
        `package` and `starting_package` are dropped from the compact shape.
        """

        with self.assertRaises(ValidationError):
            WireRequestBody(
                intent="x",
                package="com.app",  # type: ignore[call-arg]
            )


class TestWireQuestionBody(unittest.TestCase):
    """
    Pin the compact question-body wire shape for HITL prompts.
    """

    def test_question_only(self) -> None:
        """
        Question body carries only the prompt text.
        """

        body = WireQuestionBody(question="what should I do next?")
        dumped = body.model_dump(mode="json", exclude_none=True)

        self.assertEqual({"question": "what should I do next?"}, dumped)

    def test_extra_step_rejected(self) -> None:
        """
        Step number is dropped from the compact question shape.
        """

        with self.assertRaises(ValidationError):
            WireQuestionBody(
                question="x",
                step=1,  # type: ignore[call-arg]
            )


class TestWireAnswerBody(unittest.TestCase):
    """
    Pin the compact answer-body wire shape for HITL responses.
    """

    def test_answer_only(self) -> None:
        """
        Answer body carries only the response text.
        """

        body = WireAnswerBody(answer="Add Tiramisu to cart")
        dumped = body.model_dump(mode="json", exclude_none=True)

        self.assertEqual({"answer": "Add Tiramisu to cart"}, dumped)


class TestWireObservation(unittest.TestCase):
    """
    Pin the compact observation projection.
    """

    def test_all_fields_optional(self) -> None:
        """
        Observation with no evidence serializes to an empty object under exclude_none.
        """

        observation = WireObservation()
        dumped = observation.model_dump(mode="json", exclude_none=True)

        self.assertEqual({}, dumped)

    def test_summary_and_evidence_kept(self) -> None:
        """
        observation.summary and observation.evidence are both surfaced verbatim.
        """

        observation = WireObservation(summary="Cart is visible.", evidence="tap CTA")
        dumped = observation.model_dump(mode="json", exclude_none=True)

        self.assertEqual({"summary": "Cart is visible.", "evidence": "tap CTA"}, dumped)


class TestWireProgressBodyProjection(unittest.TestCase):
    """
    Pin per-schema projection from a stored audit body.
    """

    def test_projection_passes_all_visible_fields_through_verbatim(self) -> None:
        """
        Wire projector surfaces summary, rationale, and observation.summary/evidence exactly as stored — no dedup, no rewriting.
        """

        same = "I'll tap 'View Cart' to verify."
        projected = WireProgressBody.project(
            body={
                "summary": same,
                "analysis": same,
                "rationale": "verify",
                "observation": {
                    "summary": same,
                    "evidence": same,
                    "screen": "abc",
                    "changed": True,
                },
                "step": 7,
            }
        )

        assert projected is not None
        self.assertEqual(
            {
                "summary": same,
                "rationale": "verify",
                "observation": {"summary": same, "evidence": same},
            },
            projected.model_dump(mode="json", exclude_none=True),
        )

    def test_distinct_observation_survives_projection(self) -> None:
        """
        Distinct observation fields survive the projection.
        """

        projected = WireProgressBody.project(
            body={
                "summary": "tap search",
                "observation": {
                    "summary": "search bar in focus",
                    "evidence": "home screen with search bar",
                },
            }
        )

        assert projected is not None
        self.assertEqual(
            {
                "summary": "tap search",
                "observation": {
                    "summary": "search bar in focus",
                    "evidence": "home screen with search bar",
                },
            },
            projected.model_dump(mode="json", exclude_none=True),
        )

    def test_missing_summary_returns_none(self) -> None:
        """
        Body without a valid summary yields None so the caller keeps the original.
        """

        self.assertIsNone(WireProgressBody.project(body={"step": 1}))


class TestWireRequestBodyProjection(unittest.TestCase):
    """
    Pin request-body projection.
    """

    def test_intent_only(self) -> None:
        """
        Compact request body drops package/starting_package.
        """

        projected = WireRequestBody.project(
            body={"intent": "search for burgers", "starting_package": "com.delivery"}
        )

        assert projected is not None
        self.assertEqual(
            {"intent": "search for burgers"}, projected.model_dump(mode="json", exclude_none=True)
        )

    def test_missing_intent_returns_none(self) -> None:
        """
        Missing intent yields None so the caller keeps the original.
        """

        self.assertIsNone(WireRequestBody.project(body={"package": None}))


class TestWireResultBodyProjection(unittest.TestCase):
    """
    Pin result-body projection.
    """

    def test_summary_success_reason_only(self) -> None:
        """
        Compact result body drops error/steps/detail/status.
        """

        projected = WireResultBody.project(
            body={
                "error": None,
                "steps": 3,
                "detail": None,
                "reason": "done",
                "status": "succeeded",
                "success": True,
                "summary": "done",
            }
        )

        assert projected is not None
        self.assertEqual(
            {"summary": "done", "success": True, "reason": "done"},
            projected.model_dump(mode="json", exclude_none=True),
        )

    def test_missing_success_returns_none(self) -> None:
        """
        Missing boolean success yields None so the caller keeps the original.
        """

        self.assertIsNone(WireResultBody.project(body={"summary": "x"}))


class TestWireQuestionBodyProjection(unittest.TestCase):
    """
    Pin question-body projection.
    """

    def test_question_only(self) -> None:
        """
        Compact question body drops step.
        """

        projected = WireQuestionBody.project(body={"step": 7, "question": "What next?"})

        assert projected is not None
        self.assertEqual(
            {"question": "What next?"}, projected.model_dump(mode="json", exclude_none=True)
        )


class TestWireAnswerBodyProjection(unittest.TestCase):
    """
    Pin answer-body projection.
    """

    def test_answer_only(self) -> None:
        """
        Compact answer body is just the reply.
        """

        projected = WireAnswerBody.project(body={"answer": "add rasmalai to cart"})

        assert projected is not None
        self.assertEqual(
            {"answer": "add rasmalai to cart"}, projected.model_dump(mode="json", exclude_none=True)
        )
