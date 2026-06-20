from __future__ import annotations

import re
from typing import List, Set, Tuple

from pydantic import JsonValue

from fathom.constants.collaboration import Label


class PrivacyClassifier:
    """
    Deterministic regex/rule classifier for sensitive content.

    First-pass production classifier — fast, predictable, no ML. Detects:
      - 4-8 digit OTP-like numeric codes
      - email addresses
      - 13-19 digit payment-card-like numbers
      - auth/token/password-ish key tokens
      - UPI ids of the form `<vpa>@<bank>`

    Determinism is critical: the recorder runs classification on every
    message persistence path, including idempotent replays. Classification
    output must be identical for the same input so replay never produces a
    "different content" conflict.
    """

    __EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    # OTP detection requires a keyword neighborhood so progress-style
    # messages like "Step 1234 done" or "Year 2026" do not get false-flagged
    # as PRIVACY_OTP and silently dropped from memory projection.
    __OTP_RE = re.compile(
        r"\b(?:otp|one[\s-]?time[\s-]?(?:code|password|pin)|"
        r"verification[\s-]?(?:code|pin)|auth[\s-]?code|passcode|pin)\b"
        r"[^a-z0-9]{0,16}(?<!\d)\d{4,8}(?!\d)",
        re.IGNORECASE,
    )
    __CARD_RE = re.compile(r"(?<!\d)\d[\d\s-]{11,21}\d(?!\d)")
    __UPI_RE = re.compile(r"\b[\w.-]+@(?:upi|ybl|okaxis|paytm|airtel|ibl)\b", re.IGNORECASE)
    __TOKEN_HINTS = (
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "api-key",
        "apikey",
        "bearer ",
        "authorization",
    )

    def classify(self, *, body: JsonValue, existing: Tuple[Label, ...] = ()) -> Tuple[Label, ...]:
        """
        Return the union of `existing` labels with detected privacy labels.

        Order is deterministic (sorted) so two equivalent classifications
        produce identical tuples. The store's replay-equality check now
        compares labels as sets too, so callers can rely on this output
        being stable across multiple invocations.
        """

        text = self.__flatten(body=body)
        if not text:
            return self.__deduplicate(labels=existing)

        detected: Set[Label] = set(existing)
        if self.__UPI_RE.search(text):
            detected.add(Label.PRIVACY_UPI)
        if self.__EMAIL_RE.search(text):
            detected.add(Label.PRIVACY_EMAIL)
        if self.__matches_card(text=text):
            detected.add(Label.PRIVACY_PAYMENT)
        elif self.__OTP_RE.search(text):
            detected.add(Label.PRIVACY_OTP)
        if self.__matches_token_hints(text=text):
            detected.add(Label.PRIVACY_CREDENTIAL)

        return self.__deduplicate(labels=tuple(detected))

    def __deduplicate(self, *, labels: Tuple[Label, ...]) -> Tuple[Label, ...]:
        """
        Return labels deduplicated and sorted by their underlying enum
        value so two semantically equal label sets produce identical
        tuples for replay-equality checks.
        """

        seen: Set[Label] = set()
        ordered: List[Label] = []
        for label in sorted(labels, key=lambda value: value.value):
            if label in seen:
                continue
            seen.add(label)
            ordered.append(label)
        return tuple(ordered)

    def __flatten(self, *, body: JsonValue) -> str:
        """
        Flatten a JSON-safe body into a single search string. Numbers are
        emitted with whitespace separators so adjacent non-numeric content
        does not create false-positive runs.
        """

        if body is None:
            return ""
        if isinstance(body, str):
            return body
        if isinstance(body, (int, float, bool)):
            return f" {body} "
        if isinstance(body, list):
            return " ".join(self.__flatten(body=item) for item in body)
        if isinstance(body, dict):
            return " ".join(f"{key} {self.__flatten(body=value)}" for key, value in body.items())
        return ""

    def __matches_card(self, *, text: str) -> bool:
        """
        Detect payment-card-like numbers via length + Luhn check on
        digit-only candidates so noise sequences like "12345" don't fire.
        """

        for match in self.__CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            if 13 <= len(digits) <= 19 and self.__luhn_valid(digits=digits):
                return True
        return False

    def __luhn_valid(self, *, digits: str) -> bool:
        """
        Standard Luhn checksum.
        """

        total = 0
        parity = len(digits) % 2
        for index, character in enumerate(digits):
            digit = ord(character) - ord("0")
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0

    def __matches_token_hints(self, *, text: str) -> bool:
        """
        Heuristic token/credential detection by keyword neighbourhood.
        """

        lowered = text.lower()
        return any(hint in lowered for hint in self.__TOKEN_HINTS)


class ClassificationResult:
    """
    Output of one classification call: the merged label tuple and a
    boolean flag indicating whether classification added anything.
    """

    def __init__(self, *, labels: Tuple[Label, ...], added: bool) -> None:
        """
        Store the merged label set and whether classification changed it.
        """

        self.labels = labels
        self.added = added

    @classmethod
    def from_diff(
        cls,
        *,
        existing: Tuple[Label, ...],
        merged: Tuple[Label, ...],
    ) -> "ClassificationResult":
        """
        Build a result that flags whether the merged tuple introduced
        any label not present in `existing`.
        """

        added = bool(set(merged) - set(existing))
        return cls(labels=merged, added=added)
