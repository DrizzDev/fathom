from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from google import genai
from google.oauth2 import service_account

from fathom.settings.env import FathomSettings


@dataclass(frozen=True)
class ClassificationResult:
    """Result of target classification with type metadata."""

    description: str
    is_positional: bool = False


class TargetClassifier:
    """
    Uses an LLM to classify UI targets as 'Stable', 'Dynamic', or 'Positional'
    and generates generalized/positional descriptions accordingly.
    Uses generic text generation, not function calling.
    """

    def __init__(self) -> None:
        self.__settings = FathomSettings()
        self.__client: Optional[genai.Client] = None
        self.__model = "gemini-2.5-flash-lite"
        self.__initialize_client()
        self.__cache: Dict[str, ClassificationResult] = {}
        self.__goal_cache: Dict[str, str] = {}

    def __initialize_client(self) -> None:
        try:
            api_key = self.__settings.gemini_api_key
            project = self.__settings.vertex_project_id
            location = self.__settings.vertex_location or "us-central1"
            credentials = None

            if self.__settings.google_application_credentials:
                path = Path(self.__settings.google_application_credentials)
                if path.exists():
                    credentials = service_account.Credentials.from_service_account_file(
                        str(path),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )

            if api_key:
                self.__client = genai.Client(api_key=api_key)
            else:
                self.__client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                    credentials=credentials,
                )
        except Exception as e:
            print(f"DEBUG: Classifier Init Failed: {e}")
            self.__client = None

    async def classify_and_generalize(
        self, target: str, intent: str, rationale: str = "", screen_description: str = ""
    ) -> ClassificationResult:
        """
        Determines if a target is stable, dynamic, or a positional list item.
        Returns a ClassificationResult with the resolved description and
        whether the target is a positional/ordinal reference.
        """
        if not target or not self.__client:
            return ClassificationResult(description=target)

        cache_key = hashlib.md5(
            f"{target}:{intent}:{rationale}:{screen_description}".encode(),
            usedforsecurity=False,
        ).hexdigest()
        if cache_key in self.__cache:
            return self.__cache[cache_key]

        screen_line = f'\nScreen Description: "{screen_description}"' if screen_description else ""

        prompt = f"""
You are an expert mobile test automation engineer.
Your goal is to decide how a UI element should be referenced in a reusable test script.
The script must work across different runs even when list content changes.

User Intent: "{intent}"
Target Text: "{target}"
Rationale: "{rationale}"{screen_line}

Instructions:
1. CHECK INTENT: Is the "Target Text" explicitly mentioned or strongly implied by the "User Intent"?
   - YES (e.g. Intent="Play Aditi Shah...", Target="Aditi Shah Sleep Meditation") -> Output STABLE.
   - NO -> Continue to step 2.

2. CHECK VAGUE: Is the "Target Text" generic or unhelpful (e.g. "a visible item", "element", "the button", "result")?
   - YES -> Use the Rationale and Screen Description to generate a specific description. Continue to step 3.
   - NO -> Continue to step 3.

3. CHECK LIST CONTEXT: Is the target an item inside a list, grid, carousel, or search results?
   Use the Screen Description and Rationale to determine this.
   - YES -> Output POSITIONAL with an ordinal reference describing the item's position and collection type.
     Use natural ordinals: "the first search result", "the second card", "the third doctor listing".
     NEVER include the item's specific content (names, times, prices).
   - NO -> Continue to step 4.

4. CHECK TYPE: Is the text specific content that might change (dynamic) or a permanent UI element (stable)?
   - Content (e.g. "Tomorrow 10:00 AM", "Dr. Jane Smith") -> Output DYNAMIC + generic description.
   - Permanent UI (e.g. "Login", "Settings", "Filter") -> Output STABLE.

Format (output ONLY one of these lines):
- STABLE
- POSITIONAL: <ordinal_description>
- DYNAMIC: <generalized_description>

IMPORTANT: Do not output reasoning or explanation. Output ONLY the classification line.

Examples:
(Intent: "Search protein", Target: "Optimum Nutrition WGS", Screen: "Search results with product list", Rationale: "Tapping the first result") -> POSITIONAL: the first search result
(Intent: "Browse doctors", Target: "Dr. Jane Smith", Screen: "Doctor listing with cards", Rationale: "Selecting the top doctor") -> POSITIONAL: the first doctor card
(Intent: "Book", Target: "a visible item", Rationale: "Tapping the second card") -> POSITIONAL: the second card
(Intent: "Book", Target: "Tomorrow 10:00 AM", Screen: "Time slot picker", Rationale: "Selecting first slot") -> POSITIONAL: the first time slot
(Intent: "Play Aditi Shah", Target: "Aditi Shah Sleep Meditation") -> STABLE
(Intent: "Login", Target: "Submit button") -> STABLE
(Intent: "Browse", Target: "Featured banner", Screen: "Home screen with promotional banner") -> DYNAMIC: the promotional banner

Answer:
"""
        try:
            response = await self.__client.aio.models.generate_content(
                model=self.__model, contents=prompt
            )
            text = (response.text or "").strip()

            result = ClassificationResult(description=target)

            for line in text.splitlines():
                line = line.strip()
                if line.startswith("STABLE"):
                    result = ClassificationResult(description=target)
                    break
                elif line.startswith("POSITIONAL:"):
                    desc = line.split("POSITIONAL:", 1)[1].strip()
                    result = ClassificationResult(description=desc, is_positional=True)
                    break
                elif line.startswith("DYNAMIC:"):
                    desc = line.split("DYNAMIC:", 1)[1].strip()
                    result = ClassificationResult(description=desc)
                    break

            self.__cache[cache_key] = result
            return result

        except Exception:
            return ClassificationResult(description=target)

    async def infer_goal_state(self, intent: str) -> str:
        """
        Extracts the visual goal state from the user's intent.
        Example: "Book a flight and stop at payment" -> "Payment Page"
        """
        if not intent or not self.__client:
            return "Goal State"

        cache_key = hashlib.md5(f"GOAL:{intent}".encode(), usedforsecurity=False).hexdigest()
        if cache_key in self.__goal_cache:
            return self.__goal_cache[cache_key]

        prompt = f"""
You are an expert test automation engineer.
Your goal is to identify the final visual state (screen name or element) that confirms the test passed, based on the User Intent.

User Intent: "{intent}"

Instructions:
1. Identify the explicit or implicit "Goal State" (e.g. "Payment Screen", "Confirmation Message", "Search Results").
2. Output ONLY the name of that state/element.
3. Keep it concise (2-4 words).

Examples:
(Intent: "...stop at the payment page") -> Payment Page
(Intent: "Book a consultation...") -> Booking Confirmation
(Intent: "Search for doctors") -> Doctor List
(Intent: "Open Settings") -> Settings Screen

Goal State:
"""
        try:
            response = await self.__client.aio.models.generate_content(
                model=self.__model, contents=prompt
            )
            goal_state = (response.text or "").strip()
            # Clean up potential extra text
            if "Goal State:" in goal_state:
                goal_state = goal_state.split("Goal State:", 1)[1].strip()

            self.__goal_cache[cache_key] = goal_state
            return str(goal_state)

        except Exception as e:
            print(f"DEBUG: Goal Inference Error: {e}")
            return "Goal State"
