from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

from google import genai
from google.oauth2 import service_account

from fathom.settings.env import FathomSettings


class TargetClassifier:
    """
    Uses an LLM to classify UI targets as 'Stable' or 'Dynamic'
    and generates generalized descriptions for dynamic elements.
    Uses generic text generation, not function calling.
    """

    def __init__(self) -> None:
        self.__settings = FathomSettings()
        self.__client: Optional[genai.Client] = None
        self.__model = "gemini-2.5-flash-lite"
        self.__initialize_client()
        self.__cache: Dict[str, str] = {}

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

    async def classify_and_generalize(self, target: str, intent: str, rationale: str = "") -> str:
        """
        Determines if a target is dynamic.
        Returns the ORIGINAL target if stable (or matched in intent).
        Returns a GENERALIZED description if dynamic.
        """
        if not target or not self.__client:
            return target

        cache_key = hashlib.md5(
            f"{target}:{intent}:{rationale}".encode(), usedforsecurity=False
        ).hexdigest()
        if cache_key in self.__cache:
            return self.__cache[cache_key]

        prompt = f"""
You are an expert mobile test automation engineer.
Your goal is to decide if a specific UI element text should be preserved (STABLE) or generalized (DYNAMIC) for a test script.

User Intent: "{intent}"
Target Text: "{target}"
Rationale: "{rationale}"

Instructions:
1. CHECK INTENT: Is the "Target Text" explicitly mentioned or strongly implied by the "User Intent"?
   - YES (e.g. Intent="Play Aditi Shah...", Target="Aditi Shah Sleep Meditation") -> Output STABLE.
   - NO -> Continue to step 2.

2. CHECK VAGUE: Is the "Target Text" generic or unhelpful (e.g. "a visible item", "element", "the button", "result")?
   - YES -> Use the Rationale to generate a specific, descriptive name (e.g. "the second card", "the apply button", "the doctor profile"). Output DYNAMIC: <specific_description>.
   - NO -> Continue to step 3.

3. CHECK TYPE: Is the text specific content that might change (dynamic) or a permanent UI element (stable)?
   - Content (e.g. "Tomorrow 10:00 AM", "Search Output") -> Output DYNAMIC + generic description (e.g. "the first time slot").
   - Permanent UI (e.g. "Login", "Settings", "Filter") -> Output STABLE.

4. DYNAMIC OUTPUT: If dynamic/vague, provide a clear, natural language description.

Format:
- STABLE
- DYNAMIC: <generalized_description>

IMPORTANT: Do not output reasoning or explanation. Output ONLY the classification line.

Examples:
(Intent: "Book", Target: "a visible item", Rationale: "Tapping the second card") -> DYNAMIC: the second card
(Intent: "Book", Target: "Tomorrow 10:00 AM") -> DYNAMIC: the first time slot
(Intent: "Play Aditi Shah", Target: "Aditi Shah Sleep Meditation") -> STABLE
(Intent: "Browse", Target: "Aditi Shah") -> DYNAMIC: the second item

Answer:
"""
        try:
            response = await self.__client.aio.models.generate_content(
                model=self.__model, contents=prompt
            )
            text = (response.text or "").strip()

            # Parse response - look for the classification line
            final_description = target
            found = False

            for line in text.splitlines():
                line = line.strip()
                if line.startswith("STABLE"):
                    final_description = target
                    found = True
                    break
                elif line.startswith("DYNAMIC:"):
                    final_description = line.split("DYNAMIC:", 1)[1].strip()
                    found = True
                    break

            if not found:
                # If no clear prefix found, check if the whole text is likely the description?
                # Safer to fallback to target unless we are sure.
                pass

            self.__cache[cache_key] = final_description
            return final_description

        except Exception:
            return target

    async def infer_goal_state(self, intent: str) -> str:
        """
        Extracts the visual goal state from the user's intent.
        Example: "Book a flight and stop at payment" -> "Payment Page"
        """
        if not intent or not self.__client:
            return "Goal State"

        cache_key = hashlib.md5(f"GOAL:{intent}".encode(), usedforsecurity=False).hexdigest()
        if cache_key in self.__cache:
            return self.__cache[cache_key]

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

            self.__cache[cache_key] = goal_state
            return str(goal_state)

        except Exception as e:
            print(f"DEBUG: Goal Inference Error: {e}")
            return "Goal State"
