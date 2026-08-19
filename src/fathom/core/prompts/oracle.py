from __future__ import annotations


class OraclePromptBuilder:
    """
    Builds prompts for the criterion oracle's settled-screen reading.
    """

    __INSTRUCTION = (
        "You are a strict UI test oracle. You receive one criterion and one settled "
        "post-action screenshot of a mobile app. Decide whether the criterion is "
        "observably satisfied on this screen. Judge only what is visible in the "
        "pixels; never assume off-screen or future state. When the screen does not "
        "show enough to decide either way, answer unclear. Report the visible detail "
        "that supports your outcome as evidence."
    )

    def build_system_instruction(self) -> str:
        """
        Build the system instruction shared across all oracle readings.
        """

        return self.__INSTRUCTION

    def build_user_prompt(self, *, criterion: str) -> str:
        """
        Build the per-reading user prompt naming the criterion under judgement.
        """

        return f"Criterion: {criterion}"
