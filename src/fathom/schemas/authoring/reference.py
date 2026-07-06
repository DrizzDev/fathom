from __future__ import annotations

from typing import Tuple

from pydantic import Field

from fathom.constants.authoring import AuthoringExampleKind, AuthoringLexiconCategory
from fathom.constants.dialect import DialectName
from fathom.schemas.base import SealedModel


class CommandExample(SealedModel):
    """
    Example that teaches command usage without changing runtime behavior.
    """

    reason: str = Field(min_length=1, description="Why this example is preferred or avoided.")
    command: str = Field(min_length=1, description="Rendered command pattern for the situation.")
    situation: str = Field(min_length=1, description="Evidence situation the example represents.")
    kind: AuthoringExampleKind = Field(description="Whether the example is preferred or avoided.")


class AuthoringLexiconTerm(SealedModel):
    """
    UI terminology hint exposed to authoring prompts.
    """

    term: str = Field(min_length=1, description="UI term the authoring agent may use.")
    category: AuthoringLexiconCategory = Field(description="Category of the UI term.")
    guidance: str = Field(min_length=1, description="When this term describes a visible UI object.")


class AuthoringScenario(SealedModel):
    """
    Task-level example that teaches authoring judgment across commands.
    """

    title: str = Field(min_length=1, description="Short name for the authoring scenario.")
    situation: str = Field(
        min_length=1, description="Evidence pattern represented by the scenario."
    )
    preferred: str = Field(
        min_length=1, description="Preferred authoring decision or Flow pattern."
    )
    avoided: str = Field(min_length=1, description="Less reliable authoring decision to avoid.")
    reason: str = Field(min_length=1, description="Why the preferred decision is more replayable.")


class CommandDoc(SealedModel):
    """
    Command semantics and syntax exposed to authoring prompts.
    """

    name: str = Field(min_length=1, description="Command or node name.")
    purpose: str = Field(min_length=1, description="What the command means.")
    syntax: str = Field(min_length=1, description="Canonical rendered syntax.")
    example: str = Field(min_length=1, description="One valid command example.")

    rules: Tuple[str, ...] = Field(
        default_factory=tuple, description="Command-specific authoring constraints."
    )
    examples: Tuple[CommandExample, ...] = Field(
        default_factory=tuple, description="Few-shot examples for command usage."
    )


class DialectGuide(SealedModel):
    """
    Dialect-level authoring guidance shared by all commands.
    """

    principles: Tuple[str, ...] = Field(description="Core replayability principles.")
    selection: Tuple[str, ...] = Field(description="How to choose stable or dynamic targets.")

    composition: Tuple[str, ...] = Field(description="How to merge or separate commands.")
    completion: Tuple[str, ...] = Field(description="How to decide complete versus partial Flow.")
    scenarios: Tuple[AuthoringScenario, ...] = Field(
        default_factory=tuple,
        description="Few-shot task patterns for authoring judgment.",
    )
    lexicon: Tuple[AuthoringLexiconTerm, ...] = Field(
        default_factory=tuple,
        description="Prompt-only UI terminology for complete replayable command phrasing.",
    )


class AuthoringDialectReference(SealedModel):
    """
    Script dialect reference supplied to the authoring agent.
    """

    name: DialectName = Field(description="Target script dialect.")
    guide: DialectGuide = Field(description="Dialect-level authoring guide.")
    commands: Tuple[CommandDoc, ...] = Field(
        min_length=1, description="Commands supported by the target dialect."
    )
