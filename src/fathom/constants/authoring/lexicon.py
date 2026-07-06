from __future__ import annotations

from typing import Tuple

from fathom.constants.authoring import AuthoringLexiconCategory
from fathom.schemas.authoring.reference import AuthoringLexiconTerm

UI_LEXICON: Tuple[AuthoringLexiconTerm, ...] = (
    AuthoringLexiconTerm(
        term="button",
        category=AuthoringLexiconCategory.CONTROL,
        guidance="Use for tappable controls that trigger an action or submit a choice.",
    ),
    AuthoringLexiconTerm(
        term="icon button",
        category=AuthoringLexiconCategory.CONTROL,
        guidance="Use for tappable icon-only controls such as close, back, search, or menu.",
    ),
    AuthoringLexiconTerm(
        term="tab",
        category=AuthoringLexiconCategory.CONTROL,
        guidance="Use for selectable navigation items inside a tab bar.",
    ),
    AuthoringLexiconTerm(
        term="checkbox",
        category=AuthoringLexiconCategory.CONTROL,
        guidance="Use for binary selectable options.",
    ),
    AuthoringLexiconTerm(
        term="search field",
        category=AuthoringLexiconCategory.FIELD,
        guidance="Use for a unique search input; avoid copying placeholder text as the field name.",
    ),
    AuthoringLexiconTerm(
        term="input field",
        category=AuthoringLexiconCategory.FIELD,
        guidance="Use for editable text fields when a more specific role is unavailable.",
    ),
    AuthoringLexiconTerm(
        term="suggestions list",
        category=AuthoringLexiconCategory.CONTAINER,
        guidance="Use for query suggestions shown under or near a search field.",
    ),
    AuthoringLexiconTerm(
        term="product grid",
        category=AuthoringLexiconCategory.CONTAINER,
        guidance="Use for a grid of product cards or tiled shopping results.",
    ),
    AuthoringLexiconTerm(
        term="product card",
        category=AuthoringLexiconCategory.CONTENT,
        guidance="Use for a tappable product tile containing product details, image, price, or rating.",
    ),
    AuthoringLexiconTerm(
        term="list row",
        category=AuthoringLexiconCategory.CONTENT,
        guidance="Use for one repeated row inside a vertical list.",
    ),
    AuthoringLexiconTerm(
        term="dialog",
        category=AuthoringLexiconCategory.CONTAINER,
        guidance="Use for modal overlays that block the underlying screen until handled.",
    ),
    AuthoringLexiconTerm(
        term="bottom sheet",
        category=AuthoringLexiconCategory.CONTAINER,
        guidance="Use for a panel that slides from the bottom and overlays the current screen.",
    ),
    AuthoringLexiconTerm(
        term="banner",
        category=AuthoringLexiconCategory.FEEDBACK,
        guidance="Use for non-modal messages or promotional strips across part of the screen.",
    ),
    AuthoringLexiconTerm(
        term="toast",
        category=AuthoringLexiconCategory.FEEDBACK,
        guidance="Use for short transient feedback messages.",
    ),
)
