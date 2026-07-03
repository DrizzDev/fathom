from __future__ import annotations

from typing import Final

# The grammar below is the Lark dialect, not Python. The legend at its top explains
# the notation; each rule is annotated with the exact line the renderer emits for it.
DRIZZ_GRAMMAR: Final[str] = r"""
// =============================================================================
// How to read this grammar
//   lowercase    a rule (parser production)      UPPERCASE   a terminal (token)
//   "literal"    exact text, dropped from tree   -> alias    names a rule branch
//   a+  a*  a?   one-or-more / any / optional     |           alternative
//   NAME.N       terminal with lexer priority N (higher beats the FREEWORD regex)
//   _name        leading underscore: filtered out of the parse tree
//   %ignore      tokens skipped between others (inline whitespace)
// =============================================================================

// --- Entry point: a script is newline-separated statements ------------------
start: _NL* statement (_NL+ statement)* _NL*

?statement: command | if_block

// An IF block is one condition line, then a braced body of leaf commands.
// Drizz itself supports nested IF/ELSE; this grammar parses only what the renderer
// emits, and we emit a single executed branch because a recording evidences just the
// path that ran (no ELSE synthesis). Widen this rule when we render those forms.
//   IF Overlay is visible
//   {
//       Tap on "Skip"
//   }
if_block: IF_LINE _NL+ "{" _NL+ command (_NL+ command)* _NL+ "}"

?command: open_app
        | tap
        | text
        | scroll_until
        | scroll
        | wait_full
        | wait_duration
        | wait_subject
        | back
        | kill
        | clear
        | minimise
        | set_gps
        | store
        | map_action
        | validate_many
        | validate_one

// --- Action commands --------------------------------------------------------
open_app: "OPEN_APP" ":" PACKAGE                 // OPEN_APP: com.example
        | "OPEN_APP" PACKAGE                     // OPEN_APP com.example (both forms accepted)
tap: "Tap" "on" nl_target                        // Tap on Login CTA
   | "Tap" "the" nl_target                        // Tap the cart icon
   | "Tap" nl_target                              // Tap Add under Snacks header
text: "Type" STRING ("in" | "into") nl_target    // Type "John" into the name field
scroll: "Scroll" DIRECTION                        -> scroll_plain    // Scroll down
      | "Scroll" DIRECTION "by" INT "%"            -> scroll_percent  // Scroll down by 30%
      | "Scroll" DIRECTION "inside" nl_target      -> scroll_inside   // Scroll down inside product list
      | "Scroll" DIRECTION "inside" nl_target "until" STRING -> scroll_inside_until
scroll_until: "Scroll" DIRECTION "until" STRING  // Scroll down until "Proceed to Pay"
map_action: "MAP_ACTION" "Tap" "on" nl_target    // MAP_ACTION Tap on the location pin

// --- Lifecycle & synchronization --------------------------------------------
wait_full: "Wait" INT "seconds" "for" subject    // Wait 5 seconds for page content to load
wait_duration: "Wait" "for" INT "seconds"        // Wait for 5 seconds
wait_subject: "Wait" "until" STRING              // Wait until "Spinner"
back: "PRESS_DEVICE_BACK_BUTTON"                 // PRESS_DEVICE_BACK_BUTTON
kill: "KILL_APP"                                 // KILL_APP
clear: "CLEAR_APP"                               // CLEAR_APP
minimise: MINIMISE                               // MINIMISE_APP
// SET_GPS(latitude=12.34, longitude=-56.78)
set_gps: "SET_GPS" "(" "latitude" "=" SIGNED_NUMBER "," "longitude" "=" SIGNED_NUMBER ")"

// --- Data capture & assertions ----------------------------------------------
store: "Store" capture "as" NAME                 // Store 499 as savedTotal
validate_one: "Validate" subject STATE           // Validate home is visible
// Validate the following are visible: 1. "Price" 2. "Discount"
validate_many: "Validate" FOLLOWING GROUP_STATE ":" numbered+
numbered: INT "." STRING                          // 1. "Price"

// --- Targets: unquoted natural-language text to end of line -----------------
// tap / type-field / map targets are plain natural-language phrases; the
// renderer folds any ordinal ("the first ...") or container ("... under ...")
// straight into this text, so the parser reads it as one free-text run.
//   Login CTA            the first Add under Snacks header
// A quoted STRING alternative lets recorded text with reserved words, punctuation,
// or whitespace runs round-trip verbatim; the renderer quotes only when needed.
nl_target: FREEWORD+ | STRING

// --- Free-text tails: a run of words ending at the rule's terminator --------
//   capture  ends at the "as" keyword            (Store <capture> as NAME)
//   subject  ends at a STATE phrase              (<subject> is visible)
capture: FREEWORD+ | STRING
subject: FREEWORD+ | STRING

// --- Terminals --------------------------------------------------------------
// A whole IF header line, captured as one token so a condition like
// "Overlay is visible" stays free text and is never lexed as a STATE phrase.
IF_LINE: /IF [^\n]+/

// These phrases outrank the FREEWORD regex (.10/.9 > default 0) so the lexer
// prefers them where a free-text subject ends: ...home |is visible -> STATE.
FOLLOWING.10: "the following are"
STATE.9: "is visible" | "is present" | "is enabled" | "is disabled"
GROUP_STATE.9: "visible" | "present" | "enabled" | "disabled"

DIRECTION: "up" | "down" | "left" | "right"
MINIMISE: "MINIMISE_APP" | "MINIMIZE_APP"    // British spelling canonical; both parse
PACKAGE: /[A-Za-z0-9_.]+/                    // e.g. com.android.chrome
NAME: /[A-Za-z0-9_]+/                        // stored-variable identifier
// A selector or value in any Drizz delimiter; priority .5 wins over FREEWORD at a quote.
STRING.5: /"[^"]*"/ | /'[^']*'/ | /`[^`]*`/  // "double" | 'single' | `backtick`
FREEWORD: /[^\s{}";]+/                       // one word of an unquoted free-text tail (allows ':')
INT: /[0-9]+/

_NL: /(\r?\n)+/                              // statement separator (filtered)

%import common.SIGNED_NUMBER                 // GPS coordinate, e.g. -56.78
%import common.WS_INLINE                     // spaces/tabs between tokens
%ignore WS_INLINE
"""
