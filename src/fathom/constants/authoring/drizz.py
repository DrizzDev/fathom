from __future__ import annotations

from typing import Tuple

from fathom.constants.authoring import AuthoringExampleKind
from fathom.constants.authoring.lexicon import UI_LEXICON
from fathom.schemas.authoring.reference import (
    AuthoringScenario,
    CommandDoc,
    CommandExample,
    DialectGuide,
)

DRIZZ_GUIDE = DialectGuide(
    principles=(
        "Author a replayable script, not a literal transcript of every attempted step.",
        "Every command must be grounded in supplied evidence and valid for the target dialect.",
        "Prefer clear, complete command phrases that a future replay can execute without hidden context.",
        "Use recorded values for value-bearing commands; never invent values or assertions.",
    ),
    selection=(
        "Use an exact target when evidence identifies a stable semantic UI element.",
        "Use a relative or dynamic target when evidence shows selection by order, query, filter, condition, or runtime context.",
        "Combine stable and dynamic context when both are needed to make the target replayable.",
        "Avoid raw trace labels when they are generic, incomplete, or only meaningful to the recorder.",
        "Author target text from target.anchors and target.structure before using target.claim.",
        "Use target.claim only when target.claim.verified is true; otherwise treat it as recorder narrative.",
        "Treat target.name, target.export, target.generalized, and target.scroll as candidate aliases; choose the clearest replay phrase instead of concatenating aliases.",
        "Treat placeholder or hint text as evidence for locating a field, not necessarily as the field name users should replay.",
        "Use container wording only when it names a real visible UI region, such as suggestions list, product grid, dialog, bottom sheet, or form.",
    ),
    composition=(
        "Merge repeated attempts that serve one user-level purpose into one replayable command when the evidence supports it.",
        "Keep separate commands when evidence shows separate user-level purposes or different command semantics.",
        "Represent conditional behavior only when a condition and its branch were recorded.",
        "Do not turn internal memory updates, diagnostics, or observations into script commands.",
    ),
    completion=(
        "Return a complete Flow only when evidence proves the user-level goal was completed.",
        "Return a partial Flow when the run stopped, evidence is insufficient, or completion cannot be asserted faithfully.",
        "A validation command must assert a meaningful visible or data state, not merely repeat a generic control label.",
    ),
    scenarios=(
        AuthoringScenario(
            title="stable target selection",
            situation=(
                "A step taps a stable screen element whose role and label are both recorded."
            ),
            preferred="Use the semantic target and role, such as Tap on Login button.",
            avoided="Use a recorder-only phrase such as Tap on target or Tap on button.",
            reason="Stable targets replay better when the command names the actual UI object.",
        ),
        AuthoringScenario(
            title="runtime result selection",
            situation=(
                "A step chooses a result because of a prior query, ordering, filter, or runtime condition."
            ),
            preferred=(
                "Combine the relative choice with the selection context, such as "
                "Tap on first matching result in the recorded results list."
            ),
            avoided="Use a context-free target such as Tap on item.",
            reason="Dynamic selections need the reason they were selected, not only the tapped widget.",
        ),
        AuthoringScenario(
            title="repeated attempts",
            situation=(
                "Several consecutive steps in one episode scroll or tap while searching for the same state."
            ),
            preferred=(
                "Represent the episode as one replayable movement or target-seeking command when the "
                "stop state is proven; otherwise keep only faithful progress and mark the Flow partial."
            ),
            avoided="Emit every repeated attempt as independent user-level commands.",
            reason="Retries are execution mechanics; scripts should capture the user-level purpose.",
        ),
        AuthoringScenario(
            title="terminal completion",
            situation="The verifier supplied completion assertions after execution ended.",
            preferred=(
                "End a complete Flow with a CheckNode whose subject and kind match those assertions "
                "and whose assertion_ids cite the assertion identifiers."
            ),
            avoided="Invent a terminal validation from the last tapped control or a broad observation.",
            reason="Completion assertions are the trustworthy terminal proof for final Validate commands.",
        ),
        AuthoringScenario(
            title="captured value",
            situation="A command records a successful value capture for a named variable.",
            preferred="Use Store with the captured runtime value and recorded variable name.",
            avoided="Store the capture description or a value inferred from surrounding text.",
            reason="Value-bearing commands must preserve recorded values exactly.",
        ),
        AuthoringScenario(
            title="role-qualified control",
            situation=(
                "A control label is recorded, and evidence also identifies the role or purpose of "
                "the control."
            ),
            preferred="Include the role visible on screen, such as button, field, dropdown, row, or card.",
            avoided="Use only the raw label or placeholder text when the role is known.",
            reason="A replayable command should identify the UI role of the object being used.",
        ),
        AuthoringScenario(
            title="scroll stop state",
            situation="A scroll episode is meant to reveal a section, result, or condition.",
            preferred=(
                "Use a stop target that names the visible state, such as "
                'Scroll down until "Reviews section is visible".'
            ),
            avoided='Use a fragment-only target, such as Scroll down until "Reviews".',
            reason="The stop condition must tell replay what visible state ends the scroll.",
        ),
        AuthoringScenario(
            title="conditional guard wording",
            situation="A branch runs because a visible UI condition appeared.",
            preferred="Name the concrete condition, such as IF account picker dialog is visible.",
            avoided="Use a recorder-only condition, such as IF overlay is visible.",
            reason="Conditional guards should explain the UI state that makes the branch replayable.",
        ),
        AuthoringScenario(
            title="alias selection",
            situation=(
                "A step records both a placeholder-like raw name and a cleaner exported role for "
                "the same target."
            ),
            preferred="Use one clear replay target that names the field role once.",
            avoided=(
                'Combine aliases, such as Type "search query" into Search by Keyword under Search box.'
            ),
            reason=(
                "Multiple recorded aliases describe the same object; concatenating them makes the "
                "command harder to replay."
            ),
        ),
        AuthoringScenario(
            title="dynamic result selection",
            situation=(
                "A step selects a result because of query, order, filter, or another runtime "
                "condition."
            ),
            preferred=(
                "Name the visible item plus the replay context, such as Tap on the first matching "
                "result card in the results grid."
            ),
            avoided="Use only the title or only a broad area, such as Tap on result card.",
            reason=(
                "Dynamic choices need both what was tapped and why that instance was the right one."
            ),
        ),
        AuthoringScenario(
            title="unverified target claim",
            situation=(
                "The planner target phrase does not match any evidence-owned visual or "
                "accessibility anchor."
            ),
            preferred=(
                "Use verified anchors and structural context, or return a partial Flow when the "
                "target cannot be identified replayably."
            ),
            avoided="Use target.claim.text as the command target just because it was recorded.",
            reason="Planner claims are useful context, but only verified anchors are target truth.",
        ),
        AuthoringScenario(
            title="dynamic address control",
            situation=(
                "A tap opens or changes delivery location, and the recorded target claim is the "
                "current address, user data, ETA, or a full content-description sentence."
            ),
            preferred=(
                "Use the actual visible UI role and purpose from the screen, such as a dropdown, "
                "field, row, card, chip, button, icon, tab, or menu item."
            ),
            avoided=(
                "Use the runtime value, such as Tap on Selected address is Manhattan, A108 Adams "
                "St, New York, NY 10007, USA Delivering in null minutes."
            ),
            reason=(
                "Addresses, user data, and ETA text are runtime values; replay should identify the "
                "control by its visible role and purpose, not the current value."
            ),
        ),
        AuthoringScenario(
            title="incomplete evidence",
            situation="Execution stopped before the user-level goal was proven.",
            preferred="Set Flow.partial to true and return only commands supported by executed evidence.",
            avoided="Publish a complete Flow by guessing missing follow-up commands or assertions.",
            reason="A partial script is useful and honest; invented completion is not.",
        ),
    ),
    lexicon=UI_LEXICON,
)


DRIZZ_COMMANDS: Tuple[CommandDoc, ...] = (
    CommandDoc(
        name="open_app",
        syntax="OPEN_APP: <package>",
        example="OPEN_APP: com.android.chrome",
        purpose="Launch the target application.",
        rules=("Use the target package recorded by launch evidence.",),
    ),
    CommandDoc(
        name="tap",
        syntax="Tap on <target>",
        example="Tap on Login button",
        purpose="Tap a replayable UI target.",
        rules=(
            "Target must identify what replay should tap.",
            "Prefer semantic labels over recorder-only labels when evidence supports the semantic label.",
            "Include the UI role when evidence identifies it, such as button, input field, tab, or product card.",
            "Do not concatenate multiple aliases for the same target; choose the clearest one.",
            "For dynamic results, include the visible item, relative choice, and selection context when evidence supports them.",
        ),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                situation="A search suggestion was chosen after typing a query.",
                command='Tap on the first "search query" suggestion in the suggestions list',
                reason="The target combines visible text, relative position, and the visible list role.",
            ),
            CommandExample(
                command="Tap on product card",
                kind=AuthoringExampleKind.AVOID,
                reason="The command omits the available selection context needed for reliable replay.",
                situation="A recorded target only names a generic card while the evidence has query, order, or product context.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command="Tap on Buy Now button",
                reason="The command names both the visible label and the UI role.",
                situation="A button label is recorded with enough evidence to identify it as a button.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command="Tap on location dropdown",
                reason="The command names the actual visible UI role and purpose instead of the current address value.",
                situation="A delivery/location control was tapped and the screenshot shows a dropdown affordance.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                situation="A result was selected because it matched a runtime condition.",
                command="Tap on the first matching result card in the results grid",
                reason="The command carries the dynamic condition that made the selected card replayable.",
            ),
            CommandExample(
                command="Tap on Buy Now",
                kind=AuthoringExampleKind.AVOID,
                reason="The command leaves the UI role implicit even though evidence can name it.",
                situation="A button label is recorded with enough evidence to identify it as a button.",
            ),
            CommandExample(
                command=(
                    "Tap on Selected address is Manhattan, A108 Adams St, New York, NY 10007, "
                    "USA Delivering in null minutes"
                ),
                kind=AuthoringExampleKind.AVOID,
                reason="The command copies a dynamic runtime value and content-description sentence.",
                situation="A delivery/location control was tapped to change the address.",
            ),
        ),
    ),
    CommandDoc(
        name="type",
        syntax='Type "<value>" into <field>',
        example='Type "John" into name field',
        purpose="Enter a value into a focused field.",
        rules=("Use the recorded typed value and the field it was entered into.",),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                situation="A field has raw placeholder text plus a cleaner semantic role.",
                command='Type "search query" into search field',
                reason="The replay target uses the field role instead of copying placeholder text.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                situation="A field has multiple aliases for the same object.",
                command='Type "search query" into Search by Keyword under Search box',
                reason="The command concatenates aliases and reads like recorder metadata.",
            ),
        ),
    ),
    CommandDoc(
        name="scroll",
        example="Scroll down",
        syntax="Scroll <direction>",
        purpose="Move through content in a direction.",
        rules=(
            "Use plain scroll when the evidence records movement but no concrete stop target.",
            "Direction is page-motion direction, not finger-motion direction.",
        ),
        examples=(
            CommandExample(
                command="Scroll down",
                kind=AuthoringExampleKind.PREFERRED,
                reason="Plain movement is faithful when no concrete stop target is proven.",
                situation="The run performed several scroll gestures while searching within one list.",
            ),
        ),
    ),
    CommandDoc(
        name="scroll_until",
        syntax="Scroll <direction> until <target>",
        example='Scroll down until "Ratings and Reviews section"',
        purpose="Move through content until a concrete target is found.",
        rules=(
            "Use only when evidence records a concrete target that replay should stop at.",
            "The target must be complete enough to tell replay what should become visible.",
            "Prefer section, row, result, field, dialog, or condition wording over standalone fragments.",
            "If the stop target is a condition, state the visible object and condition, not only the condition text.",
        ),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command='Scroll down until "Reviews section is visible"',
                reason="The stop target is a complete visible state.",
                situation="Scrolling stopped when a concrete section became visible.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command='Scroll down until "a matching result card is visible"',
                reason="The stop target names the visible object and the condition that ends the scroll.",
                situation="Scrolling searched for an item satisfying a runtime condition.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                command='Scroll down until "Reviews"',
                situation="Several scroll attempts mention broad list context.",
                reason="The stop target is a fragment rather than a visible stopping state.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                command='Scroll down until "matches condition"',
                reason="The condition omits the visible object that replay should find.",
                situation="Scrolling searched for an item satisfying a runtime condition.",
            ),
        ),
    ),
    CommandDoc(
        name="wait",
        purpose="Wait for time or a state.",
        example='Wait until "search results are visible"',
        syntax="Wait for <seconds> seconds or Wait until <subject>",
        rules=(
            "Prefer state-based waits over fixed time when evidence records a state.",
            "When waiting for UI content, make the quoted subject a complete ready state, not a bare container or label fragment.",
        ),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command='Wait until "results list is loaded"',
                reason="The subject names the ready state that ends the wait.",
                situation="A prior command loads a list whose content is needed by the next action.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                command='Wait until "results list"',
                reason="The subject is only a container name and does not state the awaited condition.",
                situation="A wait is bounded by UI content becoming available.",
            ),
        ),
    ),
    CommandDoc(
        name="store",
        syntax="Store <value> as <name>",
        example="Store 123456 as OTP code",
        purpose="Store a captured runtime value under a variable name.",
        rules=(
            "Use the captured value, not the capture subject.",
            "Emit Store only for successful captured values recorded in evidence.",
        ),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command="Store Rs. 76 as item.price",
                reason="The command stores the actual runtime value.",
                situation="Evidence captured a runtime value for a requested variable.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                command="Store product price as item.price",
                situation="Evidence captured a runtime value for a requested variable.",
                reason="The command stores a description instead of the captured value.",
            ),
        ),
    ),
    CommandDoc(
        name="validate",
        syntax="Validate <subject> <state>",
        example="Validate checkout screen is visible",
        purpose="Assert a meaningful UI or data state.",
        rules=(
            "Subject must be a meaningful state or object proven by evidence.",
            "Do not validate generic controls unless the control itself is the user-level state.",
            "Include the role or state being asserted so the command reads as a complete assertion.",
        ),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command="Validate phone number input field is visible",
                reason="The assertion names a meaningful screen state.",
                situation="A destination screen exposes a distinctive input field.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                command="Validate dismissal option is visible",
                situation="Only a generic dismissal option was tapped during an overlay.",
                reason="The assertion repeats a generic control instead of the user-level state.",
            ),
        ),
    ),
    CommandDoc(
        name="branch",
        syntax="IF <condition> { <commands> }",
        purpose="Represent a recorded conditional branch.",
        example="IF Permission dialog is visible { Tap on Allow button }",
        rules=(
            "Condition must be recorded in evidence.",
            "Branch body must contain only commands that ran under that condition.",
        ),
    ),
    CommandDoc(
        name="back",
        syntax="PRESS_DEVICE_BACK_BUTTON",
        example="PRESS_DEVICE_BACK_BUTTON",
        purpose="Press system device back button.",
        rules=("Use only for a recorded system-back action.",),
    ),
)
