from __future__ import annotations

from typing import Optional, Tuple

from pydantic import Field, model_validator
from tests.support.vision.models import LabelSource

from fathom.constants.assessment import VisualVerdict
from fathom.constants.success import SuccessKind
from fathom.schemas.base.common import NonBlank, SealedModel

AMAZON_SHOPPING = "com.amazon.mShop.android.shopping"
AMAZON_MUSIC = "com.amazon.mp3"
MEESHO = "com.meesho.supply"
LAUNCHER = "com.android.launcher"
PERMISSION_UI = "com.google.android.permissioncontroller"
CHATGPT = "com.openai.chatgpt"
CONTROLLED = "controlled.capture"

AMAZON_RUN = "assets/traces/2026-07-31/7f64d39f10e440d5ad9334acf15094ce"
AMAZON_MUSIC_RUN = "assets/traces/2026-07-31/b2282f11417c438da2c8d0d3059b02f8"
AMAZON_SIGNIN_RUN = "assets/traces/2026-07-31/aa12a78e5a054aa2baf931e4c83b1f89"
CHATGPT_AD_RUN = "assets/traces/2026-07-31/3b4fd954ab4a4841937ed7c3a718f5a9"
MEESHO_RUN = "assets/traces/2026-07-29/4ee2b518a161418fba58feaae2952844"
MEESHO_PERMISSION_RUN = "assets/traces/2026-07-29/bdcb0d1e146b4bd696d3b300d543c629"
CONTROLLED_SOURCE = "controlled/programmatic"


class VisionCase(SealedModel):
    """
    A data-only visual-assessment case: real pixels, the threaded assertion, and its typed ground-truth.
    """

    name: NonBlank = Field(description="Stable case identifier used in reports.")
    screenshot: NonBlank = Field(
        description="File name under screens/ holding the captured pixels."
    )
    app: NonBlank = Field(description="Application exercised by the screen.")
    package: NonBlank = Field(description="Foreground package truth on the captured screen.")
    assertion: NonBlank = Field(
        description="Exact assertion threaded into the production VLM prompt."
    )
    goal_kind: SuccessKind = Field(description="Success kind of the active goal driving this turn.")
    expected_verdict: VisualVerdict = Field(
        description="Oracle verdict a correct model must return."
    )
    truth_satisfied: bool = Field(
        description="Oracle: whether the observed goal is genuinely satisfied here."
    )
    authority_package: Optional[str] = Field(
        default=None,
        description="Bound TargetAuthority package for the goal, or None when unbound.",
    )
    scenario: NonBlank = Field(description="Screen type or adversarial trap this case represents.")
    provenance: NonBlank = Field(
        description="Source of the pixels: a trace run path or a controlled tag."
    )
    label_source: LabelSource = Field(
        description="Whether the ground-truth was assigned by human or programmatic means."
    )
    critical_negative: bool = Field(
        description="Whether an effective false positive here blocks the Slice-3 cutover."
    )

    @property
    def expected_admission(self) -> bool:
        """
        Whether the effective shadow rule should advance the goal on this case.
        """

        return self.goal_kind is SuccessKind.OBSERVED and self.truth_satisfied

    @model_validator(mode="after")
    def __consistent_oracle(self) -> "VisionCase":
        """
        Keep the expected verdict consistent with the satisfaction oracle for observed goals.
        """

        if self.goal_kind is SuccessKind.OBSERVED:
            expected = (
                VisualVerdict.SATISFIED if self.truth_satisfied else VisualVerdict.NOT_SATISFIED
            )
            if self.expected_verdict is not expected:
                raise ValueError(
                    "Observed case expected_verdict must match its truth_satisfied oracle."
                )
        return self


AMAZON_CASES: Tuple[VisionCase, ...] = (
    VisionCase(
        name="launcher_home",
        screenshot="launcher_home.png",
        app="Launcher",
        package=LAUNCHER,
        assertion="The Amazon shopping app is open in the foreground.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=AMAZON_SHOPPING,
        scenario="Clean negative: launcher home with no Amazon surface present.",
        provenance=AMAZON_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="icon_visible_not_open",
        screenshot="shopping_folder_amazon_icon.png",
        app="Launcher",
        package=LAUNCHER,
        assertion="The Amazon shopping app is open in the foreground.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=AMAZON_SHOPPING,
        scenario="False positive: Amazon icon prominent in an open folder, app not launched.",
        provenance=AMAZON_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="amazon_versus_amazon_music",
        screenshot="app_drawer_amazon_and_music.png",
        app="Launcher",
        package=LAUNCHER,
        assertion="The Amazon shopping app, not Amazon Music, is open in the foreground.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=AMAZON_SHOPPING,
        scenario="Wrong-target false positive: Amazon and Amazon Music icons adjacent in the drawer.",
        provenance=AMAZON_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="amazon_home_open",
        screenshot="amazon_home.png",
        app="Amazon",
        package=AMAZON_SHOPPING,
        assertion="The Amazon shopping app home screen is open.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.SATISFIED,
        truth_satisfied=True,
        authority_package=AMAZON_SHOPPING,
        scenario="Clean positive: the Amazon home screen is genuinely in the foreground.",
        provenance=AMAZON_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=False,
    ),
    VisionCase(
        name="suggestions_not_results",
        screenshot="amazon_search_suggestions_ghar_soap.png",
        app="Amazon",
        package=AMAZON_SHOPPING,
        assertion="The Amazon search results page for 'ghar soap' is displayed with a list of purchasable product cards.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=AMAZON_SHOPPING,
        scenario="False positive: autocomplete suggestions with thumbnails are not the results grid.",
        provenance=AMAZON_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="search_text_entered",
        screenshot="amazon_search_suggestions_ghar_soap.png",
        app="Amazon",
        package=AMAZON_SHOPPING,
        assertion="The Amazon search field contains the typed query 'ghar soap'.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.SATISFIED,
        truth_satisfied=True,
        authority_package=AMAZON_SHOPPING,
        scenario="Clean positive: the assertion names exactly the observable typed-query state.",
        provenance=AMAZON_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=False,
    ),
    VisionCase(
        name="cart_not_confirmed",
        screenshot="amazon_home.png",
        app="Amazon",
        package=AMAZON_SHOPPING,
        assertion="A 'ghar soap' product has been added to the cart and an add-to-cart confirmation is visible.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=AMAZON_SHOPPING,
        scenario="False positive: cart badge shows pre-existing items, no 'ghar soap' add confirmation.",
        provenance=AMAZON_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="amazon_results_not_on_home",
        screenshot="amazon_home.png",
        app="Amazon",
        package=AMAZON_SHOPPING,
        assertion="The Amazon search results for 'ghar soap' are displayed with product listings.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=AMAZON_SHOPPING,
        scenario="Long-but-progressing: correct app and home reached, but the results goal is not yet met.",
        provenance=AMAZON_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="amazon_login_absent",
        screenshot="amazon_home_signin_promo.png",
        app="Amazon",
        package=AMAZON_SHOPPING,
        assertion="A login screen requiring the user to enter sign-in credentials is displayed.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=AMAZON_SHOPPING,
        scenario="Login absent: a dismissible 'Sign in' promo banner on home is not a login screen.",
        provenance=AMAZON_SIGNIN_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="amazon_music_wrong_app",
        screenshot="amazon_music_interstitial.png",
        app="Amazon Music",
        package=AMAZON_MUSIC,
        assertion="The Amazon shopping app is open in the foreground.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=AMAZON_SHOPPING,
        scenario="Wrong application: Amazon Music Unlimited opened instead of Amazon shopping.",
        provenance=AMAZON_MUSIC_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
)


MEESHO_CASES: Tuple[VisionCase, ...] = (
    VisionCase(
        name="meesho_results_present",
        screenshot="meesho_search_results.png",
        app="Meesho",
        package=MEESHO,
        assertion="The Meesho search results for 'ghar soap' are displayed with a grid of purchasable product cards.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.SATISFIED,
        truth_satisfied=True,
        authority_package=MEESHO,
        scenario="Clean positive: a genuine Meesho results grid with product cards and prices.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=False,
    ),
    VisionCase(
        name="meesho_results_repeated_different_goal",
        screenshot="meesho_search_results.png",
        app="Meesho",
        package=MEESHO,
        assertion="A single product detail page for one ghar soap product is open.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=MEESHO,
        scenario="Repeated screen under a different goal: the results grid is not a product detail page.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="meesho_suggestions_not_results",
        screenshot="meesho_search_typed_suggestions.png",
        app="Meesho",
        package=MEESHO,
        assertion="The Meesho search results grid for 'ghar soaps' is displayed with product cards.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=MEESHO,
        scenario="False positive: a typed-query autocomplete list is not the results grid.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="meesho_search_text_entered",
        screenshot="meesho_search_typed_suggestions.png",
        app="Meesho",
        package=MEESHO,
        assertion="The Meesho search field contains the typed query 'Ghar soaps'.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.SATISFIED,
        truth_satisfied=True,
        authority_package=MEESHO,
        scenario="Clean positive: query typed into the search field but not yet submitted.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=False,
    ),
    VisionCase(
        name="meesho_product_detail_open",
        screenshot="meesho_product_detail.png",
        app="Meesho",
        package=MEESHO,
        assertion="A product detail page for a ghar soap product is open.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.SATISFIED,
        truth_satisfied=True,
        authority_package=MEESHO,
        scenario="Visual goal already satisfied on arrival: a single-product detail page.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=False,
    ),
    VisionCase(
        name="meesho_detail_not_results",
        screenshot="meesho_product_detail.png",
        app="Meesho",
        package=MEESHO,
        assertion="A grid of multiple search-result product cards is displayed.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=MEESHO,
        scenario="Product grid versus details: a detail page is not a multi-card results grid.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="rating_wrong_attribute",
        screenshot="meesho_product_detail.png",
        app="Meesho",
        package=MEESHO,
        assertion="The product's customer rating is at least 4.2 out of 5.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=MEESHO,
        scenario="Wrong attribute above threshold: a 4.9 supplier rating must not satisfy a customer-rating predicate.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="meesho_product_not_added",
        screenshot="meesho_product_detail.png",
        app="Meesho",
        package=MEESHO,
        assertion="The product has been added to the cart and an added-to-cart confirmation is visible.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=MEESHO,
        scenario="Product not selected: the detail page shows Buy Now, with no add-to-cart confirmation.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="meesho_permission_present",
        screenshot="meesho_permission_contacts.png",
        app="Meesho",
        package=PERMISSION_UI,
        assertion="A system permission dialog requesting access to contacts is visible.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.SATISFIED,
        truth_satisfied=True,
        authority_package=None,
        scenario="Permission dialog present: an Android contacts-access prompt is on screen.",
        provenance=MEESHO_PERMISSION_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=False,
    ),
    VisionCase(
        name="meesho_permission_blocks_results",
        screenshot="meesho_permission_contacts.png",
        app="Meesho",
        package=PERMISSION_UI,
        assertion="The Meesho search results for 'ghar soap' are displayed.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=MEESHO,
        scenario="Permission dialog blocks the goal; the foreground is the permission controller, not Meesho.",
        provenance=MEESHO_PERMISSION_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="meesho_tap_command_no_assessment",
        screenshot="meesho_search_results.png",
        app="Meesho",
        package=MEESHO,
        assertion="Tap the first product card in the results.",
        goal_kind=SuccessKind.COMMAND,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=MEESHO,
        scenario="Command goal: completes only from a runtime receipt; no visual assessment must advance it.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=False,
    ),
    VisionCase(
        name="meesho_capture_goal_no_assessment",
        screenshot="meesho_product_detail.png",
        app="Meesho",
        package=MEESHO,
        assertion="Store the displayed product price as the captured value 'price'.",
        goal_kind=SuccessKind.CAPTURE,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=MEESHO,
        scenario="Capture goal: completes only from a committed STORE receipt, never from an assessment.",
        provenance=MEESHO_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=False,
    ),
)


MULTI_APP_CASES: Tuple[VisionCase, ...] = (
    VisionCase(
        name="chatgpt_ad_overlay_blocks",
        screenshot="chatgpt_ad_overlay.png",
        app="ChatGPT (ad)",
        package=CHATGPT,
        assertion="The Amazon search results for 'ghar soap' are displayed.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=None,
        scenario="Unrelated overlay: a full-screen ChatGPT advertisement, no results present.",
        provenance=CHATGPT_AD_RUN,
        label_source=LabelSource.HUMAN,
        critical_negative=True,
    ),
    VisionCase(
        name="controlled_login_present",
        screenshot="controlled_login.png",
        app="Controlled",
        package=CONTROLLED,
        assertion="A login screen with email and password fields and a sign-in button is displayed.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.SATISFIED,
        truth_satisfied=True,
        authority_package=None,
        scenario="Login present: a controlled synthetic login screen with credential fields.",
        provenance=CONTROLLED_SOURCE,
        label_source=LabelSource.PROGRAMMATIC,
        critical_negative=False,
    ),
    VisionCase(
        name="rating_threshold_met",
        screenshot="controlled_customer_rating_high.png",
        app="Controlled",
        package=CONTROLLED,
        assertion="The product's customer rating is at least 4.2 out of 5.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.SATISFIED,
        truth_satisfied=True,
        authority_package=None,
        scenario="Correct attribute above threshold: a 4.5 customer rating satisfies the predicate.",
        provenance=CONTROLLED_SOURCE,
        label_source=LabelSource.PROGRAMMATIC,
        critical_negative=False,
    ),
    VisionCase(
        name="rating_below_threshold",
        screenshot="controlled_customer_rating_low.png",
        app="Controlled",
        package=CONTROLLED,
        assertion="The product's customer rating is at least 4.2 out of 5.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=None,
        scenario="Correct attribute below threshold: a 3.8 customer rating fails the predicate.",
        provenance=CONTROLLED_SOURCE,
        label_source=LabelSource.PROGRAMMATIC,
        critical_negative=True,
    ),
    VisionCase(
        name="rating_unrelated_number",
        screenshot="controlled_unrelated_number.png",
        app="Controlled",
        package=CONTROLLED,
        assertion="The product's customer rating is at least 4.2 out of 5.",
        goal_kind=SuccessKind.OBSERVED,
        expected_verdict=VisualVerdict.NOT_SATISFIED,
        truth_satisfied=False,
        authority_package=None,
        scenario="Number in unrelated region: a 4.5% cashback figure must not satisfy a customer-rating predicate.",
        provenance=CONTROLLED_SOURCE,
        label_source=LabelSource.PROGRAMMATIC,
        critical_negative=True,
    ),
)


CASES: Tuple[VisionCase, ...] = AMAZON_CASES + MEESHO_CASES + MULTI_APP_CASES
