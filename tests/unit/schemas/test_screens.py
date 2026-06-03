from __future__ import annotations

from fathom.schemas.screens import ScreenChangeRegion, ScreenDiff, ScreenScrollTranslation


class TestActionHadEffectNoiseGate:
    """
    Behavioral pins for the action-effect verifier on noisy frames.
    """

    @staticmethod
    def __diff(
        *,
        regions: int = 0,
        scroll_dx: float = 0.0,
        scroll_dy: float = 0.0,
        phash_distance: int = 0,
        ssim_score: float = 1.0,
        activity_changed: bool = False,
        xml_hash_changed: bool = False,
        content_pixel_diff_ratio: float = 0.0,
        interaction_hash_changed: bool = False,
    ) -> ScreenDiff:
        """
        Build a :class:`ScreenDiff` with the requested per-signal values.
        """

        return ScreenDiff(
            ssim_score=ssim_score,
            phash_distance=phash_distance,
            activity_changed=activity_changed,
            xml_hash_changed=xml_hash_changed,
            interaction_hash_changed=interaction_hash_changed,
            content_pixel_diff_ratio=content_pixel_diff_ratio,
            scroll_translation=ScreenScrollTranslation(dx=scroll_dx, dy=scroll_dy),
            changed_regions=[
                ScreenChangeRegion(x=0, y=0, width=1, height=1) for _ in range(regions)
            ],
            
        )

    def test_ios_springboard_clock_tick_noise_returns_false(self) -> None:
        """
        Status-bar ``UpdatesFrequently`` clock tick flips xml_hash but every
        visual signal stays at the identical-frame floor; must not fire.
        """

        diff = self.__diff(
            ssim_score=1.0,
            phash_distance=0,
            xml_hash_changed=True,
            content_pixel_diff_ratio=0.0,
        )
        assert diff.action_had_effect is False

    def test_unity_render_loop_micro_regions_return_false(self) -> None:
        """
        Unity SurfaceView animations produce many ``changed_regions`` with
        sub-threshold ssim/content_diff; must not fire as effect.
        """

        diff = self.__diff(
            regions=15,
            ssim_score=0.99,
            phash_distance=1,
            content_pixel_diff_ratio=0.002,
            
        )
        assert diff.action_had_effect is False

    def test_activity_change_returns_true_alone(self) -> None:
        """
        Foreground package switch is always a real effect.
        """

        diff = self.__diff(activity_changed=True)
        assert diff.action_had_effect is True

    def test_visual_signal_returns_true(self) -> None:
        """
        A pHash jump above threshold is an unambiguous real effect.
        """

        diff = self.__diff(phash_distance=20, ssim_score=0.5, content_pixel_diff_ratio=0.5)
        assert diff.action_had_effect is True

    def test_scroll_signal_returns_true(self) -> None:
        """
        Scroll translation beyond the action-effect floor counts as effect.
        """

        diff = self.__diff(scroll_dy=20.0)
        assert diff.action_had_effect is True

    def test_xml_hash_with_visual_co_signal_returns_true(self) -> None:
        """
        Real UI change: xml_hash flips alongside a visual signal.
        """

        diff = self.__diff(xml_hash_changed=True, phash_distance=20)
        assert diff.action_had_effect is True
