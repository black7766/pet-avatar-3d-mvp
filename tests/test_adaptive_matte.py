import unittest

import numpy as np
from PIL import Image, ImageDraw

from poc import (
    adaptive_green_matte_frame,
    assess_rgba_frames,
    edge_profile_for_pet,
    refine_adaptive_edge,
    refine_reframed_halo,
    sharpen_alpha_transitions,
    suppress_opaque_key_halo,
)


class AdaptiveGreenMatteTest(unittest.TestCase):
    def test_alpha_contrast_sharpens_transition_without_moving_half_contour(self):
        rgba = np.zeros((1, 5, 4), dtype=np.uint8)
        rgba[0, :, :3] = 120
        rgba[0, :, 3] = (0, 64, 128, 192, 255)
        frames, stats = sharpen_alpha_transitions(
            [Image.fromarray(rgba, "RGBA")], strength=1.18
        )
        result = np.asarray(frames[0])[:, :, 3]

        self.assertEqual(int(result[0, 0]), 0)
        self.assertLess(int(result[0, 1]), 64)
        self.assertEqual(int(result[0, 2]), 128)
        self.assertGreater(int(result[0, 3]), 192)
        self.assertEqual(int(result[0, 4]), 255)
        self.assertEqual(stats["contour_anchor"], 0.5)

    def test_green_identity_detail_is_not_counted_as_edge_spill(self):
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[12:52, 12:52] = (130, 100, 70, 255)
        rgba[28:36, 28:36] = (20, 190, 30, 255)
        residual, visible = assess_rgba_frames([Image.fromarray(rgba, "RGBA")])
        self.assertEqual(residual, 0.0)
        self.assertGreater(visible, 0.3)

    def test_preserves_body_and_upper_detail_while_removing_green_background(self):
        image = Image.new("RGB", (128, 128), (12, 232, 14))
        draw = ImageDraw.Draw(image)

        # Light body with a dark non-green patch must remain opaque.
        draw.ellipse((36, 18, 92, 116), fill=(232, 224, 208))
        draw.ellipse((55, 42, 65, 52), fill=(38, 28, 24))

        # A green upper-body detail is intentionally protected from chroma keying.
        draw.ellipse((69, 42, 77, 50), fill=(30, 180, 48))

        # A lower enclosed green gap represents background visible between legs.
        draw.rectangle((59, 78, 68, 108), fill=(12, 232, 14))

        # A darker green contact shadow should still key out despite being far from key RGB.
        draw.ellipse((28, 108, 100, 123), fill=(8, 78, 9))

        profile = {
            "bg_floor": 0.90,
            "key_rgb": np.array([12, 232, 14], dtype=np.float32) / 255.0,
            "sampled_frames": 1,
        }
        result = np.asarray(adaptive_green_matte_frame(image, profile))
        alpha = result[:, :, 3]

        self.assertGreater(int(alpha[62, 50]), 245, "light body was eroded")
        self.assertGreater(int(alpha[47, 73]), 245, "upper green detail was removed")
        self.assertLess(int(alpha[92, 63]), 10, "enclosed lower green gap was retained")
        self.assertLess(int(alpha[116, 40]), 10, "dark green contact shadow was retained")

    def test_removes_mild_core_green_cast_but_preserves_saturated_green_detail(self):
        image = Image.new("RGB", (96, 96), (8, 230, 10))
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 10, 76, 88), fill=(145, 125, 100))
        draw.rectangle((35, 50, 60, 70), fill=(140, 155, 130))
        draw.ellipse((42, 24, 52, 34), fill=(100, 120, 70))
        profile = {
            "bg_floor": 0.90,
            "key_rgb": np.array([8, 230, 10], dtype=np.float32) / 255.0,
            "sampled_frames": 1,
        }

        result = np.asarray(adaptive_green_matte_frame(image, profile))
        cast_pixel = result[60, 48, :3].astype(int)
        identity_green = result[29, 47, :3].astype(int)

        self.assertLessEqual(cast_pixel[1], max(cast_pixel[0], cast_pixel[2]) + 10)
        self.assertGreater(cast_pixel[2], 130, "green bounce was reduced without restoring blue")
        self.assertGreater(identity_green[1], max(identity_green[0], identity_green[2]) + 15)

    def test_repairs_opaque_yellow_green_rim_without_changing_alpha(self):
        rgb = np.zeros((80, 80, 3), dtype=np.float32)
        alpha = np.zeros((80, 80), dtype=np.float32)
        alpha[12:68, 12:68] = 1.0
        rgb[12:68, 12:68] = (0.28, 0.22, 0.12)
        rgb[12:15, 12:68] = (0.96, 0.82, 0.32)
        rgb[65:68, 12:68] = (0.96, 0.82, 0.32)
        rgb[12:68, 12:15] = (0.96, 0.82, 0.32)
        rgb[12:68, 65:68] = (0.96, 0.82, 0.32)
        rgb[28:52, 12:15] = (1.0, 0.99, 0.96)
        identity_green = (0.10, 0.65, 0.12)
        rgb[34:46, 34:46] = identity_green

        result = suppress_opaque_key_halo(rgb, alpha)
        real_result = suppress_opaque_key_halo(rgb, alpha, profile="real")

        before_distance = np.linalg.norm(rgb[13, 40] - rgb[30, 40])
        after_distance = np.linalg.norm(result[13, 40] - result[30, 40])
        neutral_before = np.linalg.norm(rgb[40, 13] - rgb[40, 30])
        neutral_after = np.linalg.norm(result[40, 13] - result[40, 30])
        self.assertLess(after_distance, before_distance * 0.55)
        self.assertLess(neutral_after, neutral_before * 0.55)
        self.assertLess(
            np.linalg.norm(real_result[40, 13] - rgb[40, 30]),
            neutral_after,
        )
        real_luma = float(np.dot(real_result[40, 13], (0.299, 0.587, 0.114)))
        core_luma = float(np.dot(rgb[40, 30], (0.299, 0.587, 0.114)))
        self.assertLessEqual(real_luma - core_luma, 0.17)
        np.testing.assert_allclose(result[40, 40], identity_green, atol=1e-6)
        self.assertEqual(float(alpha.sum()), 56 * 56)

        _, real_alpha = refine_reframed_halo(rgb, alpha, profile="real")
        _, cartoon_alpha = refine_reframed_halo(rgb, alpha, profile="cartoon")
        self.assertLess(real_alpha[13, 40], 0.90)
        self.assertEqual(real_alpha[40, 40], 1.0)
        np.testing.assert_allclose(cartoon_alpha, alpha, atol=1e-6)

    def test_motion_mask_lifts_blurred_dark_fur_but_not_shadows(self):
        # A dark leg motion-blurred to 50% coverage keys to ~0.16 alpha on the normalized
        # ramp, visually amputating the limb. With a motion mask the linear un-mix estimate
        # must restore roughly the true coverage, while a keyed contact shadow (pure green
        # at lower brightness) stays removed even inside the motion mask.
        key = np.array([0.106, 0.886, 0.110], dtype=np.float32)
        fur = np.array([0.30, 0.25, 0.15], dtype=np.float32)
        rgb = np.tile(key, (96, 96, 1)).astype(np.float32)
        rgb[20:60, 30:44] = fur  # solid leg
        for column, t in zip(range(44, 64), np.linspace(0.9, 0.1, 20)):
            rgb[20:60, column] = t * fur + (1.0 - t) * key  # blur smear
        rgb[70:80, 30:60] = key * 0.45  # dark green contact shadow
        image = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
        profile = {
            "bg_floor": 0.8235,
            "key_rgb": key,
            "sampled_frames": 1,
        }
        motion_mask = np.zeros((96, 96), dtype=bool)
        motion_mask[10:90, 20:70] = True

        static = np.asarray(adaptive_green_matte_frame(image, profile))
        lifted = np.asarray(adaptive_green_matte_frame(image, profile, motion_mask=motion_mask))

        half_coverage_alpha_static = int(static[40, 52, 3])
        half_coverage_alpha_lifted = int(lifted[40, 52, 3])
        self.assertLess(half_coverage_alpha_static, 90)
        self.assertGreater(half_coverage_alpha_lifted, 100, "blur smear was not lifted")
        self.assertLess(
            abs(half_coverage_alpha_lifted - 128), 55,
            "lifted alpha should approximate true coverage",
        )
        self.assertLess(int(lifted[74, 45, 3]), 10, "contact shadow was lifted")
        self.assertGreater(int(lifted[40, 35, 3]), 245, "solid leg must stay opaque")

    def test_white_paw_on_dark_leg_survives_real_halo_suppression(self):
        # A thick white marking near the silhouette is pet color, not key bounce:
        # it must keep its brightness even though the nearest deep fur is much darker.
        rgb = np.zeros((90, 90, 3), dtype=np.float32)
        alpha = np.zeros((90, 90), dtype=np.float32)
        alpha[10:80, 30:52] = 1.0
        rgb[10:80, 30:52] = (0.30, 0.24, 0.16)
        paw = (slice(62, 80), slice(30, 52))
        rgb[paw] = (0.93, 0.91, 0.88)
        white_before = rgb[71, 40].copy()

        cleaned, halo, _ = suppress_opaque_key_halo(
            rgb, alpha, profile="real", return_details=True
        )
        self.assertFalse(halo[paw].any(), "white paw was flagged as key halo")
        np.testing.assert_allclose(cleaned[71, 40], white_before, atol=0.02)

        refined_rgb, _ = refine_adaptive_edge(cleaned, alpha, profile="real")
        paw_luma = float(np.dot(refined_rgb[71, 40], (0.299, 0.587, 0.114)))
        self.assertGreater(paw_luma, 0.80, "edge refine dimmed the white paw")

        _, feather_alpha = refine_reframed_halo(rgb, alpha, profile="real")
        self.assertEqual(float(feather_alpha[paw].min()), 1.0)

    def test_edge_profile_keeps_cartoon_and_real_assets_separate(self):
        self.assertEqual(edge_profile_for_pet("pet_123_real"), "real")
        self.assertEqual(edge_profile_for_pet("pet_123_paimomo"), "cartoon")

    def test_real_profile_repairs_mild_warm_exposure_without_touching_cartoon(self):
        rgb = np.zeros((72, 72, 3), dtype=np.float32)
        alpha = np.zeros((72, 72), dtype=np.float32)
        alpha[8:64, 8:64] = 1.0
        rgb[8:64, 8:64] = (0.29, 0.23, 0.15)
        rgb[8:11, 8:64] = (0.69, 0.53, 0.20)

        cartoon = suppress_opaque_key_halo(rgb, alpha, profile="cartoon")
        real = suppress_opaque_key_halo(rgb, alpha, profile="real")
        source_delta = np.linalg.norm(rgb[9, 36] - rgb[24, 36])
        cartoon_delta = np.linalg.norm(cartoon[9, 36] - rgb[24, 36])
        real_delta = np.linalg.norm(real[9, 36] - rgb[24, 36])

        self.assertAlmostEqual(cartoon_delta, source_delta, places=5)
        self.assertLess(real_delta, source_delta * 0.75)

    def test_edge_refine_adds_subpixel_alpha_without_expanding_far_background(self):
        rgb = np.zeros((80, 80, 3), dtype=np.float32)
        alpha = np.zeros((80, 80), dtype=np.float32)
        for y in range(12, 68):
            end = 55 + (2 if y % 4 < 2 else 0)
            alpha[y, 18:end] = 1.0
            rgb[y, 18:end] = (0.34, 0.27, 0.20)

        _, refined = refine_adaptive_edge(rgb, alpha, profile="real")

        soft_pixels = (refined > 0.02) & (refined < 0.98)
        self.assertGreater(int(soft_pixels.sum()), 60)
        self.assertLess(
            int(soft_pixels.sum()), 120,
            "smooth contour was expanded into an unnecessarily wide soft band",
        )
        self.assertEqual(float(refined[:, 62:].max()), 0.0)
        self.assertEqual(float(refined[30, 30]), 1.0)

    def test_edge_refine_rebuilds_green_and_overbright_rim_from_core_color(self):
        rgb = np.zeros((80, 80, 3), dtype=np.float32)
        alpha = np.zeros((80, 80), dtype=np.float32)
        alpha[12:68, 12:68] = 1.0
        rgb[12:68, 12:68] = (0.34, 0.27, 0.20)
        rgb[12:15, 12:68] = (0.80, 0.98, 0.18)

        refined_rgb, refined_alpha = refine_adaptive_edge(rgb, alpha, profile="real")
        before_bias = rgb[13, 40, 1] - max(rgb[13, 40, 0], rgb[13, 40, 2])
        after_bias = refined_rgb[13, 40, 1] - max(
            refined_rgb[13, 40, 0], refined_rgb[13, 40, 2]
        )
        before_luma = float(np.dot(rgb[13, 40], (0.299, 0.587, 0.114)))
        after_luma = float(np.dot(refined_rgb[13, 40], (0.299, 0.587, 0.114)))

        self.assertLess(after_bias, before_bias * 0.25)
        self.assertLess(after_luma, before_luma - 0.12)
        self.assertEqual(
            float(refined_alpha[12:15, 40].min()), 1.0,
            "a smooth opaque edge should be color-corrected without forced feathering",
        )
        np.testing.assert_allclose(refined_rgb[40, 40], rgb[40, 40], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
