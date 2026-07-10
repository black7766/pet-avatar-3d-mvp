import unittest

import numpy as np
from PIL import Image, ImageDraw

from poc import adaptive_green_matte_frame, assess_rgba_frames, suppress_opaque_key_halo


class AdaptiveGreenMatteTest(unittest.TestCase):
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

        before_distance = np.linalg.norm(rgb[13, 40] - rgb[30, 40])
        after_distance = np.linalg.norm(result[13, 40] - result[30, 40])
        neutral_before = np.linalg.norm(rgb[40, 13] - rgb[40, 30])
        neutral_after = np.linalg.norm(result[40, 13] - result[40, 30])
        self.assertLess(after_distance, before_distance * 0.55)
        self.assertLess(neutral_after, neutral_before * 0.55)
        np.testing.assert_allclose(result[40, 40], identity_green, atol=1e-6)
        self.assertEqual(float(alpha.sum()), 56 * 56)


if __name__ == "__main__":
    unittest.main()
