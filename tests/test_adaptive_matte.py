import unittest

import numpy as np
from PIL import Image, ImageDraw

from poc import adaptive_green_matte_frame


class AdaptiveGreenMatteTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
