import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image, ImageDraw

import poc
from poc import build_loop_frames


class LockedEndpointLoopTest(unittest.TestCase):
    def test_video_request_uses_configured_duration_and_resolution(self):
        with patch.multiple(poc, CLIP_DURATION_SECONDS=4, CLIP_RESOLUTION="480p"):
            body = poc.animate_request_body("data:image/png;base64,test", "idle")
        prompt = body["content"][0]["text"]
        self.assertIn("--duration 4", prompt)
        self.assertIn("--resolution 480p", prompt)
        self.assertIn("--generate_audio false", prompt)

    def test_preserves_locked_first_and_last_frames(self):
        frames = [
            Image.new("RGBA", (32, 32), (index, index, index, 255))
            for index in range(20)
        ]
        frames[-1] = frames[0].copy()

        result, added, meta = build_loop_frames(
            frames,
            close_frames=8,
            optimize_loop=True,
            preserve_locked_endpoints=True,
        )

        self.assertEqual(len(result), len(frames))
        self.assertEqual(added, 0)
        self.assertEqual(meta["mode"], "locked_endpoints")
        self.assertEqual(result[0].getpixel((0, 0)), result[-1].getpixel((0, 0)))

    def test_clean_seam_skips_translucent_crossfade(self):
        frames = []
        for index in range(80):
            frame = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            offset = index % 4
            draw.rectangle((8 + offset, 8, 22 + offset, 26), fill=(120, 90, 60, 255))
            frames.append(frame)
        frames[-1] = frames[0].copy()

        result, added, meta = build_loop_frames(
            frames,
            close_frames=8,
            optimize_loop=True,
        )

        self.assertEqual(added, 0)
        self.assertEqual(meta["closure"], "direct")
        self.assertTrue(all(frame.getextrema()[3][1] == 255 for frame in result))

    def test_disk_matte_fallback_defines_locked_loop(self):
        def fake_extract(_video, frames_dir):
            frames_dir.mkdir(parents=True, exist_ok=True)
            for index in range(24):
                frame = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((9, 5, 22, 28), fill=(110 + index % 2, 90, 70, 255))
                frame.save(frames_dir / f"f_{index:04d}.png")
            return {"mode": "test", "bg_floor": 0.9, "key_rgb": [0, 255, 0]}

        with TemporaryDirectory() as temp:
            output = Path(temp)
            pet_dir = output / "pet"
            pet_dir.mkdir()
            (pet_dir / "raw_idle.mp4").write_bytes(b"placeholder")
            with patch.multiple(
                poc,
                OUTPUT=output,
                MATTE_PIPELINE="disk",
                ADAPTIVE_GREEN_MATTE=True,
                LOCK_STATE_LAST_FRAME=True,
                SINGLE_SOURCE_ANIMATION=False,
                WEBP_WIDTH=32,
                WEBP_BOTTOM_MARGIN=2,
                WEBP_METHOD=0,
            ), patch.object(poc, "sample_bg_color", return_value="0x00FF00"), patch.object(
                poc, "extract_adaptive_green_frames", side_effect=fake_extract
            ), patch.object(poc, "assess_frames", return_value=(0.0, 0.25)):
                poc.step_matte("pet", "idle")

            self.assertTrue((pet_dir / "anim_idle.webp").exists())


if __name__ == "__main__":
    unittest.main()
