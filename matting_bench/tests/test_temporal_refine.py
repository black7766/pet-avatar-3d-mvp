from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poc import (  # noqa: E402
    _remove_unsupported_temporal_fragments,
    fuse_structural_support_masks,
    stabilize_alpha_temporal,
)


class TemporalFragmentVotingTest(unittest.TestCase):
    def test_preserves_flow_supported_tail_and_large_component(self) -> None:
        frames = []
        supports = []
        for index in range(5):
            frame = np.zeros((64, 64, 4), dtype=np.float32)
            frame[18:48, 22:44, :3] = 0.4
            frame[18:48, 22:44, 3] = 1.0
            tail_x = 4 + index * 4
            frame[28:34, tail_x:tail_x + 6, :3] = 0.3
            frame[28:34, tail_x:tail_x + 6, 3] = 1.0
            aligned = np.zeros((64, 64), dtype=bool)
            aligned[18:48, 22:44] = True
            aligned[28:34, tail_x:tail_x + 6] = True
            frames.append(frame)
            supports.append([aligned])

        # Unsupported one-frame speck should be removed.
        frames[2][4:7, 52:55, :3] = 0.8
        frames[2][4:7, 52:55, 3] = 1.0
        # A disconnected area larger than the safety threshold represents a body part.
        frames[2][46:58, 2:14, :3] = 0.5
        frames[2][46:58, 2:14, 3] = 1.0

        stats = _remove_unsupported_temporal_fragments(frames, supports)
        result = frames

        self.assertTrue(np.all(result[2][28:34, 12:18, 3] > 0.9))
        self.assertTrue(np.all(result[2][4:7, 52:55, 3] == 0.0))
        self.assertTrue(np.all(result[2][46:58, 2:14, 3] > 0.9))
        self.assertEqual(stats["single_frame_fragments_removed"], 1)
        self.assertEqual(stats["fragment_pixels_removed"], 9)
        self.assertEqual(stats["protected_large_components"], 1)

    def test_preserves_current_frame_lower_motion_silhouette(self) -> None:
        frames = []
        for index in range(3):
            frame = np.zeros((64, 64, 4), dtype=np.float32)
            frame[10:42, 20:44, :3] = 0.4
            frame[10:42, 20:44, 3] = 1.0
            frame[42:58, 22:30, :3] = 0.7
            frame[42:58, 22:30, 3] = 1.0 if index != 1 else 0.2
            frames.append(frame)

        expected_lower = [frame[34:, :, :].copy() for frame in frames]
        result, stats = stabilize_alpha_temporal(
            frames,
            source_rgbs=[frame[:, :, :3].copy() for frame in frames],
            flow_size=64,
            preserve_lower_motion=True,
        )

        for frame, expected in zip(result, expected_lower):
            np.testing.assert_allclose(frame[34:, :, :], expected, atol=1e-6)
        self.assertTrue(stats["preserve_lower_motion"])
        self.assertGreater(stats["restored_motion_pixels"], 0)

    def test_structural_support_repairs_hole_without_clipping_soft_detail(self) -> None:
        from PIL import Image
        from tempfile import TemporaryDirectory

        frame = np.zeros((32, 32, 4), dtype=np.float32)
        frame[7:25, 8:24, :3] = 0.5
        frame[7:25, 8:24, 3] = 1.0
        frame[14:18, 14:18, :3] = 0.0
        frame[14:18, 14:18, 3] = 0.0
        frame[15, 25:29, :3] = 0.7
        frame[15, 25:29, 3] = 0.25
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[7:25, 8:24] = 255

        with TemporaryDirectory() as directory:
            mask_path = Path(directory) / "mask.png"
            Image.fromarray(mask, "L").save(mask_path)
            result, stats = fuse_structural_support_masks(
                [Image.fromarray((frame * 255).astype(np.uint8), "RGBA")],
                [mask_path],
            )

        alpha = np.asarray(result[0])[:, :, 3]
        self.assertTrue(np.all(alpha[14:18, 14:18] >= 250))
        self.assertTrue(np.all(alpha[15, 25:29] == 63))
        self.assertEqual(stats["severe_holes"], 16)


if __name__ == "__main__":
    unittest.main()
