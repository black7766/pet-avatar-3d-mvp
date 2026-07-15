import unittest

from prompt_config.actions import ACTION_TEXT
from prompts import CLIP_PROMPTS, STATE_FRAME_PROMPTS


class PromptConfigTest(unittest.TestCase):
    def test_only_current_product_actions_are_configured(self):
        self.assertEqual(set(ACTION_TEXT), {"idle", "fast_walk", "sleep"})
        self.assertEqual(set(CLIP_PROMPTS), {"idle", "fast_walk", "sleep"})
        self.assertEqual(set(STATE_FRAME_PROMPTS), {"fast_walk", "sleep"})

    def test_action_text_does_not_own_locked_render_constraints(self):
        action_text = " ".join(
            text
            for action in ACTION_TEXT.values()
            for text in action.values()
        ).lower()
        for locked_term in (
            "#00ff00",
            "--resolution",
            "--duration",
            "--watermark",
            "--generate_audio",
            "背景必须",
            "纯绿色二维抠像底板",
        ):
            self.assertNotIn(locked_term, action_text)

    def test_composed_video_prompts_keep_production_guards(self):
        for prompt in CLIP_PROMPTS.values():
            self.assertIn("--resolution 720p", prompt)
            self.assertIn("--duration 5", prompt)
            self.assertIn("--camerafixed true", prompt)
            self.assertIn("--generate_audio false", prompt)
            self.assertIn("#00FF00", prompt)
            self.assertIn("完整全身入镜", prompt)

    def test_state_frames_keep_green_screen_and_full_body_guards(self):
        for prompt in STATE_FRAME_PROMPTS.values():
            self.assertIn("#00FF00", prompt)
            self.assertIn("完整全身入镜", prompt)
            self.assertIn("禁止地面", prompt)


if __name__ == "__main__":
    unittest.main()
