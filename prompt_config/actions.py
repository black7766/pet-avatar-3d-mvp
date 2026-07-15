# -*- coding: utf-8 -*-
"""Editable action-only prompt text.

Only edit this file when changing pose or motion. Background, lighting, framing,
identity, full-body safety, loop and API constraints live in ``locked.py``.
"""

ACTION_TEXT = {
    "idle": {
        "video": (
            "idle 静息状态循环：宠物始终正对镜头端坐，目光看向观众，只做轻微呼吸起伏、自然眨眼、"
            "尾巴尖小幅摆动；头部最多向左和向右各转动10到15度，然后回到正视镜头，"
            "动作安静稳定，最后回到与开头相同的正面端坐初始姿态"
        ),
    },
    "fast_walk": {
        "state_frame": (
            "Create a production animation keyframe of this exact pet brisk-walking in place, "
            "facing screen right in a locked near-side profile view. "
            "The camera is level with the pet's shoulder and uses a flat telephoto perspective. "
            "The spine stays horizontal and the complete body occupies 55 to 62 percent of the canvas. "
            "Use one clean mid-stride gait phase. The near front leg and far rear leg extend forward; "
            "the other pair supports the body. Offset the near and far legs slightly so all four "
            "continuous legs and all four solid paws remain readable. "
            "The tail is anatomically attached at the rump and extends toward screen left in one gentle, "
            "low curve. Keep the entire tail and rounded tail tip separated from the hind legs."
        ),
        "video": (
            "The pet brisk-walks in place facing screen right in the same near-side profile for the entire clip. "
            "Repeat one compact, mechanically consistent four-beat walking cycle. The body root is locked "
            "horizontally; only a subtle vertical body motion is allowed. The pet never turns toward the camera "
            "and never changes screen position or scale. The four legs alternate with clear near-leg and far-leg "
            "separation. Each leg stays continuously connected to the torso and every paw stays solid, sharp, "
            "and anatomically consistent. The complete rounded paw and toe silhouette remains visible even at "
            "the lowest stance phase. Keep the tail base locked to the rump. The complete tail remains outside "
            "the hind-leg silhouette and makes only a two-to-five-degree slow vertical sway, preserving one "
            "continuous tail and visible tip in every frame. Start and end on the same gait phase."
        ),
    },
    "sleep": {
        "state_frame": (
            "生成睡觉状态首帧：严格采用参考姿态式的低伏趴睡，正面偏 3/4 视角，"
            "胸腹低伏，前爪在脸下方并拢或自然叠放，头部放低并轻轻趴在两只前爪上，"
            "双眼闭合，表情放松，尾巴向后或侧后方自然伸出；"
            "不要侧躺，不要蜷缩成球，不要露出肚皮，不要睁眼，不要抬头。"
        ),
        "video": (
            "Strict sleep-loop requirement: the pet is already asleep from frame 1. Eyes must remain fully "
            "closed for the entire clip. The head must remain low and resting on or very close to the front paws. "
            "Only subtle breathing is allowed. No opening eyes, no looking at camera, no lifting head, no waking "
            "up, no getting up, no rolling, no transition into or out of sleep. "
            "sleep 睡眠状态循环：视频第一帧就已经是宠物睡着后的稳定低伏趴睡姿态，胸腹低伏，"
            "前爪在脸下方并拢或自然叠放，头部轻轻趴在两只前爪上，双眼闭合，表情放松；"
            "整段只保留轻微、规律的睡眠呼吸起伏，偶尔耳朵或爪子极轻微抽动；"
            "不要出现入睡、醒来、起身、翻身、走动或其他状态过渡。"
        ),
    },
}
