# -*- coding: utf-8 -*-
"""Locked production constraints composed around editable action text."""

from .actions import ACTION_TEXT


_VIDEO_OPTIONS = (
    " --resolution 720p --duration 5 --camerafixed true --watermark false "
    "--generate_audio false"
)

_IDENTITY_AND_ASSET_LOCK = (
    "全程保持同一只宠物的身份、毛色、花纹、五官、体型比例、毛发质感和角色风格；"
    "完整全身入镜，耳尖、四肢、每只脚掌、腹部、尾巴根部和完整尾巴尖均不得裁切、断裂、消失或贴边；"
    "禁止变脸、变体型、身体拉长、重复肢体、断腿、断尾、出现人手、道具、文字或其他动物；"
)

_GREENSCREEN_LOCK = (
    "背景必须是完全静止且每个像素色相和亮度一致的纯绿色二维抠像底板 #00FF00；"
    "它不是摄影棚、墙面、地面、无影墙、平台、跑步机或物理环境；"
    "禁止地面、地平线、接触阴影、投影、环境光遮蔽、反射、渐变、光晕、暗角、背景残影和绿色反光；"
    "宠物照明采用恒定、柔和、中性的正面漫射光；禁止轮廓光、逆光、发光白边、黄色亮边和过曝毛尖；"
)

_STATE_FRAME_BASE = (
    "基于输入宠物角色图，生成同一只宠物的状态动画首帧图。"
    "只改变姿态到指定状态，不改变品种、脸型、颜色、服装、配饰或角色风格；"
    + _IDENTITY_AND_ASSET_LOCK
    + "宠物位于画面中心，主体占画布约 55% 到 62%，四周保留充足纯绿安全区；"
    + _GREENSCREEN_LOCK
)

_SLEEP_FRAME_LOCK = (
    "使用正面偏 3/4 视角的广角全身构图，不得近景或大头特写；"
    "尾巴必须从尾根到圆润尾尖解剖连续、完整可见，不得平切、隐藏、断开或伸出画面。"
)

_FAST_WALK_FRAME_LOCK = (
    "This is a suspended walk-cycle reference preview inside animation software, not a pet standing in a studio. "
    "No paw physically touches a rendered surface. The stance phase is represented only by the lowest paw "
    "position in the gait. Keep the lowest paw at least 10 percent of the image height above the bottom edge. "
    "A continuous band of exactly uniform green must remain directly below every paw, with each toe and the "
    "complete rounded underside of every paw visible. Keep generous uniform green clearance around ears, paws, "
    "belly, and tail tip. Preserve realistic fur and the exact paw markings."
)

_IDLE_VIDEO_PREFIX = (
    "动作生成统一规范：把首帧当作已经标准化好的同一只宠物角色模型，只继承首帧外观，"
    "不要根据原始上传照片重新推断姿态；从正面端坐标准姿态开始，前0.3秒保持静止；"
    + _IDENTITY_AND_ASSET_LOCK
)

_IDLE_VIDEO_TAIL = (
    "身体保持在画面原位，镜头完全固定；最后0.8秒完全回到首帧正面端坐姿态并保持静止，"
    "尾巴、耳朵、身体轮廓和前爪位置与首帧一致；"
    + _GREENSCREEN_LOCK
    + _VIDEO_OPTIONS
)

_STATE_LOOP_BASE = (
    "产品状态动画统一规则：硬件上报状态时，宠物已经处于该状态；视频第一帧必须继承首帧的目标姿态，"
    "不要展示从静息进入该状态或离开该状态的过程；"
    + _IDENTITY_AND_ASSET_LOCK
    + "镜头固定，宠物保持画面中心，动作连续稳定；禁止跳帧、闪回初始坐姿和镜头切换；"
    + _GREENSCREEN_LOCK
)

_STATE_LOOP_TAIL = (
    "首帧和尾帧必须保持同一状态下的相近姿态，循环时不能突然切回静息姿态；"
    + _VIDEO_OPTIONS
)

_FAST_WALK_VIDEO_LOCK = (
    "Generate one fixed-camera production animation shot of the exact pet in frame 1. "
    "This is a suspended locomotion-cycle preview, not physical walking on a rendered surface. "
    "All paws remain above the background with a continuous uniform green band below the lowest paw in every "
    "frame. Stance is indicated by limb timing only, never by floor contact, paw compression, a contact patch, "
    "or a shadow. No idle transition, turn, camera move, motion blur, frame blending, duplicated anatomy, or "
    "background disturbance. "
)


STATE_FRAME_PROMPTS = {
    "fast_walk": (
        _STATE_FRAME_BASE
        + ACTION_TEXT["fast_walk"]["state_frame"]
        + _FAST_WALK_FRAME_LOCK
    ),
    "sleep": (
        _STATE_FRAME_BASE
        + ACTION_TEXT["sleep"]["state_frame"]
        + _SLEEP_FRAME_LOCK
    ),
}

CLIP_PROMPTS = {
    "idle": (
        _IDLE_VIDEO_PREFIX
        + ACTION_TEXT["idle"]["video"]
        + _IDLE_VIDEO_TAIL
    ),
    "fast_walk": (
        _STATE_LOOP_BASE
        + _FAST_WALK_VIDEO_LOCK
        + ACTION_TEXT["fast_walk"]["video"]
        + _STATE_LOOP_TAIL
    ),
    "sleep": (
        _STATE_LOOP_BASE
        + ACTION_TEXT["sleep"]["video"]
        + _STATE_LOOP_TAIL
    ),
}
