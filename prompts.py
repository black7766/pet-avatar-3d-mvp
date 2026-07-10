# -*- coding: utf-8 -*-
"""PoC prompt 模板集中地 — 调优只改这里。

绿幕铁律：所有生成环节都必须强调「纯绿色背景、无阴影、无渐变」，
抠图环节按实际采样色 key（生成模型给的绿不会是标准 0x00FF00）。

2026-06-11 用户反馈修订：
- 冒烟的「治愈手办」偏动漫 → 风格改为三档矩阵（real/cute/figurine），照片到位后跑矩阵供用户挑档；
- 还原度三旋钮：照片输入(i2i 主引擎) > 风格档位 > 编辑幅度措辞；支持多图输入提升还原度。
"""

# ---------- ① 形象风格化（图生图，Seedream-4.5）----------
# 影视大师级共同尾缀（2026-06-11 用户反馈第三轮定稿：宠物要像车机萌宠一样
# "生动地站在屏幕里"、与 App 其它组件产生分离的 3D 感）。
# 视觉机制拆解：
#   轮廓光 rim light  = 主体从背景剥离的"割裂感"来源（电影标准手法）
#   眼神光 catchlight = "生动/有灵魂"的来源
#   微仰英雄视角+大占比 = "站在屏幕上的存在感"
#   皮克斯级全局光照+SSS 毛发 = 大银幕质感（拒绝证件照式平光）
# v5（2026-06-11 深夜定稿）：v4 的皮克斯/CG 词汇把图拽向动漫质感、丢了真实感
# （用户：与自己宠物像才能产生感情）→ 全部换成真实摄影语言。
# 割裂感照样成立：摄影里轮廓光叫 hair light，是影棚宠物摄影标准布光，不依赖 CG 措辞。
_COMMON_TAIL = (
    "真实照片质感：这是一张专业宠物摄影棚实拍的高清照片，"
    "不是CG、不是3D动画、不是插画、不是皮克斯风格；"
    "神态生动：目光温和地平视镜头看着主人，眼睛里有自然的眼神光，耳朵自然放松；"
    "构图：与宠物眼睛同高的平视视角（绝不仰拍、绝不俯拍、宠物不抬头），"
    "正面端坐面向镜头、全身完整入镜居中，占画面约七到八成高度，头顶留呼吸空间；"
    "影棚布光：柔和主光，背后轮廓光（hair light）沿毛发边缘勾出细细的自然亮边、"
    "把宠物从背景中清晰分离，毛发根根分明、真实细腻；"
    "明亮的标准绿幕背景（高饱和高亮度纯绿色 #00FF00，禁止深绿暗绿，"
    "无阴影、无渐变、无地面、无任何环境元素）"
)

_IDENTITY_LOCK = (
    "只从输入照片提取宠物身份特征：毛色、花纹、斑纹分布、耳朵形状、眼睛颜色、鼻子、脸型、体型、尾巴和毛发质感；"
    "不要继承输入照片里的动作、姿态、拍摄角度、背景、光线、裁切、坐卧站趴状态、张嘴闭眼等瞬时表情；"
    "如果原图里有牵引绳、人的手、地面、植物、墙面、路面、家具或背景杂物，必须全部移除，不要变成身体两侧的色块或残片；"
    "If the pet is wearing visible clothing, a vest, harness, collar, tag, bow, or leash attached to the body, preserve it as an identity feature. "
    "Keep the dominant color, position, and silhouette of clear pet clothing or harnesses, especially a purple shirt, purple vest, chest garment, or collar. "
    "Do not turn a dressed pet into a plain undressed pet unless the item is clearly only background clutter. "
)

_CANONICAL_POSE = (
    "无论原图是站立、趴卧、侧身、仰头、低头、奔跑、睡觉、张嘴、被抱着还是只拍到半身，"
    "输出都必须统一为手机 App 动画资产的标准首帧：正面端坐面向镜头，身体垂直稳定，"
    "四肢自然收拢，前爪并排落在身体正下方，头部平视镜头，双眼睁开，嘴巴闭合或轻微微笑，"
    "尾巴完整自然放在身体一侧或身后，脚掌、耳朵、尾巴和全身轮廓完整入镜；"
    "宠物轮廓外必须只剩纯绿幕，不允许出现米黄色、灰白色、道路色、墙面色、布料色、光晕、翅膀状残片或背景块；"
)

STYLE_PROMPTS = {
    # PaiMomo 档：保留猫的纹理识别点，但输出为手机 App 里可长期陪伴的 3D 萌宠形象
    "paimomo3d": (
        _IDENTITY_LOCK + _CANONICAL_POSE +
        "把照片中的宠物转化成类似 PaiMomo 手机宠物软件里的 3D 萌宠角色："
        "整体是高质量 3D 卡通写实风，头身比例轻微萌化，眼睛更圆润有神，"
        "表情亲近、干净、温和，适合放在手机 App 首页长期陪伴；"
        "必须严格保留原猫的金棕虎斑毛色、胸口浅色长毛、脸部纹路、耳朵形状、"
        "绿色眼睛和蓬松尾巴等识别特征，主人一眼能认出是同一只猫；"
        "不要二次元扁平插画，不要贴纸风，不要玩具塑料质感；"
        "使用电影级 3D 渲染质感，柔软真实毛发，圆润体积光，轮廓清晰，"
        "正面端坐、全身完整、脚掌和尾巴完整入镜，透明资产友好构图；"
        "明亮的标准绿幕背景（高饱和高亮度纯绿色 #00FF00，禁止深绿暗绿，"
        "无阴影、无渐变、无地面、无任何环境元素）"
    ),
    # A 档：最大还原（用户定稿档）— 真实照片质感，特征一根毛都不许动
    "real": (
        _IDENTITY_LOCK + _CANONICAL_POSE +
        "Realistic entity version, close to a studio pet photo. Do not make it PaiMomo style, cute cartoon style, toy style, plush style, chibi style, or oversized-eye mascot style. "
        "Preserve natural proportions, realistic fur, real eyes, real nose, and any visible pet clothing or harness from the reference photo. "
        "把照片中的宠物身份原样还原：完整保留它的毛色、花纹、斑纹分布、毛发长短质感、"
        "耳朵眼睛鼻子的形状比例和体型特征，主人必须一眼认出就是自家宠物；"
        "绝不卡通化、绝不动漫化、不改变任何身份特征；" + _COMMON_TAIL
    ),
    # B 档：写实偏可爱 — 保留备选
    "cute": (
        _IDENTITY_LOCK + _CANONICAL_POSE +
        "把照片中的宠物做成写实偏可爱的电影 CG 主角：严格保留原宠物的毛色、"
        "花纹、斑纹分布、耳朵形状、五官特征和整体比例，毛发真实蓬松，"
        "仅做轻微讨喜化（神态更灵动），主人一眼认出是自家宠物，"
        "不要动漫风、不要大眼睛卡通比例；" + _COMMON_TAIL
    ),
    # C 档：治愈手办 — 保留备选（实测漂移最大）
    "figurine": (
        _IDENTITY_LOCK + _CANONICAL_POSE +
        "把照片中的宠物变成3D治愈系玩偶手办：高度保留原宠物的毛色、花纹、斑纹分布、"
        "耳朵形状和五官特征，让主人一眼认出是自家宠物；柔和讨喜、生动可爱；" + _COMMON_TAIL
    ),
}

# Keep the companion-avatar prompt species and color agnostic. Earlier versions
# accidentally described one golden tabby sample and could recolor other pets.
STYLE_PROMPTS["paimomo3d"] = (
    _IDENTITY_LOCK
    + _CANONICAL_POSE
    + "Create a premium PaiMomo-like 3D companion avatar for a mobile pet app. "
      "Use only the uploaded pet as the identity source. Preserve its actual species, breed cues, coat base color, exact marking and stripe layout, eye color, ear shape, muzzle, nose, body build, leg length, fur length, and tail shape. "
      "Do not recolor the pet, do not invent golden fur or green eyes, and do not copy identity traits from any previous example. "
      "Make the styling softly rounded and appealing with expressive but believable eyes, detailed layered fur, soft subsurface lighting, and polished high-end animated-film rendering. "
      "Keep recognizability stronger than cuteness: no extreme chibi proportions, no giant head, no toy-plastic surface, no flat illustration, and no generic mascot face. "
      "Show one complete pet in a calm front-facing seated canonical pose. Both ears, every paw, the full torso, and the entire tail including its tip must be visible with generous clearance from every edge. "
      "Use a uniform bright chroma green #00FF00 background. No floor, contact shadow, cast shadow, gradient, halo, props, text, people, or extra animals."
)

_STATE_SHEET_RULE = (
    "Generate one high-resolution 2x2 pet pose reference layout, not a game sprite sheet, not pixel art, not an icon sheet, not a comic storyboard, no text labels. "
    "The whole image is 2048x2048 with four equal quadrants. Every quadrant contains the same pet identity, same markings, same face, same body ratio, same tail, and same 3D style. "
    "Each pet must be a smooth high-detail 3D render or realistic studio avatar with natural fur detail; absolutely no pixel art, no 8-bit style, no low-resolution sprite, no blocky edges, no retro game asset, no posterized flat color. "
    "If the input pet is wearing visible clothing, a vest, harness, collar, leash, bow, tag, or other pet accessory, treat it as an identity feature and preserve its dominant color, position, and silhouette in every quadrant unless it is only a background object. "
    "Do not turn a dressed pet into a plain generic pet. Do not remove a clearly visible purple shirt, vest, harness, collar, or chest garment. "
    "All quadrants use a pure bright green #00FF00 background with no floor, no ground shadow, no contact shadow, no black motion shadow, no gradients, no props, no people, no extra animals, and no text. "
    "Each quadrant must show the full body centered with safe green margins; ears, paws, body, and tail tip must not be cropped. "
    "Top-left quadrant: idle standard avatar, front-facing seated pose, looking at the user. "
    "Top-right quadrant: fast_walk state, mid-stride brisk walking pose, natural alternating legs, stable body, forward energy, but not running or jumping, no dark shadow or motion trail. "
    "Bottom-left quadrant: sleep state, low prone sleeping pose, eyes fully closed, head resting gently on front paws, relaxed sleeping expression, full tail visible. "
    "Bottom-right quadrant: repeat idle reference, same as top-left for identity consistency, no additional pet."
)

STATE_SHEET_PROMPTS = {
    "real": (
        _IDENTITY_LOCK
        + "Realistic version: generate a realistic 3D / studio pet avatar close to the original pet. Strictly preserve fur color, face shape, eyes, ears, tail, body type, and identity markers. "
        + _STATE_SHEET_RULE
    ),
    "paimomo3d": (
        _IDENTITY_LOCK
        + "Cute version: generate a PaiMomo-like high quality 3D cute pet avatar for a mobile companion app. Make it rounded and friendly while strictly preserving fur color, face shape, eyes, ears, tail, body type, and identity markers. "
        + _STATE_SHEET_RULE
    ),
}

DEFAULT_STYLE = "real"

# ---------- ② 动作片段库（图生视频）----------
# idle 使用首尾帧锁同一张图，保证静息状态机切换不跳变。
# sleep / walk / run 这类硬件状态动画只锁首帧，不锁尾帧；状态已发生后直接播放稳定循环。
_ACTION_LOCK = (
    "动作生成统一规范：把首帧当作已经标准化好的同一只宠物角色模型，只继承首帧外观，"
    "不要根据原始上传照片重新推断姿态；每段动画必须从正面端坐的标准姿态开始，前0.3秒保持静止，"
    "全程保持同一体型比例、同一五官、同一毛色花纹、同一尾巴长度，不能站起来、趴下、翻身、跳出画面、身体拉长或脸部变形；"
)

_CLIP_TAIL = (
    "；身体保持在画面原位不走出画面；镜头完全固定不动；"
    "最后0.8秒必须完全回到首帧的正面端坐姿态并保持静止，尾巴、耳朵、身体轮廓、前爪位置与首帧一致；"
    "纯绿色背景始终保持纯色静止无任何变化"
    " --resolution 720p --duration 5 --camerafixed true --watermark false --generate_audio false"
)

CLIP_PROMPTS = {
    # 静息循环（正脸陪伴 + 适当左右转头互动，2026-06-11 用户定稿）
    "idle": (
        _ACTION_LOCK +
        "idle 静息状态循环：宠物始终正对镜头端坐，目光看向观众，只做轻微呼吸起伏、自然眨眼、"
        "尾巴尖小幅摆动；头部最多向左和向右各转动10到15度，然后回到正视镜头，"
        "动作安静稳定，最后回到与开头相同的正面端坐初始姿态" + _CLIP_TAIL
    ),
}
DEFAULT_CLIP = "idle"

# ---------- ③ 状态首帧图（Seedream 图生图）----------
# run / walk / sleep 不能直接拿端坐 chosen.png 做 Seedance first_frame；
# 需要先生成对应状态的静态首帧，否则视频必然从静息端坐过渡到目标状态。
_STATE_FRAME_RULE = (
    "基于输入宠物角色图，生成同一只宠物的状态动画首帧图。"
    "必须严格保持同一只宠物的身份、毛色、花纹、五官、体型比例、毛发质感和当前3D/写实风格；"
    "只改变姿态到指定状态，不改变品种、脸型、颜色或角色风格；"
    "完整全身入镜，宠物位于画面中心，手机App动画资产构图；"
    "明亮标准绿幕背景 #00FF00，无阴影、无接触阴影、无地面投影、无灰黑拖影、无运动残影、无渐变、无地面、无道具、无文字、无其他动物、无人手。"
)

STATE_FRAME_PROMPTS = {
    "run": (
        _STATE_FRAME_RULE +
        "生成奔跑状态首帧：宠物已经处于奔跑或小跑中的中间步态，四肢呈自然奔跑姿势，身体略有前进动势，"
        "耳朵、尾巴和毛发有速度感，但不要生成脚底阴影、接触阴影、地面暗斑或黑色运动拖影；宠物仍完整居中，不要从端坐姿态开始，不要站立静止，不要坐下。"
    ),
    "walk": (
        _STATE_FRAME_RULE +
        "生成走动状态首帧：宠物已经处于自然行走中的中间步态，一只前爪自然前伸、另一侧后腿配合迈步，"
        "身体平稳，尾巴和耳朵自然摆动，宠物完整居中；不要从端坐姿态开始，不要站立静止，不要坐下。"
    ),
    "sleep": (
        _STATE_FRAME_RULE +
        "生成睡觉状态首帧：严格采用参考姿态式的低伏趴睡，正面偏 3/4 视角，全身完整入镜，广角全身构图，宠物整体只占画面约 60%，上下左右都有明显绿色留白，"
        "胸腹低伏，前爪在脸下方并拢或自然叠放，头部放低并轻轻趴在两只前爪上，双眼闭合，表情放松，尾巴向后或侧后方自然伸出且完整可见；"
        "不要近景特写，不要大头照，不要让脸部占满画面，不要裁切耳朵、身体、前爪或尾巴，不要侧躺，不要蜷缩成球，不要露出肚皮，不要睁眼，不要抬头。"
    ),
}

# ---------- ④ 产品状态循环提示词（2026-06-23）----------
# 硬件上报 sleep/walk/run 时，宠物已经处于该状态；App 不需要展示入睡、起步或停止的过渡过程。
# 因此这些片段应从第一帧就处于目标状态，并在目标状态内无缝循环。
_STATE_LOOP_RULE = (
    "产品状态动画统一规则：硬件上报状态时，宠物已经处于该状态；首帧图已经是目标状态首帧，"
    "视频第一帧必须继承首帧图的目标姿态，"
    "不要展示从静息进入该状态的过渡过程，也不要展示离开该状态的过程；"
    "全程保持同一只宠物的身份、毛色、花纹、五官、体型比例和3D风格；"
    "镜头固定，完整全身入镜，宠物保持画面中心，纯绿色背景，动作稳定、连续、可无缝循环；"
    "禁止跳帧、闪回到初始坐姿、变脸、变体型、出现人手、碗、毯子、窝或其他道具；"
)

_STATE_LOOP_TAIL = (
    "首帧和尾帧必须保持同一状态下的相近姿态，循环时不能突然切回静息姿态；"
    "纯绿色背景始终保持纯色静止无任何变化"
    " --resolution 720p --duration 5 --camerafixed true --watermark false --generate_audio false"
)

CLIP_PROMPTS["sleep"] = (
    _STATE_LOOP_RULE +
    "Strict sleep-loop requirement: the pet is already asleep from frame 1. Eyes must remain fully closed for the entire clip. The head must remain low and resting on or very close to the front paws. Only subtle breathing is allowed. No opening eyes, no looking at camera, no lifting head, no waking up, no getting up, no rolling, no transition into or out of sleep. "
    "sleep 睡眠状态循环：视频第一帧就已经是宠物睡着后的稳定低伏趴睡姿态，正面偏 3/4 视角，全身完整入镜，广角全身构图，宠物整体只占画面约 60%，上下左右都有明显绿色留白，"
    "胸腹低伏，前爪在脸下方并拢或自然叠放，头部放低并轻轻趴在两只前爪上，双眼闭合，表情放松，尾巴向后或侧后方自然伸出且完整可见；"
    "整段只保留轻微、规律的睡眠呼吸起伏，偶尔耳朵或爪子极轻微抽动；不要侧躺，不要蜷缩成球，不要翻身；"
    "不要出现从坐姿趴下、从站姿倒下、打哈欠入睡、抬头醒来、睁眼、起身、翻身、走动或任何入睡/醒来过渡；"
    "不要出现床、毯子、宠物窝、碗、食物、人手或其他道具；"
    + _STATE_LOOP_TAIL
)

CLIP_PROMPTS["walk"] = (
    _STATE_LOOP_RULE +
    "walk 行走状态循环：视频第一帧就已经是宠物正在自然行走的状态，四肢按稳定步态循环，"
    "身体有轻微上下起伏，尾巴和耳朵随步态自然摆动；宠物保持画面中心，可以原地踏步或极小幅位移，"
    "但不要走出画面，不要切换镜头；不要展示从静息站起来、开始走、停下、坐回静息或回头看主人；"
    + _STATE_LOOP_TAIL
)

CLIP_PROMPTS["run"] = (
    _STATE_LOOP_RULE +
    "run 奔跑状态循环：视频第一帧就已经是宠物正在奔跑或小跑的状态，四肢有清晰、连续、有节奏的奔跑循环，"
    "身体弹跳自然，耳朵、尾巴和毛发随速度轻微摆动；不要生成脚底阴影、地面暗斑、接触阴影、黑色拖影或背景残影；宠物保持画面中心，可以原地奔跑或极小幅位移，"
    "但不要冲出画面，不要改变镜头；不要展示从静息启动、加速过程、刹停、坐下或回到静息姿态；"
    + _STATE_LOOP_TAIL
)

# Current product state set: idle / fast_walk / sleep.
# Keep a single locomotion state to avoid duplicated walking/running assets.
STATE_FRAME_PROMPTS["fast_walk"] = (
    _STATE_FRAME_RULE
    + "Generate the first frame for fast_walk: the pet is already in a brisk walking loop pose, mid-stride, full body visible, centered, with a stable body, natural alternating legs, ears and tail subtly following motion. "
    "It must look like fast walking, not running, jumping, sitting, standing still, or transitioning from idle. "
    "Use pure bright green #00FF00 background, no floor, no ground shadow, no contact shadow, no dark shadow under the pet, no motion trail, no props, no text."
)

CLIP_PROMPTS["fast_walk"] = (
    _STATE_LOOP_RULE
    + "fast_walk brisk-walk state loop: the first frame is already a stable brisk-walking pose, and the entire 5-second clip remains inside that state. "
    "Create a smooth treadmill-style in-place brisk walking cycle with natural alternating legs, subtle body bob, ear and tail secondary motion, full body always centered and fully visible. "
    "Lock the pet's body yaw, facing direction, and three-quarter viewing angle to the first frame for the entire clip. The head and torso must keep facing the same screen direction. "
    "Never rotate around the vertical axis, never turn around, never show the back view, never orbit, never walk in a circle, never cross the frame, and never drift sideways. Only the gait cycle and subtle secondary motion may change. "
    "Do not show acceleration, start-from-idle, stop, sit-down, jump, run, sprint, or return to idle. "
    "Avoid any floor, black shadow, contact shadow, dark smear, motion trail, props, text, or camera movement. "
    + _STATE_LOOP_TAIL
)

CLIP_PROMPTS.pop("run", None)
CLIP_PROMPTS.pop("walk", None)
STATE_FRAME_PROMPTS.pop("run", None)
STATE_FRAME_PROMPTS.pop("walk", None)
