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
_COMMON_TAIL_BASE = (
    "真实照片质感：这是一张专业宠物摄影棚实拍的高清照片，"
    "不是CG、不是3D动画、不是插画、不是皮克斯风格；"
    "神态生动：目光温和地平视镜头看着主人，眼睛里有自然的眼神光，耳朵自然放松；"
    "构图：与宠物眼睛同高的平视视角（绝不仰拍、绝不俯拍、宠物不抬头），"
    "正面端坐面向镜头、全身完整入镜居中，占画面约七到八成高度，头顶留呼吸空间；"
)

# 2026-07-13：实体版（real）与轮廓光解耦。hair light 会把白/黄亮边烘焙进不透明毛尖，
# 绿幕抠除后形成任何 alpha 后处理都无法完全修复的光圈（见 matting_bench 上游提示词研究
# §2.3）。real 档改用主体/背景分离照明的哑光绿幕描述；cute/figurine 保留 hair light。
_COMMON_TAIL_RIMLIGHT = (
    _COMMON_TAIL_BASE +
    "影棚布光：柔和主光，背后轮廓光（hair light）沿毛发边缘勾出细细的自然亮边、"
    "把宠物从背景中清晰分离，毛发根根分明、真实细腻；"
    "明亮的标准绿幕背景（高饱和高亮度纯绿色 #00FF00，禁止深绿暗绿，"
    "无阴影、无渐变、无地面、无任何环境元素）"
)

_COMMON_TAIL_MATTE = (
    _COMMON_TAIL_BASE +
    "影棚布光：柔和中性正面主光打在宠物身上，毛发根根分明、真实细腻，"
    "宠物照明与背景照明彼此独立；禁止轮廓光、逆光、发光亮边、过曝毛尖和绿色反光；"
    "这是用于透明合成的单色抠像底板，不是摄影棚场景、不是无缝背景纸，也没有地面平面或地平线；"
    "背景从四角到宠物脚底、尾巴下方都必须是完全相同的均匀哑光中高亮度标准色度绿，"
    "禁止接触阴影、投影、脚底暗区、渐变、光晕、反射和任何环境元素；"
    "宠物全身占画面高度百分之七十到七十八，脚掌下方保留百分之十以上纯绿安全区，"
    "耳尖、脚掌、腹部和完整尾巴均不得贴边或被裁切"
)

# 兼容旧引用：默认尾缀保持带轮廓光的原行为。
_COMMON_TAIL = _COMMON_TAIL_RIMLIGHT

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
        "Use soft neutral diffuse studio lighting from the camera side. No rim light, no backlight, no edge glow, no overexposed fur outline, no green reflected light, and no green or yellow color spill on fur. "
        "把照片中的宠物身份原样还原：完整保留它的毛色、花纹、斑纹分布、毛发长短质感、"
        "耳朵眼睛鼻子的形状比例和体型特征，主人必须一眼认出就是自家宠物；"
        "绝不卡通化、绝不动漫化、不改变任何身份特征；" + _COMMON_TAIL_MATTE
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

# ---------- ② 动作配置 ----------
# 动作文字与锁定生产约束位于 prompt_config/。修改动作只编辑 actions.py。
from prompt_config import CLIP_PROMPTS, STATE_FRAME_PROMPTS

DEFAULT_CLIP = "idle"
