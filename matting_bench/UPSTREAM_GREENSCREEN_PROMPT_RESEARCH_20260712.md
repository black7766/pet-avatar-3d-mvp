# Seedream / Seedance 上游绿幕生成优化研究

> 日期：2026-07-12
> 范围：实体版 `fast_walk` / `sleep`；只研究上游首帧和视频生成，不修改现有生成、抠图或展示代码。
> 目标：减少毛发绿边、白/黄边缘发光、尾巴锯齿、身体与尾巴裁切、动作模糊，使本地自适应绿幕算法更容易得到稳定 alpha。

## 1. 执行结论

1. **当前生产背景仍选绿幕，不改蓝幕、灰底或黑底。** 本地算法和质量指标均围绕绿色色度距离、边框绿色采样和去绿溢色设计；换底会同时失去速度与既有调参收益。
2. 上游目标不应写成“发光、荧光、高亮度 `#00FF00`”，应改成：**均匀、哑光、稳定的中高亮标准色度绿，背景与主体分开照明，无反射、无渐变、无压缩色块**。模型不会严格输出指定十六进制值，后处理仍应实采边框 key 色。
3. **实体版禁用 rim light / hair light / backlight。** 它们会把白毛、浅毛和半透明毛尖生成成不属于真实前景的白黄光圈；绿幕抠除后尤其明显。
4. **快走降低单帧位移与毛发甩动，不使用“速度感”“动态模糊”等词。** 采用原地、固定朝向、四肢清晰、短曝光观感的 brisk-walk cycle，尾巴只做低幅摆动。
5. **睡眠保持极低运动。** 只允许胸腹微弱呼吸，尾巴完整静止或极轻抽动；这天然比快走更容易获得干净边缘。
6. **所有首帧把主体限制在画面约 55%–65%，四周至少 15% 纯绿安全区。** 尾巴尖、耳尖、脚掌不能进入边缘 10% 区域。
7. 当前 API 没有可确认的“绿幕质量”“运动模糊强度”“负面提示词”“透明背景”独立参数。这些只能通过输入首帧和自然语言约束，不应伪造成 API 参数。
8. 推荐先做 **A0/A1/A2 三组、每动作各 1 条**。若 A2 明显减少光圈和锯齿，再扩大到多宠物；不要一开始全矩阵重复调用。

## 2. 当前链路审计

### 2.1 当前模型与请求

| 环节 | 当前实现 | 已确认请求项 |
|---|---|---|
| 实体形象 / 状态首帧 | `doubao-seedream-4-5-251128` | `image`、`prompt`、`size=2048x2048`、`response_format=url`、`watermark=false` |
| 对照图模型 | `doubao-seedream-5-0-260128` | 仅作为备选模型 ID |
| 动作视频 | `doubao-seedance-1-5-pro-251215` | `content` 中含文本和 `first_frame`；`generate_audio=false` |
| 视频规格 | 提示词尾部控制 | `--resolution 720p --duration 5 --camerafixed true --watermark false --generate_audio false` |
| 尾帧 | 当前 `SINGLE_SOURCE_ANIMATION=1` 时复用首帧为 `last_frame` | 有助于闭环，但可能压制大动作或造成回拉 |

代码位置：`prompts.py`、`poc.py::animate_request_body()`、`poc.py::step_state_sheet()`。

### 2.2 当前提示词中的有效项

- 已要求纯绿背景、无地面、无阴影、固定镜头、完整全身、无音频。
- 已要求实体版避免卡通化，并禁止背光、轮廓光、绿色反射。
- `fast_walk` 已限制原地快走、禁止转身和横向漂移。
- `sleep` 已限制第一帧即睡眠，只保留呼吸，不展示入睡或醒来过渡。

### 2.3 当前提示词中的冲突与风险

1. 早期通用实体提示曾使用 `hair light` 和“自然亮边”，后续实体段又禁止 rim light。即使最终字符串覆盖了部分语义，长提示中的冲突仍可能让模型生成白边或过曝毛尖。
2. 多处使用“高饱和高亮度纯绿色 `#00FF00`”。生成模型不保证精确色值，“高亮度/明亮”还可能诱发绿色环境反射、边缘 bloom 和视频编码溢色。
3. `fast_walk` 的“耳朵和尾巴随速度摆动”若没有幅度限制，容易导致尾巴细结构高速位移、运动模糊和时序锯齿。
4. 首尾帧锁同图有利于循环，但对快走可能在尾段强行回到特定腿位。A/B 中必须同时记录循环缝和尾段动作是否减速/闪回。
5. 一张 2×2 状态表再裁切，单格有效分辨率和背景一致性都弱于独立状态图。若目标是验证最高边缘质量，A/B 应使用独立 2048×2048 状态首帧，状态表仅保留经济档。

## 3. 官方能力边界

### 3.1 可确认的 Seedream 控制项

火山方舟图片生成 API 文档列出：`model`、`prompt`、`image`、`size`、`seed`、`sequential_image_generation`、`guidance_scale`、`response_format`、`watermark` 等。`guidance_scale` 范围为 1–10；`seed` 可提高相对稳定性。当前代码没有显式传 `seed` 或 `guidance_scale`。

Seedream 官方提示词指南建议使用自然语言清晰描述 **主体 + 行为 + 环境**，再补充风格、色彩、光影和构图；不建议把提示词写成松散关键词堆叠。对本项目意味着应把“受控绿幕资产”作为明确用途，并用不冲突的完整句子说明背景、主体光照和安全留白。

### 3.2 可确认的 Seedance 控制项

方舟视频生成 API 使用 `POST /api/v3/contents/generations/tasks`，请求体含 `model`、`content`，支持图片角色（如 `first_frame` / `last_frame`）；可选 `return_last_frame`。任务结果会返回 `resolution`、`ratio`、`duration` 或 `frames`、`framespersecond` 和 `usage`。

Seedance 1.5 官方资料确认支持首尾帧图生视频。当前项目实际请求还明确传了 `generate_audio=false`。官方资料未给出可独立调节的快门、运动模糊、绿幕、边缘锐度、负面提示词或 alpha 通道参数，因此本研究不把这些写成 API 字段。

### 3.3 关于提示词内 `--参数`

当前项目把 `--resolution`、`--duration`、`--camerafixed`、`--watermark`、`--generate_audio` 拼在文本中；官方示例也使用 `--ratio`、`--dur` 一类文本尾参数。生产代码已实测当前写法可用，但必须以任务查询结果中的 `resolution`、`duration`、`framespersecond` 和 `generate_audio`/媒体轨道检查为准，不能仅相信提示词文字。

## 4. 背景方案判断

| 背景 | 与宠物毛发的分离 | 对当前算法 | 上游风险 | 结论 |
|---|---|---|---|---|
| 均匀中高亮绿幕 | 通常最好；棕、橙、白、黑、灰宠物与绿通道差异明显 | 完全匹配，现有边框采样、色度距离和 despill 可复用 | 浅色毛易受绿反射；绿色项圈/衣物会误伤 | **主方案** |
| 纯亮 `#00FF00` / 荧光绿 | 理论色差大 | 可抠 | 易诱发 bloom、绿溢色和压缩色块；模型也不保证精确色值 | 不推荐作为文案目标 |
| 蓝幕 | 对黄、橙、棕毛可能有优势 | 需重写 key、despill、指标和阈值 | 深灰/黑毛、蓝眼或蓝色配件可能粘连；视频蓝通道噪声通常更敏感 | 仅在宠物/服饰主体含明显绿色时作为兜底 A/B |
| 中性灰底 | 无色溢优势 | 不能靠单一色度快速可靠抠图 | 灰、白、黑毛与背景亮度重叠，胡须和软毛难分 | 不适合当前高速主链 |
| 黑底 | 无绿边 | 需 luma key 或模型抠图 | 黑毛、阴影、鼻子、瞳孔、深色尾巴易丢；半透明浅毛会产生黑边 | 不适合通用宠物 |

专业后期资料一致强调：绿/蓝幕应均匀照明，主体不能与背景同色，并需要 spill suppression。FFmpeg 的 `chromakey`/`colorkey` 本质上按 key 色距离、`similarity` 和 `blend` 生成 alpha；所以**背景色稳定性比追求理论上的绝对 `#00FF00` 更重要**。

### 推荐的上游绿幕描述

> A uniform matte chroma-green studio backdrop, medium-high brightness and stable hue across the entire frame. The background is lit separately from the pet. No glow, bloom, gradient, vignette, texture, compression blocks, green bounce, contact shadow, cast shadow, or floor plane.

## 5. 可直接替换的提示词

以下文本应替换对应的 **状态首帧 prompt + Seedance 动作 prompt** 中相关部分；不要与旧的 `hair light`、`bright glowing green` 等冲突语句叠加。

### 5.1 实体版共用首帧规则

```text
Create a production-ready realistic pet animation keyframe for chroma-key compositing. Preserve the exact identity, species, breed cues, natural body proportions, coat color, markings, fur length, face, ears, paws, clothing, and complete tail of the reference pet. Keep realistic photographic fur; do not make the pet cartoon, chibi, plush, toy-like, or mascot-like.

Show exactly one complete pet, centered, occupying about 55% to 65% of the frame. Keep at least 15% clean green clearance around the entire silhouette. Both ear tips, every paw, the full belly, and the entire tail including the tail tip must be visible. No body part may enter the outer 10% border area.

Use soft neutral diffuse frontal subject lighting with restrained highlights and preserved fur texture. The pet lighting and background lighting are separate. No rim light, hair light, backlight, edge light, bloom, glow, overexposed fur tips, green reflection, green bounce, yellow-white halo, or colored spill on the fur, paws, belly, or tail.

Use a uniform matte chroma-green studio backdrop with medium-high brightness and one stable hue across the whole frame. No floor plane, contact shadow, cast shadow, gradient, vignette, texture, noise pattern, compression blocks, props, text, people, or other animals. The pet silhouette must be crisp but naturally anti-aliased, with individual fur tips visible and no artificially sharpened outline.
```

### 5.2 `fast_walk` 状态首帧

```text
The pet is already in a controlled in-place brisk-walking gait at a front three-quarter angle. Use a readable mid-stride pose: one front paw and the opposite rear paw move forward while the other pair supports the body. Keep all four legs anatomically clear and separated. The torso remains level and stable. The complete tail stays inside the safe area with a gentle natural curve and generous green clearance around its tip.

This is a clean animation keyframe, not an action photograph. Use a short-exposure appearance: no motion blur, directional blur, ghosting, smear, speed lines, dust, floor shadow, dark plate under the paws, or duplicated limbs. Do not crop, enlarge, or stretch the pet.
```

### 5.3 `fast_walk` Seedance 视频

```text
Generate a single continuous 5-second production loop of the same realistic pet brisk-walking in place. Frame 1 is already inside the stable gait cycle; do not start from idle and do not stop or sit down. Keep the same front three-quarter body yaw and the same screen position for the entire clip. The camera is locked.

Use a moderate brisk-walk cadence with small per-frame displacement and clearly readable alternating legs. Keep the torso stable with only subtle vertical movement. Ears move minimally. The complete tail makes only a low-amplitude, slow secondary sway; it must never whip, flick rapidly, blur, split, duplicate, cross the frame edge, or disappear behind the body. Preserve individual fur detail on the tail and body in every frame.

Maintain a short-exposure, clean-frame appearance throughout: no motion blur, temporal ghosting, frame blending, duplicated paws, smeared fur, speed streaks, dark motion trails, floor, contact shadow, or background disturbance. Keep at least 15% uniform green clearance around the moving silhouette at all times. Every frame must contain the complete ears, paws, torso, and full tail tip.

The matte chroma-green background remains one stable, evenly lit hue across every frame. Subject lighting remains soft, neutral, frontal, and constant. No rim light, backlight, bloom, edge glow, green bounce, yellow-white fur halo, or color spill. End on a naturally compatible phase of the same walking cycle; never flash back to idle.

--resolution 720p --duration 5 --camerafixed true --watermark false --generate_audio false
```

### 5.4 `sleep` 状态首帧

```text
The pet is already asleep in a low prone pose at a front three-quarter angle. The chest and belly rest low; the head rests naturally on or immediately above the two front paws; both eyes are fully closed. Show the complete body and the full tail including its tip. Place the tail beside or behind the body as one continuous, clearly separated silhouette with green clearance around it; do not hide it under the torso.

The pet occupies about 55% to 60% of the frame. Keep generous clean green space on every side. No bed, blanket, cushion, pet house, bowl, floor, contact shadow, or cropped anatomy.
```

### 5.5 `sleep` Seedance 视频

```text
Generate a single continuous 5-second production loop of the same realistic pet already sleeping in the exact low prone pose from frame 1. The camera and composition are locked. The eyes remain fully closed and the head remains resting on the front paws for the entire clip.

Only a very small, slow, regular breathing motion is allowed in the chest and belly. Optionally allow one extremely subtle ear twitch. The paws and complete tail remain stable, sharp, fully visible, and inside the safe green area. No tail whipping, no rolling, no pose transition, no lifting the head, no waking, no opening the eyes, no sitting, and no getting up.

Maintain clean-frame detail with no motion blur, temporal ghosting, fur smearing, outline sharpening, duplicated anatomy, floor, contact shadow, or dark residue. The matte chroma-green background remains one uniform stable hue across every frame. Subject lighting remains soft, neutral, frontal, and constant. No rim light, backlight, bloom, edge glow, green bounce, yellow-white fur halo, or color spill. End at the same breathing phase as the beginning for a seamless loop.

--resolution 720p --duration 5 --camerafixed true --watermark false --generate_audio false
```

## 6. A/B 测试矩阵

### 6.1 最小有效矩阵

同一只长毛浅色宠物优先，因为最容易暴露绿边、白边和尾巴锯齿。每个动作共享同一身份图；先测快走，再测睡眠。

| 组 | 首帧 | 视频提示 | 背景 | 目的 | 预期 | 风险 |
|---|---|---|---|---|---|---|
| A0 基线 | 当前提示词 | 当前提示词 | 当前纯亮绿 | 保留可比基准 | 复现现有绿边/光圈 | 无新增成本以外风险 |
| A1 去光圈 | 新共用规则 + 动作首帧 | 当前动作描述 + 新照明/背景约束 | 均匀哑光绿 | 隔离首帧和照明收益 | 白黄边、绿反射下降 | 动作模糊仍可能存在 |
| A2 完整优化 | 新首帧 | 新动作 prompt | 均匀哑光绿 | 验证完整上游方案 | 尾巴更完整、边缘更稳、模糊更低 | 快走可能显得不够激烈 |
| A3 蓝幕兜底 | A2，仅把背景改标准蓝 | A2 蓝幕版 | 均匀蓝 | 仅验证主体有绿色配件/毛色的样本 | 绿色主体保留更好 | 当前算法需蓝幕适配，不纳入主生产 |

### 6.2 第二阶段变量

只有 A2 通过后再做：

| 变量 | B0 | B1 | 判断标准 |
|---|---|---|---|
| 状态首帧 | 2×2 状态表裁切 | 独立 2048×2048 单状态图 | 独立图是否显著改善尾尖、毛尖和安全留白 |
| 快走尾帧 | 仅首帧 | 首尾同图 | 循环缝下降是否值得尾段回拉/动作减速 |
| 视频分辨率 | 480p | 720p | 最终 640px WebP 下，尾巴锯齿和毛发保留是否仍可辨 |
| 绿幕亮度措辞 | bright pure green | medium-high matte chroma green | 边框色方差、绿溢色、光圈是否下降 |
| 动作幅度 | 当前 brisk walk | moderate cadence + low-amplitude tail | 尾巴时序 alpha 与动作可读性的平衡 |

### 6.3 评价指标

每条视频至少取 96 帧、覆盖完整 5 秒；不要只看首中尾三帧。

| 类别 | 指标 | 通过建议 |
|---|---|---|
| 构图 | 任一帧主体触边 / 尾尖缺失 | 0 帧 |
| 背景 | 边框 key 色 RGB 标准差、背景色度距离 P95 | 相比 A0 下降至少 20% |
| 溢色 | 半透明边缘 green fringe ratio | 相比 A0 下降至少 30% |
| 发光 | 外轮廓 1–5 px 亮度相对内侧毛发的异常增量 | 相比 A0 下降至少 30% |
| 锯齿 | 尾巴外轮廓曲率高频能量 / staircase score | 相比 A0 下降至少 20% |
| 时序 | consecutive alpha MAE、尾巴 ROI alpha MAE | 不高于 A0，快走重点看尾巴 ROI |
| 模糊 | 拉普拉斯清晰度、尾巴/爪部局部清晰度 | A2 不低于 A0 的 90% |
| 身份 | 脸、花纹、服饰和体型人工盲评 | 无明显漂移 |
| 动作 | 快走步态可读性 / 睡眠呼吸自然度 | 不因去模糊约束退化成静态图 |
| 循环 | 首尾帧差、尾段闪回/减速 | 无可见闪回，接缝不可察觉 |

## 7. 成本与时间估算

以下使用项目 2026-07-10 已记录口径，不替代方舟实际账单：Seedream 4.5 约 `0.25 元/张`；Seedance 1.5 Pro 无声视频约 `8 元/百万 tokens`。已有 720p 4 秒样本约 `87,300 tokens/条`；5 秒实际 token 以任务 `usage` 为准。

### 最小矩阵成本

| 内容 | 数量 | 粗估费用 | 墙钟时间说明 |
|---|---:|---:|---|
| A0/A1/A2，2 动作独立状态首帧 | 6 张 | 约 ¥1.50 | 现有记录单张约 20–180 秒，受重试/排队影响大 |
| A0/A1/A2，2 动作 720p/5s 视频 | 6 条 | 约 ¥4.2–5.3 | 可并发提交；单条历史约 60–130 秒，云端长尾明显 |
| 本地抠图与指标 | 6 条 | API 费用 0 | 96 帧单条历史约 30–40 秒，可并行 |
| **合计** | **6 图 + 6 视频** | **约 ¥5.7–6.8** | 并发条件下预计 8–20 分钟完成，不含失败重试 |

降本顺序：

1. 先只对 `fast_walk` 跑 A0/A1/A2；尾巴问题改善后再跑 `sleep`，可把首轮视频请求减半。
2. 首轮筛查可用 480p，但最终结论必须回到 720p；项目旧探针显示 480p token 可下降约 55%，同时可能出现转身和边缘质量下降。
3. 固定输入、prompt 版本和可用时的 `seed`，保留失败样本；不要只挑最好的一条得出结论。
4. 独立状态图只在 A2 通过后扩大。若状态表与独立图差异不显著，生产继续使用状态表以节省生图调用。

## 8. 推荐落地顺序

1. 建立 A0 基线：冻结当前原始视频、任务 `usage`、首帧和 96 帧质量指标。
2. 先生成 **A1 快走**，确认去掉发光措辞后白黄光圈和绿反射是否下降。
3. 再生成 **A2 快走**，重点复核尾巴 ROI、爪部模糊和动作幅度。
4. A2 快走通过后，以同样共用规则生成 A2 睡眠；睡眠只需验证呼吸自然度和毛发边缘稳定。
5. 只有宠物本身存在大面积绿色毛色、衣物、项圈时才触发蓝幕支线；不要把蓝幕变成常规双跑成本。
6. 上游合格后仍保留本地自适应绿幕 + 时序平滑；提示词只能降低坏输入概率，不能替代 alpha 后处理。

## 9. 资料来源

### 火山方舟官方

- [创建视频生成任务 API](https://api.volcengine.com/api-docs/view?action=CreateContentsGenerationsTasks&serviceCode=ark&version=2024-01-01)：请求结构、图片角色、`return_last_frame`。
- [查询视频生成任务 API](https://api.volcengine.com/api-docs/view?action=GetContentsGenerationsTask&serviceCode=ark&version=2024-01-01)：结果中的分辨率、时长/帧数、帧率和 usage。
- [图片生成 API](https://api.volcengine.com/api-docs/view?action=ImageGenerations&serviceCode=ark&version=2024-01-01)：`image`、`size`、`seed`、`guidance_scale`、`watermark` 等参数。
- [Seedance 1.5 Pro 提示词指南](https://www.volcengine.com/docs/82379/2168087?lang=zh)：当前模型对应官方提示词资料，2026-07-07 更新。
- [Seedream 4.0–5.0 提示词指南](https://www.volcengine.com/docs/82379/1829186)：自然语言、主体/行为/环境、风格/光影/构图建议。
- [Seedream 助力 Seedance 生视频最佳实践](https://www.volcengine.com/docs/82379/1951250)：图像资产到视频生成的官方工作流资料。
- [Seedance 产品页](https://www.volcengine.com/activity/seedance)：Seedance 1.5 首尾帧能力。

### 色键一手/厂商资料

- [Adobe Premiere Ultra Key](https://helpx.adobe.com/sg/premiere/desktop/add-video-effects/effects-and-transitions-library/apply-and-customize-chromakey-using-the-ultra-key-effect.html)：均匀照明绿/蓝背景、主体避开同色、边缘清理和 spill suppression。
- [FFmpeg Filters: chromakey / colorkey](https://ffmpeg.org/ffmpeg-filters.html#chromakey)：key 色、`similarity`、`blend` 的正式定义，说明均匀稳定 key 色对 alpha 的直接作用。
- [Apple Core Image Chroma Key](https://developer.apple.com/documentation/coreimage/applying-a-chroma-key-effect)：以颜色范围映射 alpha 的官方实现示例。

## 10. 明确不作的承诺

- 提示词无法保证模型逐像素输出 `#00FF00`。
- `camerafixed` 无法保证宠物朝向绝对不漂移；它主要约束镜头，不是骨骼/身体 yaw 锁。
- “no motion blur” 是生成约束，不是已确认的独立数值参数。
- 单次 A/B 好样本不能证明稳定性。正式定版至少覆盖短毛/长毛、深色/浅色、蓬松尾/细尾和带绿色配件样本。
- 上游优化不能完全消除半透明毛发的色污染，最终仍需要本地 despill、边缘融合和时序处理。
## 11. 最小实测结果（2026-07-12）

- Seedance 任务：`cgt-20260712172952-stlls`，720p、5 秒、无声，usage `108,900` tokens。
- 云端提交至成功：`107.8s`；下载：`3.6s`。
- Edge v2 绿边：A0 `0.006302` → A1/A2 `0.006154`（改善 `2.3%`）。
- Edge v2 碎片：A0 `0.0127%` → A1/A2 `0.0122%`（改善 `3.5%`）。
- Edge v2 时序 alpha MAE/s：A0 `0.510197` → A1/A2 `0.448271`（改善 `12.1%`）。
- 主体触边帧：A0 `0/96` → A1/A2 `2/96`；最小留白 `41px` → `4px`。
- 尾巴/侧向附属区域 alpha 保留代理：A0 `0.9937` → A1/A2 `0.9942`；主连通域占比 `1.0000` → `1.0000`。
- 绿幕 profile `bg_floor`：A0 `0.8591` → A1/A2 `0.5833`，新视频背景均匀性/可分性反而下降。
- 人工复核：A1/A2 仍生成明显地面与接触阴影，背景存在亮度渐变，宠物由侧向快走漂移到正面；模型未稳定遵循无地面、均匀绿幕和固定朝向。
- 决策：**不直接替换生产提示词**。保留其中的低幅尾摆、短曝光和去轮廓光约束，但下一轮必须单独强化无地面/无接触阴影并缩小主体运动范围。
- 说明：尾巴指标是无语义 GT 条件下的侧向附属区域代理，必须配合动图人工复核，不能视作尾巴语义分割准确率。
- 完整机器结果：`poc_output/upstream_prompt_ab_20260712/result.json`。
