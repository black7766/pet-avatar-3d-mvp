# 宠物动效 V4 生产方案

更新日期：2026-07-15

## 生产基线

当前唯一生产版本为 `airgap_motion_v4`。旧的 state sheet、多模型抠图、SAM2 默认增强、地面绿幕和历史 WebP MVP 均不参与默认生成。

动作提示词已拆分到 `prompt_config/`：产品动作调整只编辑 `prompt_config/actions.py`，背景、绿幕、灯光、完整身体、身份一致性、循环和 API 参数由 `prompt_config/locked.py` 自动拼接。`prompts.py` 保留为生成流程的兼容入口。

| 项目 | 当前值 |
|---|---|
| 形象生成 | Seedream 4.5，实体版 / 萌宠版各 1 张 |
| 状态首帧 | `idle` 使用选定形象；`fast_walk`、`sleep` 分别生成目标状态首帧 |
| 动作模型 | `doubao-seedance-1-5-pro-251215` |
| 动作资产 | `idle`、`fast_walk`、`sleep` |
| 视频规格 | 720P、5 秒、24fps、无声、固定相机 |
| 生产提示词 | 无物理地面、无接触阴影、无环境遮蔽、主体全身完整、纯色绿幕 |
| 抠图 | 自研 adaptive green matte + 运动 alpha 反混合恢复 |
| 时序 | 双向光流融合，工作尺寸 384px，快走保护下半身当前帧轮廓 |
| 实体版边缘 | odds-domain alpha 对比 1.18 |
| 结构模型 | SAM2 默认关闭 |
| 输出 | 640px、24fps、质量 94 的透明 WebP |

## 固定环境参数

`petavatar_server.py` 的 `subprocess_env()` 是服务端生产参数的权威入口：

```text
PETAVATAR_CLIP_RESOLUTION=720p
PETAVATAR_CLIP_DURATION=5
PETAVATAR_WEBP_FPS=24
PETAVATAR_WEBP_WIDTH=640
PETAVATAR_WEBP_QUALITY=94
PETAVATAR_MATTE_PIPELINE=memory
PETAVATAR_PARALLEL_MATTE=3
PETAVATAR_PRODUCTION_PIPELINE=airgap_motion_v4
PETAVATAR_GREEN_TEMPORAL_REFINE=1
PETAVATAR_TEMPORAL_FLOW_SIZE=384
PETAVATAR_GREEN_MOTION_ALPHA_REFINE=1
PETAVATAR_GREEN_MOTION_THRESHOLD=10
PETAVATAR_GREEN_MOTION_DILATE=4
PETAVATAR_GREEN_PRESERVE_LOWER_MOTION=1
PETAVATAR_REAL_ALPHA_EDGE_CONTRAST=1.18
PETAVATAR_SAM2_STRUCTURAL_REFINE=0
```

所有 Seedance 请求必须保持 `generate_audio=false`。下载后仍会通过 ffmpeg 再次移除音轨。

## 完整流程

1. 上传原图，同时保存到 `inputs/` 和宠物结果目录。
2. Seedream 生成实体版或萌宠版标准形象。
3. 为快走、睡眠分别生成目标状态首帧；静息使用标准形象。
4. 三段 Seedance 任务并行提交，生成稳定循环态，不生成状态过渡。
5. ffmpeg 一次解码到内存，执行 V4 自适应绿幕抠图。
6. 对 alpha 执行运动恢复、双向时序融合、碎片投票和实体版边缘收紧。
7. 重排主体尺寸，编码透明 WebP，并写入 `metrics.json`。
8. 页面只做内嵌预览，不提供媒体下载入口。

## 运行与复现

```powershell
python petavatar_server.py --host 127.0.0.1 --port 8792
```

当前 V4 实体版样例：

```text
poc_output/pet_20260715_155012_95feab9d_real/
```

本地预览：

```text
http://127.0.0.1:8792/view/pet_20260715_155012_95feab9d_real
```

## 高清档原则

如启用 1080P，必须同时把最终 WebP 提升到 960px；如果仍输出 640px，大部分源视频分辨率收益会在缩放阶段丢失。

```powershell
$env:PETAVATAR_CLIP_RESOLUTION='1080p'
$env:PETAVATAR_WEBP_WIDTH='960'
python petavatar_server.py --host 127.0.0.1 --port 8792
```

建议 720P/640px 作为默认档，1080P/960px 作为高清可选档，不直接替换默认档。

### 720P 与 1080P token

当前 1:1、5 秒、24fps 的 720P 三段任务实测每段均为 `108,900 completion_tokens`。方舟 1:1 规格中，720P 为 960×960，1080P 为 1440×1440，因此像素面积和 token 约为 2.25 倍。

| 规格 | 单段 5 秒 | 三段 | 在线无声费用 |
|---|---:|---:|---:|
| 720P 实测 | 108,900 | 326,700 | ¥2.6136 |
| 1080P 估算 | 243,000–245,025 | 729,000–735,075 | ¥5.832–¥5.8806 |
| 升级增量 | +134,100–136,125 | +402,300–408,375 | +¥3.2184–¥3.267 |

计价按 Seedance 1.5 Pro 在线无声 `¥8/百万 tokens`。准确账单必须以任务返回的 `usage.completion_tokens` 为准；以上 1080P 数据未调用付费 API，是按官方分辨率和当前 121 帧实际行为推算。

- [方舟模型列表](https://docs.volcengine.com/docs/82379/1330310?lang=zh)
- [创建视频生成任务与分辨率规格](https://docs.volcengine.com/docs/82379/1520757?lang=zh)
- [模型价格与 token 公式](https://docs.volcengine.com/docs/82379/1544106?lang=zh)
- [查询任务与 usage](https://docs.volcengine.com/docs/82379/1521309?lang=zh)

## 回滚开关

```text
PETAVATAR_REAL_ALPHA_EDGE_CONTRAST=1.0
PETAVATAR_GREEN_MOTION_ALPHA_REFINE=0
PETAVATAR_GREEN_PRESERVE_LOWER_MOTION=0
PETAVATAR_GREEN_TEMPORAL_REFINE=0
```

不要恢复旧的地面或接触阴影提示词。产品内的落地感应由独立稳定阴影层实现。
