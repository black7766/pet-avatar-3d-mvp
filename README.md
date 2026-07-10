# 宠物 3D 动效生成 MVP

这是把当前“宠物照片生成 3D 动效形象”方案、代码、前端 Demo、提示词和样例产物整理到一起的独立项目目录。

目标能力：

- 上传一张宠物照片，自动生成两版形象：实体版和萌宠版。
- 每版生成无声循环状态资产：`idle`（静息）、`fast_walk`（快走，合并走动/奔跑表现）、`sleep`（睡眠）。
- 使用 Seedream 做图生图形象生成，Seedance 做视频动作生成，本地完成抠图、透明 WebP 合成和前端展示。
- 前端只做页面内预览，不触发浏览器下载。

## 目录

```text
.
├── petavatar_server.py       # 本地上传生成 API + 静态预览服务
├── poc.py                    # 生成主流程：stylize / state_frames / animate / matte
├── prompts.py                # Seedream / Seedance 提示词集中维护
├── serve_poc_output.py       # 只预览 poc_output 的轻量静态服务
├── inputs/                   # 样例输入图和上传图
├── poc_output/               # 前端页面和已生成的最终动效资产
├── legacy_webp_mvp/          # 早期 imagegen + 本地裁帧 WebP MVP
└── docs/                     # 需求、方案、状态机和上下文记忆
```

## 运行

1. 准备 Python 3.10+ 和 ffmpeg。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 复制 `.env.example` 为 `.env`，填入火山方舟 Ark Key：

```bash
ARK_API_KEY=your_ark_key_here
```

4. 启动完整上传生成服务：

```bash
python petavatar_server.py --host 127.0.0.1 --port 8792
```

5. 打开：

```text
http://127.0.0.1:8792/paimomo_compare.html
```

只看已有产物，不跑生成：

```bash
python serve_poc_output.py --host 127.0.0.1 --port 8792
```

## 当前链路

```mermaid
flowchart LR
  A["上传宠物照片"] --> B["Seedream 图生图"]
  B --> C["实体版 / 萌宠版形象"]
  C --> D["Seedream 目标状态首帧"]
  D --> E["Seedance 4 秒状态循环"]
  E --> F["内存并行绿幕抠图"]
  F --> G["透明 WebP / WebM"]
  G --> H["前端状态机展示"]
```

## 关键约束

- 不提交 `.env`，API Key 只放本地环境变量或本地 `.env`。
- `generate_audio` 必须保持 `false`，生成无声视频。
- WebP 是展示资产，不能再触发下载。
- 后续新增动作时，优先改 `prompts.py`，不要把提示词散落在页面里。
- `sleep`、`fast_walk` 按“硬件状态已发生后的稳定循环”生成，不做入睡、起步、停止、醒来等过渡过程。
- 中间帧和 raw 视频默认忽略，避免 git 仓库膨胀。
- 上传原图会保存两份：`inputs/<pet_id>.<ext>` 作为全局索引，`poc_output/<pet_id>/uploaded_original.<ext>` 作为结果目录内副本。历史展示优先读取结果目录内副本，避免迁移或分享时出现“上传原图未找到”。

## 2026-07-10 A/B 优化结论

- 快走和睡眠先生成独立状态首帧，Seedance 使用相同目标状态首尾帧；不再用端坐图驱动全部动作。
- Seedance 1.5 Pro 使用最短合法时长 4 秒，明确 `generate_audio=false`。
- 本地抠图改为 FFmpeg 一次解码到内存，三个动作并行处理，不再落盘 RGB/RGBA 中间 PNG。
- 抠图在透明过渡带做前景色恢复，并按风格拆分边缘策略：实体版在轮廓内侧 12px 做邻近深层毛色回填、黄绿/暖白曝光抑制、亮度上限与 1-3px Alpha 羽化；萌宠版保持保守的 8px 策略。高饱和绿色虹膜和饰品不处理。
- 快走首尾已经足够接近时直接循环，不再强制做 Alpha 交叉淡化，避免尾帧出现半透明双影。
- 锁定首尾帧时保留完整端点，不再执行旧的头尾裁剪和交叉淡化。
- 同源 A/B：视频 tokens `653,400 → 523,800`，增加最终轮廓光晕修复后本地抠图墙钟 `75.7s → 46.5s`，平均循环缝 `0.991% → 0.286%`，单宠刊例成本约 `¥5.73 → ¥5.69`。测试猫实体版的内部边缘曝光像素率从 `20.21%` 降至 `2.22%`。
- 480p 经济档探针：同一快走片段 `87,300 → 38,800` tokens；按六段投影单宠约 `¥3.36`（较旧链路约低 41.3%）。本次 480p 出现完整转身，动作质量不等价；720p 保留为质量默认档，并仍需对快走朝向漂移做自动 QA。
- 对比页：`poc_output/ab_compare_pet_20260710_121221_5ce7716e.html`。

## 文档入口

- [需求说明](docs/REQUIREMENTS.md)
- [四大状态与提示词方案](docs/STATE_AND_PROMPT_STRATEGY.md)
- [项目记忆](docs/MEMORY.md)
