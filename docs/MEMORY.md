# 项目记忆

## 项目入口

- 本地独立项目目录：`D:\work\公司分析\company_anls\产品分析\宠物3D动效生成MVP`
- 当前主页面：`poc_output/paimomo_compare.html`
- 当前完整服务：`petavatar_server.py`
- 当前生成主流程：`poc.py`
- 当前提示词集中维护：`prompts.py`
- 早期本地 WebP MVP：`legacy_webp_mvp/`

## 当前决策

- 不再把这套 Demo 散落在 `D:\work\ai_tools` 多个目录里，新项目目录作为后续交接和投资人 Demo 的统一入口。
- 火山方舟 API Key 不进入 git。使用 `.env` 或环境变量 `ARK_API_KEY`。
- Seedance 调用必须保持无声：`generate_audio=false`。
- 展示资产优先使用透明 WebP/WebM，页面内预览，不提供媒体下载入口。
- 当前展示方案属于“视频资产 + 状态机”的 2.5D 方案，不是最终 skeleton/IK 实时 3D 方案。
- 2026-06-23 更新：上传原图必须同时保存到 `inputs/` 和对应 `poc_output/<pet_id>/uploaded_original.*`。历史卡片优先读取输出目录内副本，避免迁移项目后源图丢失。
- 2026-06-23 更新：动作体系正式收敛为四种产品状态：`idle` 静息、`run` 奔跑、`walk` 走动、`sleep` 睡觉。非四状态入口不再作为生成入口或页面入口。

## 方案脉络

1. 先用 Seedream 把用户宠物照片标准化为实体版和萌宠版。
2. 用 Seedance 生成动作视频。
3. 本地绿幕/背景抠图，处理毛发边缘。
4. 合成透明 WebP/WebM。
5. 页面通过状态按钮和历史卡片展示原图、实体版、萌宠版。
6. 当前动作体系只保留 `idle/run/walk/sleep` 四大状态；四大状态都以稳定循环资产为主，过渡动画不是当前默认规格。

## 待办

- 把 `prompts.py` 里的终端乱码注释清理成标准 UTF-8 中文，避免后续协作阅读困难。
- 基于《猫狗四大状态与静息动作池设计表.xlsx》实现静息动作池权重、冷却和采样安全标记。
- 基于当前四状态生成更多宠物样例，并补充自动 QA 结果。
- 增加自动 QA：首尾帧差异、主体可见面积、绿边残留、文件体积、帧数。
