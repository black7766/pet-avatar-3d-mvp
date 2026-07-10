# 宠物动效抠图模型本地部署与对比报告

测试日期：2026-07-10

## 结论

不建议用某一个通用抠图模型直接替换当前自适应绿幕算法。当前最适合产品的方案是：

1. 当前自适应绿幕算法继续作为默认主链，负责高置信度绿幕、细毛保留和 RGB 去绿。
2. `BiRefNet-matting` 常驻 GPU，作为低置信度帧的结构修复与回退模型。
3. 模型 alpha 不与当前 alpha 全局平均，只用于补主体缺口、去除非主体碎片和约束主连通域。
4. 边缘颜色仍使用绿幕前景颜色估计；模型不能修复 Seedance 已烘焙进 RGB 的白色曝光边和轮廓光。
5. `MatAnyone` 和 `SAM 2.1` 可作为目标传播研究工具，不应直接输出最终毛发 alpha。

`BiRefNet-matting` 是本轮最值得保留的成熟模型：24 个连续快走帧上的时序 alpha 误差为
`0.01197`，优于当前算法的 `0.01382`；主体碎片率也从当前算法的 `0.1345%`
降到 `0.0010%`。代价是背景 alpha 泄漏略高，因此应作为结构先验，而不是无条件覆盖当前结果。

## 测试环境

| 项目 | 环境 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti 11GB |
| Driver | 591.86 |
| CUDA Toolkit / NVCC | 12.1 / 12.1 |
| Python | 3.11.9 |
| 静态样本 | `idle / fast_walk / sleep` 各取第 0、48、96 帧，共 9 帧 |
| 静态分辨率 | 960x960 RGBA |
| 时序样本 | `fast_walk` 连续前 24 帧 |
| 时序分辨率 | 640x640 RGBA |

输入来自同一只实体版宠物的三段 Seedance 绿幕视频：

- `poc_output/pet_20260710_121221_5ce7716e_real_after/raw_idle.mp4`
- `poc_output/pet_20260710_121221_5ce7716e_real_after/raw_fast_walk.mp4`
- `poc_output/pet_20260710_121221_5ce7716e_real_after/raw_sleep.mp4`

本测试没有人工绘制的 alpha 真值。`pseudo MAE` 只在高置信绿幕背景和高置信非绿前景上计算，
可用于发现明显缺口、背景泄漏和碎片，但不能替代毛发边缘人工复核。

## 已部署模型

| 模型 | 本地状态 | 许可证结论 | 官方链接 |
|---|---|---|---|
| BiRefNet-matting | 已部署、已测 9 帧与 24 连续帧 | 上游 MIT；HF 检查点卡未单列许可证字段，商用前保留卡片和上游 LICENSE | [HF](https://huggingface.co/ZhengPeng7/BiRefNet-matting) / [GitHub](https://github.com/ZhengPeng7/BiRefNet) |
| BiRefNet General | 已部署、已测 9 帧 | MIT | [GitHub](https://github.com/ZhengPeng7/BiRefNet) |
| ViTMatte-S | 已部署、已测 9 帧 | 代码 MIT；HF 模型卡 Apache-2.0 | [GitHub](https://github.com/hustvl/ViTMatte) / [HF](https://huggingface.co/hustvl/vitmatte-small-composition-1k) |
| Paddle PP-MattingV2 | 已部署、CUDA/CPU 均实测 | PaddleSeg Apache-2.0；官方权重压缩包无独立 LICENSE | [PaddleSeg Matting](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/Matting) |
| BEN2 Base | 已部署、重复跑结果一致 | MIT；只测试 Base，未测试商业 Full 版 | [GitHub](https://github.com/PramaLLC/BEN2) / [HF](https://huggingface.co/PramaLLC/BEN2) |
| U2Net | 已部署 ONNX、已测 9 帧 | Apache-2.0 | [GitHub](https://github.com/xuebinqin/U-2-Net) |
| ISNet General | 已部署 ONNX、已测 9 帧 | Apache-2.0 | [GitHub](https://github.com/xuebinqin/DIS) |
| BiRefNet General Lite | 已部署 ONNX、已测 9 帧 | MIT | [GitHub](https://github.com/ZhengPeng7/BiRefNet) |
| MatAnyone v1 | 已部署、已测 24 连续帧 | S-Lab 1.0，非商业；商用需授权 | [GitHub](https://github.com/pq-yang/MatAnyone) |
| SAM 2.1 Small | 已部署、已测 24 连续帧 | Apache-2.0 | [GitHub](https://github.com/facebookresearch/sam2) |
| RMBG-2.0 | 适配器已完成，权重未下载 | HF 权重 gated；自托管非商业，商业使用需另签协议 | [HF](https://huggingface.co/briaai/RMBG-2.0) / [GitHub](https://github.com/Bria-AI/RMBG-2.0) |

### 百度方案

百度智能抠图公有云 API 官方说明支持人、动物、食物和物品，但它是外部云服务，不提供可直接下载的
同等本地权重。本轮未使用百度云凭证调用该 API。为验证百度本地开源路线，已部署官方 PaddleSeg
`PP-MattingV2-STDC1-human-512`；它速度很快，但发布检查点明确属于人像抠图，宠物测试是跨域使用。

- [百度智能抠图 API](https://ai.baidu.com/ai-doc/IMAGEPROCESS/rm8zl3koj)
- [百度智能抠图产品页](https://ai.baidu.com/tech/imageprocess/segment)

## 静态质量指标

下表中 `raw 绿边` 是模型原始 RGBA 的边缘污染；`去绿后` 使用同一套只作用于画布相连外轮廓的
颜色清理，模型 alpha 保持不变。ViTMatte 本身就是“当前 alpha + 窄 trimap 细化”的混合结果。

| 方法 | pseudo MAE ↓ | 背景 alpha ↓ | 前景损失 ↓ | raw 绿边 ↓ | 去绿后 ↓ | 碎片率 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 当前自研绿幕 | **0.00179** | **0.00083** | 0.00275 | 0.01452 | 0.01452 | 0.1345% |
| BiRefNet General | 0.00239 | 0.00274 | 0.00205 | 0.01879 | 0.01736 | 0.0021% |
| BiRefNet-matting | 0.00265 | 0.00342 | 0.00188 | 0.01845 | 0.01775 | **0.0010%** |
| ViTMatte-S 混合 | 0.00223 | 0.00181 | 0.00265 | 0.01710 | 0.01710 | 0.1097% |
| Paddle PP-MattingV2 | 0.00234 | 0.00346 | **0.00123** | 0.10751 | 0.01474 | 0.0081% |
| BEN2 Base | 0.00545 | 0.00056 | 0.01034 | 0.03693 | **0.00899** | 0.0128% |
| U2Net | 0.00442 | 0.00230 | 0.00654 | 0.04125 | 0.01502 | 0.0022% |
| ISNet General | 0.00345 | 0.00169 | 0.00520 | 0.05966 | 0.02430 | 0.0041% |
| BiRefNet General Lite | 0.00379 | 0.00287 | 0.00472 | 0.08461 | 0.02124 | 0.0002% |

解读：

- 当前算法在“背景干净”和“高置信前景保留”上仍然最好。
- BiRefNet 系列的优势是主体连贯、碎片少，并非所有像素误差都更低。
- BEN2 的去绿后颜色指标很好，但 alpha 已丢掉较多细毛，不能只看绿边分数。
- Paddle 在猫身上意外保持了较完整主体，但其人像训练域和权重许可证说明不足，不宜直接商用主链。
- ViTMatte 只有在 trimap 已经可靠时才有意义；当前 9 帧提升不足以抵消整条链路复杂度。

## 连续帧结果

| 方法 | 时序 alpha MAE ↓ | pseudo MAE ↓ | 前景损失 ↓ | 绿边 ↓ | 结论 |
|---|---:|---:|---:|---:|---|
| 当前自研绿幕 | 0.01382 | 0.00301 | 0.00492 | 0.01068 | 毛发完整，但独立碎边可能逐帧闪动 |
| BiRefNet-matting 原始前景重建 | **0.01197** | **0.00277** | **0.00221** | **0.00662** | 本轮连续帧最佳 |
| MatAnyone v1 | 0.01402 | 0.00353 | 0.00357 | 0.01143 | 鼻口出现内部半透明缺口，未改善本样本时序 |
| SAM 2.1 Small | 不直接可比 | - | - | - | 传播稳定，但输出只有 0/255 二值 mask，细毛会硬切 |

`BiRefNet-matting` 再叠加现有外轮廓去绿后，24 帧绿边指标反而升到 `0.01587`。原因是模型官方
前景重建已经处理了部分颜色，二次处理会过度修正。因此生产使用时应按 provider 选择 RGB 策略，
不能把同一个后处理无条件套给所有模型。

## 速度与资源

不同模型官方接口的计时边界并不完全相同。下表优先列“热模型推理”；全流程数据包含解码、
预处理、前景估计或 PNG 写盘，只用于估算生产吞吐。

| 方法 | 热推理/调用 | 已测全流程 | 峰值显存 | 说明 |
|---|---:|---:|---:|---|
| 当前自研绿幕 | CPU | 440.7 ms/帧 @ 640p | 0 GPU | 三段 96 帧并行生产实测墙钟约 35.3s |
| Paddle PP-MattingV2 | **49.1 ms/帧** | 398.5 ms/帧 @ 960p | 未统一测量 | 512 输入，人像检查点 |
| ViTMatte-S | 78.4 ms/帧 | 1139 ms/帧 @ 960p | 597 MB | 全流程先运行当前绿幕生成 trimap |
| U2Net ONNX | 94.2 ms/帧 | - | 未统一测量 | 去掉首帧冷启动 |
| BiRefNet-matting | 197.6 ms/帧 | 421.8 ms/帧 @ 640p | 1636 MB allocated | 模型加载约 1.94s |
| BiRefNet General | 208.4 ms/帧 | 715.5 ms/帧 @ 960p | 约 1636 MB allocated | 官方 PyTorch FP16 |
| ISNet ONNX | 315.6 ms/帧 | - | 未统一测量 | 去掉首帧冷启动 |
| BEN2 Base | 361.7 ms/帧 | 668.1 ms/帧 @ 960p | 2587 MB allocated | 首次 CUDA warm-up 很慢 |
| BiRefNet General Lite ONNX | 5043.3 ms/帧 | - | 未统一测量 | 当前 Windows ONNX 路径不具备生产价值 |
| MatAnyone v1 | 11.33 fps @ 640p | 冷启动总计 38.65s/24 帧 | 698 MB allocated | 包含较重导入、初始化和 warm-up |
| SAM 2.1 Small | 9.70 fps @ 960p | 冷启动总计 48.12s/24 帧 | 719 MB allocated | 二值目标传播 |

所有本地模型都不消耗 API token，成本来自 GPU 时间、磁盘和工程维护。生产时必须常驻模型并预热，
否则 Paddle、BEN2、MatAnyone、SAM2 的冷启动会抵消热推理速度优势。

## 建议生产链路

### 1. 默认路径

1. 对整段视频建立一次绿幕颜色 profile。
2. 使用当前算法生成 `alpha_chroma`、去绿 RGB 和置信度图。
3. 高置信帧直接进入时序清理与 WebP 编码，不调用模型。

### 2. BiRefNet-matting 回退

只在以下情况调用模型：主体内部出现洞、主连通域断裂、边缘碎片超阈值、绿幕纯度不足、宠物颜色
接近背景色，或 QA 评分低于阈值。模型结果作为 `alpha_structure`：

- 只在 `alpha_structure > 0.9` 且靠近当前主连通域时补洞。
- 只在 `alpha_structure < 0.05` 且不属于主连通域时删除碎片。
- 当前算法识别出的细长毛发、胡须和尾毛不因模型低置信度而自动删除。
- 不做 `0.5 * alpha_chroma + 0.5 * alpha_structure` 这类全局平均。

### 3. 时序处理

对上一帧 alpha 做光流反向映射，只在当前帧的低置信边缘带内融合；主体内部和明确背景不做平滑。
模型可每 4 至 6 帧运行一次，中间帧传播结构 mask，并在场景突变、动作幅度增大或 QA 失败时提前刷新。

### 4. RGB 与毛发边缘

- 去绿只处理与画布背景连通的外轮廓，不能处理鼻孔、口腔、眼睛等内部低 alpha 区域。
- 使用局部前景颜色估计替换绿幕污染，不要仅降低 alpha 隐藏亮边。
- Seedance 提示词继续明确：均匀纯绿、无地面阴影、无反射、无背光轮廓、完整身体和尾巴。
- 如果源视频已有白色曝光轮廓，任何 alpha 模型都无法恢复未生成的真实毛色，需要从视频生成阶段修复。

## 进一步达到“接近完美”的必要工作

当前最大缺口不是再增加更多通用模型，而是缺少宠物毛发真值数据。建议人工精修 24 至 50 个代表帧，
覆盖长毛猫、卷毛犬、白毛、黑毛、快速尾巴、睡姿和四肢交叉，用这些真值计算 SAD、MSE、Gradient、
Connectivity 和时序 warp error。随后再决定是否：

1. 微调 BiRefNet-matting 为宠物专用模型；
2. 训练轻量结构修复网络，只修洞和碎片；
3. 将当前绿幕 alpha 与宠物专用网络蒸馏到单个快速模型。

## 本地文件

- 统一实验入口：`matting_bench/README.md`
- 模型与结论元数据：`matting_bench/benchmark_catalog.json`
- 统一评估：`matting_bench/evaluate.py`
- 外轮廓去绿：`matting_bench/postprocess_green.py`
- 本地对比页生成器：`matting_bench/render_html.py`
- 各模型适配器：`matting_bench/providers/`
- 统一指标：`matting_bench/outputs/final_smoke_metrics.json`（本地生成，不提交大输出）
- 连续帧指标：`matting_bench/outputs/temporal_final_compare.json`（本地生成）
- 可视化页：`poc_output/matting_benchmark_pet_20260710.html`

模型权重保存在 `.models/`，虚拟环境保存在 `.venvs/`。两者均已加入 `.gitignore`，不会进入仓库。
