# ViTMatte 本地部署与绿幕混合测试报告

测试日期：2026-07-10

## 结论

官方 ViTMatte-S 已完成本地离线部署，统一 `infer.py --input-dir --output-dir
--device` 接口可用。9 帧 smoke 样本已在 RTX 2080 Ti 上实跑，输出 9/9 同名、
`960x960`、8-bit RGBA PNG。

最终采用保守混合方案：现有自适应绿幕 alpha 负责锁定背景和生成窄 trimap，
ViTMatte 只细化现有主体支持域内的 unknown 区；RGB 沿用现有自适应去绿溢色结果。
该方案避免了模型向纯绿背景扩张和低 alpha 反合成导致的洋红边。

本次 9 帧上，毛发边缘改善较轻微，不足以证明可以替换当前基线。前景损失和碎片
指标有小幅改善，但背景 alpha、绿边和 pseudo MAE 变差。建议保留为实验 provider，
在有人工 alpha 真值和连续视频时再决定是否进入主流程。

## 模型与许可证

| 项目 | 记录 |
|---|---|
| 模型 | `hustvl/vitmatte-small-composition-1k` |
| checkpoint | ViTMatte-S，Composition-1k |
| HF revision | `6a58ad7646403c1df626fbd746900aec7361ea1d` |
| 实现 | Hugging Face Transformers `VitMatteForImageMatting` |
| 权重 | `model.safetensors`，103,294,572 bytes |
| SHA-256 | `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb` |
| 上游代码 | <https://github.com/hustvl/ViTMatte> |
| 上游代码许可证 | MIT |
| HF 模型页 | <https://huggingface.co/hustvl/vitmatte-small-composition-1k> |
| HF 模型卡许可证 | Apache-2.0 |
| Transformers 许可证 | Apache-2.0 |

上游代码与 HF 模型卡的许可证标签不同。若随产品分发代码或权重，应同时保留两处
notice，并让法务按实际分发方式确认，而不是只采用其中一个标签。

下载器固定 revision，只取 `README.md`、`config.json`、
`preprocessor_config.json` 和 safetensors 权重；下载后同时校验文件大小与 SHA-256。
推理阶段设置 HF/Transformers offline，所有缓存位于 `.models/vitmatte/`。

## Trimap 与混合参数

| 参数 | 最终值 |
|---|---|
| 绿幕 profile | 9 帧共享，避免逐帧阈值跳动 |
| profile `bg_floor` | `0.801619` |
| profile key RGB | 约 `[22, 253, 16]` |
| 确定背景 | 自适应 alpha `<= 0.02`，trimap 值 0 |
| 确定前景种子 | 自适应 alpha `>= 0.98` |
| unknown 带 | 对确定前景做 6px 椭圆核内蚀，trimap 值 128 |
| 确定前景 | 内蚀后的核心，trimap 值 255 |
| 平均 unknown 占比 | `2.681%` |
| 模型后约束 | known background 强制 0，known foreground 强制 1 |
| RGB | `adaptive_green_matte_frame` 已清理 RGB + `refine_reframed_halo` |

保守背景锁定是必要约束。曾测试对主体外侧做 6px/12px 对称扩张，unknown 占比
增至约 `4.81%/8.63%`，虽然碎片数显著下降，但会吸入纯绿背景，并在白色胡须的
低 alpha 反合成中产生洋红 RGB，故未采用。

## 运行环境与性能

| 项目 | 结果 |
|---|---:|
| OS / Python | Windows 10 / Python 3.11.9 |
| GPU | NVIDIA GeForce RTX 2080 Ti 11GB，driver 591.86 |
| Torch / CUDA | `2.5.1+cu121` / CUDA 12.1，FP16 |
| Transformers | `4.47.1` |
| 模型加载 | `0.392 s`（缓存已热） |
| 绿幕 profile | `0.552 s` |
| 9 帧 batch wall | `10.315 s` |
| 端到端均值 | `1.139 s/帧`（自适应 alpha、ViTMatte、halo、PNG） |
| 首帧模型推理 | `0.675 s`（缓存已热） |
| 后 8 帧模型推理均值 | `78.4 ms/帧` |
| Torch 峰值 allocated | `597.49 MiB` |
| Torch 峰值 reserved | `776.00 MiB` |
| 独立 venv 磁盘占用 | `4.692 GiB` |
| 模型目录磁盘占用 | `0.096 GiB` |

首次 CUDA 使用的调参运行中曾观察到 `8.21 s` 首帧推理，属于驱动/内核冷启动；
服务化时应在接流量前 warm up。显存数字是从模型加载前 reset 的 PyTorch allocator
峰值，不包括驱动上下文和其他进程占用。

当前 CPU 基线本次为 `11.9566 s / 9 帧`、`1.2437 s/帧`。ViTMatte 最终运行若把
模型加载、profile 和 batch 相加约 `11.26 s`，单次 9 帧任务与基线接近；常驻模型
时才有明显吞吐优势。CPU ViTMatte 性能未实测。

## 质量对比

以下是中央 `evaluate.py` 的绿幕伪真值指标，无人工 alpha ground truth；除明确标注外
均为越低越好。

| 指标 | 自适应基线 | ViTMatte 混合 | 变化 |
|---|---:|---:|---:|
| pseudo MAE | 0.001791 | 0.002229 | +24.42%，变差 |
| background alpha mean | 0.000831 | 0.001811 | +0.000980，变差 |
| foreground loss mean | 0.002752 | 0.002647 | -3.81%，改善 |
| green fringe | 0.014523 | 0.017105 | +17.78%，变差 |
| opaque green leak | 0.1037% | 0.2579% | +0.1542 个百分点，变差 |
| soft alpha（越高不必然越好） | 0.9979% | 1.0082% | +0.0103 个百分点 |
| coverage | 28.8215% | 28.8207% | 基本不变 |
| fragment ratio | 0.1345% | 0.1097% | -18.45%，改善 |
| fragment count | 27.56 | 24.67 | -10.48%，改善 |

模型只处理平均 `2.681%` 的像素，unknown 区 alpha 平均绝对改变量为 `0.04758`。
checkerboard 肉眼检查显示耳尖、身体外轮廓的过渡略平滑，胡须仍保留；整体差异很小。
fragment 指标会把合法的分离胡须/毛尖也计为碎片，因此下降不能单独视为质量提升。

## 失败风险

1. **错误 trimap 不可恢复**：纯绿背景被锁定后，ViTMatte 无法找回被自适应 alpha
   误删的绿色饰物、绿色眼部细节或漏检毛发。
2. **不可靠绿幕会直接失败**：边框绿色占比或置信度不足时，现有 profile 会抛错；
   这比静默输出错误 alpha 更安全，但调用方需要处理失败。
3. **不透明光晕不是 alpha 问题**：源帧已烘焙的白/黄/绿 rim light 不能仅靠 matting
   消除。激进降低 alpha 还可能暴露错误的直通 RGB。
4. **时序未验证**：smoke 是每段 3 个非连续抽样帧，中央 temporal 指标为 null。
   逐帧 ViTMatte 可能闪烁，必须用完整连续序列验证。
5. **伪真值偏差**：绿幕指标可能把真实的绿色反光或细毛当背景。生产决策需要人工
   alpha 真值、黑白/彩色多背景合成和局部放大检查。
6. **域差异**：Composition-1k 与当前 AI 生成猫、强绿幕、明显 rim light 不完全同域。
7. **许可证双标签**：代码 MIT、HF 模型卡 Apache-2.0，分发前需同时核对。

## 证据文件

- 最终 RGBA 与运行指标：`evidence/smoke_rgba/`
- 每帧 baseline alpha、trimap、模型 alpha、最终 alpha：`evidence/diagnostics/`
- 同帧自适应基线：`evidence/baseline_rgba/`
- 中央评估结果：`evidence/evaluation.json`
- 保守最终版与被否决激进版对比：`evidence/evaluation_variants.json`
- 被否决的对称 6px 调参输出：`evidence/aggressive_symmetric_radius6_rgba/`
- checkerboard 对照：`evidence/contact_sheet.jpg`
- 完整安装版本：`evidence/pip_freeze.txt`
- 权重下载与哈希记录：`.models/vitmatte/hustvl--vitmatte-small-composition-1k/local_manifest.json`

复现实跑命令：

```powershell
.venvs/vitmatte/Scripts/python.exe `
  matting_bench/providers/vitmatte/infer.py `
  --input-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --output-dir matting_bench/providers/vitmatte/evidence/smoke_rgba `
  --diagnostics-dir matting_bench/providers/vitmatte/evidence/diagnostics `
  --device cuda
```
