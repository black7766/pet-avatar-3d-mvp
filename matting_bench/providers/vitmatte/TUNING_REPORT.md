# ViTMatte-S 混合方案调参与实测报告

测试日期：2026-07-11
模型：`hustvl/vitmatte-small-composition-1k`，revision `6a58ad7646403c1df626fbd746900aec7361ea1d`

## 结论

推荐并已设为 `infer.py` 默认值：

```text
background_threshold = 0.02
foreground_threshold = 0.98
unknown_radius_px = 2
fusion_weight = 0.35
fusion_max_delta = 0.25
```

配置 ID 为 `r02_tight_w35_d25`。它在 20 组 smoke 交叉扫描中取得最低加权质量排名，并在独立持锁复跑中得到：

- `pseudo_mae`: `0.00190598`
- `background_alpha_mean`: `0.00110018`
- `foreground_loss_mean`: `0.00271178`
- `green_fringe`: `0.01446983`
- `fragment_pct`: `0.11057270%`
- `soft_alpha_pct`: `1.00040%`
- unknown 区占比：`1.40437%`
- 24 帧时序误差：`0.01372414`
- 9 帧平均模型推理：`110.81 ms/frame`，去首帧后 `74.00 ms/frame`
- 9 帧端到端：`723.62 ms/frame`
- PyTorch peak allocated VRAM：`597.49 MiB`

较强融合 `weight=0.70/max_delta=0.50` 虽略降前景损失和碎片，但明显增加背景 alpha、pseudo MAE 和绿边。radius 从 2 增至 12 会把 unknown 占比从约 `1.4%` 扩至 `4.5%`，未带来综合质量收益。

## 官方语义与输入约束

参考资料：

- [Transformers v4.47.1 ViTMatte 文档](https://huggingface.co/docs/transformers/v4.47.1/model_doc/vitmatte)：模型同时接收图像和 trimap，示例输出为 `1x1x640x960` alpha。
- [Transformers v4.47.1 processor 源码](https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/vitmatte/image_processing_vitmatte.py)：RGB 与 trimap 都按 `1/255` 缩放；只标准化 RGB；随后拼成四通道输入；右侧和底部补齐到 32 的倍数。
- [ViTMatte 官方单图入口](https://github.com/hustvl/ViTMatte/blob/main/run_one_image.py)：图像使用 `RGB`，trimap 使用单通道 `L`，两者通过 `to_tensor` 转为 `0..1`。
- [固定 checkpoint 的 preprocessor_config.json](https://huggingface.co/hustvl/vitmatte-small-composition-1k/blob/6a58ad7646403c1df626fbd746900aec7361ea1d/preprocessor_config.json)：`do_pad=true`、`size_divisibility=32`，RGB mean/std 均为 `0.5`。

本 provider 的 trimap 使用精确三值：黑色 `0` 为确定背景，灰色 `128` 为 unknown，白色 `255` 为确定前景；processor 缩放后分别约为 `0`、`0.502`、`1`。RGB 和 trimap 必须同高同宽，否则四通道拼接失败。Transformers v4.47.1 不做固定尺寸 resize，只补齐到 32 的倍数；模型输出需裁回源尺寸。本次正式 smoke 输入为 `960x960`，已能被 32 整除，实测 padding 为 `0x0`。

混合约束为：

```text
fused = adaptive + fusion_weight * clip(model - adaptive, +/-fusion_max_delta)
```

公式只作用于 unknown 区；融合后确定背景强制回到 `0`，确定前景强制回到 `1`，随后中央 `refine_reframed_halo` 仍可能对 halo 像素做统一 feather。推荐配置中，ViTMatte 在 unknown 区的原始平均绝对修正为 `0.09149`，融合后为 `0.02472`，约 `12.57%` 的 unknown 像素触发 `0.25` 限幅。

## 实测设计

- smoke：`matting_bench/data/pet_20260710_121221_5ce7716e/smoke`，9 帧，`960x960`。
- temporal：`temporal_fast_walk_24_640`，24 个连续帧，`640x640`。
- 交叉参数：radius `2/4/6/8/12` × 阈值 `0.02/0.98`、`0.04/0.96` × 融合 `0.35/0.25`、`0.70/0.50`，共 20 组。
- 质量与时序：统一调用中央 `matting_bench/evaluate.py`。
- 排名：对 `pseudo_mae/background_alpha_mean/foreground_loss_mean/green_fringe/fragment_pct` 做归一化名次加权，权重为 `5/2/2/3/1`，越低越好；`soft_alpha_pct` 仅记录，不参与排名。
- 前两名重新创建输出目录，各自复跑 9 帧 smoke，再跑 24 帧 temporal；推荐项速度与显存只采用该最终复跑。

环境：Windows 10、Python `3.11.9`、RTX 2080 Ti、Torch `2.5.1+cu121`、Transformers `4.47.1`、FP16。

## 20 组结果

`score` 越低越好。前两行质量值来自最终 smoke 复跑，其 score 保留初筛排名；其余行为持锁初筛值。

| ID | radius | bg/fg | weight | max delta | pseudo MAE | green fringe | fragment | unknown | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r02_tight_w35_d25 | 2 | 0.02/0.98 | 0.35 | 0.25 | 0.001906 | 0.014470 | 0.1106% | 1.404% | 2.579 |
| r02_tight_w70_d50 | 2 | 0.02/0.98 | 0.70 | 0.50 | 0.002071 | 0.015402 | 0.1097% | 1.404% | 7.842 |
| r02_relaxed_w35_d25 | 2 | 0.04/0.96 | 0.35 | 0.25 | 0.001901 | 0.014581 | 0.1160% | 1.363% | 3.211 |
| r02_relaxed_w70_d50 | 2 | 0.04/0.96 | 0.70 | 0.50 | 0.002058 | 0.015501 | 0.1157% | 1.363% | 8.000 |
| r04_tight_w35_d25 | 4 | 0.02/0.98 | 0.35 | 0.25 | 0.001919 | 0.014439 | 0.1104% | 2.068% | 3.895 |
| r04_tight_w70_d50 | 4 | 0.02/0.98 | 0.70 | 0.50 | 0.002094 | 0.015337 | 0.1097% | 2.068% | 7.895 |
| r04_relaxed_w35_d25 | 4 | 0.04/0.96 | 0.35 | 0.25 | 0.001915 | 0.014544 | 0.1160% | 2.025% | 4.263 |
| r04_relaxed_w70_d50 | 4 | 0.04/0.96 | 0.70 | 0.50 | 0.002086 | 0.015423 | 0.1158% | 2.025% | 8.474 |
| r06_tight_w35_d25 | 6 | 0.02/0.98 | 0.35 | 0.25 | 0.001924 | 0.014427 | 0.1110% | 2.681% | 4.737 |
| r06_tight_w70_d50 | 6 | 0.02/0.98 | 0.70 | 0.50 | 0.002103 | 0.015291 | 0.1098% | 2.681% | 8.895 |
| r06_relaxed_w35_d25 | 6 | 0.04/0.96 | 0.35 | 0.25 | 0.001919 | 0.014533 | 0.1161% | 2.638% | 4.579 |
| r06_relaxed_w70_d50 | 6 | 0.04/0.96 | 0.70 | 0.50 | 0.002093 | 0.015378 | 0.1158% | 2.638% | 8.684 |
| r08_tight_w35_d25 | 8 | 0.02/0.98 | 0.35 | 0.25 | 0.001926 | 0.014419 | 0.1119% | 3.311% | 5.421 |
| r08_tight_w70_d50 | 8 | 0.02/0.98 | 0.70 | 0.50 | 0.002105 | 0.015266 | 0.1097% | 3.311% | 9.263 |
| r08_relaxed_w35_d25 | 8 | 0.04/0.96 | 0.35 | 0.25 | 0.001921 | 0.014520 | 0.1161% | 3.267% | 5.579 |
| r08_relaxed_w70_d50 | 8 | 0.04/0.96 | 0.70 | 0.50 | 0.002098 | 0.015347 | 0.1157% | 3.267% | 9.421 |
| r12_tight_w35_d25 | 12 | 0.02/0.98 | 0.35 | 0.25 | 0.001924 | 0.014416 | 0.1119% | 4.547% | 4.737 |
| r12_tight_w70_d50 | 12 | 0.02/0.98 | 0.70 | 0.50 | 0.002103 | 0.015252 | 0.1090% | 4.547% | 8.632 |
| r12_relaxed_w35_d25 | 12 | 0.04/0.96 | 0.35 | 0.25 | 0.001920 | 0.014515 | 0.1162% | 4.503% | 5.000 |
| r12_relaxed_w70_d50 | 12 | 0.04/0.96 | 0.70 | 0.50 | 0.002096 | 0.015341 | 0.1150% | 4.503% | 8.895 |

完整指标见 `tuning_results.json`，每项均含用户要求的六项质量指标、三项 runtime、unknown 占比、时序误差、状态和相对输出路径。

## 前两名持锁复测

| 配置 | pseudo MAE | bg alpha | fg loss | fringe | fragment | soft alpha | unknown | temporal MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r02_tight_w35_d25` | 0.00190598 | 0.00110018 | 0.00271178 | 0.01446983 | 0.11057% | 1.00040% | 1.40437% | **0.01372414** |
| `r02_relaxed_w35_d25` | **0.00190109** | **0.00108909** | 0.00271310 | 0.01458059 | 0.11605% | 0.99035% | 1.36263% | 0.01374952 |

| 配置 | smoke inference | smoke warm | smoke E2E | temporal inference | temporal warm | temporal E2E | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r02_tight_w35_d25` | 110.81 ms | 74.00 ms | 723.62 ms | 64.39 ms | 49.26 ms | 333.19 ms | 597.49 MiB |
| `r02_relaxed_w35_d25` | 120.66 ms | 75.55 ms | 756.71 ms | 64.25 ms | 49.71 ms | 339.86 ms | 597.49 MiB |

放宽阈值的第二名在 pseudo MAE 和背景 alpha 上略优，但推荐项有更低的绿边、碎片率和时序误差，且 unknown 区仍保持窄带，因此综合选择 tight 方案。

## GPU 锁与计时说明

最终命令：

```powershell
python matting_bench/run_with_gpu_lock.py -- `
  .venvs/vitmatte/Scripts/python.exe `
  matting_bench/providers/vitmatte/sweep.py `
  --device cuda --gpu-lock-held
```

最终采纳轮第一次锁请求即成功，等待 `7.037 s`；之后 20 组初筛、两组 smoke 复跑和两组 temporal 复跑在同一个锁周期内串行完成，20 组均一次成功，最后正常释放。结果 JSON 的 `benchmark.gpu_lock_held=true`，时间范围为 `2026-07-11T02:43:05Z` 至 `02:48:50Z`。更早的 640px 功能筛选和一轮含单进程异常的 960px 运行均不用于最终推荐或计时。

`mean_inference_ms` 按要求保留全部帧均值，包含每个新进程的首帧 CUDA warm-up；额外记录 `mean_inference_excluding_first_ms` 便于常驻服务估算。显存为模型加载前 reset 后的 PyTorch allocator peak allocated，不含驱动上下文和其他进程显存。

## 限制

- 中央质量指标是绿幕伪真值，不是人工 alpha ground truth；边缘视觉判断仍需人工放大检查。
- 数据只覆盖一个 640px 宠物绿幕样本集和一段 fast-walk 连续序列。
- ViTMatte 是逐帧模型；当前方案没有引入跨帧状态，时序指标仅用于验证参数没有放大闪烁。
- 当前推荐优先控制背景污染和绿边，不追求最低前景损失或最低碎片单项值。
