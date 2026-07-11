# BiRefNet-matting 调参与实测报告

实测日期：2026-07-10 至 2026-07-11。模型固定为 `ZhengPeng7/BiRefNet-matting@57f9f68b43ba337c75762b14cf3075d659007268`，CUDA 使用 FP16，设备为 RTX 2080 Ti 11 GB。

## 结论

- **本 checkpoint 内最佳配置：**`1024 + official-auto`；在 CUDA 上解析为官方 GPU foreground refinement（两阶段半径 `90/6`）。
- **生产结论：不推荐。**保留为研究对照，不替换当前 adaptive green baseline，也不优先于 BiRefNet General。
- 优点是前景保留与时序稳定：smoke `foreground_loss=0.0018832`，24 帧 `temporal_alpha_mae=0.0119748`，均为三者最佳。
- 主要问题是背景抑制：smoke `pseudo_mae=0.0026530`、`background_alpha_mean=0.0034229`，均差于 baseline 和 General。
- 关闭 refinement 会把绿边从 `0.0184581` 放大到 `0.1058571`（5.74 倍），不可用于生产。

## 官方出处

| 官方来源 | 核实到的参数 | 本地实现 |
|---|---|---|
| [固定 revision 的 Matting 模型卡](https://huggingface.co/ZhengPeng7/BiRefNet-matting/blob/57f9f68b43ba337c75762b14cf3075d659007268/README.md) | 模型为 trimap-free general matting；训练/验证集以人像抠图数据为主 | 将其作为独立 checkpoint 评测，不与 General 混用结论 |
| [固定 revision 的 Matting handler](https://huggingface.co/ZhengPeng7/BiRefNet-matting/blob/57f9f68b43ba337c75762b14cf3075d659007268/handler.py) | `usage='Matting'`、`1024x1024`、CUDA FP16、foreground estimator `r=90/6` | 独立 `infer.py` 默认固定该 repo/revision，复用统一 runtime |
| [官方 GitHub image_proc.py](https://github.com/ZhengPeng7/BiRefNet/blob/d83f35576a13fd90f8cca7c5e73d0907c2297328/image_proc.py) | 官方 CPU/GPU 两种 foreground estimator，半径为 `90/6` | 暴露 `official-cpu` 与 `official-gpu` |
| [官方 HF Demo](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo/blob/main/app.py) | 自定义分辨率按 32 的倍数归整；标准 checkpoint 推荐 `1024x1024` | 暴露 `512/768/1024`，默认 `official-auto` |

模型卡列出的 P3M-10k、TR-humans、AM-2k、AIM-500、Human-2k、Distinctions-646、HIM2K、PPM-100 以人像 matting 为主；本次对象是宠物毛发，因此不能把官方人像验证集指标外推为本场景结论。

## CLI 变更

`birefnet_matting/infer.py` 现在是独立 provider 入口，默认固定 matting checkpoint，同时共享经过验证的 General runtime：

- `--input-resolution {512,768,1024}`，默认 `1024`。
- `--foreground-refinement {official-auto,official-cpu,official-gpu,none}`，默认 `official-auto`。
- `official-auto` 在 CUDA 上解析为 `official-gpu`，在 CPU 上解析为 `official-cpu`。
- `none` 仅作消融；官方 handler 与 Demo 都会执行 foreground refinement。

## 数据与方法

- Smoke：`matting_bench/data/pet_20260710_121221_5ce7716e/smoke`，9 张 `960x960`。
- Temporal：`matting_bench/data/pet_20260710_121221_5ce7716e/temporal_fast_walk_24_640`，24 张连续 `640x640` fast-walk 帧。
- 统一评估：中央 `matting_bench/evaluate.py`；无手绘 alpha GT，质量值为受控绿幕 pseudo 指标。
- Matting 与 General 使用同一帧、同一预处理、同一输出/评估合同。

## Smoke 质量

| 配置 | pseudo MAE | 背景 alpha | 前景损失 | 绿边 | 碎片 % | soft alpha % |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **0.0017913** | **0.0008311** | 0.0027515 | **0.0145227** | 0.134511 | 0.9979 |
| Matting 1024 + official-auto | 0.0026530 | 0.0034229 | 0.0018832 | 0.0184581 | 0.000725 | 1.8761 |
| Matting 768 + official-auto | 0.0029068 | 0.0041140 | **0.0016996** | 0.0255979 | **0.000160** | 1.9479 |
| Matting 512 + official-auto | 0.0037615 | 0.0044157 | 0.0031072 | **0.0178688** | 0.000877 | 2.4178 |
| Matting 1024 + none | 0.0026530 | 0.0034229 | 0.0018832 | 0.1058571 | 0.000725 | 1.8761 |

768 只改善前景损失和碎片，却恶化总体 pseudo MAE、背景泄漏和绿边；512 的总体质量最差。1024 是唯一合理候选。

## CUDA 计时

推荐配置由 `matting_bench/run_with_gpu_lock.py` 串行复跑。下表均为**持锁最终值**，均值排除首帧。

| 数据集 | 纯推理 ms/帧 | 端到端 ms/帧 | foreground refinement ms/帧 | 峰值 allocated / reserved |
|---|---:|---:|---:|---:|
| smoke 9 帧 | 200.429 | 446.180 | 70.125 | 1715.158 / 2980.053 MB |
| temporal 24 帧 | 205.847 | 300.653 | 28.157 | 1715.158 / 2980.053 MB |

Smoke 首帧纯推理为 `1079.269 ms`，全 9 帧纯推理均值为 `298.078 ms`；模型加载为 `2385.539 ms`。Smoke 立即拿锁（`0.000 s`）；Temporal 等待 `7.065 s` 后拿锁。锁等待位于 infer 子进程外，不计入 metrics。

无锁 sweep 仅用于筛选：1024/768/512/1024-none 的稳态纯推理筛选值约为 `166.9/120.5/113.3/168.6 ms`，对应峰值约 `1715/1172/779/1715 MB`。这些不是最终计时，JSON 的正式 `runtime` 对非入选组均为 `null`。

## Temporal 对比

| Provider | pseudo MAE | 背景 alpha | 前景损失 | 绿边 | 碎片 % | temporal alpha MAE |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0030075 | **0.0010978** | 0.0049172 | 0.0106784 | 0.006354 | 0.0138207 |
| General 1024 + official-auto | **0.0027374** | 0.0025853 | 0.0028895 | **0.0055447** | **0.002621** | 0.0128395 |
| Matting 1024 + official-auto | 0.0027724 | 0.0033317 | **0.0022131** | 0.0066124 | 0.003770 | **0.0119748** |

Matting 的时序稳定性最好，但它仍有最高的背景 alpha；在当前合成/绿幕管线中，时序优势不足以抵消背景泄漏风险。

## 推荐决策

1. **不将 BiRefNet-matting 上线为当前生产默认。**
2. 若研究或 A/B 必须使用，固定 `1024 + official-auto`；不要采用 768/512，也不要关闭 refinement。
3. 两个 learned checkpoint 中优先 General：General 的 smoke pseudo MAE 和背景抑制更好，Matting 只在前景保留和 temporal alpha MAE 上领先。
4. 后续只有在带手绘 GT 的宠物毛发/摆尾数据上仍能复现时序优势，并针对背景泄漏设门槛后，才值得重新评估上线。

## 证据

- 全组 smoke 评估：`matting_bench/providers/birefnet_matting/evidence/smoke_evaluation.json`
- 持锁 smoke 复核：`matting_bench/providers/birefnet_matting/evidence/locked_smoke_evaluation.json`
- Temporal 评估：`matting_bench/providers/birefnet_matting/evidence/temporal_evaluation.json`
- 持锁运行日志：`matting_bench/providers/birefnet_matting/evidence/run_logs/`
- 机器可读结果：`matting_bench/providers/birefnet_matting/tuning_results.json`
