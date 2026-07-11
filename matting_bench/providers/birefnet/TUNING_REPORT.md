# BiRefNet General 调参与实测报告

实测日期：2026-07-10 至 2026-07-11。模型固定为 `ZhengPeng7/BiRefNet@e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`，CUDA 使用 FP16，设备为 RTX 2080 Ti 11 GB。

## 结论

- **配置推荐：**`1024 + official-auto`；在 CUDA 上解析为官方 GPU foreground refinement（两阶段半径 `90/6`）。
- **模型结论：有条件推荐。**它适合作为非绿幕输入或自适应绿幕基线失败时的 learned fallback；**不推荐替换当前受控绿幕 baseline**。
- 9 帧 smoke 上，General 最佳组的 `pseudo_mae=0.0023932`，差于 baseline 的 `0.0017913`；背景 alpha 和绿边也差于 baseline，但前景损失及碎片显著更低。
- 24 帧 fast-walk 上，General 的 `temporal_alpha_mae=0.0128395`，优于 baseline 的 `0.0138207`。
- 关闭官方 refinement 不改变 alpha 指标，却把 `green_fringe` 从 `0.0187170` 放大到 `0.1011023`（5.40 倍），因此不可作为生产配置。

## 官方出处

| 官方来源 | 核实到的参数 | 本地实现 |
|---|---|---|
| [固定 revision 的 HF README](https://huggingface.co/ZhengPeng7/BiRefNet/blob/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/README.md) | 官方示例使用 `1024x1024`、ImageNet mean/std、CUDA FP16、末级输出 `sigmoid` | 默认 `--input-resolution 1024`，相同归一化与 FP16 |
| [固定 revision 的 HF handler](https://huggingface.co/ZhengPeng7/BiRefNet/blob/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/handler.py) | General 为 `1024x1024`；foreground estimator 两阶段 `r=90`、`r=6` | `official-cpu` 精确复用该 OpenCV 路径 |
| [官方 GitHub image_proc.py](https://github.com/ZhengPeng7/BiRefNet/blob/d83f35576a13fd90f8cca7c5e73d0907c2297328/image_proc.py) | 官方新增 CPU/GPU 两种 foreground estimator，半径仍为 `90/6` | `official-gpu` 移植其 torch 路径 |
| [官方 HF Demo](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo/blob/main/app.py) | 自定义分辨率按 32 的倍数归整；标准 checkpoint 推荐 `1024x1024`；CUDA 走 GPU refinement | 暴露 `512/768/1024` 与 `official-auto/cpu/gpu/none` |

`512`、`768` 是本次速度/质量消融值，不是官方对标准 checkpoint 的推荐值；`none` 也是本次消融开关，官方 handler 与 Demo 均会执行 foreground refinement。

## CLI 变更

`infer.py` 现在支持：

- `--input-resolution {512,768,1024}`，默认 `1024`。
- `--foreground-refinement {official-auto,official-cpu,official-gpu,none}`，默认 `official-auto`。
- `official-auto` 在 CUDA 上解析为 `official-gpu`，在 CPU 上解析为 `official-cpu`。
- metrics 增加预处理、纯推理、foreground refinement、端到端耗时及峰值显存；原 provider 合约参数保持不变。

## 数据与方法

- Smoke：`matting_bench/data/pet_20260710_121221_5ce7716e/smoke`，9 张 `960x960`，覆盖 idle、fast_walk、sleep 的首/中/末帧。
- Temporal：`matting_bench/data/pet_20260710_121221_5ce7716e/temporal_fast_walk_24_640`，24 张连续 `640x640` fast-walk 帧。
- 统一评估：中央 `matting_bench/evaluate.py`；无手绘 alpha GT，因此 `pseudo_mae`、前景损失、背景泄漏均为受控绿幕高置信区域指标。
- 所有 smoke 配置与 baseline 在同一次中央评估中比较；最佳组才进入 temporal。

## Smoke 质量

数值越低越好；`soft alpha` 仅描述 alpha 分布，不单独判优。

| 配置 | pseudo MAE | 背景 alpha | 前景损失 | 绿边 | 碎片 % | soft alpha % |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **0.0017913** | **0.0008311** | 0.0027515 | **0.0145227** | 0.134511 | 0.9979 |
| General 1024 + official-auto | 0.0023932 | 0.0027397 | **0.0020468** | 0.0187170 | 0.002060 | 1.5131 |
| General 768 + official-auto | 0.0033624 | 0.0037612 | 0.0029635 | 0.0375066 | 0.002154 | 1.5187 |
| General 512 + official-auto | 0.0032936 | 0.0035351 | 0.0030520 | 0.0147156 | **0.000000** | 1.9847 |
| General 1024 + none | 0.0023932 | 0.0027397 | 0.0020468 | 0.1011023 | 0.002060 | 1.5131 |

1024 是 General 的明确质量最优点。512 的绿边和碎片较低，但 pseudo MAE 与前景损失恶化；768 在本数据上没有形成有效折中。

## CUDA 计时

推荐配置由 `matting_bench/run_with_gpu_lock.py` 串行复跑。下表均为**持锁最终值**，均值排除首帧；锁等待位于子进程外，不计入 metrics。

| 数据集 | 纯推理 ms/帧 | 端到端 ms/帧 | foreground refinement ms/帧 | 峰值 allocated / reserved |
|---|---:|---:|---:|---:|
| smoke 9 帧 | 200.957 | 352.143 | 63.819 | 1715.158 / 2980.053 MB |
| temporal 24 帧 | 204.168 | 289.275 | 27.369 | 1715.158 / 2980.053 MB |

Smoke 首帧纯推理为 `898.380 ms`，全 9 帧纯推理均值为 `278.448 ms`；模型加载为 `2098.744 ms`。Smoke 获取锁前等待 `2.023 s`。Temporal 前 5 次因 Windows 锁占用返回 `OSError 36`，第 6 次成功并在锁内等待 `1.014 s`；只采用成功子进程的 metrics。

无锁 sweep 仅用于筛选，不能作为最终性能结论：1024/768/512/1024-none 的稳态纯推理筛选值分别约为 `166.0/111.4/103.4/166.2 ms`，对应峰值约 `1715/1172/779/1715 MB`。这些值在 `tuning_results.json.screening_runtime_unlocked` 中单独标为 `not_final`。

## Temporal 对比

| Provider | pseudo MAE | 背景 alpha | 前景损失 | 绿边 | 碎片 % | temporal alpha MAE |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0030075 | **0.0010978** | 0.0049172 | 0.0106784 | 0.006354 | 0.0138207 |
| General 1024 + official-auto | **0.0027374** | 0.0025853 | 0.0028895 | **0.0055447** | **0.002621** | 0.0128395 |
| Matting 1024 + official-auto | 0.0027724 | 0.0033317 | **0.0022131** | 0.0066124 | 0.003770 | **0.0119748** |

General 在 24 帧上取得最低 pseudo MAE、绿边和碎片；Matting 的 temporal alpha MAE 与前景保留更好。当前数据不足以证明任一 learned model 能全面替代 baseline。

## 推荐决策

1. 在 BiRefNet General 内部，采用 `1024 + official-auto`，不要关闭 refinement。
2. 在当前受控绿幕生产链路中继续使用 baseline；General 作为非绿幕/异常样本 fallback。
3. 不采用 768：它没有提供可靠质量收益。512 只适合显存受限的降级路径，并需接受更高 pseudo MAE/前景损失。
4. 若要推动 General 成为主路径，需要补充手绘 alpha GT、多只长短毛宠物、快速摆尾/毛发运动和非绿幕场景，不应仅依据当前 pseudo 指标上线。

## 证据

- 全组 smoke 评估：`matting_bench/providers/birefnet_matting/evidence/smoke_evaluation.json`
- 持锁 smoke 复核：`matting_bench/providers/birefnet_matting/evidence/locked_smoke_evaluation.json`
- Temporal 评估：`matting_bench/providers/birefnet_matting/evidence/temporal_evaluation.json`
- 持锁运行日志：`matting_bench/providers/birefnet/evidence/run_logs/`
- 机器可读结果：`matting_bench/providers/birefnet/tuning_results.json`
