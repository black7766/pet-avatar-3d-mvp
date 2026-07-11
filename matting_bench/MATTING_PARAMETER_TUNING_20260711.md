# 宠物动效抠图模型部署与参数调优报告

更新时间：2026-07-11
测试硬件：NVIDIA GeForce RTX 2080 Ti 11 GB
测试对象：当前 3D 宠物绿幕动作素材

## 1. 本轮完成范围

- 本地部署并跑通 8 组外部 provider、10 条模型/检查点路径。
- 加入当前自研绿幕算法作为统一基线。
- 共实测 9 个 provider、69 组参数配置。
- 静态集：`idle / fast_walk / sleep` 各取首、中、尾帧，共 9 张 960×960 RGBA 输出。
- 时序集：`fast_walk` 连续 24 帧、640×640。
- GPU 模型通过全局文件锁串行计时，避免多个进程抢显存导致数据失真。
- 所有推荐配置均做黑底、白底、棋盘格和毛发边缘人工复核。

模型权重、虚拟环境、测试帧和批量输出均保存在本地 `.models/`、`.venvs/`、`matting_bench/data/` 与 `matting_bench/outputs/`，不纳入 Git。

## 2. 结论

### 当前生产主链

继续使用自研自适应绿幕算法，默认参数把 `GREEN_CORE_DESPILL_STRENGTH` 从 `0.90` 调整为 `1.10`。

原因：在当前 Seedance 绿幕素材上，它仍然是毛发保留、背景纯净、绿边、部署成本和工程可控性最均衡的方案。参数调整后：

- `green_fringe`：`0.014523 → 0.013816`，下降约 4.9%。
- `pseudo_mae`：`0.001791 → 0.001788`，基本持平并略有改善。
- `foreground_loss_mean`：`0.002752 → 0.002745`，没有通过削毛换取干净边缘。
- 连续帧误差：`0.013821 → 0.013823`，变化可忽略。

### 学习型后备

选择 **BiRefNet General 1024 + 官方 foreground refinement**，仅用于非绿幕输入、绿幕判断失败或主体连通性异常的样本。

它的优势是主体连通和连续帧稳定性较好；劣势是本测试中背景 alpha 和绿边都高于自研主链。关闭官方 refinement 后，绿边会恶化约 5.4 倍，因此不能省略该步骤。

### 不作为最终 alpha 的模型

- **SAM 2.1 Small**：目标传播稳定，但输出是二值分割，无法表达半透明毛发。可作为主体支持区域或断裂修复提示，不能直接输出 WebP alpha。
- **MatAnyone v1**：时序表现可用，但当前上游 S-Lab License 1.0 不适合直接商用；保留为研究对照。
- **PP-MattingV2**：推理最快，但公开检查点是人像域，宠物绿边明显，不进入生产候选。
- **BiRefNet-matting**：时序误差最低，但本轮宠物静态背景泄漏高于 General；检查点训练域以人像抠图为主，保留研究对照。
- **BEN2**：去绿后边缘颜色干净，但前景损失约为当前基线的 3.7 倍，会削弱低对比毛发。

## 3. 推荐配置实测对比

以下数值越低越好。`端到端`包含各 provider 自己的预处理、后处理和写盘，纯模型耗时与端到端耗时不能混用。自研算法当前是 CPU 全流程，外部模型主要是 CUDA 推理。

| Provider / 推荐配置 | pseudo MAE | 背景 alpha | 前景损失 | 绿边 | 模型推理 | 时序误差 | 生产判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| 自研 `despill_1_10` | 0.001788 | 0.000831 | 0.002745 | 0.013816 | CPU 全流程 577.9 ms | 0.013823 | 默认主链 |
| BiRefNet General `1024_auto` | 0.002393 | 0.002740 | 0.002047 | 0.018717 | 201.0 ms | 0.012839 | 非绿幕后备 |
| rembg U2Net `alpha_default` | 0.002057 | 0.003135 | 0.000979 | 0.012666 | 935.6 ms | 0.012753 | 可用但不划算 |
| ViTMatte `r02_tight_w35_d25` | 0.001906 | 0.001100 | 0.002712 | 0.014470 | 110.8 ms | 0.013724 | 混合细化实验 |
| BiRefNet-matting `1024_auto` | 0.002653 | 0.003423 | 0.001883 | 0.018458 | 200.4 ms | 0.011975 | 人像域研究对照 |
| Paddle PP-MattingV2 `512` | 0.002345 | 0.003455 | 0.001235 | 0.107512 | 18.5 ms | 0.012077 | 不采用当前检查点 |
| BEN2 `refine_r90` | 0.005304 | 0.000506 | 0.010101 | 0.004709 | 264.6 ms | 0.014703 | 前景损失过高 |
| MatAnyone `warmup1` | 0.002175 | 0.003434 | 0.000916 | 0.012534 | 83.5 ms | 0.012900 | 非商用研究 |
| SAM2.1 Small `state_cpu` | 0.000935 | 0.000707 | 0.001162 | 0.000000* | 79.8 ms | 0.014994 | 二值辅助 mask |

\* SAM2 的绿边为零是二值输出造成的结构性结果，不代表毛发质量最好。

## 4. 各模型参数扫测结果

### 自研绿幕算法

扫测 12 组：前景阈值、边缘分位数、alpha gamma、核心去绿强度、核心半径、halo profile。

- 推荐：`core_despill=1.10`，其余保持生产默认值。
- `gamma=1.40 + despill=1.10` 指标更干净，但连续帧误差略升，且更容易让细软毛变薄，不作为默认。
- `halo_none` 与 `halo_cartoon` 的代理误差很低，但人工复核发现抬起的腿和毛发有烘焙白边，已从最终 Pareto 排除。

### BiRefNet General / BiRefNet-matting

扫测 512、768、1024 输入，以及官方自动 refinement、CPU/GPU refinement、禁用 refinement。

- 两个检查点都应使用 1024 输入和官方 refinement。
- General 更适合作为宠物异常样本后备。
- matting 检查点的连续帧更稳，但本测试宠物背景泄漏更高。

### ViTMatte

扫测 20 组：unknown 半径 `2/4/6/8/12`、两组前景/背景阈值、两档融合强度。

- 推荐：`background=0.02`、`foreground=0.98`、`unknown_radius=2`、`fusion_weight=0.35`、`fusion_max_delta=0.25`。
- 结论：只细化窄 trimap 时有小幅改善，但仍依赖自研算法先生成可靠 trimap，增加模型成本后收益有限。

### rembg

实测 U2Net、ISNet General、BiRefNet General Lite，每个模型扫测默认、官方 alpha matting、窄 unknown 区和二值后处理。

- 推荐：U2Net + `foreground=240 / background=10 / erode=10`。
- 二值后处理虽然代理指标漂亮，但会硬切胡须和耳缘，已排除。
- “fur safe”窄 unknown 参数实际删除的软细节多于恢复的软细节，已排除。
- BiRefNet General Lite 的 Windows ONNX 路径约 4.9–5.0 秒/帧，没有质量收益。

### Paddle PP-MattingV2 / BEN2

- PP-MattingV2 推荐官方 `short=512`；640 会出现独立绿色碎片，384 质量下降。
- BEN2 的官方固定输入为 1024；推荐 refinement 半径 90，但毛发前景损失仍过高。

### MatAnyone / SAM2.1 Small

- MatAnyone 推荐首帧阈值 128、1 次 recurrent warmup、FP16、`mem_every=5`。
- SAM2 推荐首帧阈值 128、logit 阈值 0、FP16、video/state CPU offload；质量与默认一致并节省约 16.5 MiB 显存。
- 两者都不能取代最终毛发 alpha：前者受许可限制，后者只有二值 mask。

## 5. 百度方案说明

百度云“人像/视频人像分割”属于在线服务，并不是可下载到本地的公开权重。本轮本地部署采用百度 PaddlePaddle 官方开源链路 **PaddleSeg Matting / PP-MattingV2** 作为百度系代表。它证明了速度优势，但公开人像检查点不适合直接承担宠物毛发主链。

## 6. 官方链接

- [PaddleSeg Matting / PP-MattingV2](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/Matting)
- [百度智能云图像分割文档](https://ai.baidu.com/ai-doc/IMAGEPROCESS/rm8zl3koj)
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- [BiRefNet-matting 模型卡](https://huggingface.co/ZhengPeng7/BiRefNet-matting)
- [ViTMatte](https://github.com/hustvl/ViTMatte)
- [Transformers ViTMatte 文档](https://huggingface.co/docs/transformers/model_doc/vitmatte)
- [BEN2](https://github.com/PramaLLC/BEN2)
- [rembg](https://github.com/danielgatis/rembg)
- [U-2-Net](https://github.com/xuebinqin/U-2-Net)
- [IS-Net / DIS](https://github.com/xuebinqin/DIS)
- [MatAnyone](https://github.com/pq-yang/MatAnyone)
- [SAM 2](https://github.com/facebookresearch/sam2)

## 7. 复现与查看

```powershell
python matting_bench/aggregate_tuning.py `
  --output matting_bench/outputs/tuning/aggregate_final.json `
  --strict-outputs

python matting_bench/render_tuning_html.py `
  --aggregate matting_bench/outputs/tuning/aggregate_final.json `
  --output poc_output/matting_tuning_report_20260711.html
```

本地总览：`http://127.0.0.1:8792/matting_tuning_report_20260711.html`

实体版动图对比：`http://127.0.0.1:8792/matting_animated_compare_real_20260711.html`

- 支持快走 / 睡眠两种动作切换，均包含 9 条抠图路径。
- 每个动作使用 96 张真实连续帧，以 640×640、19.2 FPS、5 秒无声循环展示。
- 切换动作时，动图、耗时和质量指标同步更新。

## 8. 下一阶段建议

1. 保持自研绿幕主链，不为当前受控素材增加默认神经网络成本。
2. 接入“失败检测器”：主体断裂、背景不纯、alpha 空洞、非绿幕输入命中时，才调用 BiRefNet General。
3. 把 SAM2 仅用于主体支持区域或断裂提示，不参与最终软 alpha。
4. 后续采集 20–30 个宠物、不同毛色与动作的人工 alpha 真值小集；目前代理指标只能做相对比较，不能替代真值评测。
5. 若未来视频源不再是稳定绿幕，再评估可商用的宠物域视频 matting 微调，而不是继续堆叠通用人像模型。
