# MatAnyone 宠物视频时序抠图实跑报告

## 结论

官方 MatAnyone v1 已在 Windows + RTX 2080 Ti 上部署并完成 `fast_walk`
连续 24 帧实跑。模型源码、权重和 `InferenceCore` 均来自官方仓库，没有使用
RVM 或其他仅人像模型冒充宠物方案。

技术部署通过，但当前结果**不建议直接作为宠物生产默认抠图器**：目标追踪稳定，
没有丢帧、空帧或主体触边，不过鼻口区域存在持续的小块半透明误修，模型还会平滑
部分毛发细节；同时官方 S-Lab License 1.0 的商业使用需要另行取得许可。

最终推荐跑测产物：

- 指标：`runs/fast_walk_24_final/metrics.json`
- RGBA：`runs/fast_walk_24_final/rgba/frame_0000.png` 至 `frame_0023.png`
- Alpha：`runs/fast_walk_24_final/alpha/frame_0000.png` 至 `frame_0023.png`
- 视觉检查：`runs/fast_walk_24_final/contact_sheet_rgba.png` 和
  `contact_sheet_alpha.png`

## 模型来源与许可证

| 项目 | 固定值 |
|---|---|
| 官方仓库 | `https://github.com/pq-yang/MatAnyone` |
| 固定 commit | `e5ddc534c1fff9bb9e54cf476095d29071b7cb4f` |
| 官方权重 | GitHub release `v1.0.0/matanyone.pth` |
| 权重大小 | `141429992` bytes |
| 权重 SHA-256 | `dd26b991d020ed5eb4be50996f97354c45cfdfc0f59958e8983ac6a198f4809d` |
| 参数量 | `35,249,195` |
| 许可证 | S-Lab License 1.0，仅明确允许非商业用途；商业用途需联系作者 |

官方项目将 MatAnyone 描述为支持首帧目标指定的 human video matting 框架。
目标指定机制可用于宠物，但该 checkpoint 并不是宠物专用模型，域外泛化不能按人像
基准推断。

## Windows 部署

| 项目 | 实跑环境 |
|---|---|
| OS | Windows 10 build 26200 |
| Python | 3.11.9 |
| Torch / TorchVision | 2.5.1+cu124 / 0.20.1+cu124 |
| GPU | NVIDIA GeForce RTX 2080 Ti 11GB |
| 驱动 / CUDA runtime | 591.86 / 12.4 |
| 其余直接依赖 | 见 `requirements-runtime.txt` |
| venv | `.venvs/video_matting/` |
| 源码、权重、缓存 | `.models/video_matting/` |

没有安装官方 `pyproject.toml` 中与推理无关的 Gradio、PySide6、训练数据增强和 GUI
依赖。最小包装路径为 OpenCV 解码 -> 官方 `InferenceCore.step` -> PNG alpha/RGBA。
当前系统没有全局 FFmpeg，因此不调用上游依赖 PyAV/FFmpeg 的视频读写入口。模型、
memory manager、recurrent refinement 和 checkpoint loading 都是官方实现，这不是替换
模型的 fallback。

上游模型构造器默认先下载 ImageNet ResNet-18/50，再由完整 release checkpoint 覆盖。
包装器设置 `cfg.model.pretrained_resnet=False`，仅取消这两次冗余下载，不改变网络层或
checkpoint 参数。

## 输入与初始化

| 项目 | 值 |
|---|---|
| 输入 | `poc_output/pet_20260710_121221_5ce7716e_real_after/raw_fast_walk.mp4` |
| 源视频 | 97 帧，24fps，960x960 |
| 本次范围 | frame 0-23，连续 24 帧 |
| 推理尺寸 | 640x640 |
| 首帧 profile | `profile_green_arrays([frame0])` |
| 首帧 alpha | `adaptive_green_matte_frame(frame0, profile)` |
| 模型初始化 | alpha >= 128 的二值 mask，覆盖率 27.7251% |
| recurrent refinement | 10 次 |

首帧初始化直接动态加载仓库现有 `poc.py`，不复制、不修改中央实现。RGBA 的颜色通道
使用现有绿幕算法的 clip-level 去绿结果；最终 A 通道始终是未替换、未后处理的
MatAnyone alpha。

## 性能结果

最终跑测 `runs/fast_walk_24_final`：

| 指标 | 结果 |
|---|---:|
| 完整 CLI 冷启动 | 38.651s |
| Torch import | 7.918s |
| 官方模块 import | 7.748s |
| 模型加载 | 5.081s |
| 10 次 warmup + 24 帧模型段 | 5.227s |
| 24 个输出帧净推理 | 2.119s |
| 输出吞吐 | 11.327 fps |
| 去掉首输出帧后的稳态吞吐 | 11.284 fps |
| 单帧 p50 / p95 | 74.4ms / 127.2ms |
| Torch 峰值 allocated | 697.736MiB |
| Torch 峰值 reserved | 912.000MiB |
| RGBA 写盘 | 2.975s |
| RGBA 序列 SHA-256 | `cb2d9097bb73cb87ba15cb03eb27a2d912a93c81691126344d3a1bb2261cc91b` |

冷启动受 Windows Torch DLL 导入和实时扫描影响明显。若进入服务链路，应常驻模型，
不能把完整 CLI 墙钟当作稳态逐帧性能；11.3fps 仍低于源视频 24fps，离线处理可用，
单卡实时处理不达标。

## 时序质量

| 指标 | MatAnyone | 现有逐帧绿幕参考 |
|---|---:|---:|
| 可见覆盖均值 | 25.6352% | 25.3094% |
| 可见覆盖 CV | 8.6473% | 8.4277% |
| 相邻帧 alpha MAE 均值 | 1.1087% | 1.0763% |
| 最大连通分量数 | 2 | 20 |
| 空帧 / 触边帧 | 0 / 0 | 0 / 0 |

相邻帧 MAE 同时包含真实步态和位移，不能单独解释成 flicker。连通分量减少说明主体
语义更连贯，但也可能代表胡须、毛尖等细小结构被平滑，不能简单视为越少越好。

以受控绿幕结果作为诊断参考，不作为真实 alpha ground truth：

| 对照指标 | MatAnyone 二值 mask 初始化 |
|---|---:|
| 平均 alpha MAE | 0.3612% |
| 参考前景内部变软 | 1.8060% |
| 参考前景内部重度漏透明 | 0.0947% |
| 参考背景泄漏 | 0.3594% |

### 优点

- 连续 24 帧始终锁定同一只猫，尾巴、四肢和主体没有整块丢失。
- 无空帧、无画布触边，步态中的轮廓变化连续。
- 640x640 峰值 Torch 显存不到 1GiB，11GB 卡有充足余量。
- 首帧目标可由现有绿幕 mask 自动提供，不需要人工点选。

### 缺点

- 鼻口区域在多数帧出现小块半透明误修；首帧输入内部原本完全不透明，因此属于模型
  recurrent refinement 的宠物域误判。
- 对细碎毛发/胡须偏平滑。语义稳定性提升与细节损失同时存在。
- MatAnyone 只预测 alpha，不估计去污染后的 foreground RGB；本项目仍需现有绿幕颜色
  清理。腿内侧已经烘焙进源视频的黄白高光不能由 alpha 模型修复。
- 11.3fps 不能覆盖 24fps 实时链路。
- 商业许可证尚未闭合。

## 初始化与 warmup 对照

| 配置 | 结果 |
|---|---|
| soft alpha + warmup 10 | 平均 alpha MAE 0.4696%，内部重度漏透明 0.4633%；劣于二值 mask |
| binary mask + warmup 10 | 最终推荐；平均 alpha MAE 0.3612%，内部重度漏透明 0.0947% |
| binary mask + warmup 0 | 第 0 帧出现更多散布的内部半透明斑，后续才收敛；不推荐 |

对照结果保存在 `runs/fast_walk_24`、`runs/fast_walk_24_mask_init` 和
`runs/fast_walk_24_mask_warmup0`。最终默认因此采用 binary mask + warmup 10。

## 官方代码状态与最小包装

官方模型核心已在 Windows 本地跑通，因此没有切换到替代模型，也没有引入 RVM。
上游完整 CLI 的 PyAV/FFmpeg 视频 I/O 和混合 GUI/训练依赖没有作为本次部署入口；
`setup.ps1` 安装最小推理依赖，`run.ps1` 固定所有缓存和写入目录，`matanyone_cli.py`
直接调用官方模型核心并输出逐帧 PNG。官方 checkout 在跑测后 `git status` 为空。

## 证据索引

| 文件 | 内容 |
|---|---|
| `evidence/00_bootstrap_venv.log` | venv / pip bootstrap |
| `evidence/01_install_torch.log` | 官方 cu124 Torch wheel 安装 |
| `evidence/02_install_runtime.log` | 最小运行时依赖安装 |
| `evidence/03_cuda_model_smoke.log` | CUDA kernel、权重加载、参数量冒烟 |
| `evidence/04_fast_walk_24_run.log` | soft-alpha 初始化 24 帧对照 |
| `evidence/05_fast_walk_24_mask_init_run.log` | binary-mask 初始化 24 帧对照 |
| `evidence/06_fast_walk_24_mask_warmup0_run.log` | warmup=0 对照 |
| `evidence/07_fast_walk_24_final_run.log` | 最终推荐配置完整 stdout |
| `evidence/08_setup_idempotent.log` | setup 二次运行和 `pip check` |

## 建议

当前受控绿幕素材继续以现有绿幕算法作为生产主链路。MatAnyone 可保留为非均匀背景、
遮挡或需要首帧目标传播时的候选，但接入中央 harness 前至少需要：

1. 扩展到多品种猫狗、深浅毛色、快速转身和遮挡的 100+ 片段评测。
2. 增加内部漏透明检测，避免鼻口/眼睛等主体内部被误修后静默交付。
3. 将模型常驻服务，单独测稳态吞吐、并发和长视频 memory 行为。
4. 取得商业使用许可，或选择许可证兼容且明确覆盖任意物体的视频 matting 模型。
