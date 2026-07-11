# BiRefNet local provider

状态：已完成本地部署、CUDA/CPU 实跑和 9 帧小样校验。写入仅位于：

- `matting_bench/providers/birefnet/`
- `.models/birefnet/`
- `.venvs/birefnet/`

未修改中央 harness，未创建 Git commit。

## 模型与许可证

| 项目 | 值 |
|---|---|
| 官方权重 | `ZhengPeng7/BiRefNet` |
| HF revision | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` |
| 模型类型 | 标准通用 BiRefNet，Swin-L，输入 `1024x1024` |
| 许可证 | MIT（HF metadata 与官方 GitHub `LICENSE`） |
| 权重 | `model.safetensors`，444,473,596 bytes / 423.883 MiB |
| SHA-256 | `9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154` |
| 本地路径 | `D:\work\pet-avatar-3d-mvp\.models\birefnet\ZhengPeng7--BiRefNet` |

来源：[官方 Hugging Face 模型](https://huggingface.co/ZhengPeng7/BiRefNet)、[官方 GitHub 仓库](https://github.com/ZhengPeng7/BiRefNet)、[MIT LICENSE](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE)。`download_model.py` 固定 revision，并校验权重大小与 SHA-256。

输出使用模型 alpha，同时复用官方 HF `handler.py` 的两次 blur-fusion 前景估计（`r=90`、`r=6`）减轻半透明毛发边缘的绿幕溢色。

## 安装

PowerShell：

```powershell
Set-Location D:\work\pet-avatar-3d-mvp

New-Item -ItemType Directory -Force -Path `
  .models\birefnet\pip-cache, `
  .models\birefnet\tmp, `
  .models\birefnet\hf-cache, `
  .models\birefnet\torch-cache | Out-Null

py -3.11 -m venv .venvs\birefnet

$root = (Resolve-Path .).Path
$env:PIP_CACHE_DIR = Join-Path $root '.models\birefnet\pip-cache'
$env:TEMP = Join-Path $root '.models\birefnet\tmp'
$env:TMP = $env:TEMP
$env:HF_HOME = Join-Path $root '.models\birefnet\hf-cache'
$python = Join-Path $root '.venvs\birefnet\Scripts\python.exe'

& $python -m pip install --disable-pip-version-check --no-cache-dir --upgrade pip
& $python -m pip install --disable-pip-version-check --no-cache-dir `
  -r matting_bench\providers\birefnet\requirements.lock.txt
& $python matting_bench\providers\birefnet\download_model.py
& $python -m pip check
```

已安装核心版本：Python 3.11.9、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、Transformers 4.47.1、timm 1.0.12。完整锁定见 `requirements.lock.txt`。

## CLI

必需接口：

```powershell
.venvs\birefnet\Scripts\python.exe `
  matting_bench\providers\birefnet\infer.py `
  --input-dir <PNG目录> `
  --output-dir <输出目录> `
  --device cuda `
  --input-resolution 1024 `
  --foreground-refinement official-auto
```

`--device` 可取 `cuda` 或 `cpu`；`--input-resolution` 可取 `512/768/1024`；`--foreground-refinement` 可取 `official-auto/official-cpu/official-gpu/none`。`official-auto` 在 CUDA 上使用官方 GPU 两阶段前景估计，在 CPU 上使用官方 OpenCV 路径；`none` 只用于消融。CLI 读取输入目录直属的所有 `.png`，在输出目录生成同名 RGBA PNG；CUDA 使用 FP16，CPU 使用 FP32。运行完全离线，默认读取固定本地模型目录。可选 `--metrics-json` 写出逐帧时间和显存记录。

完整参数出处、9 帧 sweep、持锁计时与 24 帧时序结论见 `TUNING_REPORT.md` 和 `tuning_results.json`。

本次两条实跑命令：

```powershell
$python = '.venvs\birefnet\Scripts\python.exe'
$provider = 'matting_bench\providers\birefnet'

& $python $provider\infer.py `
  --input-dir $provider\sample_inputs `
  --output-dir $provider\sample_outputs\cuda `
  --device cuda `
  --metrics-json $provider\run_metrics\cuda.json

& $python $provider\infer.py `
  --input-dir $provider\sample_inputs `
  --output-dir $provider\sample_outputs\cpu `
  --device cpu `
  --metrics-json $provider\run_metrics\cpu.json
```

## 小样

来源：`poc_output\pet_20260710_121221_5ce7716e_real_after` 下：

- `raw_idle.mp4`
- `raw_sleep.mp4`
- `raw_fast_walk.mp4`

三段视频均为 960x960、24 FPS、97 帧。每段按 25% / 50% / 75% 抽取零基帧 24、48、72，即 1.0、2.0、3.0 秒，共 9 帧。系统无 `ffmpeg/ffprobe`，因此使用独立 venv 内的 OpenCV 抽帧：

```powershell
& $python $provider\extract_samples.py `
  --video-dir poc_output\pet_20260710_121221_5ce7716e_real_after `
  --output-dir $provider\sample_inputs
```

详细来源清单：`sample_inputs/samples.json`。

## 实测结果

机器：Intel64 Family 6 Model 158，16 逻辑核（PyTorch 8 线程）；RTX 2080 Ti 11GB；驱动 591.86。

| 指标 | CUDA FP16 | CPU FP32 |
|---|---:|---:|
| 模型加载 | 14.65 s | 4.51 s |
| 首帧纯推理 | 13.95 s | 45.28 s |
| 后续帧平均纯推理 | 0.263 s | 47.07 s |
| 9 帧纯推理合计 | 16.06 s | 421.87 s |
| 9 帧端到端合计 | 24.20 s | 431.71 s |
| 完整命令墙钟时间 | 70.9 s | 449.4 s |
| 峰值显存 allocated | 1.60 GiB | N/A |
| 峰值显存 reserved | 2.78 GiB | N/A |

CUDA 首帧包含内核冷启动。CPU 在两次真实运行中测得平均纯推理 33.47–46.87 s/帧，受同机负载影响明显；表内为最终 RGBA 产物对应的第二次运行。显存是从模型加载前到最后输出的 PyTorch allocator 峰值，不含驱动上下文的不可见开销。

原始逐帧结果：`run_metrics/cuda.json`、`run_metrics/cpu.json`。完整汇总见 `metrics.json`。

## 输出与校验

- CUDA RGBA：`sample_outputs/cuda/`
- CPU RGBA：`sample_outputs/cpu/`
- 灰底合成预览：`diagnostics/cuda_contact_sheet_gray.png`

校验结果：两侧均为 9 个同名 960x960 RGBA PNG；alpha 覆盖 0–255，共 179,489 个半透明像素。CPU/CUDA alpha 全像素 MAE 为 0.012/255，95% 像素差为 0，最大差为 14。前景细化使半透明像素上的绿色过量均值从 84.40 降到 3.49。

未提供真值 alpha，因此没有计算 SAD、MSE 或 IoU；当前结论限于结构校验、CPU/CUDA 一致性和灰底视觉抽查。强背光边缘仍可见少量源素材白色 halo。

## 失败点

1. 系统 PATH 无 `ffmpeg/ffprobe`。已改用锁定版本 `opencv-python-headless==4.10.0.84`，9 帧全部抽取成功。
2. 首次 CUDA 运行把无索引 `torch.device('cuda')` 传给 `torch.cuda.set_device()`，PyTorch 2.5.1 报错。CLI 已在内部规范化为 `cuda:0`，外部接口仍是 `--device cuda`；原始错误在 `run_metrics/cuda_attempt1_error.json`。
3. CPU 重复测量波动较大。`metrics.json` 同时记录两次实测范围，不将单次结果解释为稳定吞吐。
