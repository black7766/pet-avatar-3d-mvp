# rembg 9-frame smoke results

Generated: `2026-07-10T09:08:40.623398+00:00`

Each model ran in a cold Python process with no warm-up frame. `remove` timing covers rembg preprocessing, ONNX inference, and mask postprocessing; end-to-end timing also includes PNG load/save.

CUDA host: `NVIDIA GeForce RTX 2080 Ti`, driver `591.86`, 11264 MiB VRAM.

## Runtime

| Model | Device / active EP | Weight MiB | Session load s | 9-frame remove s | Mean ms/frame | Mean excl. first ms | End-to-end s | Output |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `u2net` | `cuda` / `CUDAExecutionProvider, CPUExecutionProvider` | 167.84 | 1.144 | 6.107 | 678.6 | 94.2 | 7.285 | `D:\work\pet-avatar-3d-mvp\matting_bench\providers\rembg\outputs\smoke\u2net` |
| `isnet-general-use` | `cuda` / `CUDAExecutionProvider, CPUExecutionProvider` | 170.37 | 2.485 | 3.795 | 421.7 | 315.6 | 5.072 | `D:\work\pet-avatar-3d-mvp\matting_bench\providers\rembg\outputs\smoke\isnet-general-use` |
| `birefnet-general-lite` | `cuda` / `CUDAExecutionProvider, CPUExecutionProvider` | 213.63 | 9.017 | 43.289 | 4809.9 | 5043.3 | 44.824 | `D:\work\pet-avatar-3d-mvp\matting_bench\providers\rembg\outputs\smoke\birefnet-general-lite` |

## Provenance

License values below are the licenses declared by the upstream model repositories. rembg itself is MIT-licensed.

| Model | Upstream source | License | Official rembg weight | MD5 |
|---|---|---|---|---|
| `u2net` | [repository](https://github.com/xuebinqin/U-2-Net) | [Apache-2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE) | [ONNX](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx) | `60024c5c889badc19c04ad937298a77b` |
| `isnet-general-use` | [repository](https://github.com/xuebinqin/DIS) | [Apache-2.0](https://github.com/xuebinqin/DIS/blob/main/LICENSE.md) | [ONNX](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx) | `fc16ebd8b0c10d971d3513d564d01e29` |
| `birefnet-general-lite` | [repository](https://github.com/ZhengPeng7/BiRefNet) | [MIT](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE) | [ONNX](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx) | `4fab47adc4ff364be1713e97b7e66334` |
