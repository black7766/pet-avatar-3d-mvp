# rembg family parameter tuning report

Generated: 2026-07-11

## Recommendation

Primary: `u2net__alpha_default`

```text
model = u2net
alpha_matting = true
alpha_matting_foreground_threshold = 240
alpha_matting_background_threshold = 10
alpha_matting_erode_size = 10
post_process_mask = false
```

Secondary: `isnet-general-use__alpha_default`, with the same `remove()` parameters.

The primary keeps the most soft detail outside the binary core, preserves visible
whiskers in the review sheets, matches ISNet's 24-frame temporal stability, and is
faster. The secondary has marginally lower pseudo error and fragment ratio and uses a
1024x1024 model input, but it removes slightly more soft detail and is slower.

Do not enable `post_process_mask` for the final pet alpha. Its pseudo score looks best
only because the operation makes alpha binary: `soft_alpha_pct` becomes exactly zero,
the green-fringe metric has no soft pixels to inspect, and whiskers/fur are hard-cut.

## Test protocol

- Hardware: NVIDIA GeForce RTX 2080 Ti 11264 MiB, driver 591.86.
- Runtime: `rembg==2.0.76`, `onnxruntime-gpu==1.23.2`.
- Execution providers requested and verified active: `CUDAExecutionProvider`, then
  `CPUExecutionProvider`. TensorRT was available but not requested.
- One cold process and one model session per config; the session was reused across all
  frames in that config.
- Smoke dataset: 9 fixed 960x960 frames, three each from fast walk, idle, and sleep.
- Temporal dataset: 24 consecutive 640x640 fast-walk frames for the two finalists.
- Quality was computed only by central `matting_bench/evaluate.py`.
- Runtime values below are per-frame means excluding the first frame. Session loading
  is not included; end-to-end includes PNG load, `remove()`, and save.
- Every final CUDA inference command ran through
  `python matting_bench/run_with_gpu_lock.py -- ...`. All 14 successful run logs were
  checked for both lock acquisition and release. Lock-contention attempts exited
  before inference and are not used as measurements.
- Peak VRAM was not instrumented in this adapter, so `peak_vram_mb` is `null`; this
  report makes no VRAM claim.

The source has a controlled green background but no hand-painted alpha ground truth.
`pseudo_mae` therefore combines confident-green background leakage and confident
non-green foreground loss. Boundary quality still requires the visual and soft-detail
checks below.

## Smoke results

| Config | Pseudo MAE | BG alpha | FG loss | Green fringe | Fragment % | Soft alpha % | Infer ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `u2net__default` | 0.0044215 | 0.0022985 | 0.0065445 | 0.0412470 | 0.00238 | 1.7002 | 61.4 | 116.1 |
| `u2net__alpha_default` | 0.0020570 | 0.0031347 | 0.0009793 | 0.0126660 | 0.00125 | 1.6413 | 935.6 | 1042.7 |
| `u2net__alpha_fur_safe` | 0.0016293 | 0.0027428 | 0.0005158 | 0.0182488 | 0.00341 | 1.1989 | 561.6 | 654.4 |
| `u2net__postprocess_binary` | 0.0011317 | 0.0009950 | 0.0012683 | 0.0000000 | 0.00000 | 0.0000 | 101.4 | 160.2 |
| `isnet-general-use__default` | 0.0034490 | 0.0016937 | 0.0052042 | 0.0596592 | 0.00416 | 1.1476 | 165.3 | 223.5 |
| `isnet-general-use__alpha_default` | 0.0019785 | 0.0030876 | 0.0008694 | 0.0131505 | 0.00123 | 1.5648 | 956.0 | 1054.3 |
| `isnet-general-use__alpha_fur_safe` | 0.0014174 | 0.0025847 | 0.0002501 | 0.0255230 | 0.00163 | 0.9952 | 628.0 | 723.0 |
| `isnet-general-use__postprocess_binary` | 0.0007726 | 0.0012214 | 0.0003239 | 0.0000000 | 0.00000 | 0.0000 | 183.5 | 236.1 |
| `birefnet-general-lite__default` | 0.0037917 | 0.0028655 | 0.0047178 | 0.0846074 | 0.00041 | 0.9835 | 5034.1 | 5095.2 |
| `birefnet-general-lite__alpha_default` | 0.0022832 | 0.0031293 | 0.0014372 | 0.0139116 | 0.00128 | 1.5808 | 5003.9 | 5096.1 |
| `birefnet-general-lite__alpha_fur_safe` | 0.0018313 | 0.0029540 | 0.0007086 | 0.0320194 | 0.00221 | 1.0119 | 4875.6 | 4966.9 |
| `birefnet-general-lite__postprocess_binary` | 0.0015822 | 0.0023988 | 0.0007655 | 0.0000000 | 0.00000 | 0.0000 | 4309.6 | 4363.1 |

BiRefNet General Lite is not competitive on this Windows ONNX path: it takes about
4.9-5.0 seconds per hot 960px frame and does not beat the two alpha-default finalists
on quality.

## Temporal results

| Config | Frames | Temporal alpha MAE | Pseudo MAE | Green fringe | Fragment % | Soft alpha % | Infer ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `u2net__alpha_default` | 24 | 0.0127528 | 0.0026812 | 0.0076542 | 0.00209 | 1.9372 | 463.1 | 516.9 |
| `isnet-general-use__alpha_default` | 24 | 0.0127571 | 0.0026534 | 0.0077454 | 0.00162 | 1.9126 | 535.4 | 581.0 |

The temporal difference is negligible (`0.0000043`). ISNet is marginally tighter on
pseudo MAE and fragments; U2Net is about 13.5% faster on hot inference and retains
slightly more soft alpha.

## Fine-hair guardrail

The `alpha_fur_safe` hypothesis used `225/5/3`: lower known-background threshold and
smaller erosion. Its aggregate pseudo score improved, but the narrower unknown trimap
band deleted soft detail that the official `240/10/10` alpha settings retained.

The diagnostic below counts alpha > 0.08 pixels outside each model's binary
post-process core. It is a relative detail-retention check, not ground truth.

| Model | Alpha-default detail px/frame | 225/5/3 detail px/frame | Default detail lost by 225/5/3 | New detail from 225/5/3 |
|---|---:|---:|---:|---:|
| U2Net | 6520.7 | 5729.6 | 280.6 | 6.9 |
| ISNet General | 5728.9 | 4620.2 | 589.0 | 8.7 |
| BiRefNet General Lite | 4683.4 | 4004.0 | 267.2 | 12.4 |

This is why the lower-pseudo `alpha_fur_safe` variants are not recommended. The visual
review reaches the same conclusion: alpha defaults preserve more whiskers and ear/body
fur, while the binary variants remove them.

Review artifacts:

- `visual_review/smoke_candidates_contact.jpg`
- `visual_review/fast_walk_mid_full_600.jpg`
- `visual_review/idle_mid_head_600.jpg`

## Official parameter behavior

Current `rembg` code is authoritative for the pinned version:

- `remove()` defaults are `alpha_matting=False`, foreground threshold `240`,
  background threshold `10`, erosion size `10`, and `post_process_mask=False`.
- Alpha matting marks mask values above the foreground threshold as known foreground,
  values below the background threshold as known background, erodes both known
  regions, and solves the remaining trimap with PyMatting. It also estimates foreground
  RGB, so RGBA must be reviewed through explicit compositing.
- `post_process_mask` performs a radius-1 morphological opening, Gaussian filtering
  with sigma 2, then thresholds at 127. The final mask is binary.
- The official usage page contains an older alpha example with foreground threshold
  `270`. That exceeds an 8-bit mask and was not used; the current 2.0.76 API/code
  defaults were tested instead.

Model/session behavior:

| Model | Fixed input | Normalization | Output handling |
|---|---|---|---|
| U2Net | 320x320 | ImageNet mean/std | first output, per-image min-max, LANCZOS back to source |
| ISNet General | 1024x1024 | mean 0.5, std 1.0 | first output, per-image min-max, LANCZOS back to source |
| BiRefNet General Lite | 1024x1024 | ImageNet mean/std | sigmoid, per-image min-max, LANCZOS back to source |

`new_session()` uses default ONNX Runtime session options. If `OMP_NUM_THREADS` is set,
rembg assigns it to both inter-op and intra-op thread counts. It was unset for these
CUDA runs. The adapter explicitly supplies CUDA then CPU providers and verifies that
CUDA remains first after every frame.

Official sources:

- [rembg README](https://github.com/danielgatis/rembg/blob/main/README.md)
- [rembg usage/API examples](https://github.com/danielgatis/rembg/blob/main/USAGE.md)
- [remove, alpha matting, and post-process implementation](https://github.com/danielgatis/rembg/blob/main/rembg/bg.py)
- [session factory](https://github.com/danielgatis/rembg/blob/main/rembg/session_factory.py)
- [U2Net rembg session](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/u2net.py)
- [ISNet General rembg session](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/dis_general_use.py)
- [BiRefNet rembg session](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/birefnet_general.py)
- [U2Net upstream inference](https://github.com/xuebinqin/U-2-Net/blob/master/u2net_test.py)
- [DIS/ISNet upstream inference](https://github.com/xuebinqin/DIS/blob/main/IS-Net/Inference.py)
- [BiRefNet upstream documentation](https://github.com/ZhengPeng7/BiRefNet)

## Reproduction

Run the complete 12-config smoke sweep, the two selected temporal runs, and collection:

```powershell
.venvs/rembg/Scripts/python.exe `
  matting_bench/providers/rembg/tuning_sweep.py `
  --phase all --device cuda --overwrite
```

The sweep wraps every CUDA inference subprocess as:

```powershell
python matting_bench/run_with_gpu_lock.py -- <rembg infer command>
```

Run only the production recommendation directly:

```powershell
python matting_bench/run_with_gpu_lock.py -- `
  .venvs/rembg/Scripts/python.exe `
  matting_bench/providers/rembg/infer.py `
  --model u2net --device cuda `
  --alpha-matting `
  --alpha-matting-foreground-threshold 240 `
  --alpha-matting-background-threshold 10 `
  --alpha-matting-erode-size 10 `
  --input-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --output-dir matting_bench/providers/rembg/outputs/reproduction/u2net_alpha_default
```

Compile checks:

```powershell
.venvs/rembg/Scripts/python.exe -m py_compile `
  matting_bench/providers/rembg/infer.py `
  matting_bench/providers/rembg/tuning_sweep.py
```

Machine-readable results, complete parameter/session records, relative output paths,
and per-config notes are in `tuning_results.json`.
