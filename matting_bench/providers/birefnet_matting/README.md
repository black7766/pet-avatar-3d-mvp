# BiRefNet-matting provider

Pinned local deployment of the official `ZhengPeng7/BiRefNet-matting` checkpoint.
It is evaluated separately from the general BiRefNet checkpoint because this model
was trained specifically for general matting datasets.

```powershell
.venvs\birefnet\Scripts\python.exe matting_bench\providers\birefnet_matting\download_model.py
.venvs\birefnet\Scripts\python.exe matting_bench\providers\birefnet_matting\infer.py `
  --input-dir matting_bench\data\pet_20260710_121221_5ce7716e\smoke `
  --output-dir matting_bench\outputs\birefnet_matting\smoke `
  --device cuda `
  --input-resolution 1024 `
  --foreground-refinement official-auto `
  --metrics-json matting_bench\outputs\birefnet_matting\metrics.json
```

`infer.py` 默认固定 matting checkpoint 与 revision。输入分辨率可取
`512/768/1024`；foreground refinement 可取
`official-auto/official-cpu/official-gpu/none`，其中 `none` 只用于消融。
完整参数出处、9 帧 sweep、持锁计时与 24 帧时序结论见
`TUNING_REPORT.md` 和 `tuning_results.json`。

Source: <https://huggingface.co/ZhengPeng7/BiRefNet-matting>

The upstream repository declares the MIT license. The checkpoint card links to
that repository and does not declare a separate model license field, so legal
review should retain the pinned card and repository license before commercial use.
