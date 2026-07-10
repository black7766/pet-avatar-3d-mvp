# BiRefNet-matting provider

Pinned local deployment of the official `ZhengPeng7/BiRefNet-matting` checkpoint.
It is evaluated separately from the general BiRefNet checkpoint because this model
was trained specifically for general matting datasets.

```powershell
.venvs\birefnet\Scripts\python.exe matting_bench\providers\birefnet_matting\download_model.py
.venvs\birefnet\Scripts\python.exe matting_bench\providers\birefnet\infer.py `
  --input-dir matting_bench\data\pet_20260710_121221_5ce7716e\smoke `
  --output-dir matting_bench\outputs\birefnet_matting\smoke `
  --device cuda `
  --model-dir .models\birefnet\ZhengPeng7--BiRefNet-matting `
  --model-repo ZhengPeng7/BiRefNet-matting `
  --model-revision 57f9f68b43ba337c75762b14cf3075d659007268 `
  --metrics-json matting_bench\outputs\birefnet_matting\metrics.json
```

Source: <https://huggingface.co/ZhengPeng7/BiRefNet-matting>

The upstream repository declares the MIT license. The checkpoint card links to
that repository and does not declare a separate model license field, so legal
review should retain the pinned card and repository license before commercial use.
