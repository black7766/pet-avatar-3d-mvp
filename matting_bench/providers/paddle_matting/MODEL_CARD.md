# PP-MattingV2 local model card

## Selection

- Model: `PP-MattingV2-STDC1-human-512`
- PaddleSeg release: `v2.10.0`
- Source commit: `d459390adcec7fa6dd010c21b71aeb73f2afded9`
- Official model archive: <https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmattingv2-stdc1-human_512.zip>
- Archive SHA-256: `daff48b08c61958b9a21093791f6aed8eb3939b34b7418e40c18b2348136893d`
- Downloaded archive size: 33,379,434 bytes

PP-MattingV2 is the official PaddleSeg lightweight real-time matting model. The
PaddleSeg model zoo explicitly describes the released model family as **human
matting**, and this checkpoint name also contains `human_512`. It is not an
official pet or category-agnostic model. The pet runs in this provider are
cross-domain probes only.

## Extracted inference files

| File | SHA-256 |
|---|---|
| `model.pdmodel` | `350b9c6c68e1ce9f57ab7853f23ee30f55ddafc5f3e8dea14f1bb8c020486e7d` |
| `model.pdiparams` | `49ab8f0c77138ca5f0dcfd19c3f578704d1e386b399d90f985ef7f20704ec606` |
| `deploy.yaml` | `181ad89e515671cd3b3b4a4c712def0319448523bd5502b938114e2a62459e30` |

## Preprocessing

The CLI reads `deploy.yaml` and implements the official transform chain:

1. `LoadImages` with BGR-to-RGB conversion.
2. `LimitShort(max_short=512)`.
3. `ResizeToIntMult(mult_int=32)`.
4. `Normalize(mean=0.5, std=0.5)` and HWC-to-NCHW conversion.

The predicted alpha is bilinearly restored to the original image dimensions and
packed with the source RGB into an 8-bit RGBA PNG. Fully transparent RGB pixels
are zeroed; visible RGB pixels are not recolored or foreground-estimated.

## License

The downloaded model ZIP does not contain a separate license file. It is hosted
and linked by the official PaddleSeg model zoo. PaddleSeg source is Apache-2.0;
the pinned source checkout and a copied license are retained under
`.models/paddle_matting/`. Confirm artifact licensing with Baidu/PaddlePaddle
before commercial redistribution if a separate model-weight grant is required.
