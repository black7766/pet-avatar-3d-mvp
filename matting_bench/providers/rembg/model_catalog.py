"""Pinned rembg model metadata used by the local benchmark adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    weight_url: str
    expected_md5: str
    upstream_repo: str
    upstream_license: str
    upstream_license_url: str
    input_size: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_size"] = list(self.input_size)
        return payload


MODEL_SPECS = {
    "u2net": ModelSpec(
        name="u2net",
        filename="u2net.onnx",
        weight_url=(
            "https://github.com/danielgatis/rembg/releases/download/"
            "v0.0.0/u2net.onnx"
        ),
        expected_md5="60024c5c889badc19c04ad937298a77b",
        upstream_repo="https://github.com/xuebinqin/U-2-Net",
        upstream_license="Apache-2.0",
        upstream_license_url="https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE",
        input_size=(320, 320),
    ),
    "isnet-general-use": ModelSpec(
        name="isnet-general-use",
        filename="isnet-general-use.onnx",
        weight_url=(
            "https://github.com/danielgatis/rembg/releases/download/"
            "v0.0.0/isnet-general-use.onnx"
        ),
        expected_md5="fc16ebd8b0c10d971d3513d564d01e29",
        upstream_repo="https://github.com/xuebinqin/DIS",
        upstream_license="Apache-2.0",
        upstream_license_url="https://github.com/xuebinqin/DIS/blob/main/LICENSE.md",
        input_size=(1024, 1024),
    ),
    "birefnet-general-lite": ModelSpec(
        name="birefnet-general-lite",
        filename="birefnet-general-lite.onnx",
        weight_url=(
            "https://github.com/danielgatis/rembg/releases/download/"
            "v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
        ),
        expected_md5="4fab47adc4ff364be1713e97b7e66334",
        upstream_repo="https://github.com/ZhengPeng7/BiRefNet",
        upstream_license="MIT",
        upstream_license_url="https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        input_size=(1024, 1024),
    ),
}
