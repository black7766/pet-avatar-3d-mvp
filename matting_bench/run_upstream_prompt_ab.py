"""Run one paid upstream-prompt A/B probe and compare local green matting.

The script is deliberately self-contained: it does not mutate the production prompt
or pipeline modules. It reads ARK_API_KEY from the process environment or repository
.env, resumes a completed task without resubmitting, and writes only below the
dedicated experiment directory plus the result section in the research Markdown.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "matting_bench"
OUTPUT = ROOT / "poc_output" / "upstream_prompt_ab_20260712"
FIRST_FRAME = (
    ROOT
    / "poc_output"
    / "pet_20260710_121221_5ce7716e_real_after"
    / "state_fast_walk.png"
)
A0_VIDEO = (
    ROOT
    / "poc_output"
    / "pet_20260710_121221_5ce7716e_real_after"
    / "raw_fast_walk.mp4"
)
RESEARCH_MD = BENCH / "UPSTREAM_GREENSCREEN_PROMPT_RESEARCH_20260712.md"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seedance-1-5-pro-251215"
FRAME_COUNT = 96
FRAME_SIZE = 640
POLL_SECONDS = 8

PROMPT = """Generate a single continuous 5-second production loop of the exact same realistic pet from the first frame, brisk-walking in place. Frame 1 is already inside the stable gait cycle; do not start from idle, stop, sit, jump, run, or return to idle. Preserve the exact identity, natural body proportions, coat color, markings, realistic fur, face, ears, paws, clothing, and complete tail. Do not make the pet cartoon, chibi, plush, toy-like, or mascot-like.

Keep the same front three-quarter body yaw, scale, and screen position for the entire clip. The camera is completely locked. Use a moderate brisk-walk cadence with small per-frame displacement, clearly readable alternating legs, a stable torso with only subtle vertical movement, minimal ear motion, and a low-amplitude slow tail sway. The full tail including its tip must remain sharp, continuous, visible, and separated from the frame edge. It must never whip, flick rapidly, blur, split, duplicate, disappear behind the body, or leave the frame.

Show exactly one complete pet, centered, occupying about 55% to 65% of the frame. Keep at least 15% clean green clearance around the moving silhouette in every frame. Both ear tips, every paw, the full torso and belly, and the entire tail tip must remain visible. No body part may enter the outer 10% border area.

Use a clean short-exposure appearance in every frame: no motion blur, directional blur, temporal ghosting, frame blending, duplicated paws, smeared fur, speed streaks, dark motion trail, dust, floor, contact shadow, cast shadow, or dark plate under the paws. Preserve individual fur strands without an artificially sharpened outline.

Use soft neutral diffuse frontal lighting on the pet with restrained highlights and preserved fur texture. Keep subject lighting separate from background lighting and constant in every frame. No rim light, hair light, backlight, edge light, bloom, glow, overexposed fur tips, green reflection, green bounce, yellow-white halo, or colored spill on the fur, paws, belly, or tail.

Use a uniform matte chroma-green studio backdrop with medium-high brightness and one stable hue across the entire frame and every frame. No gradient, vignette, texture, noise pattern, compression blocks, background movement, props, text, people, or other animals. End on a naturally compatible phase of the same walking cycle; never flash back to idle.

--resolution 720p --duration 5 --camerafixed true --watermark false --generate_audio false"""


def load_api_key() -> str:
    if os.environ.get("ARK_API_KEY"):
        return os.environ["ARK_API_KEY"].strip()
    env_path = ROOT / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().lstrip("\ufeff") == "ARK_API_KEY":
                return value.strip().strip('"').strip("'")
    raise RuntimeError("ARK_API_KEY is missing from the environment and repository .env")


def data_uri(path: Path) -> str:
    import base64

    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def ark_request(
    key: str,
    path: str,
    body: dict[str, Any] | None = None,
    method: str = "POST",
    timeout: int = 300,
    retries: int = 3,
) -> dict[str, Any]:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(ARK_BASE + path, data=payload, method=method)
        request.add_header("Authorization", f"Bearer {key}")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"Ark HTTP {exc.code}: {detail[:800]}") from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= retries:
                raise
            last_error = exc
        time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"Ark request failed: {last_error}")


def public_request_body() -> dict[str, Any]:
    return {
        "model": MODEL,
        "generate_audio": False,
        "content": [
            {"type": "text", "text": PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": data_uri(FIRST_FRAME)},
                "role": "first_frame",
            },
        ],
    }


def redacted_request_body(body: dict[str, Any]) -> dict[str, Any]:
    saved = json.loads(json.dumps(body))
    saved["content"][1]["image_url"]["url"] = (
        f"local-file://{FIRST_FRAME.relative_to(ROOT).as_posix()}"
    )
    return saved


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "pet-avatar-ab/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        destination.write_bytes(response.read())


def create_or_resume_video() -> tuple[Path, dict[str, Any], dict[str, float]]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    video_path = OUTPUT / "raw_fast_walk_a1a2.mp4"
    task_path = OUTPUT / "task.json"
    submitted_path = OUTPUT / "task_submitted.json"
    timing_path = OUTPUT / "timing.json"
    if video_path.is_file() and task_path.is_file() and timing_path.is_file():
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task.get("status") == "succeeded" and video_path.stat().st_size > 0:
            return video_path, task, json.loads(timing_path.read_text(encoding="utf-8"))

    key = load_api_key()
    body = public_request_body()
    (OUTPUT / "request.json").write_text(
        json.dumps(redacted_request_body(body), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    submitted_at = time.time()
    if submitted_path.is_file():
        submitted = json.loads(submitted_path.read_text(encoding="utf-8"))
        task_id = submitted["id"]
        submit_seconds = 0.0
        print(f"[seedance] resume submitted task {task_id}", flush=True)
    else:
        task = ark_request(key, "/contents/generations/tasks", body)
        task_id = task["id"]
        submit_seconds = time.time() - submitted_at
        submitted_path.write_text(
            json.dumps({"id": task_id}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    poll_started = time.time()
    while True:
        time.sleep(POLL_SECONDS)
        task = ark_request(key, f"/contents/generations/tasks/{task_id}", method="GET")
        status = task.get("status")
        print(f"[seedance] {task_id} {status}", flush=True)
        if status == "succeeded":
            break
        if status in {"failed", "cancelled"}:
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(f"Seedance task {status}: {task.get('error')}")
    generation_seconds = time.time() - submitted_at
    poll_seconds = time.time() - poll_started
    task_path.write_text(
        json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    download_started = time.time()
    download(task["content"]["video_url"], video_path)
    timing = {
        "submit_seconds": round(submit_seconds, 3),
        "poll_seconds": round(poll_seconds, 3),
        "generation_wall_seconds": round(generation_seconds, 3),
        "download_seconds": round(time.time() - download_started, 3),
    }
    timing_path.write_text(
        json.dumps(timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return video_path, task, timing


def video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "frames": frames,
        "fps": round(fps, 4),
        "duration_seconds": round(duration, 4),
        "width": width,
        "height": height,
    }


def extract_uniform_frames(video: Path, destination: Path) -> dict[str, Any]:
    info = video_info(video)
    if info["frames"] < FRAME_COUNT:
        raise RuntimeError(f"{video} contains only {info['frames']} frames")
    indices = np.rint(np.linspace(0, info["frames"] - 1, FRAME_COUNT)).astype(int)
    destination.mkdir(parents=True, exist_ok=True)
    existing = sorted(destination.glob("f_*.png"))
    if len(existing) == FRAME_COUNT:
        return info
    for path in existing:
        path.unlink()
    capture = cv2.VideoCapture(str(video))
    for output_index, source_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"failed reading frame {source_index} from {video}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(
            (FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS
        )
        image.save(destination / f"f_{output_index:04d}.png", compress_level=2)
    capture.release()
    return info


def run_matte(source: Path, destination: Path, edge_refine: bool) -> None:
    expected = sorted(destination.glob("f_*.png"))
    if len(expected) == FRAME_COUNT and (destination / "metrics.json").is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(BENCH / "providers" / "baseline" / "infer.py"),
        "--input-dir",
        str(source),
        "--output-dir",
        str(destination),
        "--device",
        "cpu",
        "--core-despill",
        "1.10",
    ]
    if edge_refine:
        command.append("--edge-refine")
    subprocess.run(command, cwd=ROOT, check=True)


def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def green_score(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (g - np.maximum(r, b)) / np.maximum(g, 1.0 / 255.0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def geometry_metrics(source_dir: Path, output_dir: Path) -> dict[str, float]:
    border_touch = 0
    source_border_risk = 0
    tail_proxy_retention: list[float] = []
    tail_proxy_connected: list[float] = []
    min_clearances: list[float] = []
    subject_coverage: list[float] = []
    for source_path in sorted(source_dir.glob("f_*.png")):
        source = load_rgb(source_path)
        output = load_rgba(output_dir / source_path.name)
        alpha = output[:, :, 3]
        mask = alpha > 0.08
        main = largest_component(mask)
        ys, xs = np.where(main)
        if not len(xs):
            border_touch += 1
            min_clearances.append(0.0)
            tail_proxy_retention.append(0.0)
            tail_proxy_connected.append(0.0)
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        clearance = min(x0, y0, FRAME_SIZE - 1 - x1, FRAME_SIZE - 1 - y1)
        min_clearances.append(float(clearance))
        subject_coverage.append(float(main.mean() * 100.0))
        if clearance <= 4:
            border_touch += 1

        score = green_score(source)
        source_fg = score < 0.10
        sy, sx = np.where(source_fg)
        if not len(sx):
            tail_proxy_retention.append(0.0)
            tail_proxy_connected.append(0.0)
            continue
        sx0, sx1, sy0, sy1 = int(sx.min()), int(sx.max()), int(sy.min()), int(sy.max())
        if min(sx0, sy0, FRAME_SIZE - 1 - sx1, FRAME_SIZE - 1 - sy1) <= 4:
            source_border_risk += 1
        width = max(1, sx1 - sx0 + 1)
        left = source_fg & (np.arange(FRAME_SIZE)[None, :] <= sx0 + width * 0.22)
        right = source_fg & (np.arange(FRAME_SIZE)[None, :] >= sx1 - width * 0.22)
        # The larger lateral appendage band is a deterministic tail/paw proxy. It is
        # intentionally reported as a proxy because no semantic tail GT is available.
        candidate = left if int(left.sum()) >= int(right.sum()) else right
        if candidate.any():
            tail_proxy_retention.append(float(alpha[candidate].mean()))
            tail_proxy_connected.append(float(main[candidate].mean()))
        else:
            tail_proxy_retention.append(0.0)
            tail_proxy_connected.append(0.0)
    return {
        "border_touch_frames": float(border_touch),
        "source_border_risk_frames": float(source_border_risk),
        "min_subject_clearance_px": float(min(min_clearances, default=0.0)),
        "mean_subject_clearance_px": float(np.mean(min_clearances)),
        "subject_coverage_pct": float(np.mean(subject_coverage)),
        "lateral_appendage_alpha_retention": float(np.mean(tail_proxy_retention)),
        "lateral_appendage_main_component_ratio": float(np.mean(tail_proxy_connected)),
    }


def evaluate(source_dir: Path, output_dir: Path, duration: float) -> dict[str, Any]:
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import evaluate as benchmark  # type: ignore

    result = benchmark.evaluate_provider(source_dir, output_dir)
    temporal = result.get("temporal_alpha_mae")
    sample_delta = duration / max(1, FRAME_COUNT - 1)
    result["temporal_alpha_mae_per_second"] = (
        float(temporal) / sample_delta if temporal is not None and sample_delta else None
    )
    result["geometry"] = geometry_metrics(source_dir, output_dir)
    return result


def change_pct(value: float, baseline: float, lower_is_better: bool = True) -> float | None:
    if baseline == 0:
        return None
    raw = (value - baseline) / baseline * 100.0
    return -raw if lower_is_better else raw


def append_research_summary(result: dict[str, Any]) -> None:
    marker = "## 11. 最小实测结果（2026-07-12）"
    text = RESEARCH_MD.read_text(encoding="utf-8")
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n\n"
    usage = result["api"]["usage"]
    timing = result["timing"]
    a0 = result["variants"]["A0"]["adaptive_green_edge_v2"]
    a2 = result["variants"]["A1A2"]["adaptive_green_edge_v2"]
    a0_mean, a2_mean = a0["mean"], a2["mean"]
    a0_geo, a2_geo = a0["geometry"], a2["geometry"]
    a0_bg_floor = a0["runtime"]["profile"]["bg_floor"]
    a2_bg_floor = a2["runtime"]["profile"]["bg_floor"]
    fragment_delta = change_pct(a2_mean["fragment_pct"], a0_mean["fragment_pct"])
    temporal_delta = change_pct(
        a2["temporal_alpha_mae_per_second"],
        a0["temporal_alpha_mae_per_second"],
    )
    fringe_delta = change_pct(a2_mean["green_fringe"], a0_mean["green_fringe"])
    lines = [
        marker,
        "",
        f"- Seedance 任务：`{result['api']['task_id']}`，720p、{result['api']['duration']} 秒、无声，usage `{usage.get('total_tokens', usage.get('completion_tokens', 0)):,}` tokens。",
        f"- 云端提交至成功：`{timing['generation_wall_seconds']:.1f}s`；下载：`{timing['download_seconds']:.1f}s`。",
        f"- Edge v2 绿边：A0 `{a0_mean['green_fringe']:.6f}` → A1/A2 `{a2_mean['green_fringe']:.6f}`（改善 `{fringe_delta:.1f}%`）。" if fringe_delta is not None else f"- Edge v2 绿边：A0 `{a0_mean['green_fringe']:.6f}` → A1/A2 `{a2_mean['green_fringe']:.6f}`。",
        f"- Edge v2 碎片：A0 `{a0_mean['fragment_pct']:.4f}%` → A1/A2 `{a2_mean['fragment_pct']:.4f}%`" + (f"（改善 `{fragment_delta:.1f}%`）。" if fragment_delta is not None else "。"),
        f"- Edge v2 时序 alpha MAE/s：A0 `{a0['temporal_alpha_mae_per_second']:.6f}` → A1/A2 `{a2['temporal_alpha_mae_per_second']:.6f}`" + (f"（改善 `{temporal_delta:.1f}%`）。" if temporal_delta is not None else "。"),
        f"- 主体触边帧：A0 `{int(a0_geo['border_touch_frames'])}/96` → A1/A2 `{int(a2_geo['border_touch_frames'])}/96`；最小留白 `{a0_geo['min_subject_clearance_px']:.0f}px` → `{a2_geo['min_subject_clearance_px']:.0f}px`。",
        f"- 尾巴/侧向附属区域 alpha 保留代理：A0 `{a0_geo['lateral_appendage_alpha_retention']:.4f}` → A1/A2 `{a2_geo['lateral_appendage_alpha_retention']:.4f}`；主连通域占比 `{a0_geo['lateral_appendage_main_component_ratio']:.4f}` → `{a2_geo['lateral_appendage_main_component_ratio']:.4f}`。",
        f"- 绿幕 profile `bg_floor`：A0 `{a0_bg_floor:.4f}` → A1/A2 `{a2_bg_floor:.4f}`，新视频背景均匀性/可分性反而下降。",
        "- 人工复核：A1/A2 仍生成明显地面与接触阴影，背景存在亮度渐变，宠物由侧向快走漂移到正面；模型未稳定遵循无地面、均匀绿幕和固定朝向。",
        "- 决策：**不直接替换生产提示词**。保留其中的低幅尾摆、短曝光和去轮廓光约束，但下一轮必须单独强化无地面/无接触阴影并缩小主体运动范围。",
        "- 说明：尾巴指标是无语义 GT 条件下的侧向附属区域代理，必须配合动图人工复核，不能视作尾巴语义分割准确率。",
        f"- 完整机器结果：`{(OUTPUT / 'result.json').relative_to(ROOT).as_posix()}`。",
        "",
    ]
    RESEARCH_MD.write_text(text + "\n".join(lines), encoding="utf-8")


def main() -> None:
    if not FIRST_FRAME.is_file() or not A0_VIDEO.is_file():
        raise FileNotFoundError("required first frame or A0 video is missing")
    overall_started = time.time()
    new_video, task, api_timing = create_or_resume_video()
    videos = {"A0": A0_VIDEO, "A1A2": new_video}
    video_infos: dict[str, dict[str, Any]] = {}
    variants: dict[str, dict[str, Any]] = {}
    local_started = time.time()
    for variant, video in videos.items():
        source_dir = OUTPUT / "frames" / variant.lower()
        info = extract_uniform_frames(video, source_dir)
        video_infos[variant] = info
        variants[variant] = {}
        for provider, edge_refine in (
            ("adaptive_green_baseline", False),
            ("adaptive_green_edge_v2", True),
        ):
            matte_dir = OUTPUT / "matte" / variant.lower() / provider
            run_matte(source_dir, matte_dir, edge_refine=edge_refine)
            variants[variant][provider] = evaluate(
                source_dir, matte_dir, float(info["duration_seconds"])
            )
    local_seconds = time.time() - local_started
    usage = task.get("usage") or {}
    result = {
        "schema_version": 1,
        "experiment": "A0 current prompt vs A1+A2 upstream prompt",
        "first_frame": FIRST_FRAME.relative_to(ROOT).as_posix(),
        "a0_video": A0_VIDEO.relative_to(ROOT).as_posix(),
        "a1a2_video": new_video.relative_to(ROOT).as_posix(),
        "sampling": {
            "frames_per_video": FRAME_COUNT,
            "size": [FRAME_SIZE, FRAME_SIZE],
            "method": "uniform over each complete clip",
            "temporal_normalization": "optical-flow alpha MAE divided by sample interval",
        },
        "api": {
            "task_id": task.get("id"),
            "model": task.get("model", MODEL),
            "resolution": task.get("resolution"),
            "duration": task.get("duration"),
            "framespersecond": task.get("framespersecond"),
            "generate_audio": task.get("generate_audio"),
            "usage": usage,
        },
        "timing": {
            **api_timing,
            "local_extract_matte_evaluate_seconds": round(local_seconds, 3),
            "overall_wall_seconds": round(time.time() - overall_started, 3),
        },
        "video_info": video_infos,
        "variants": variants,
        "limitations": [
            "No hand-painted alpha ground truth; green/fragments are controlled-screen proxy metrics.",
            "Lateral appendage retention is a deterministic tail/paw proxy, not semantic tail segmentation.",
            "A0 is a 4-second historical clip while A1A2 is requested as 5 seconds; temporal MAE is normalized by sample interval.",
        ],
    }
    result_path = OUTPUT / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    append_research_summary(result)
    print(result_path)


if __name__ == "__main__":
    main()
