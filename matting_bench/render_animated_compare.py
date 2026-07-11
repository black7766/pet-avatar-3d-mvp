"""Build an entity-only animated matting comparison page.

The page uses the same 24-frame fast-walk sequence for every provider. Generated
WebPs are page-local assets so the existing poc_output HTTP server can serve them.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


RECOMMENDATION_KEYS = (
    "config_id",
    "recommended_config_id",
    "selected_config_id",
    "primary",
    "id",
)

PROVIDER_ORDER = {
    "adaptive_green_baseline": 0,
    "ZhengPeng7/BiRefNet (General)": 1,
    "vitmatte_adaptive_green_hybrid": 2,
    "rembg": 3,
    "ben2": 4,
    "ZhengPeng7/BiRefNet-matting": 5,
    "paddle_matting": 6,
    "official_matanyone_v1": 7,
    "official_meta_sam2_1_small_video": 8,
}

PROVIDER_VIEW = {
    "adaptive_green_baseline": {
        "name": "自研自适应绿幕",
        "role": "当前生产主链",
        "summary": "不占 GPU，毛发保留、背景纯净和工程成本最均衡。",
        "alpha": "软 alpha",
    },
    "ZhengPeng7/BiRefNet (General)": {
        "name": "BiRefNet General",
        "role": "异常样本后备",
        "summary": "主体连通和时序较好，适合非绿幕或绿幕失败样本。",
        "alpha": "软 alpha",
    },
    "vitmatte_adaptive_green_hybrid": {
        "name": "ViTMatte 混合细化",
        "role": "候选实验",
        "summary": "只修正自研 trimap 的 2px 窄边缘，收益有限但可控。",
        "alpha": "软 alpha",
    },
    "rembg": {
        "name": "rembg U2Net",
        "role": "成熟部署对照",
        "summary": "官方 alpha matting 能保留软毛发，但当前本机路径较慢。",
        "alpha": "软 alpha",
    },
    "ben2": {
        "name": "BEN2 Base",
        "role": "轮廓模型对照",
        "summary": "绿边较少，但低对比毛发和前景颜色损失明显。",
        "alpha": "软 alpha",
    },
    "ZhengPeng7/BiRefNet-matting": {
        "name": "BiRefNet-matting",
        "role": "人像域研究",
        "summary": "时序误差最低，但宠物背景泄漏高于 General。",
        "alpha": "软 alpha",
    },
    "paddle_matting": {
        "name": "Paddle PP-MattingV2",
        "role": "百度系速度对照",
        "summary": "CUDA 推理最快，公开人像检查点的宠物绿边较明显。",
        "alpha": "软 alpha",
    },
    "official_matanyone_v1": {
        "name": "MatAnyone v1",
        "role": "视频时序研究",
        "summary": "连续传播稳定，但当前上游许可不适合直接商用。",
        "alpha": "软 alpha",
    },
    "official_meta_sam2_1_small_video": {
        "name": "SAM 2.1 Small",
        "role": "主体 mask 辅助",
        "summary": "轮廓传播稳定，但二值 mask 无法表达半透明毛发。",
        "alpha": "二值 mask",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recommendation_id(provider: dict[str, Any]) -> str | None:
    normalized = provider.get("recommended_config_id")
    if normalized:
        return str(normalized)
    recommendation = provider.get("recommendation") or {}
    for key in RECOMMENDATION_KEYS:
        value = recommendation.get(key)
        if value:
            return str(value)
    return None


def resolve_temporal_dir(repo_root: Path, provider: dict[str, Any], config: dict[str, Any]) -> Path:
    name = provider["provider"]
    recommendation = provider.get("recommendation") or {}
    candidates = [
        config.get("temporal_output_dir"),
        recommendation.get("temporal_output_dir"),
    ]
    if name == "adaptive_green_baseline":
        candidates.append(
            f"matting_bench/outputs/tuning/baseline/temporal/{config['id']}"
        )
    candidates.append(config.get("output_dir"))
    for value in candidates:
        if not value:
            continue
        directory = repo_root / str(value)
        if directory.is_dir():
            direct = sorted(directory.glob("*.png"))
            rgba = sorted((directory / "rgba").glob("*.png"))
            if len(direct) == 24:
                return directory
            if len(rgba) == 24:
                return directory / "rgba"
    raise FileNotFoundError(f"24-frame temporal output not found for {name}:{config['id']}")


def encode_webp(frame_dir: Path, output: Path, duration_ms: int) -> dict[str, Any]:
    frame_paths = sorted(frame_dir.glob("*.png"))
    if len(frame_paths) != 24:
        raise ValueError(f"expected 24 PNGs in {frame_dir}, got {len(frame_paths)}")
    frames: list[Image.Image] = []
    sizes: set[tuple[int, int]] = set()
    for path in frame_paths:
        with Image.open(path) as image:
            frame = image.convert("RGBA")
            sizes.add(frame.size)
            frames.append(frame.copy())
    if len(sizes) != 1:
        raise ValueError(f"mixed frame sizes in {frame_dir}: {sorted(sizes)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=90,
        method=4,
        lossless=False,
    )
    return {
        "frames": len(frames),
        "width": frames[0].width,
        "height": frames[0].height,
        "duration_ms": duration_ms * len(frames),
        "file_mb": output.stat().st_size / (1024 * 1024),
    }


def copy_image(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return f"/{destination.parent.name}/{destination.name}"


def temporal_metrics_path(frame_dir: Path) -> Path:
    if frame_dir.name == "rgba":
        return frame_dir.parent / "metrics.json"
    return frame_dir / "metrics.json"


def runtime_metrics(provider: str, data: dict[str, Any]) -> dict[str, float | None | str]:
    if provider == "adaptive_green_baseline":
        return {
            "task_seconds": float(data["total_seconds"]),
            "core_seconds": float(data["total_seconds"]),
            "inference_ms": None,
            "frame_ms": float(data["mean_ms_per_frame"]),
            "vram_mb": 0.0,
            "note": "CPU 全流程，含 alpha、去绿和 PNG 写盘",
        }
    if provider == "ben2":
        summary = data["summary"]
        task_ms = (
            summary["model_load_ms"]
            + summary["warmup_total_ms"]
            + summary["measured_wall_ms"]
        )
        return {
            "task_seconds": task_ms / 1000,
            "core_seconds": summary["measured_wall_ms"] / 1000,
            "inference_ms": summary["inference_mean_excluding_first_ms"],
            "frame_ms": summary["mean_total_ms"],
            "vram_mb": summary["max_cuda_pipeline_peak_allocated_mib"],
            "note": "总任务含模型加载、一次预热和 24 帧写盘",
        }
    if provider in {
        "ZhengPeng7/BiRefNet (General)",
        "ZhengPeng7/BiRefNet-matting",
    }:
        return {
            "task_seconds": data["model_load_seconds"] + data["batch_wall_seconds"],
            "core_seconds": data["batch_wall_seconds"],
            "inference_ms": data["inference_mean_excluding_first_seconds"] * 1000,
            "frame_ms": data["end_to_end_mean_excluding_first_seconds"] * 1000,
            "vram_mb": data["cuda_peak_memory"]["allocated_bytes"] / (1024 * 1024),
            "note": "总任务含模型加载；核心处理含官方前景 refinement 与写盘",
        }
    if provider == "paddle_matting":
        return {
            "task_seconds": data["process_seconds_excluding_cli_import"],
            "core_seconds": data["measured_seconds"],
            "inference_ms": data["mean_inference_ms"],
            "frame_ms": data["mean_total_ms"],
            "vram_mb": data["peak_vram_mb"],
            "note": "总任务含 predictor 加载和预热；核心处理为 24 帧批次",
        }
    if provider == "rembg":
        return {
            "task_seconds": data["run_wall_seconds"],
            "core_seconds": data["batch_wall_seconds"],
            "inference_ms": data["remove_mean_excluding_first_seconds"] * 1000,
            "frame_ms": data["end_to_end_mean_excluding_first_seconds"] * 1000,
            "vram_mb": None,
            "note": "总任务含 ONNX session 加载；核心处理含 PyMatting 与写盘",
        }
    if provider == "official_meta_sam2_1_small_video":
        timing = data["timing"]
        inference = timing["inference"]
        return {
            "task_seconds": timing["end_to_end_seconds"],
            "core_seconds": inference["propagation_wall_seconds"],
            "inference_ms": inference["propagation_wall_seconds"] * 1000 / 24,
            "frame_ms": inference["propagation_wall_seconds"] * 1000 / 24,
            "vram_mb": inference["peak_memory"]["peak_allocated_mib"],
            "note": "总任务含初始化、首帧提示、传播、诊断和写盘；输出为二值 mask",
        }
    if provider == "official_matanyone_v1":
        timing = data["timing"]
        return {
            "task_seconds": timing["end_to_end_seconds"],
            "core_seconds": timing["output_inference_seconds"],
            "inference_ms": timing["output_inference_seconds"] * 1000 / 24,
            "frame_ms": timing["output_inference_seconds"] * 1000 / 24,
            "vram_mb": timing["cuda_peak_during_inference"]["peak_allocated_mib"],
            "note": "总任务含模型加载、绿幕 RGB、一次 warmup、诊断和写盘",
        }
    if provider == "vitmatte_adaptive_green_hybrid":
        return {
            "task_seconds": (
                data["model_load_seconds"]
                + data["green_profile_seconds"]
                + data["batch_wall_seconds"]
            ),
            "core_seconds": data["batch_wall_seconds"],
            "inference_ms": data["inference_mean_excluding_first_seconds"] * 1000,
            "frame_ms": data["end_to_end_mean_seconds"] * 1000,
            "vram_mb": data["cuda_memory"]["peak_allocated_bytes"] / (1024 * 1024),
            "note": "总任务含模型加载、绿幕 profile 和自研+ViTMatte 混合处理",
        }
    raise KeyError(provider)


def fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def source_pipeline_metrics(path: Path) -> dict[str, Any]:
    data = read_json(path)
    stylize = sum(float(item.get("seconds", 0)) for item in data.get("stylize", []))
    state = sum(float(item.get("seconds", 0)) for item in data.get("state_frame", []))
    animate = sum(float(item.get("seconds", 0)) for item in data.get("animate", []))
    matte = sum(float(item.get("seconds", 0)) for item in data.get("matte", []))
    tokens = 0
    for section in ("stylize", "state_frame", "animate"):
        for item in data.get(section, []):
            tokens += int((item.get("usage") or {}).get("total_tokens", 0))
    return {
        "stylize_seconds": stylize,
        "state_seconds": state,
        "animate_seconds": animate,
        "api_seconds": stylize + state + animate,
        "matte_seconds": matte,
        "total_seconds": stylize + state + animate + matte,
        "tokens": tokens,
        "image_tasks": len(data.get("stylize", [])) + len(data.get("state_frame", [])),
        "video_tasks": len(data.get("animate", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("matting_bench/outputs/tuning/aggregate_final.json"),
    )
    parser.add_argument(
        "--source-metrics",
        type=Path,
        default=Path("poc_output/pet_20260710_121221_5ce7716e_real_after/metrics.json"),
    )
    parser.add_argument(
        "--source-frames",
        type=Path,
        default=Path(
            "matting_bench/data/pet_20260710_121221_5ce7716e/temporal_fast_walk_24_640"
        ),
    )
    parser.add_argument(
        "--original-image",
        type=Path,
        default=Path("inputs/pet_20260710_121221_5ce7716e_real_after.jpg"),
    )
    parser.add_argument(
        "--entity-image",
        type=Path,
        default=Path("poc_output/pet_20260710_121221_5ce7716e_real_after/chosen.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poc_output/matting_animated_compare_real_20260711.html"),
    )
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    if not 1.0 <= args.fps <= 24.0:
        raise SystemExit("--fps must be between 1 and 24")
    repo_root = Path(__file__).resolve().parent.parent
    aggregate = read_json(repo_root / args.aggregate)
    pipeline = source_pipeline_metrics(repo_root / args.source_metrics)
    output = repo_root / args.output
    assets_dir = output.parent / f"{output.stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 / args.fps)

    original_url = copy_image(
        repo_root / args.original_image,
        assets_dir / "original_pet.jpg",
    )
    entity_url = copy_image(
        repo_root / args.entity_image,
        assets_dir / "entity_keyframe.png",
    )
    source_webp = assets_dir / "green_source_fast_walk.webp"
    source_asset = encode_webp(repo_root / args.source_frames, source_webp, duration_ms)
    source_url = f"/{assets_dir.name}/{source_webp.name}"

    cards: list[dict[str, Any]] = []
    for provider in aggregate.get("providers", []):
        config_id = recommendation_id(provider)
        config = next(
            (item for item in provider.get("configs", []) if item.get("id") == config_id),
            None,
        )
        if config is None:
            continue
        name = provider["provider"]
        if name not in PROVIDER_VIEW:
            continue
        frame_dir = resolve_temporal_dir(repo_root, provider, config)
        metrics_path = temporal_metrics_path(frame_dir)
        runtime = runtime_metrics(name, read_json(metrics_path))
        slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")
        webp_path = assets_dir / f"{slug}.webp"
        asset = encode_webp(frame_dir, webp_path, duration_ms)
        cards.append(
            {
                "provider": name,
                "config": config_id,
                "view": PROVIDER_VIEW[name],
                "runtime": runtime,
                "quality": config.get("quality") or {},
                "temporal_alpha_mae": config.get("temporal_alpha_mae"),
                "eligible": bool(
                    config.get("eligible_for_final")
                    and config.get("passes_all_guardrails")
                ),
                "asset": asset,
                "url": f"/{assets_dir.name}/{webp_path.name}",
            }
        )
    cards.sort(key=lambda item: PROVIDER_ORDER[item["provider"]])

    card_html: list[str] = []
    for item in cards:
        view = item["view"]
        runtime = item["runtime"]
        quality = item["quality"]
        category = "candidate" if item["eligible"] else "research"
        badge_class = "primary" if item["provider"] == "adaptive_green_baseline" else category
        card_html.append(
            f"""
            <article class="model-card" data-category="{category}">
              <header><div><strong>{html.escape(view['name'])}</strong><small>{html.escape(str(item['config']))}</small></div><span class="badge {badge_class}">{html.escape(view['role'])}</span></header>
              <div class="motion-stage"><img class="motion" src="{html.escape(item['url'])}" data-src="{html.escape(item['url'])}" alt="{html.escape(view['name'])} 实体版快走抠图动图"></div>
              <p class="summary">{html.escape(view['summary'])}</p>
              <section class="runtime-grid">
                <div><span>总任务耗时</span><strong>{fmt(runtime['task_seconds'], 2, 's')}</strong></div>
                <div><span>24帧核心处理</span><strong>{fmt(runtime['core_seconds'], 2, 's')}</strong></div>
                <div><span>稳态推理/帧</span><strong>{fmt(runtime['inference_ms'], 1, 'ms')}</strong></div>
                <div><span>峰值显存</span><strong>{fmt(runtime['vram_mb'], 0, 'MB')}</strong></div>
              </section>
              <table class="quality"><tbody>
                <tr><th>pseudo MAE</th><td>{fmt(quality.get('pseudo_mae'), 6)}</td><th>背景 alpha</th><td>{fmt(quality.get('background_alpha_mean'), 6)}</td></tr>
                <tr><th>前景损失</th><td>{fmt(quality.get('foreground_loss_mean'), 6)}</td><th>绿边</th><td>{fmt(quality.get('green_fringe'), 6)}</td></tr>
                <tr><th>碎片率</th><td>{fmt(quality.get('fragment_pct'), 4, '%')}</td><th>软 alpha</th><td>{fmt(quality.get('soft_alpha_pct'), 3, '%')}</td></tr>
                <tr><th>时序误差</th><td>{fmt(item.get('temporal_alpha_mae'), 6)}</td><th>动图体积</th><td>{fmt(item['asset']['file_mb'], 2, 'MB')}</td></tr>
              </tbody></table>
              <footer><span>{html.escape(view['alpha'])}</span><span>{html.escape(str(runtime['note']))}</span></footer>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>实体版宠物 · 抠图模型动图对比</title>
<style>
:root{{--bg:#f2f4f1;--panel:#fff;--text:#18201c;--muted:#68726c;--line:#d8ddd9;--green:#176b55;--green-soft:#e7f3ee;--warm:#8b583f;--warn:#8a5a24}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 Inter,"Microsoft YaHei",sans-serif;letter-spacing:0}}button{{font:inherit}}.shell{{width:min(1540px,calc(100% - 32px));margin:auto;padding:22px 0 60px}}
.top{{display:flex;justify-content:space-between;align-items:end;gap:28px;padding-bottom:16px;border-bottom:1px solid var(--line)}}h1{{margin:3px 0;font-size:28px}}h2{{margin:24px 0 10px;font-size:18px}}p{{margin:0;color:var(--muted)}}.eyebrow{{color:var(--green);font-size:11px;font-weight:700}}.top>p{{max-width:530px}}
.pipeline{{display:grid;grid-template-columns:1.3fr repeat(4,minmax(125px,.7fr));gap:8px;margin:14px 0}}.pipeline>div{{padding:11px 12px;border:1px solid var(--line);border-radius:7px;background:#fff}}.pipeline .decision{{background:var(--green-soft);border-color:#afd1c5}}.pipeline span{{display:block;color:var(--muted);font-size:11px}}.pipeline strong{{display:block;margin-top:2px;font-size:20px;font-variant-numeric:tabular-nums}}.pipeline .decision strong{{font-size:17px}}
.source-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.source-item{{overflow:hidden;border:1px solid var(--line);border-radius:8px;background:#fff}}.source-item header{{padding:9px 11px;border-bottom:1px solid var(--line)}}.source-media,.motion-stage{{aspect-ratio:1;background-color:#f7f8f7;background-image:linear-gradient(45deg,#e8ebe8 25%,transparent 25%),linear-gradient(-45deg,#e8ebe8 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#e8ebe8 75%),linear-gradient(-45deg,transparent 75%,#e8ebe8 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0}}.source-media img,.motion-stage img{{width:100%;height:100%;object-fit:contain;display:block}}.source-item footer{{display:flex;justify-content:space-between;padding:8px 11px;color:var(--muted);font-size:11px}}
.section-row{{display:flex;align-items:center;justify-content:space-between;gap:12px}}.controls{{display:flex;gap:5px;overflow:auto;padding:5px 0}}button{{border:1px solid var(--line);border-radius:6px;background:#fff;padding:6px 9px;white-space:nowrap;cursor:pointer}}button.active{{color:#fff;background:var(--green);border-color:var(--green)}}
.model-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.model-card{{overflow:hidden;border:1px solid var(--line);border-radius:8px;background:var(--panel)}}.model-card>header{{display:flex;justify-content:space-between;align-items:start;gap:10px;padding:10px 12px;border-bottom:1px solid var(--line)}}.model-card small{{display:block;color:var(--muted)}}.badge{{padding:2px 6px;border-radius:5px;font-size:10px;white-space:nowrap;background:#f1eee9;color:var(--warm)}}.badge.primary{{background:var(--green-soft);color:var(--green)}}.badge.candidate{{background:#edf3f0;color:#326a59}}.badge.research{{background:#f5eee7;color:var(--warn)}}
.motion-stage{{aspect-ratio:1}}.model-grid[data-background="white"] .motion-stage{{background:#fff;background-image:none}}.model-grid[data-background="black"] .motion-stage{{background:#111;background-image:none}}.summary{{min-height:48px;padding:9px 12px;border-top:1px solid var(--line)}}
.runtime-grid{{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}.runtime-grid div{{padding:8px 9px;border-right:1px solid var(--line)}}.runtime-grid div:last-child{{border-right:0}}.runtime-grid span{{display:block;color:var(--muted);font-size:10px}}.runtime-grid strong{{display:block;margin-top:2px;font-size:15px;font-variant-numeric:tabular-nums}}
.quality{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}.quality th,.quality td{{padding:6px 8px;border-bottom:1px solid #edf0ed;text-align:right}}.quality th{{color:var(--muted);font-size:10px;font-weight:500;text-align:left}}.model-card>footer{{display:flex;justify-content:space-between;gap:12px;padding:8px 11px;color:var(--muted);font-size:10px}}.model-card>footer span:last-child{{text-align:right}}.method{{margin-top:12px;padding:11px 0;border-top:1px solid var(--line);color:var(--muted)}}
@media(max-width:1100px){{.pipeline{{grid-template-columns:repeat(2,1fr)}}.pipeline .decision{{grid-column:1/-1}}.model-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:700px){{.top,.section-row{{display:block}}.source-grid,.model-grid,.pipeline{{grid-template-columns:1fr}}.pipeline .decision{{grid-column:auto}}.runtime-grid{{grid-template-columns:repeat(2,1fr)}}.runtime-grid div:nth-child(2){{border-right:0}}.runtime-grid div:nth-child(-n+2){{border-bottom:1px solid var(--line)}}}}
</style></head><body><main class="shell">
<section class="top"><div><span class="eyebrow">ENTITY VERSION · FAST WALK</span><h1>实体版宠物抠图模型动图对比</h1><p>同一只宠物、同一段 AI 快走绿幕视频、同一组 24 帧输入。页面不包含萌宠版。</p></div><p>全部动图统一为 640×640、24 帧、{args.fps:g} FPS、{source_asset['duration_ms']/1000:.1f} 秒循环。质量数值越低越好；软 alpha 比例不是越低越好。</p></section>
<section class="pipeline"><div class="decision"><span>当前生产结论</span><strong>自研绿幕主链 + BiRefNet General 异常后备</strong><span>模型只在失败样本触发，避免默认增加 GPU 时间和边缘污染。</span></div><div><span>单只宠物整套生成</span><strong>{pipeline['total_seconds']:.1f}s</strong><span>约 {pipeline['total_seconds']/60:.1f} 分钟</span></div><div><span>API 图像/视频生成</span><strong>{pipeline['api_seconds']:.1f}s</strong><span>{pipeline['image_tasks']} 个图像任务 + {pipeline['video_tasks']} 个视频任务</span></div><div><span>本地抠图与 WebP</span><strong>{pipeline['matte_seconds']:.1f}s</strong><span>idle / sleep / fast_walk 三段</span></div><div><span>API token</span><strong>{pipeline['tokens']:,}</strong><span>图像与视频任务 usage 合计</span></div></section>
<h2>实体版素材链路</h2><section class="source-grid"><article class="source-item"><header><strong>1. 上传原图</strong><p>用户真实宠物照片</p></header><div class="source-media"><img src="{original_url}" alt="上传的真实宠物照片"></div><footer><span>实体版输入</span><span>不展示萌宠版</span></footer></article><article class="source-item"><header><strong>2. 实体形象首帧</strong><p>Seedream 图生图结果</p></header><div class="source-media"><img src="{entity_url}" alt="生成的实体宠物形象"></div><footer><span>图生图 {pipeline['stylize_seconds']:.1f}s</span><span>全身居中</span></footer></article><article class="source-item"><header><strong>3. 快走绿幕动作源</strong><p>所有模型使用同一段连续帧</p></header><div class="source-media"><img class="motion" src="{source_url}" data-src="{source_url}" alt="实体宠物快走绿幕动作"></div><footer><span>24 帧对比样本</span><span>原视频 720p / 无声</span></footer></article></section>
<div class="section-row"><h2>九种抠图路径动图对比</h2><div class="controls"><button type="button" data-filter="all" class="active">全部</button><button type="button" data-filter="candidate">生产候选</button><button type="button" data-filter="research">研究对照</button><button type="button" data-background="checker" class="active">棋盘格</button><button type="button" data-background="white">白底</button><button type="button" data-background="black">黑底</button><button type="button" id="replay">↻ 同步重播</button></div></div>
<section class="model-grid" data-background="checker">{''.join(card_html)}</section>
<p class="method">计时说明：总任务耗时包含各 provider 实际记录的模型加载、预处理、诊断和写盘；24 帧核心处理用于观察模型批次本身；稳态推理排除首帧冷启动。pseudo MAE、背景 alpha、前景损失、绿边、碎片率和软 alpha 来自 idle / fast_walk / sleep 共 9 张 960×960 固定帧；时序误差和本页耗时来自当前 24 帧快走序列。当前没有人工逐像素 alpha 真值，质量指标只用于同源相对比较，并经过黑底、白底和棋盘格人工复核。</p>
</main><script>
const grid=document.querySelector('.model-grid');const cards=[...document.querySelectorAll('.model-card')];const filterButtons=[...document.querySelectorAll('button[data-filter]')];const backgroundButtons=[...document.querySelectorAll('button[data-background]')];
filterButtons.forEach(button=>button.addEventListener('click',()=>{{filterButtons.forEach(item=>item.classList.toggle('active',item===button));cards.forEach(card=>card.hidden=button.dataset.filter!=='all'&&card.dataset.category!==button.dataset.filter)}}));
backgroundButtons.forEach(button=>button.addEventListener('click',()=>{{backgroundButtons.forEach(item=>item.classList.toggle('active',item===button));grid.dataset.background=button.dataset.background}}));
document.getElementById('replay').addEventListener('click',()=>{{const stamp=Date.now();document.querySelectorAll('img.motion').forEach(image=>{{const source=image.dataset.src;image.src='';requestAnimationFrame(()=>image.src=source+'?sync='+stamp)}})}});
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(output)
    print(f"providers={len(cards)} assets={assets_dir}")


if __name__ == "__main__":
    main()
