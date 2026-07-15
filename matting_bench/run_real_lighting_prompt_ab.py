"""Run a controlled Seedream A/B for the real-profile lighting prompt.

Only the lighting tail differs between variants. Both requests use the same source,
model, seed and guidance scale, then pass through the production green matte.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "matting_bench"
OUTPUT = ROOT / "poc_output" / "real_lighting_prompt_ab_20260714"
DEFAULT_SOURCE = ROOT / "inputs" / "desktop_tabby_cat_20260713_real.jpg"
MODEL = "doubao-seedream-4-5-251128"
MOTION_PET = "real_lighting_prompt_v2_20260714"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def prompt_variants() -> dict[str, dict[str, str]]:
    import prompts

    current = prompts.STYLE_PROMPTS["real"]
    if prompts._COMMON_TAIL_MATTE not in current:
        raise RuntimeError("real prompt no longer contains _COMMON_TAIL_MATTE")
    legacy = current.replace(prompts._COMMON_TAIL_MATTE, prompts._COMMON_TAIL_RIMLIGHT)
    matte_v1_tail = (
        prompts._COMMON_TAIL_BASE
        + "影棚布光：柔和中性正面主光打在宠物身上，毛发根根分明、真实细腻，"
        "宠物照明与背景照明彼此独立；禁止轮廓光、逆光、发光亮边、过曝毛尖和绿色反光；"
        "背景是均匀哑光的中高亮度标准色度绿幕，整幅画面绿色色相稳定一致，"
        "无阴影、无渐变、无光晕、无地面、无任何环境元素"
    )
    matte_v1 = current.replace(prompts._COMMON_TAIL_MATTE, matte_v1_tail)
    return {
        "legacy_rim": {
            "label": "旧提示词：冲突轮廓光",
            "prompt": legacy,
        },
        "matte_light": {
            "label": "新提示词：哑光分离照明",
            "prompt": matte_v1,
        },
        "matte_plate_v2": {
            "label": "生产优化：无地面单色底板",
            "prompt": current,
        },
    }


def generate(
    source: Path, variant: str, prompt: str, seed: int, guidance_scale: float
) -> tuple[Path, dict[str, Any]]:
    import poc

    destination = OUTPUT / variant / "generated.jpeg"
    metadata_path = OUTPUT / variant / "generation.json"
    if destination.is_file() and metadata_path.is_file():
        return destination, json.loads(metadata_path.read_text(encoding="utf-8"))

    destination.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "model": MODEL,
        "prompt": prompt,
        "image": poc.data_uri(source),
        "size": "2K",
        "response_format": "url",
        "watermark": False,
        "seed": seed,
        "guidance_scale": guidance_scale,
    }
    started = time.perf_counter()
    response = poc.ark_request("/images/generations", body)
    poc.download(response["data"][0]["url"], destination)
    metadata = {
        "model": MODEL,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "seconds": round(time.perf_counter() - started, 3),
        "usage": response.get("usage"),
        "prompt": prompt,
        "source": source.resolve().relative_to(ROOT).as_posix(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination, metadata


def background_metrics(path: Path) -> dict[str, float]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    score = (green - np.maximum(red, blue)) / np.maximum(green, 1.0 / 255.0)
    candidate = (score > 0.32) & (green > 0.18)
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), 8)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    background = np.isin(labels, border_labels[border_labels != 0]) & candidate
    pixels = rgb[background]
    if not pixels.size:
        return {"coverage_pct": 0.0, "rgb_std": 1.0, "green_score_std": 1.0}
    return {
        "coverage_pct": float(background.mean() * 100.0),
        "rgb_std": float(np.mean(np.std(pixels, axis=0))),
        "green_score_std": float(np.std(score[background])),
        "green_score_p05": float(np.quantile(score[background], 0.05)),
    }


def run_matte(source: Path, variant: str) -> tuple[Path, dict[str, Any]]:
    import poc

    source_dir = OUTPUT / variant / "source"
    output_dir = OUTPUT / variant / "matte"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = source_dir / "frame.png"
    # Match production: normalize model-dependent green before extracting alpha.
    poc.normalize_bg(source, frame)
    output = output_dir / "frame.png"
    if not output.is_file() or not (output_dir / "metrics.json").is_file():
        subprocess.run(
            [
                sys.executable,
                str(BENCH / "providers" / "baseline" / "infer.py"),
                "--input-dir",
                str(source_dir),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
                "--edge-refine",
                "--halo-profile",
                "real",
            ],
            cwd=ROOT,
            check=True,
        )
    sys.path.insert(0, str(BENCH))
    import evaluate as benchmark  # type: ignore

    return output, benchmark.evaluate_provider(source_dir, output_dir)


def edge_metrics(source: Path, matte: Path) -> dict[str, float]:
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32) / 255.0
    rgba = np.asarray(Image.open(matte).convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[:, :, 3]
    visible = alpha > 0.04
    distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 5)
    edge = visible & (distance <= 4.0)
    inner = visible & (distance >= 8.0) & (distance <= 18.0) & (alpha > 0.96)
    luminance = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    neutral = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    bright_edge = edge & (luminance > 0.82) & (neutral < 0.22)
    edge_mean = float(luminance[edge].mean()) if edge.any() else 0.0
    inner_mean = float(luminance[inner].mean()) if inner.any() else edge_mean
    return {
        "edge_luma_mean": edge_mean,
        "inner_luma_mean": inner_mean,
        "edge_luma_excess": edge_mean - inner_mean,
        "bright_neutral_edge_pct": float(bright_edge.sum() * 100.0 / max(1, edge.sum())),
    }


def load_motion_demo() -> dict[str, Any] | None:
    pet_dir = ROOT / "poc_output" / MOTION_PET
    metrics_path = pet_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_clip = {
        section: {row["clip"]: row for row in metrics.get(section, [])}
        for section in ("state_frame", "animate", "matte")
    }
    actions = []
    for clip, label in (("fast_walk", "快走循环"), ("sleep", "睡眠循环")):
        asset = pet_dir / f"anim_{clip}.webp"
        state = by_clip["state_frame"].get(clip)
        animate = by_clip["animate"].get(clip)
        matte = by_clip["matte"].get(clip)
        if not asset.is_file() or not state or not animate or not matte:
            continue
        state_usage = state.get("usage") or {}
        animate_usage = animate.get("usage") or {}
        temporal = (matte.get("reframe") or {}).get("temporal_refine") or {}
        structural = (matte.get("reframe") or {}).get("structural_support") or {}
        structural_fusion = structural.get("fusion") or {}
        actions.append(
            {
                "clip": clip,
                "label": label,
                "url": asset.relative_to(OUTPUT.parent).as_posix(),
                "state_seconds": state.get("seconds"),
                "animate_seconds": animate.get("seconds"),
                "matte_seconds": matte.get("seconds"),
                "tokens": int(state_usage.get("total_tokens", 0))
                + int(animate_usage.get("total_tokens", 0)),
                "frames": matte.get("webp_frames"),
                "fps": matte.get("webp_fps"),
                "mb": matte.get("webp_mb"),
                "seam": (matte.get("loop_meta") or {}).get("seam_diff"),
                "green_residual": matte.get("green_residual"),
                "fragments_removed": temporal.get("single_frame_fragments_removed"),
                "flow_size": temporal.get("flow_size"),
                "structural_provider": structural.get("provider"),
                "structural_seconds": structural.get("seconds"),
                "structural_pixels": structural_fusion.get("repaired_pixels"),
                "silent": animate.get("generate_audio") is False,
            }
        )
    return {"pet": MOTION_PET, "actions": actions} if actions else None


def render_html(payload: dict[str, Any]) -> Path:
    cards = []
    for variant in payload["variants"]:
        mean = variant["matting"]["mean"]
        edge = variant["edge"]
        background = variant["background"]
        usage = variant["generation"].get("usage") or {}
        recommended = variant["id"] == "matte_plate_v2"
        cards.append(
            f"""
            <article class="card{' recommended' if recommended else ''}">
              <header><h2>{variant['label']}</h2><span>{'生产推荐 · ' if recommended else ''}{variant['generation']['seconds']:.1f}s</span></header>
              <div class="views">
                <figure><img src="/{variant['generated_url']}" alt="{variant['label']} 原始绿幕"><figcaption>Seedream 原始输出</figcaption></figure>
                <figure class="checker"><img src="/{variant['matte_url']}" alt="{variant['label']} 透明抠图"><figcaption>生产抠图结果</figcaption></figure>
              </div>
              <dl>
                <div><dt>背景色波动</dt><dd>{background['rgb_std']:.5f}</dd></div>
                <div><dt>绿幕置信 P05</dt><dd>{background['green_score_p05']:.4f}</dd></div>
                <div><dt>边缘亮度过量</dt><dd>{edge['edge_luma_excess']:+.4f}</dd></div>
                <div><dt>高亮中性边缘</dt><dd>{edge['bright_neutral_edge_pct']:.2f}%</dd></div>
                <div><dt>抠图绿边</dt><dd>{mean['green_fringe']:.6f}</dd></div>
                <div><dt>背景 alpha 泄漏</dt><dd>{mean['background_alpha_mean']:.6f}</dd></div>
                <div><dt>前景损失</dt><dd>{mean['foreground_loss_mean']:.6f}</dd></div>
                <div><dt>生成 token</dt><dd>{usage.get('total_tokens', usage.get('output_tokens', '-'))}</dd></div>
              </dl>
            </article>
            """
        )
    motion = payload.get("motion_demo")
    motion_html = ""
    if motion:
        motion_cards = []
        for action in motion["actions"]:
            motion_cards.append(
                f"""
                <article class="motion-card">
                  <header><h2>{action['label']}</h2><span>{action['frames']} 帧 · {action['fps']} FPS · {action['mb']:.2f} MB</span></header>
                  <div class="motion-stage checker"><img src="/{action['url']}?v=structural2" alt="生产优化版 {action['label']}"></div>
                  <dl>
                    <div><dt>Seedream 首帧</dt><dd>{action['state_seconds']:.1f}s</dd></div>
                    <div><dt>Seedance 动作</dt><dd>{action['animate_seconds']:.1f}s</dd></div>
                    <div><dt>本地抠图</dt><dd>{action['matte_seconds']:.1f}s</dd></div>
                    <div><dt>API token</dt><dd>{action['tokens']:,}</dd></div>
                    <div><dt>循环缝</dt><dd>{action['seam']:.6f}</dd></div>
                    <div><dt>绿边残留</dt><dd>{action['green_residual']:.4f}</dd></div>
                    <div><dt>时序碎片移除</dt><dd>{action['fragments_removed']}</dd></div>
                    <div><dt>输出规格</dt><dd>{'无声' if action['silent'] else '有声'} · flow {action['flow_size']}</dd></div>
                    <div><dt>结构支撑</dt><dd>{action['structural_provider'] or '未启用'}</dd></div>
                    <div><dt>补强像素</dt><dd>{action['structural_pixels'] or 0:,}</dd></div>
                  </dl>
                </article>
                """
            )
        motion_html = f"""
        <section class="motion">
          <div class="motion-title"><div><h1>生产优化版动效验证</h1><p>使用无地面单色底板提示词，分别生成动作首帧、5 秒 Seedance 循环，再经 temporal_v3 抠图。</p></div><strong>已生成真实动效，不是静态预览</strong></div>
          <div class="motion-grid">{''.join(motion_cards)}</div>
        </section>
        """
    path = OUTPUT.with_suffix(".html")
    path.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>实体版生成照明提示词 A/B</title><style>
        :root{{--bg:#f3f5f2;--panel:#fff;--line:#d7ddd8;--ink:#17201b;--muted:#66736b;--green:#1f725b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1560px;margin:auto;padding:28px}}.intro{{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:18px}}h1,h2,p{{margin:0}}h1{{font-size:24px}}h2{{font-size:16px}}.intro p,.motion-title p{{color:var(--muted);max-width:760px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.card,.motion-card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}}.card.recommended{{border-color:#79ad9c;box-shadow:0 0 0 2px #d9ece5}}header{{display:flex;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid var(--line)}}header span{{color:var(--green);font-weight:700;white-space:nowrap}}.views{{display:grid;grid-template-columns:1fr;gap:1px;background:var(--line)}}figure{{margin:0;background:#eef1ed}}figure img{{display:block;width:100%;aspect-ratio:4/3;object-fit:contain}}figcaption{{padding:8px 12px;background:#fff;color:var(--muted);border-top:1px solid var(--line)}}.checker{{background-color:#fff;background-image:linear-gradient(45deg,#e7e9e7 25%,transparent 25%),linear-gradient(-45deg,#e7e9e7 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#e7e9e7 75%),linear-gradient(-45deg,transparent 75%,#e7e9e7 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0}}dl{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:12px;margin:0}}dl div{{padding:9px;border:1px solid var(--line);border-radius:6px}}dt{{color:var(--muted);font-size:12px}}dd{{margin:3px 0 0;font-weight:750}}.motion{{margin-top:28px;padding-top:24px;border-top:1px solid var(--line)}}.motion-title{{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:14px}}.motion-title strong{{color:var(--green)}}.motion-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.motion-stage{{width:100%;aspect-ratio:1 / 1;overflow:hidden}}.motion-stage img{{display:block;width:100%;height:100%;object-fit:contain}}@media(max-width:1100px){{.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:720px){{main{{padding:14px}}.grid,.motion-grid{{grid-template-columns:1fr}}.intro,.motion-title{{display:block}}.intro p,.motion-title p{{margin-top:8px}}.motion-title strong{{display:block;margin-top:8px}}}}
        </style></head><body><main><div class="intro"><div><h1>实体版生成照明提示词 A/B</h1><p>同一原图、同一 Seedream 模型、同一 seed 与 guidance；仅替换轮廓光描述。数值越低通常越好，最终以毛尖自然度与身份一致性目视判断。</p></div><p>seed {payload['seed']} · guidance {payload['guidance_scale']} · {payload['model']}</p></div>{motion_html}<section class="grid">{''.join(cards)}</section></main></body></html>""",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"missing source: {source}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    variants = []
    for variant, config in prompt_variants().items():
        generated, generation = generate(
            source, variant, config["prompt"], args.seed, args.guidance_scale
        )
        matte, matting = run_matte(generated, variant)
        variants.append(
            {
                "id": variant,
                "label": config["label"],
                "generated_url": generated.resolve().relative_to(OUTPUT.parent).as_posix(),
                "matte_url": matte.resolve().relative_to(OUTPUT.parent).as_posix(),
                "generation": generation,
                "background": background_metrics(generated),
                "edge": edge_metrics(generated, matte),
                "matting": matting,
            }
        )
    payload = {
        "schema_version": 1,
        "model": MODEL,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "source": source.relative_to(ROOT).as_posix(),
        "variants": variants,
        "motion_demo": load_motion_demo(),
    }
    result = OUTPUT / "result.json"
    result.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(render_html(payload))


if __name__ == "__main__":
    main()
