#!/usr/bin/env python3
"""Analyze before/after pet WebPs and build a local comparison report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "poc_output"
INPUTS = ROOT / "inputs"
CLIPS = ("idle", "fast_walk", "sleep")
STYLES = ("real", "paimomo")
STYLE_LABELS = {"real": "实体版", "paimomo": "萌宠版"}
FLOW_LABELS = {"before": "优化前", "after": "优化后"}

# Public list prices checked on 2026-07-10. Billing-console discounts may differ.
SEEDANCE_SILENT_RMB_PER_M_TOKENS = 8.0
SEEDREAM_45_RMB_PER_IMAGE = 0.25


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def analyze_webp(path: Path) -> dict:
    image = Image.open(path)
    frame_count = getattr(image, "n_frames", 1)
    durations: list[int] = []
    coverage: list[float] = []
    aspects: list[float] = []
    green: list[float] = []
    semi_alpha: list[float] = []
    fragments: list[float] = []
    edge_risk = 0
    temporal_diffs: list[float] = []
    first_thumb = None
    previous_thumb = None
    last_thumb = None

    for index in range(frame_count):
        image.seek(index)
        durations.append(int(image.info.get("duration", 42) or 42))
        rgba = np.asarray(image.convert("RGBA"))
        alpha = rgba[:, :, 3]
        visible = alpha > 16
        visible_count = int(visible.sum())
        coverage.append(float(visible.mean()))

        if visible_count:
            ys, xs = np.where(visible)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            aspects.append((y1 - y0) / max(1, x1 - x0))
            if x0 <= 4 or y0 <= 4 or x1 >= rgba.shape[1] - 4 or y1 >= rgba.shape[0] - 4:
                edge_risk += 1

            rgb = rgba[:, :, :3].astype(np.int16)
            eroded = cv2.erode(visible.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
            silhouette = visible & ~eroded
            transition = (alpha > 16) & (alpha < 245)
            edge = silhouette | transition
            greenish = (
                (rgb[:, :, 1] > rgb[:, :, 0] + 24)
                & (rgb[:, :, 1] > rgb[:, :, 2] + 24)
                & edge
            )
            green.append(float(greenish.sum() / visible_count))
            semi_alpha.append(float(((alpha > 0) & (alpha < 245)).sum() / visible_count))

            small_mask = cv2.resize(
                visible.astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST
            )
            count, _, stats, _ = cv2.connectedComponentsWithStats(small_mask, 8)
            component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([])
            total_area = int(component_areas.sum()) if component_areas.size else 0
            largest = int(component_areas.max()) if component_areas.size else 0
            fragments.append((total_area - largest) / max(1, total_area))
        else:
            aspects.append(0.0)
            green.append(1.0)
            semi_alpha.append(0.0)
            fragments.append(1.0)
            edge_risk += 1

        thumb = cv2.resize(rgba, (128, 128), interpolation=cv2.INTER_AREA)
        alpha_f = thumb[:, :, 3:4].astype(np.float32) / 255.0
        premultiplied = np.concatenate(
            [thumb[:, :, :3].astype(np.float32) * alpha_f, thumb[:, :, 3:4].astype(np.float32)],
            axis=2,
        ).astype(np.uint8)
        if first_thumb is None:
            first_thumb = premultiplied
        if previous_thumb is not None:
            temporal_diffs.append(frame_distance(previous_thumb, premultiplied))
        previous_thumb = premultiplied
        last_thumb = premultiplied

    duration_ms = sum(durations)
    coverage_mean = statistics.fmean(coverage) if coverage else 0.0
    coverage_cv = statistics.pstdev(coverage) / coverage_mean if coverage_mean else 0.0
    aspect_mean = statistics.fmean(aspects) if aspects else 0.0
    aspect_cv = statistics.pstdev(aspects) / aspect_mean if aspect_mean else 0.0
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "frames": frame_count,
        "duration_seconds": round(duration_ms / 1000.0, 2),
        "fps": round(frame_count / max(0.001, duration_ms / 1000.0), 2),
        "file_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "green_residual_max_pct": round(max(green, default=0.0) * 100, 3),
        "semi_alpha_p95_pct": round(percentile(semi_alpha, 0.95) * 100, 3),
        "fragment_p95_pct": round(percentile(fragments, 0.95) * 100, 3),
        "edge_risk_frames": edge_risk,
        "coverage_mean_pct": round(coverage_mean * 100, 2),
        "coverage_cv_pct": round(coverage_cv * 100, 2),
        "bbox_aspect_median": round(statistics.median(aspects), 3),
        "bbox_aspect_cv_pct": round(aspect_cv * 100, 2),
        "temporal_jump_p95_pct": round(percentile(temporal_diffs, 0.95) * 100, 3),
        "temporal_jump_max_pct": round(max(temporal_diffs, default=0.0) * 100, 3),
        "loop_seam_pct": round(frame_distance(first_thumb, last_thumb) * 100, 3),
    }


def load_metrics(pet_id: str) -> dict:
    path = OUTPUT / pet_id / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def usage_tokens(rows: list[dict]) -> int:
    return int(sum((row.get("usage") or {}).get("total_tokens", 0) or 0 for row in rows))


def build_report(base: str, run_meta: dict) -> dict:
    report = {"base": base, "assets": {}, "flows": {}, "run_meta": run_meta}
    for flow in ("before", "after"):
        flow_assets = []
        video_tokens = 0
        image_calls = 0
        for style in STYLES:
            pet_id = f"{base}_{style}_{flow}"
            metrics = load_metrics(pet_id)
            video_tokens += usage_tokens(metrics.get("animate", []))
            image_calls += len(metrics.get("stylize", [])) + len(metrics.get("state_frame", []))
            for clip in CLIPS:
                key = f"{style}_{flow}_{clip}"
                asset = analyze_webp(OUTPUT / pet_id / f"anim_{clip}.webp")
                asset.update({
                    "style": style,
                    "flow": flow,
                    "clip": clip,
                    "pet_id": pet_id,
                    "web_path": f"/{pet_id}/anim_{clip}.webp",
                })
                report["assets"][key] = asset
                flow_assets.append(asset)

        video_cost = video_tokens / 1_000_000 * SEEDANCE_SILENT_RMB_PER_M_TOKENS
        image_cost = image_calls * SEEDREAM_45_RMB_PER_IMAGE
        report["flows"][flow] = {
            "video_tokens": video_tokens,
            "image_calls": image_calls,
            "video_cost_rmb": round(video_cost, 2),
            "image_cost_rmb": round(image_cost, 2),
            "estimated_cost_rmb": round(video_cost + image_cost, 2),
            "total_webp_mb": round(sum(item["file_mb"] for item in flow_assets), 2),
            "mean_green_residual_max_pct": round(statistics.fmean(item["green_residual_max_pct"] for item in flow_assets), 3),
            "edge_risk_frames": sum(item["edge_risk_frames"] for item in flow_assets),
            "mean_fragment_p95_pct": round(statistics.fmean(item["fragment_p95_pct"] for item in flow_assets), 3),
            "mean_loop_seam_pct": round(statistics.fmean(item["loop_seam_pct"] for item in flow_assets), 3),
        }

    before = report["flows"]["before"]
    after = report["flows"]["after"]
    report["delta"] = {
        "video_tokens_pct": round((after["video_tokens"] / before["video_tokens"] - 1) * 100, 1),
        "estimated_cost_pct": round((after["estimated_cost_rmb"] / before["estimated_cost_rmb"] - 1) * 100, 1),
        "webp_size_pct": round((after["total_webp_mb"] / before["total_webp_mb"] - 1) * 100, 1),
        "matte_wall_pct": round((run_meta["after"]["matte_wall_seconds"] / run_meta["before"]["matte_wall_seconds"] - 1) * 100, 1),
    }
    extra_video_tokens = int(run_meta.get("experiment_extra_video_tokens", 0) or 0)
    report["experiment_actual"] = {
        "image_calls": after["image_calls"],
        "video_tokens": before["video_tokens"] + after["video_tokens"] + extra_video_tokens,
        "discarded_iteration_video_calls": int(run_meta.get("experiment_extra_video_calls", 0) or 0),
        "discarded_iteration_video_tokens": extra_video_tokens,
        "invalid_three_second_video_tasks": 0,
        "estimated_spend_rmb": round(
            after["image_calls"] * SEEDREAM_45_RMB_PER_IMAGE
            + (before["video_tokens"] + after["video_tokens"] + extra_video_tokens)
            / 1_000_000 * SEEDANCE_SILENT_RMB_PER_M_TOKENS,
            2,
        ),
    }
    probe_id = f"{base}_paimomo_480probe"
    probe_dir = OUTPUT / probe_id
    if (probe_dir / "anim_fast_walk.webp").exists() and (probe_dir / "metrics.json").exists():
        probe_metrics = load_metrics(probe_id)
        probe_row = next(row for row in probe_metrics.get("animate", []) if row.get("clip") == "fast_walk")
        reference_row = next(
            row for row in load_metrics(f"{base}_paimomo_after").get("animate", [])
            if row.get("clip") == "fast_walk"
        )
        probe_asset = analyze_webp(probe_dir / "anim_fast_walk.webp")
        probe_asset["web_path"] = f"/{probe_id}/anim_fast_walk.webp"
        reference_tokens = int((reference_row.get("usage") or {}).get("total_tokens", 0) or 0)
        probe_tokens = int((probe_row.get("usage") or {}).get("total_tokens", 0) or 0)
        token_ratio = probe_tokens / max(1, reference_tokens)
        projected_video_tokens = round(after["video_tokens"] * token_ratio)
        projected_cost = (
            projected_video_tokens / 1_000_000 * SEEDANCE_SILENT_RMB_PER_M_TOKENS
            + after["image_calls"] * SEEDREAM_45_RMB_PER_IMAGE
        )
        report["resolution_probe"] = {
            "reference_720": report["assets"]["paimomo_after_fast_walk"],
            "candidate_480": probe_asset,
            "reference_tokens": reference_tokens,
            "candidate_tokens": probe_tokens,
            "token_delta_pct": round((token_ratio - 1) * 100, 1),
            "reference_seconds": reference_row.get("seconds"),
            "candidate_seconds": probe_row.get("seconds"),
            "projected_six_video_tokens": projected_video_tokens,
            "projected_pet_cost_rmb": round(projected_cost, 2),
            "projected_vs_baseline_cost_pct": round((projected_cost / before["estimated_cost_rmb"] - 1) * 100, 1),
        }
        report["experiment_actual"]["video_tokens"] += probe_tokens
        report["experiment_actual"]["estimated_spend_rmb"] = round(
            report["experiment_actual"]["estimated_spend_rmb"]
            + probe_tokens / 1_000_000 * SEEDANCE_SILENT_RMB_PER_M_TOKENS,
            2,
        )
    return report


def metric_table_rows(report: dict) -> str:
    rows = []
    for style in STYLES:
        for clip in CLIPS:
            before = report["assets"][f"{style}_before_{clip}"]
            after = report["assets"][f"{style}_after_{clip}"]
            rows.append(
                "<tr>"
                f"<td>{STYLE_LABELS[style]}</td><td>{clip}</td>"
                f"<td>{before['frames']} / {after['frames']}</td>"
                f"<td>{before['file_mb']:.2f} / {after['file_mb']:.2f}</td>"
                f"<td>{before['green_residual_max_pct']:.3f}% / {after['green_residual_max_pct']:.3f}%</td>"
                f"<td>{before['edge_risk_frames']} / {after['edge_risk_frames']}</td>"
                f"<td>{before['bbox_aspect_median']:.3f} / {after['bbox_aspect_median']:.3f}</td>"
                f"<td>{before['loop_seam_pct']:.3f}% / {after['loop_seam_pct']:.3f}%</td>"
                "</tr>"
            )
    return "".join(rows)


def build_html(report: dict) -> str:
    base = report["base"]
    before = report["flows"]["before"]
    after = report["flows"]["after"]
    delta = report["delta"]
    source = f"/inputs/{base}_real_before.jpg"
    report_json = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    style_sections = []
    for style in STYLES:
        cards = []
        for flow in ("before", "after"):
            pet_id = f"{base}_{style}_{flow}"
            first = report["assets"][f"{style}_{flow}_idle"]
            cards.append(f"""
              <article class="demo-card" data-style="{style}" data-flow="{flow}">
                <div class="card-head"><div><strong>{FLOW_LABELS[flow]}</strong><span>{'同一坐姿驱动全部动作' if flow == 'before' else '目标状态首尾帧 + 内存并行抠图'}</span></div><b>{first['frames']} 帧</b></div>
                <div class="stage"><img draggable="false" src="/{pet_id}/anim_idle.webp?v=ab2" alt="{STYLE_LABELS[style]}{FLOW_LABELS[flow]}" /></div>
                <div class="live-metrics"><span>idle</span><span>{first['file_mb']:.2f} MB</span><span>绿残留 {first['green_residual_max_pct']:.3f}%</span></div>
              </article>
            """)
        style_sections.append(f"""
          <section class="style-section">
            <div class="section-title"><div><span class="eyebrow">{STYLE_LABELS[style]}</span><h2>同一身份资产，直接比较动作机制</h2></div></div>
            <div class="demo-grid">{''.join(cards)}</div>
          </section>
        """)

    probe_section = ""
    probe = report.get("resolution_probe")
    if probe:
        ref = probe["reference_720"]
        candidate = probe["candidate_480"]
        probe_section = f"""
          <section class="style-section">
            <div class="section-title"><div><span class="eyebrow">低成本模式探针</span><h2>同一萌宠快走首尾帧：720p 对比 480p</h2></div></div>
            <div class="demo-grid">
              <article class="demo-card resolution-card">
                <div class="card-head"><div><strong>720p 质量模式</strong><span>当前默认，毛发细节更稳</span></div><b>{probe['reference_tokens']:,} tokens</b></div>
                <div class="stage"><img draggable="false" src="{ref['web_path']}?v=probe720_fixed" alt="720p 快走" /></div>
                <div class="live-metrics"><span>fast_walk</span><span>{ref['file_mb']:.2f} MB</span><span>循环缝 {ref['loop_seam_pct']:.3f}%</span></div>
              </article>
              <article class="demo-card resolution-card">
                <div class="card-head"><div><strong>480p 低成本模式</strong><span>最终仍输出 640px WebP，细毛略软</span></div><b>{probe['candidate_tokens']:,} tokens</b></div>
                <div class="stage"><img draggable="false" src="{candidate['web_path']}?v=probe480_fixed" alt="480p 快走" /></div>
                <div class="live-metrics"><span>fast_walk</span><span>{candidate['file_mb']:.2f} MB</span><span>循环缝 {candidate['loop_seam_pct']:.3f}%</span></div>
              </article>
            </div>
            <p class="note">单段 token {probe['token_delta_pct']:.1f}%；若六段全部使用 480p，按本次比例投影单宠约 ¥{probe['projected_pet_cost_rmb']:.2f}，较旧链路 {probe['projected_vs_baseline_cost_pct']:.1f}%。建议作为批量/经济档，投资人展示和最终资产仍用 720p。</p>
            <p class="note"><strong>动作质量不是等价项：</strong>本次 480p 快走仍出现完整转身，720p 修正版降为正面到 3/4 视角漂移。因此 480p 只证明 token 降本潜力，不能直接替代质量档。</p>
          </section>
        """
        cost_finding = f"<div class=\"finding\"><strong>双档成本策略</strong><p>720p 优化链路成本基本持平；480p 探针单段 token 下降 {probe['token_delta_pct']:.1f}% ，可将单宠投影成本降至约 ¥{probe['projected_pet_cost_rmb']:.2f}，但细毛更软且本次出现完整转身，只保留为需抽检的经济档。</p></div>"
    else:
        cost_finding = "<div class=\"finding\"><strong>成本基本持平</strong><p>4 秒视频降低了视频 token，但独立状态首帧抵消了大部分节省；后续应单独验证低分辨率经济档。</p></div>"

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>宠物动效链路 A/B 对比</title>
<style>
:root{{--bg:#f4f6f3;--panel:#fff;--line:#dce2dc;--text:#18211d;--muted:#637069;--green:#176b55;--warm:#9a694c;--bad:#9a453c}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,"Microsoft YaHei",sans-serif;letter-spacing:0}}
.shell{{width:min(1180px,calc(100% - 32px));margin:auto;padding:28px 0 72px}} header{{display:grid;grid-template-columns:1fr 220px;gap:24px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:22px}}
.eyebrow{{color:var(--green);font-size:12px;font-weight:700}} h1{{font-size:34px;line-height:1.1;margin:8px 0 10px}} h2{{font-size:18px;margin:4px 0 0}} p{{color:var(--muted);margin:0;max-width:760px}}
.source{{height:132px;background:#e9ede9;border:1px solid var(--line);overflow:hidden;border-radius:8px}} .source img{{width:100%;height:100%;object-fit:cover}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0}} .metric{{background:var(--panel);border:1px solid var(--line);padding:14px;border-radius:8px}} .metric span{{display:block;color:var(--muted);font-size:12px}} .metric strong{{display:block;font-size:22px;margin-top:5px}} .metric em{{font-style:normal;color:var(--green);font-size:12px}}
.toolbar{{position:sticky;top:12px;z-index:5;display:flex;justify-content:center;margin:20px 0 26px;pointer-events:none}} .segmented{{display:flex;padding:4px;background:#fff;border:1px solid var(--line);box-shadow:0 8px 22px #1b2b2117;border-radius:8px;pointer-events:auto}} button{{border:0;background:transparent;padding:9px 18px;color:var(--muted);font-weight:700;cursor:pointer;border-radius:6px}} button.active{{background:var(--green);color:#fff}}
.style-section{{margin-top:32px}} .section-title{{display:flex;justify-content:space-between;margin-bottom:12px}} .demo-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .demo-card{{background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.card-head{{display:flex;justify-content:space-between;padding:13px 15px;border-bottom:1px solid var(--line)}} .card-head span{{display:block;color:var(--muted);font-size:12px;margin-top:2px}} .card-head b{{color:var(--green)}}
.stage{{height:390px;display:grid;place-items:center;background-color:#edf0ed;background-image:linear-gradient(45deg,#dfe4df 25%,transparent 25%),linear-gradient(-45deg,#dfe4df 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#dfe4df 75%),linear-gradient(-45deg,transparent 75%,#dfe4df 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}}
.stage img{{width:100%;height:100%;object-fit:contain}} .live-metrics{{display:flex;gap:18px;padding:10px 15px;border-top:1px solid var(--line);color:var(--muted)}} .live-metrics span:first-child{{color:var(--green);font-weight:700}}
.findings{{margin:34px 0 0;padding:22px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);display:grid;grid-template-columns:repeat(3,1fr);gap:24px}} .finding strong{{display:block;margin-bottom:5px}} .finding p{{font-size:13px}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:8px;margin-top:30px}} table{{border-collapse:collapse;width:100%;min-width:900px}} th,td{{padding:11px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:12px}} th{{background:#f0f3f0;color:var(--muted)}} td:nth-child(n+3){{font-variant-numeric:tabular-nums}}
.note{{margin-top:14px;color:var(--muted);font-size:12px}}
@media(max-width:760px){{header{{grid-template-columns:1fr}}.source{{height:180px}}.summary{{grid-template-columns:1fr 1fr}}.demo-grid{{grid-template-columns:1fr}}.stage{{height:310px}}.findings{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
</style></head><body><main class="shell">
<header><div><span class="eyebrow">PET AVATAR PIPELINE · CONTROLLED A/B</span><h1>同一只宠物，链路优化前后对比</h1><p>共享同一实体版和萌宠版身份首帧，仅改变动作首帧机制、视频时长与本地抠图实现。页面中的 WebP 均为本次实跑结果。</p></div><div class="source"><img src="{source}" alt="上传原图"></div></header>
<section class="summary">
 <div class="metric"><span>本地抠图墙钟</span><strong>{report['run_meta']['before']['matte_wall_seconds']:.1f}s → {report['run_meta']['after']['matte_wall_seconds']:.1f}s</strong><em>{delta['matte_wall_pct']:.1f}%</em></div>
 <div class="metric"><span>视频 tokens</span><strong>{before['video_tokens']:,} → {after['video_tokens']:,}</strong><em>{delta['video_tokens_pct']:.1f}%</em></div>
 <div class="metric"><span>单宠估算成本</span><strong>¥{before['estimated_cost_rmb']:.2f} → ¥{after['estimated_cost_rmb']:.2f}</strong><em>{delta['estimated_cost_pct']:.1f}%</em></div>
 <div class="metric"><span>6 个 WebP 体积</span><strong>{before['total_webp_mb']:.2f} → {after['total_webp_mb']:.2f} MB</strong><em>{delta['webp_size_pct']:.1f}%</em></div>
</section>
<div class="toolbar"><div class="segmented" role="group" aria-label="动作切换"><button class="active" data-clip="idle">静息</button><button data-clip="fast_walk">快走</button><button data-clip="sleep">睡眠</button></div></div>
{''.join(style_sections)}
{probe_section}
<section class="findings">
 <div class="finding"><strong>动作正确性提升</strong><p>旧快走包含坐姿启停，旧睡眠多数帧仍端坐；新链路从目标状态首帧开始，快走和睡眠全程保持在对应状态内。</p></div>
 <div class="finding"><strong>本地速度确定性提升</strong><p>内存解码与三动作并行使抠图墙钟下降约 {abs(delta['matte_wall_pct']):.0f}%。云端本次受排队尾延迟影响，新组反而更慢，不能把单次 API 墙钟当稳定收益。</p></div>
 {cost_finding}
</section>
<div class="table-wrap"><table><thead><tr><th>风格</th><th>动作</th><th>帧数 旧/新</th><th>MB 旧/新</th><th>最大绿残留 旧/新</th><th>触边帧 旧/新</th><th>姿态高宽比 旧/新</th><th>循环缝 旧/新</th></tr></thead><tbody>{metric_table_rows(report)}</tbody></table></div>
<p class="note">计费估算按 2026-07-10 公开刊例：Seedance 1.5 Pro 无声视频 8 元/百万 tokens，Seedream 4.5 为 0.25 元/张；实际账单以账号折扣为准。所有 MP4 已在本地强制移除音轨。</p>
</main><script>
const report={report_json};
const labels={{idle:'静息',fast_walk:'快走',sleep:'睡眠'}};
document.querySelectorAll('button[data-clip]').forEach(button=>button.addEventListener('click',()=>{{
 const clip=button.dataset.clip; document.querySelectorAll('button[data-clip]').forEach(b=>b.classList.toggle('active',b===button));
 document.querySelectorAll('.demo-card[data-style][data-flow]').forEach(card=>{{
  const style=card.dataset.style,flow=card.dataset.flow,key=`${{style}}_${{flow}}_${{clip}}`,asset=report.assets[key];
  card.querySelector('img').src=`${{asset.web_path}}?v=ab2_${{clip}}`; const spans=card.querySelectorAll('.live-metrics span');
  spans[0].textContent=labels[clip]; spans[1].textContent=`${{asset.file_mb.toFixed(2)}} MB`; spans[2].textContent=`绿残留 ${{asset.green_residual_max_pct.toFixed(3)}}%`;
 }});
}}));
</script></body></html>"""


def build_markdown(report: dict) -> str:
    before = report["flows"]["before"]
    after = report["flows"]["after"]
    delta = report["delta"]
    probe = report.get("resolution_probe") or {}
    return f"""# 宠物 3D 动效链路 A/B 测试报告

测试 ID：`{report['base']}`
测试原则：优化前后共享同一实体版与萌宠版身份首帧，仅比较动作状态首帧、视频时长和抠图实现。

## 结论

- 动作质量：优化前的快走包含端坐启停，睡眠多数帧仍端坐；优化后快走和睡眠从第一帧起就在目标状态内。720p 修正版仍有正面到 3/4 视角漂移，不能宣称朝向完全锁定。
- 本地速度：抠图墙钟从 {report['run_meta']['before']['matte_wall_seconds']:.1f}s 降至 {report['run_meta']['after']['matte_wall_seconds']:.1f}s（{delta['matte_wall_pct']:.1f}%）。
- 视频用量：{before['video_tokens']:,} → {after['video_tokens']:,} tokens（{delta['video_tokens_pct']:.1f}%）。
- 总成本：约 ¥{before['estimated_cost_rmb']:.2f} → ¥{after['estimated_cost_rmb']:.2f}（{delta['estimated_cost_pct']:.1f}%），基本持平。
- 文件体积：6 个 WebP 合计 {before['total_webp_mb']:.2f} → {after['total_webp_mb']:.2f} MB（{delta['webp_size_pct']:.1f}%）。
- 云端速度：本次优化组受服务端长尾影响，实际比旧组慢；本地优化收益确定，云端单次墙钟不具代表性。

## 480p 低成本探针

- 同一萌宠快走状态首尾帧：`{probe.get('reference_tokens', 0):,} → {probe.get('candidate_tokens', 0):,}` tokens（{probe.get('token_delta_pct', 0):.1f}%）。
- 本次生成计时：`{probe.get('reference_seconds', 0)}s → {probe.get('candidate_seconds', 0)}s`；仅为一次云端样本，不作为 SLA。
- 轮廓绿残留：`{probe.get('reference_720', {}).get('green_residual_max_pct', 0):.3f}% → {probe.get('candidate_480', {}).get('green_residual_max_pct', 0):.3f}%`；循环缝：`{probe.get('reference_720', {}).get('loop_seam_pct', 0):.3f}% → {probe.get('candidate_480', {}).get('loop_seam_pct', 0):.3f}%`。
- 六段全部使用 480p 的投影成本约 ¥{probe.get('projected_pet_cost_rmb', 0):.2f}，较旧链路 {probe.get('projected_vs_baseline_cost_pct', 0):.1f}%。建议作为经济档，720p 保留为最终质量档。
- 本次 480p 快走出现完整转身，动作质量不与 720p 等价；该探针只用于验证 token/成本，不用于证明成片质量。

## 流程对比

| 项目 | 优化前 | 优化后 |
|---|---:|---:|
| 身份图 | 实体/萌宠各 1 张 | 同一批身份图 |
| 状态首帧 | 无，3 动作共用端坐图 | 快走/睡眠各自独立状态图 |
| Seedance 时长 | 5 秒 | 4 秒（1.5 Pro 最短合法时长） |
| 抠图 | 串行、RGB/RGBA PNG 落盘 | 一次解码进内存、3 动作并行 |
| 视频 tokens | {before['video_tokens']:,} | {after['video_tokens']:,} |
| 图片调用 | {before['image_calls']} 张 | {after['image_calls']} 张 |
| 本地抠图墙钟 | {report['run_meta']['before']['matte_wall_seconds']:.1f}s | {report['run_meta']['after']['matte_wall_seconds']:.1f}s |
| 估算成本 | ¥{before['estimated_cost_rmb']:.2f} | ¥{after['estimated_cost_rmb']:.2f} |

## 下一步

1. 保留独立状态首帧与内存并行抠图，这是质量和速度的有效改动。
2. 快走提示词已经锁定 3/4 朝向、禁止转身和绕圈，但 Seedance 仍有随机方向漂移；若产品要求严格固定朝向，需要增加多候选自动筛选、动作参考或骨骼动画路径，不能只依赖文本提示词。
3. 单独做 480p/720p 一动作盲测；最终 WebP 为 640px，480p 可能显著降费，但必须先验证毛发和爪部边缘。
4. 用 BiRefNet/SAM2 作为非绿幕或疑难帧的回退路径，不替换当前快速绿幕主路径。

## 计费口径

按 2026-07-10 公开刊例估算：Seedance 1.5 Pro 无声视频 `8 元/百万 tokens`，Seedream 4.5 `0.25 元/张`。实际账单以账号折扣为准。本次完整 A/B 实验估算支出约 ¥{report['experiment_actual']['estimated_spend_rmb']:.2f}。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--meta", required=True, type=Path)
    args = parser.parse_args()
    run_meta = json.loads(args.meta.read_text(encoding="utf-8"))
    report = build_report(args.base, run_meta)
    json_path = OUTPUT / f"ab_report_{args.base}.json"
    html_path = OUTPUT / f"ab_compare_{args.base}.html"
    markdown_path = ROOT / f"AB_REPORT_{args.base}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(build_html(report), encoding="utf-8")
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(json_path)
    print(html_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
