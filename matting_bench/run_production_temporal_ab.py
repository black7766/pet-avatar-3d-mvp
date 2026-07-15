"""Compare production edge_v2 and temporal_v3 on complete pet action clips."""

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
DEFAULT_PETS = (
    "desktop_gray_cat_20260713_real",
    "desktop_tabby_cat_20260713_real",
)
DEFAULT_ACTIONS = ("fast_walk", "sleep")
PROVIDERS = {
    "edge_v2": ("边缘优化 v2", False),
    "temporal_v3": ("时序稳定 v3", True),
}
PET_LABELS = {
    "desktop_gray_cat_20260713_real": "银灰短毛猫 · 实体版",
    "desktop_tabby_cat_20260713_real": "虎斑白袜猫 · 实体版",
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def extract_frames(video: Path, destination: Path, size: int = 640) -> int:
    existing = sorted(destination.glob("f_*.png"))
    if existing:
        return len(existing)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).resize(
            (size, size), Image.Resampling.LANCZOS
        ).save(destination / f"f_{count:04d}.png", compress_level=2)
        count += 1
    capture.release()
    if count < 2:
        raise RuntimeError(f"too few frames in {video}: {count}")
    return count


def run_provider(
    source: Path, destination: Path, temporal: bool, flow_size: int
) -> None:
    expected = len(sorted(source.glob("f_*.png")))
    complete = len(sorted(destination.glob("f_*.png"))) == expected
    metrics_path = destination / "metrics.json"
    if complete and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        current_size = (metrics.get("parameters") or {}).get("temporal_flow_size")
        if not temporal or current_size == flow_size:
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
        "--edge-refine",
    ]
    if temporal:
        command.extend(["--temporal-refine", "--temporal-flow-size", str(flow_size)])
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate(source: Path, output: Path) -> dict[str, Any]:
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import evaluate as benchmark  # type: ignore

    return benchmark.evaluate_provider(source, output)


def encode_webp(source: Path, output: Path, fps: int = 24) -> dict[str, Any]:
    paths = sorted(source.glob("f_*.png"))
    newest = max(path.stat().st_mtime for path in paths)
    if output.is_file() and output.stat().st_mtime >= newest:
        with Image.open(output) as image:
            return {"frames": image.n_frames, "mb": output.stat().st_size / 1048576}
    frames = [Image.open(path).convert("RGBA") for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        quality=92,
        alpha_quality=100,
        method=4,
        exact=True,
    )
    return {"frames": len(frames), "mb": output.stat().st_size / 1048576}


def fmt(value: Any, digits: int = 5) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def change(current: float, baseline: float) -> float | None:
    if not baseline:
        return None
    return (current - baseline) / baseline * 100.0


def build_html(result: dict[str, Any], output: Path) -> None:
    sections = []
    for pet in result["pets"]:
        action_rows = []
        for action in pet["actions"]:
            cards = []
            for key in ("edge_v2", "temporal_v3"):
                item = action["providers"][key]
                mean = item["evaluation"]["mean"]
                temporal = item["evaluation"].get("temporal_alpha_mae")
                runtime = item["runtime"]
                fragment_stats = runtime.get("temporal") or {}
                cards.append(f"""
                <article class="variant">
                  <header><h3>{item['label']}</h3><span>{item['asset']['frames']} 帧 · {item['asset']['mb']:.2f} MB</span></header>
                  <div class="stage checker"><img src="{item['url']}" alt="{pet['label']} {action['label']} {item['label']}"></div>
                  <dl>
                    <div><dt>时序误差</dt><dd>{fmt(temporal, 6)}</dd></div>
                    <div><dt>碎片率</dt><dd>{fmt(mean.get('fragment_pct'), 4)}%</dd></div>
                    <div><dt>背景泄漏</dt><dd>{fmt(mean.get('background_alpha_mean'), 6)}</dd></div>
                    <div><dt>前景损失</dt><dd>{fmt(mean.get('foreground_loss_mean'), 6)}</dd></div>
                    <div><dt>绿边</dt><dd>{fmt(mean.get('green_fringe'), 6)}</dd></div>
                    <div><dt>CPU 总耗时</dt><dd>{runtime['total_seconds']:.2f}s</dd></div>
                    <div><dt>移除碎片</dt><dd>{fragment_stats.get('single_frame_fragments_removed', 0)}</dd></div>
                    <div><dt>保护大部件</dt><dd>{fragment_stats.get('protected_large_components', 0)}</dd></div>
                  </dl>
                </article>""")
            delta = action["delta"]
            action_rows.append(f"""
            <section class="action">
              <div class="action-title"><h2>{action['label']}</h2><p>v3 相对 v2：时序 {delta['temporal']:+.1f}% · 碎片 {delta['fragment']:+.1f}% · 耗时 {delta['runtime']:+.1f}%</p></div>
              <div class="variants">{''.join(cards)}</div>
            </section>""")
        sections.append(f"<section class='pet'><h1>{pet['label']}</h1>{''.join(action_rows)}</section>")
    payload = json.dumps(result, ensure_ascii=False).replace("</", "<\\/")
    output.write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>实体宠物抠图生产 A/B</title><style>
:root{{--bg:#f4f6f4;--panel:#fff;--line:#d9dfda;--text:#172019;--muted:#657067;--green:#246b56}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{width:min(1180px,calc(100% - 32px));margin:28px auto 64px}}.intro{{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:18px}}
h1,h2,h3,p{{margin:0}}.intro h1{{font-size:26px}}.intro p,.action-title p,header span{{color:var(--muted)}}.pet>h1{{font-size:20px;margin:30px 0 12px}}
.action{{margin-bottom:20px}}.action-title{{display:flex;justify-content:space-between;align-items:center;margin:0 0 8px}}.action-title h2{{font-size:16px}}
.variants{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.variant{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.variant header{{display:flex;justify-content:space-between;padding:12px 14px;border-bottom:1px solid var(--line)}}.variant h3{{font-size:15px}}
.stage{{width:100%;aspect-ratio:1 / 1;display:grid;place-items:center;overflow:hidden;min-width:0;min-height:0}}.stage img{{display:block;width:100%;height:100%;min-width:0;min-height:0;object-fit:contain}}.checker{{background-color:#edf0ed;background-image:linear-gradient(45deg,#dde2de 25%,transparent 25%),linear-gradient(-45deg,#dde2de 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#dde2de 75%),linear-gradient(-45deg,transparent 75%,#dde2de 75%);background-size:22px 22px;background-position:0 0,0 11px,11px -11px,-11px 0}}
dl{{display:grid;grid-template-columns:repeat(4,1fr);margin:0;border-top:1px solid var(--line)}}dl div{{padding:9px 10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}dt{{font-size:11px;color:var(--muted)}}dd{{margin:2px 0 0;font-weight:650}}
@media(max-width:760px){{.intro,.action-title{{align-items:flex-start;flex-direction:column}}.variants{{grid-template-columns:1fr}}dl{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><section class="intro"><div><h1>实体宠物抠图生产 A/B</h1><p>同源完整动作：edge_v2 对比光流门控 temporal_v3</p></div><p>{result['generated_at']}</p></section>{''.join(sections)}</main>
<script type="application/json" id="metrics">{payload}</script></body></html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pets", nargs="+", default=list(DEFAULT_PETS))
    parser.add_argument("--actions", nargs="+", default=list(DEFAULT_ACTIONS))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("poc_output/temporal_v3_production_ab_20260714"),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("poc_output/temporal_v3_production_ab_20260714.html"),
    )
    parser.add_argument("--flow-size", type=int, default=384)
    args = parser.parse_args()
    output_root = ROOT / args.output_root
    assets = output_root / "assets"
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pets": [],
    }
    for pet_id in args.pets:
        pet_dir = ROOT / "poc_output" / pet_id
        pet_record = {
            "id": pet_id,
            "label": PET_LABELS.get(pet_id, pet_id),
            "actions": [],
        }
        for action in args.actions:
            video = pet_dir / f"raw_{action}.mp4"
            source = output_root / pet_id / action / "source"
            frame_count = extract_frames(video, source)
            action_record: dict[str, Any] = {
                "id": action,
                "label": "快走" if action == "fast_walk" else "睡眠",
                "source": relative(video),
                "frames": frame_count,
                "providers": {},
            }
            for key, (label, temporal) in PROVIDERS.items():
                directory_name = key if not temporal else f"temporal_v3_f{args.flow_size}"
                destination = output_root / pet_id / action / directory_name
                run_provider(source, destination, temporal, args.flow_size)
                evaluation = evaluate(source, destination)
                runtime = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                webp = assets / f"{pet_id}__{action}__{key}.webp"
                asset = encode_webp(destination, webp)
                action_record["providers"][key] = {
                    "label": label,
                    "output": relative(destination),
                    "url": f"/{output_root.name}/assets/{webp.name}",
                    "evaluation": evaluation,
                    "runtime": runtime,
                    "asset": asset,
                }
            edge = action_record["providers"]["edge_v2"]
            temporal = action_record["providers"]["temporal_v3"]
            action_record["delta"] = {
                "temporal": change(
                    temporal["evaluation"]["temporal_alpha_mae"],
                    edge["evaluation"]["temporal_alpha_mae"],
                ),
                "fragment": change(
                    temporal["evaluation"]["mean"]["fragment_pct"],
                    edge["evaluation"]["mean"]["fragment_pct"],
                ),
                "runtime": change(
                    temporal["runtime"]["total_seconds"],
                    edge["runtime"]["total_seconds"],
                ),
            }
            pet_record["actions"].append(action_record)
        result["pets"].append(pet_record)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html = ROOT / args.html
    build_html(result, html)
    print(html)


if __name__ == "__main__":
    main()
