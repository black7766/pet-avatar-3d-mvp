"""Build a synchronized local HTML comparison for matting providers."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


def parse_provider(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("provider must be NAME=OUTPUT_DIR")
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path).resolve()


def compact_runtime(provider: dict, metadata: dict) -> dict:
    runtime = provider.get("runtime") or {}
    mean_ms = metadata.get("runtime_ms")
    if mean_ms is None:
        mean_ms = runtime.get("mean_ms_per_frame") or runtime.get("mean_ms")
    if mean_ms is None and runtime.get("inference_mean_excluding_first_seconds"):
        mean_ms = float(runtime["inference_mean_excluding_first_seconds"]) * 1000
    return {
        "mean_ms": mean_ms,
        "peak_vram_mb": metadata.get("vram_mb"),
        "scope": metadata.get("runtime_scope", "单帧耗时"),
    }


def fmt(value, digits=3, suffix="") -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--provider", type=parse_provider, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    catalog = (
        json.loads(args.catalog.read_text(encoding="utf-8"))
        if args.catalog
        else {"providers": {}, "summary": {}, "video_models": []}
    )
    sources = sorted(args.source_dir.glob("*.png"))
    if not sources:
        raise SystemExit("source directory has no PNG files")
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    source_assets = args.asset_dir / "source"
    source_assets.mkdir(exist_ok=True)
    for path in sources:
        shutil.copy2(path, source_assets / path.name)

    cards = []
    provider_payload = {}
    for name, output_dir in args.provider:
        provider_report = report["providers"].get(name, {})
        metadata = catalog.get("providers", {}).get(name, {})
        model_assets = args.asset_dir / name
        model_assets.mkdir(exist_ok=True)
        files = {}
        for source in sources:
            candidate = output_dir / source.name
            if candidate.exists():
                shutil.copy2(candidate, model_assets / source.name)
                files[source.name] = f"{args.asset_dir.name}/{name}/{source.name}"
        means = provider_report.get("mean") or {}
        runtime = compact_runtime(provider_report, metadata)
        display = metadata.get("display", name)
        badge = metadata.get("badge", "已测试")
        status = metadata.get("status", "candidate")
        source_url = metadata.get("source", "")
        title = (
            f'<a href="{html.escape(source_url)}" target="_blank" rel="noreferrer">{html.escape(display)}</a>'
            if source_url
            else html.escape(display)
        )
        provider_payload[name] = {"files": files, "report": provider_report}
        cards.append(
            f"""
            <article class="model-card {html.escape(status)}" data-provider="{html.escape(name)}">
              <header><strong>{title}</strong><span class="badge">{html.escape(badge)}</span></header>
              <div class="preview"><img alt="{html.escape(name)} output"></div>
              <dl>
                <div><dt>伪真值 MAE</dt><dd>{fmt(means.get('pseudo_mae'), 4)}</dd></div>
                <div><dt>绿边分数</dt><dd>{fmt(means.get('green_fringe'), 4)}</dd></div>
                <div><dt>前景损失</dt><dd>{fmt(means.get('foreground_loss_mean'), 4)}</dd></div>
                <div><dt>碎片率</dt><dd>{fmt(means.get('fragment_pct'), 3, '%')}</dd></div>
                <div><dt>{html.escape(runtime.get('scope', '单帧耗时'))}</dt><dd>{fmt(runtime.get('mean_ms'), 1, ' ms')}</dd></div>
                <div><dt>峰值显存</dt><dd>{fmt(runtime.get('peak_vram_mb'), 0, ' MB')}</dd></div>
              </dl>
              <p class="finding">{html.escape(metadata.get('finding', ''))}</p>
            </article>
            """
        )

    frames = [{"file": path.name, "label": path.stem} for path in sources]
    relative_source = f"{args.asset_dir.name}/source/"
    payload = json.dumps(
        {"frames": frames, "providers": provider_payload, "source": relative_source},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    frame_buttons = "".join(
        f'<button data-frame="{index}" class="{("active" if index == 0 else "")}">{html.escape(item["label"])}</button>'
        for index, item in enumerate(frames)
    )
    summary = catalog.get("summary", {})
    video_rows = "".join(
        f"<tr><td><a href='{html.escape(item.get('source', ''))}' target='_blank' rel='noreferrer'>{html.escape(item.get('name', ''))}</a></td><td>{html.escape(item.get('speed', '-'))}</td><td>{html.escape(item.get('vram', '-'))}</td><td>{html.escape(item.get('license', '-'))}</td><td>{html.escape(item.get('result', '-'))}</td></tr>"
        for item in catalog.get("video_models", [])
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>宠物抠图模型本地实测</title>
<style>
:root{{--bg:#f3f5f2;--panel:#fff;--line:#d9ded9;--text:#17201c;--muted:#68736d;--green:#176b55;--warm:#9b6548}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,"Microsoft YaHei",sans-serif;letter-spacing:0}}
.shell{{width:min(1440px,calc(100% - 32px));margin:auto;padding:24px 0 64px}}h1{{font-size:28px;margin:4px 0}}p{{margin:0;color:var(--muted)}}a{{color:inherit;text-decoration:none}}a:hover{{color:var(--green)}}
.top{{display:flex;justify-content:space-between;gap:18px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:18px}}.eyebrow{{font-size:12px;color:var(--green);font-weight:700}}
.controls{{position:sticky;top:8px;z-index:4;background:rgba(243,245,242,.96);padding:10px 0;display:grid;gap:8px}}.row{{display:flex;gap:6px;overflow:auto}}
button{{border:1px solid var(--line);background:#fff;color:var(--muted);padding:7px 10px;border-radius:6px;white-space:nowrap;cursor:pointer}}button.active{{background:var(--green);border-color:var(--green);color:#fff}}
.compare{{display:grid;grid-template-columns:260px 1fr;gap:12px;margin-top:12px}}.source-card,.model-card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.source-card header,.model-card header{{min-height:48px;padding:10px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;justify-content:space-between;align-items:center}}header span{{color:var(--muted);font-size:12px}}.badge{{border:1px solid var(--line);padding:2px 6px;border-radius:6px;white-space:nowrap}}.recommended .badge{{color:var(--green);border-color:#a7d4c8;background:#eef8f5}}.limited .badge{{color:#875133;border-color:#dcc3b4;background:#fff7f1}}
.source-card .preview,.model-card .preview{{height:300px;display:grid;place-items:center;background-color:#edf0ed;background-image:linear-gradient(45deg,#d9ded9 25%,transparent 25%),linear-gradient(-45deg,#d9ded9 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#d9ded9 75%),linear-gradient(-45deg,transparent 75%,#d9ded9 75%);background-size:22px 22px;background-position:0 0,0 11px,11px -11px,-11px 0}}
.preview.white{{background:#fff;background-image:none}}.preview.black{{background:#121412;background-image:none}}.preview img{{width:100%;height:100%;object-fit:contain}}
.models{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}dl{{display:grid;grid-template-columns:repeat(3,1fr);margin:0;padding:10px;gap:6px}}dl div{{border:1px solid var(--line);padding:7px;border-radius:6px}}dt{{color:var(--muted);font-size:11px}}dd{{margin:2px 0 0;font-weight:700;font-variant-numeric:tabular-nums}}.finding{{min-height:58px;padding:0 12px 12px;font-size:12px}}
.decision{{margin:18px 0 4px;border-left:4px solid var(--green);padding:12px 16px;background:#fff;border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}.decision strong{{display:block;margin-bottom:4px}}.video{{margin-top:22px;overflow:auto}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:10px;border:1px solid var(--line);text-align:left;vertical-align:top}}th{{font-size:12px;color:var(--muted);background:#f8faf8}}.note{{margin-top:18px;border-top:1px solid var(--line);padding-top:14px}}
@media(max-width:1000px){{.compare{{grid-template-columns:1fr}}.source-card{{display:none}}.models{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:620px){{.models{{grid-template-columns:1fr}}.top{{display:block}}}}
</style></head><body><main class="shell">
<section class="top"><div><span class="eyebrow">LOCAL MODEL BENCHMARK</span><h1>宠物毛发抠图模型对比</h1><p>同一批 960×960 Seedance 绿幕帧；所有模型同步切帧，支持棋盘、白色、黑色背景检查。</p></div><p>指标用于自动筛查，最终选择以毛发边缘和视频稳定性人工复核为准。</p></section>
<section class="decision"><strong>当前建议：{html.escape(summary.get('recommended', ''))}</strong><p>{html.escape(summary.get('reason', ''))} 测试范围：{html.escape(summary.get('test_scope', ''))}</p></section>
<section class="controls"><div class="row" aria-label="帧切换">{frame_buttons}</div><div class="row" aria-label="背景切换"><button data-bg="checker" class="active">棋盘</button><button data-bg="white">白底</button><button data-bg="black">黑底</button></div></section>
<section class="compare"><article class="source-card"><header><strong>输入原帧</strong><span>绿幕源</span></header><div class="preview"><img alt="source frame"></div></article><div class="models">{''.join(cards)}</div></section>
<section class="video"><h2>视频传播模型</h2><table><thead><tr><th>模型</th><th>速度</th><th>峰值显存</th><th>许可证</th><th>本地实测结论</th></tr></thead><tbody>{video_rows}</tbody></table></section>
<p class="note">伪真值 MAE 仅使用确定绿幕背景和确定非绿前景；绿边分数衡量半透明边缘中的绿色污染。二者越低越好，但不能替代毛发细节视觉判断。</p>
</main><script>
const data={payload};let frameIndex=0;let background='checker';
function render(){{const frame=data.frames[frameIndex];document.querySelector('.source-card img').src=data.source+frame.file;document.querySelectorAll('.model-card').forEach(card=>{{const item=data.providers[card.dataset.provider];const img=card.querySelector('img');const src=item.files[frame.file];if(src){{img.src=src;img.alt=card.dataset.provider+' '+frame.label}}else{{img.removeAttribute('src');img.alt='待生成'}}}});}}
document.querySelectorAll('[data-frame]').forEach(button=>button.addEventListener('click',()=>{{frameIndex=Number(button.dataset.frame);document.querySelectorAll('[data-frame]').forEach(item=>item.classList.toggle('active',item===button));render()}}));
document.querySelectorAll('[data-bg]').forEach(button=>button.addEventListener('click',()=>{{background=button.dataset.bg;document.querySelectorAll('[data-bg]').forEach(item=>item.classList.toggle('active',item===button));document.querySelectorAll('.preview').forEach(stage=>{{stage.classList.toggle('white',background==='white');stage.classList.toggle('black',background==='black')}})}}));render();
</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
