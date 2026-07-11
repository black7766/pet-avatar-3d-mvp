"""Render a self-contained tuning dashboard from aggregate_tuning.py output."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any


RECOMMENDATION_KEYS = (
    "config_id",
    "recommended_config_id",
    "selected_config_id",
    "primary",
    "id",
)


def fmt(value: object, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def compact_parameters(parameters: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in parameters.items()) or "default"


def recommended_id(provider: dict[str, Any]) -> str | None:
    normalized = provider.get("recommended_config_id")
    if normalized:
        return str(normalized)
    recommendation = provider.get("recommendation") or {}
    for key in RECOMMENDATION_KEYS:
        value = recommendation.get(key)
        if value:
            return str(value)
    return None


def preview_url(
    repo_root: Path,
    config: dict[str, Any],
    assets_dir: Path,
) -> str | None:
    output_dir = config.get("output_dir")
    if not output_dir:
        return None
    directory = repo_root / str(output_dir)
    candidates = (
        directory / "idle__f_0048.png",
        directory / "idle__f_0000.png",
        directory / "contact_sheet_rgba.png",
        directory / "contact_sheet_alpha.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            break
    else:
        pngs = sorted(directory.rglob("*.png")) if directory.is_dir() else []
        candidate = pngs[0] if pngs else None
    if candidate is None:
        return None
    slug = re.sub(
        r"[^a-z0-9._-]+",
        "-",
        f"{config['provider']}__{config['id']}".lower(),
    ).strip("-")
    destination = assets_dir / f"{slug}.png"
    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, destination)
    return f"/{assets_dir.name}/{destination.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.aggregate.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parent.parent
    assets_dir = args.output.parent / f"{args.output.stem}_assets"
    pareto = set(payload.get("pareto_frontier") or [])
    recommendations: dict[str, str] = {}
    doc_links: dict[str, str] = {}
    recommended_configs: list[dict[str, Any]] = []

    for provider in payload.get("providers", []):
        config_id = recommended_id(provider)
        if config_id:
            recommendations[provider["provider"]] = config_id
            match = next(
                (item for item in provider.get("configs", []) if item.get("id") == config_id),
                None,
            )
            if match:
                recommended_configs.append(match)
        docs = provider.get("official_docs") or []
        if docs:
            doc_links[provider["provider"]] = docs[0].get("url", "")

    preview_cards: list[str] = []
    for item in recommended_configs:
        quality = item.get("quality") or {}
        source = doc_links.get(item["provider"], "")
        title = html.escape(item["provider"])
        if source:
            title = (
                f'<a href="{html.escape(source)}" target="_blank" rel="noreferrer">'
                f"{title}</a>"
            )
        image_url = preview_url(repo_root, item, assets_dir)
        if image_url:
            media = f'<img src="{html.escape(image_url)}" alt="{html.escape(item["provider"])} 推荐配置输出">'
        else:
            media = '<div class="missing">未找到预览图</div>'
        candidate = bool(item.get("eligible_for_final") and item.get("passes_all_guardrails"))
        eligibility = "通过候选门槛" if candidate else "仅研究/对照"
        preview_cards.append(
            f"""
            <article class="preview-card">
              <header><div><strong>{title}</strong><small>{html.escape(item['id'])}</small></div><span class="{'ok' if candidate else 'warn'}">{eligibility}</span></header>
              <div class="preview-media">{media}</div>
              <dl><div><dt>pseudo MAE</dt><dd>{fmt(quality.get('pseudo_mae'))}</dd></div><div><dt>绿边</dt><dd>{fmt(quality.get('green_fringe'))}</dd></div><div><dt>推理</dt><dd>{fmt(item.get('runtime', {}).get('mean_inference_ms'), 1, ' ms')}</dd></div></dl>
            </article>
            """
        )

    rows: list[str] = []
    provider_names: list[str] = []
    for item in payload.get("configs", []):
        provider = item["provider"]
        if provider not in provider_names:
            provider_names.append(provider)
        quality = item.get("quality") or {}
        runtime = item.get("runtime") or {}
        guardrails = item.get("guardrails") or {}
        key = item["key"]
        classes: list[str] = []
        badges: list[str] = []
        if key in pareto:
            classes.append("pareto")
            badges.append("Pareto")
        if recommendations.get(provider) == item["id"]:
            classes.append("recommended")
            badges.append("模型内推荐")
        if item.get("passes_all_guardrails"):
            classes.append("pass")
        if not item.get("eligible_for_final"):
            classes.append("excluded")
        guardrail_text = " / ".join(
            f"{label}{'通过' if guardrails.get(field) else '未过'}"
            for field, label in (
                ("foreground_retention", "前景"),
                ("background_control", "背景"),
                ("green_edge", "绿边"),
            )
        )
        exclusion = "；".join(item.get("exclusion_reasons") or [])
        selection_text = "可候选" if item.get("eligible_for_final") else exclusion or "不进入最终选择"
        provider_label = html.escape(provider)
        source = doc_links.get(provider, "")
        if source:
            provider_label = (
                f'<a href="{html.escape(source)}" target="_blank" rel="noreferrer">'
                f"{provider_label}</a>"
            )
        rows.append(
            f"""
            <tr data-provider="{html.escape(provider)}" data-eligible="{str(bool(item.get('eligible_for_final'))).lower()}" class="{' '.join(classes)}">
              <td><strong>{provider_label}</strong><small>{html.escape(item['id'])}</small></td>
              <td><span class="badges">{' '.join(f'<b>{html.escape(badge)}</b>' for badge in badges)}</span>{html.escape(compact_parameters(item.get('parameters') or {}))}</td>
              <td>{fmt(quality.get('pseudo_mae'))}</td>
              <td>{fmt(quality.get('background_alpha_mean'))}</td>
              <td>{fmt(quality.get('foreground_loss_mean'))}</td>
              <td>{fmt(quality.get('green_fringe'))}</td>
              <td>{fmt(quality.get('fragment_pct'), 3, '%')}</td>
              <td>{fmt(runtime.get('mean_inference_ms'), 1, ' ms')}</td>
              <td>{fmt(runtime.get('end_to_end_ms'), 1, ' ms')}</td>
              <td>{fmt(item.get('temporal_alpha_mae'), 5)}</td>
              <td class="guardrails">{html.escape(guardrail_text)}</td>
              <td class="selection" title="{html.escape(selection_text)}">{html.escape(selection_text)}</td>
            </tr>
            """
        )

    provider_buttons = "".join(
        f'<button type="button" data-provider="{html.escape(name)}">{html.escape(name)}</button>'
        for name in provider_names
    )
    eligible_count = sum(
        1 for item in payload.get("configs", []) if item.get("eligible_for_final")
    )
    guardrail_count = sum(
        1
        for item in payload.get("configs", [])
        if item.get("passes_all_guardrails") and item.get("eligible_for_final")
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>宠物抠图模型参数调优</title>
<style>
:root{{--bg:#f3f5f2;--panel:#fff;--line:#d8ded9;--text:#17201c;--muted:#68736d;--green:#176b55;--warm:#965b3d;--soft:#edf6f2}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 Inter,"Microsoft YaHei",sans-serif;letter-spacing:0}}a{{color:inherit}}button{{font:inherit}}
.shell{{width:min(1680px,calc(100% - 32px));margin:auto;padding:24px 0 60px}}h1{{margin:3px 0;font-size:28px}}h2{{margin:26px 0 10px;font-size:18px}}p{{margin:0;color:var(--muted)}}
.top{{display:flex;justify-content:space-between;gap:28px;align-items:end;padding-bottom:18px;border-bottom:1px solid var(--line)}}.top>p{{max-width:520px}}.eyebrow{{color:var(--green);font-size:11px;font-weight:700}}
.decision{{display:grid;grid-template-columns:minmax(260px,1.4fr) repeat(3,minmax(150px,.6fr));gap:8px;margin:16px 0}}.decision>div{{border:1px solid var(--line);background:var(--panel);padding:12px;border-radius:7px}}.decision .primary{{background:var(--soft);border-color:#acd0c4}}.decision strong{{display:block;font-size:20px}}.decision span{{color:var(--muted)}}
.section-title{{display:flex;align-items:center;justify-content:space-between;gap:12px}}.background-controls{{display:flex;gap:5px}}.background-controls button{{padding:5px 9px}}
.preview-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.preview-card{{overflow:hidden;border:1px solid var(--line);border-radius:8px;background:var(--panel)}}.preview-card header{{display:flex;justify-content:space-between;gap:10px;align-items:start;padding:10px 12px;border-bottom:1px solid var(--line)}}small{{display:block;color:var(--muted)}}.preview-card header span{{padding:2px 6px;border-radius:5px;white-space:nowrap;font-size:10px}}.preview-card header .ok{{color:var(--green);background:#e5f3ee}}.preview-card header .warn{{color:#7d4a31;background:#f6ebe5}}.preview-media{{height:220px;background-color:#f7f8f7;background-image:linear-gradient(45deg,#e9ece9 25%,transparent 25%),linear-gradient(-45deg,#e9ece9 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#e9ece9 75%),linear-gradient(-45deg,transparent 75%,#e9ece9 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0}}.preview-grid[data-background="white"] .preview-media{{background:#fff;background-image:none}}.preview-grid[data-background="black"] .preview-media{{background:#111;background-image:none}}.preview-media img{{width:100%;height:100%;object-fit:contain}}.missing{{display:grid;height:100%;place-items:center;color:var(--muted)}}dl{{display:grid;grid-template-columns:repeat(3,1fr);margin:0;padding:9px 12px;gap:8px}}dt{{color:var(--muted);font-size:10px}}dd{{margin:1px 0 0;font-weight:700;font-variant-numeric:tabular-nums}}
.controls{{position:sticky;top:0;z-index:3;display:flex;gap:6px;overflow:auto;padding:10px 0;background:rgba(243,245,242,.96)}}button{{border:1px solid var(--line);border-radius:6px;background:#fff;padding:7px 10px;white-space:nowrap;cursor:pointer}}button.active{{color:#fff;background:var(--green);border-color:var(--green)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}table{{width:100%;min-width:1650px;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid var(--line);vertical-align:top;text-align:right;font-variant-numeric:tabular-nums}}th{{position:sticky;top:47px;z-index:2;background:#f8faf8;color:var(--muted);font-size:11px}}th:first-child,th:nth-child(2),td:first-child,td:nth-child(2),td:last-child{{text-align:left}}td:first-child{{min-width:190px}}td:nth-child(2){{max-width:380px}}tr.recommended{{background:#f1faf7}}tr.pareto td:first-child{{box-shadow:inset 3px 0 var(--warm)}}tr.excluded{{color:#6d7470;background:#fafafa}}.badges b{{display:inline-block;margin:0 5px 3px 0;padding:1px 5px;border:1px solid #9bcdbf;border-radius:5px;color:var(--green);font-size:10px}}.guardrails{{white-space:nowrap}}.selection{{max-width:250px}}.note{{padding-top:14px}}
@media(max-width:980px){{.top{{display:block}}.decision{{grid-template-columns:repeat(2,1fr)}}.preview-grid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:620px){{.decision,.preview-grid{{grid-template-columns:1fr}}.preview-media{{height:190px}}}}
</style></head><body><main class="shell">
<section class="top"><div><span class="eyebrow">PARAMETER SWEEP · 2026-07-11</span><h1>宠物抠图模型参数调优</h1><p>同一组宠物静态帧与 24 帧快走序列，统一质量指标、GPU 串行计时和人工毛发检查。</p></div><p>不使用单一综合分。二值掩膜可以通过删除细毛获得漂亮数值，因此最终候选还必须满足软 alpha、商用许可和人工视觉检查。</p></section>
<section class="decision"><div class="primary"><span>当前生产结论</span><strong>自研绿幕算法 + core despill 1.10</strong><span>受控绿幕素材质量、速度与成本最均衡；BiRefNet General 作为非绿幕或失败样本的学习型后备。</span></div><div><strong>{payload.get('provider_count', 0)}</strong><span>模型/provider</span></div><div><strong>{payload.get('config_count', 0)}</strong><span>实测参数配置</span></div><div><strong>{guardrail_count} / {eligible_count}</strong><span>通过门槛 / 可选配置</span></div></section>
<div class="section-title"><h2>各模型推荐参数实图</h2><div class="background-controls" aria-label="预览背景"><button type="button" data-background="checker" class="active">棋盘格</button><button type="button" data-background="white">白底</button><button type="button" data-background="black">黑底</button></div></div><section class="preview-grid" data-background="checker">{''.join(preview_cards)}</section>
<h2>完整参数结果</h2><nav class="controls"><button type="button" data-provider="all" class="active">全部</button><button type="button" data-provider="eligible">仅生产候选</button>{provider_buttons}</nav>
<section class="table-wrap"><table><thead><tr><th>模型 / 配置</th><th>参数</th><th>pseudo MAE</th><th>背景 alpha</th><th>前景损失</th><th>绿边</th><th>碎片率</th><th>模型推理</th><th>端到端</th><th>时序误差</th><th>质量门槛</th><th>选择约束</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<p class="note">门槛以调优后的自研基线为参照：前景损失不超过 1.5 倍；背景 alpha 不超过 4 倍或 0.004；绿边不超过 1.5 倍。生产 Pareto 同时考虑 pseudo MAE、前景损失、背景泄漏、绿边和推理时间，并先排除二值、非商用及人工复核失败配置。</p>
</main><script>
const buttons=[...document.querySelectorAll('button[data-provider]')];const rows=[...document.querySelectorAll('tbody tr[data-provider]')];
buttons.forEach(button=>button.addEventListener('click',()=>{{buttons.forEach(item=>item.classList.toggle('active',item===button));const selected=button.dataset.provider;rows.forEach(row=>row.hidden=selected==='eligible'?row.dataset.eligible!=='true':selected!=='all'&&row.dataset.provider!==selected)}}));
const backgroundButtons=[...document.querySelectorAll('button[data-background]')];const previewGrid=document.querySelector('.preview-grid');backgroundButtons.forEach(button=>button.addEventListener('click',()=>{{backgroundButtons.forEach(item=>item.classList.toggle('active',item===button));previewGrid.dataset.background=button.dataset.background}}));
</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
