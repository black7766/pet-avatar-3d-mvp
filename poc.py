#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""萌宠动态形象 PoC — 照片 → 风格化形象 → 状态首帧 → 透明动态 WebP

用法（在本目录执行，密钥读 .env）：
  python3 poc.py --pet pet1 --step stylize --style cute      # 出 4 张候选（real|cute|figurine）
  python3 poc.py --pet pet1 --step stylize --style all       # 三档矩阵全跑（每档 2 张）
  python3 poc.py --pet pet1 --choose cute_2                  # 把候选 cute_2 定为 chosen.png
  python3 poc.py --pet pet1 --step state_sheet --style real   # 一张总表裁出 idle/fast_walk/sleep 首帧
  python3 poc.py --pet pet1 --step animate --clip all        # 逐段生成四状态视频片段
  python3 poc.py --pet pet1 --step matte --clip all          # 各段抠图合成 anim_<clip>.webp

输入：inputs/<pet>.jpg（同宠多角度可加 <pet>_2.jpg <pet>_3.jpg，自动并入多图参考）
输出：poc_output/<pet>/
机制：新流程用一张状态总表裁出 chosen.png/state_fast_walk.png/state_sleep.png，再生成直接循环态视频。
计划：doc/计划/2026-06-11_萌宠动态形象_块0_PoC脚本_执行计划.md
"""
import argparse
import base64
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "poc_output"

ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
# 2026-06-11 经 GET /models + 实弹验证的模型 ID
MODEL_STYLIZE = "doubao-seedream-4-5-251128"        # 图生图主测（2K 原生，最小 368 万像素）
MODEL_STYLIZE_ALT = "doubao-seedream-5-0-260128"    # 对照组
MODEL_ANIMATE = "doubao-seedance-1-5-pro-251215"    # 已验证支持首尾帧 flf2v
IMAGE_SIZE = "2048x2048"
IMAGE_SIZE_PX = 2048
CLIP_ORDER = ("idle", "fast_walk", "sleep")
STATE_FRAME_CLIPS = {"sleep", "fast_walk"}
STATE_SHEET_CLIPS = ("idle", "fast_walk", "sleep")
N_CANDIDATES = int(os.environ.get("PETAVATAR_CANDIDATES", "4"))   # 单档候选数；--style all 时每档 2 张

# 抠图参数（matte 时按首帧角点实采绿色作 key，这里是容差）
# 2026-06-11 冒烟扫参定版：0.10/0.02 = 主体完整+阴影被吃+胡须保留；
# ≥0.14 浅色毛区破洞，≤0.06 脚下阴影残留
CHROMA_SIMILARITY_STEPS = tuple(
    x.strip() for x in os.environ.get("PETAVATAR_CHROMA_STEPS", "0.085,0.105,0.125,0.145").split(",")
    if x.strip()
)
CHROMA_BLEND = os.environ.get("PETAVATAR_CHROMA_BLEND", "0.04")
WEBP_FPS = int(os.environ.get("PETAVATAR_WEBP_FPS", "24"))
WEBP_WIDTH = int(os.environ.get("PETAVATAR_WEBP_WIDTH", "640"))
LOOP_CLOSE_FRAMES = int(os.environ.get("PETAVATAR_LOOP_CLOSE_FRAMES", "8"))
LOOP_TRIM_HEAD_FRAMES = int(os.environ.get("PETAVATAR_LOOP_TRIM_HEAD_FRAMES", "6"))
LOOP_TRIM_TAIL_FRAMES = int(os.environ.get("PETAVATAR_LOOP_TRIM_TAIL_FRAMES", "6"))
WEBP_QUALITY = int(os.environ.get("PETAVATAR_WEBP_QUALITY", "90"))
WEBP_METHOD = int(os.environ.get("PETAVATAR_WEBP_METHOD", "2"))
WEBP_SUBJECT_WIDTH_RATIO = float(os.environ.get("PETAVATAR_SUBJECT_WIDTH_RATIO", "0.86"))
WEBP_SUBJECT_HEIGHT_RATIO = float(os.environ.get("PETAVATAR_SUBJECT_HEIGHT_RATIO", "0.78"))
WEBP_BOTTOM_MARGIN = int(os.environ.get("PETAVATAR_BOTTOM_MARGIN", "38"))
WEBP_EDGE_RISK_MARGIN = int(os.environ.get("PETAVATAR_EDGE_RISK_MARGIN", "8"))
STATE_FRAME_ATTEMPTS = max(1, int(os.environ.get("PETAVATAR_STATE_ATTEMPTS", "3")))
STATE_SAFE_MARGIN_RATIO = float(os.environ.get("PETAVATAR_STATE_SAFE_MARGIN_RATIO", "0.04"))

sys.path.insert(0, str(ROOT))
from prompts import (  # noqa: E402
    CLIP_PROMPTS, DEFAULT_CLIP, DEFAULT_STYLE, STATE_FRAME_PROMPTS,
    STATE_SHEET_PROMPTS, STYLE_PROMPTS,
)


def load_env():
    env = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.lstrip("\ufeff")
                env[k.strip()] = v.strip()
    if os.environ.get("ARK_API_KEY"):
        env["ARK_API_KEY"] = os.environ["ARK_API_KEY"]
    return env


ENV = load_env()


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("Missing ffmpeg. Install imageio-ffmpeg or put ffmpeg in PATH.") from exc


def ark_request(path, body=None, method="POST", timeout=300, retries=2):
    """i2i 带多参考图实测 30~60s+，timeout 必须宽；超时/限流/5xx 自动重试。"""
    url = ARK_BASE + path
    payload = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=payload, method=method)
        req.add_header("Authorization", "Bearer " + ENV["ARK_API_KEY"])
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                last = f"HTTP {e.code}: {detail[:200]}"
                print(f"[retry] {path} 第 {attempt + 1} 次失败（{last}），{10 * (attempt + 1)}s 后重试…")
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"Ark HTTP {e.code}: {detail[:500]}") from e
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
            if attempt < retries:
                last = repr(e)
                print(f"[retry] {path} 第 {attempt + 1} 次失败（{last}），{10 * (attempt + 1)}s 后重试…")
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Ark 重试耗尽: {last}")


def download(url, dest: Path, retries=3):
    """下载产物。SSL EOF 类瞬断实测会出现在任意环节（2026-06-11），必须重试。"""
    tmp = dest.with_name(dest.name + ".download")
    for attempt in range(retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            if not tmp.exists() or tmp.stat().st_size <= 0:
                raise RuntimeError(f"download produced empty file: {dest.name}")
            tmp.replace(dest)
            return
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                print(f"[retry] 下载第 {attempt + 1} 次失败（{e!r:.120}），{8 * (attempt + 1)}s 后重试…")
                time.sleep(8 * (attempt + 1))
                continue
            raise
        except RuntimeError as e:
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                print(f"[retry] 下载第 {attempt + 1} 次失败（{e!r:.120}），{8 * (attempt + 1)}s 后重试…")
                time.sleep(8 * (attempt + 1))
                continue
            raise


def strip_audio_track(video: Path):
    """源 MP4 也必须无声；展示端虽然用 WebP，但保留 raw 文件时不携带音频。"""
    tmp = video.with_name(video.stem + "_silent_tmp" + video.suffix)
    subprocess.run([
        ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
        "-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart", str(tmp)
    ], check=True)
    tmp.replace(video)


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def find_inputs(pet: str) -> list:
    """主照片 <pet>.* 必须有；<pet>_2.* <pet>_3.* 可选（多角度参考，提升还原度）。"""
    exts = ("jpg", "jpeg", "png", "heic")
    photos = []
    for ext in exts:
        p = INPUTS / f"{pet}.{ext}"
        if p.exists():
            photos.append(p)
            break
    if not photos:
        sys.exit(f"找不到输入照片 inputs/{pet}.jpg|jpeg|png")
    for n in (2, 3):
        for ext in exts:
            p = INPUTS / f"{pet}_{n}.{ext}"
            if p.exists():
                photos.append(p)
                break
    return photos


def record_metric(pet_dir: Path, step: str, info: dict):
    mf = pet_dir / "metrics.json"
    m = json.loads(mf.read_text()) if mf.exists() else {}
    rows = m.setdefault(step, [])
    key = None
    if "clip" in info:
        key = ("clip", info["clip"])
    elif "file" in info:
        key = ("file", info["file"])
    if key:
        field, value = key
        for idx, row in enumerate(rows):
            if row.get(field) == value:
                rows[idx] = info
                break
        else:
            rows.append(info)
    else:
        rows.append(info)
    mf.write_text(json.dumps(m, ensure_ascii=False, indent=2))


def subject_bbox_from_green(path: Path, green_margin=24):
    """Return subject bbox and margins for green-background state frames."""
    import cv2
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green = (g > r + green_margin) & (g > b + green_margin) & (g > 90)
    subject = ~green
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        subject.astype(np.uint8), 8
    )
    if component_count <= 1:
        return None
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == main
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = subject.shape
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    return {
        "bbox": [x0, y0, x1, y1],
        "margins": {
            "left": x0,
            "top": y0,
            "right": w - x1,
            "bottom": h - y1,
        },
        "size": [w, h],
    }


def state_frame_safe(path: Path):
    info = subject_bbox_from_green(path)
    if not info:
        return False, None
    w, h = info["size"]
    min_margin = int(min(w, h) * STATE_SAFE_MARGIN_RATIO)
    margins = info["margins"]
    ok = min(margins.values()) >= min_margin
    info["min_required_margin"] = min_margin
    info["safe"] = ok
    return ok, info


# ---------- step 1: stylize ----------

def step_stylize(pet: str, style: str, model: str = MODEL_STYLIZE):
    photos = find_inputs(pet)
    pet_dir = OUTPUT / pet
    cand_dir = pet_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    uris = [data_uri(p) for p in photos]
    styles = list(STYLE_PROMPTS) if style == "all" else [style]
    per_style = 2 if style == "all" else N_CANDIDATES
    print(f"[stylize] 输入 {len(photos)} 张照片，档位 {styles} × {per_style} 张")
    for st in styles:
        for i in range(1, per_style + 1):
            dest = cand_dir / f"{st}_{i}.jpeg"
            if dest.exists():
                print(f"[stylize] 跳过已存在 {dest.name}（断点续跑）")
                continue
            t0 = time.time()
            body = {
                "model": model,
                "prompt": STYLE_PROMPTS[st],
                "image": uris if len(uris) > 1 else uris[0],
                "size": IMAGE_SIZE,
                "response_format": "url",
                "watermark": False,
            }
            resp = ark_request("/images/generations", body)
            download(resp["data"][0]["url"], dest)
            dt = round(time.time() - t0, 1)
            print(f"[stylize] 候选 {dest.name} 完成 {dt}s")
            record_metric(pet_dir, "stylize", {"model": model, "style": st,
                                               "file": dest.name, "seconds": dt,
                                               "usage": resp.get("usage")})
    print(f"[stylize] 完成 → {cand_dir}/，人工挑选后执行：python3 poc.py --pet {pet} --choose <档位_序号>")


def normalize_bg(src: Path, dest: Path, margin=12):
    """把生成图的背景统一归一化为标准亮绿。

    生成模型给的绿色不可控（real 档实测深绿 0x09xxxx，色度太弱导致 chromakey
    连灰猫奶油狗一起抠掉）；朴素 floodfill 也翻车（深绿 seed 与宠物暗部阴影在
    阈值内连通，灌进胸口）。2026-06-11 两次翻车后定版算法：
      背景 := 「绿色显著占优（G>R+margin 且 G>B+margin）」∩「与图像边界连通」
    连通性保证宠物内部的绿色系像素（如猫的黄绿眼睛）不被误伤。"""
    import cv2
    import numpy as np
    from PIL import Image
    img = Image.open(src).convert("RGB")
    arr = np.asarray(img).astype(np.int16)
    g_dom = (arr[:, :, 1] > arr[:, :, 0] + margin) & (arr[:, :, 1] > arr[:, :, 2] + margin)
    _, labels = cv2.connectedComponents(g_dom.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    border = border[border != 0]
    bg = np.isin(labels, border)
    # 背景向内吃 1px，消掉暗绿描边 rim
    bg = cv2.dilate(bg.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    out = np.asarray(img).copy()
    out[bg] = (0, 255, 0)
    ratio = bg.mean()
    Image.fromarray(out).save(dest)
    print(f"[normalize] 背景占比 {ratio:.0%} → 标准亮绿（异常时检查：<30% 或 >90% 都可疑）")


def safe_pad_green_frame(path: Path, min_margin_ratio=0.12):
    """Shrink green-screen state frames when the generated pet is too close to an edge."""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green = (g > r + 24) & (g > b + 24) & (g > 90)
    subject = ~green
    ys, xs = np.where(subject)
    if len(xs) == 0:
        return
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    min_margin = int(min(w, h) * min_margin_ratio)
    margins = (x0, y0, w - x1, h - y1)
    if min(margins) >= min_margin:
        return

    crop = img.crop((x0, y0, x1, y1))
    crop_w, crop_h = crop.size
    max_w = max(1, w - 2 * min_margin)
    max_h = max(1, h - 2 * min_margin)
    scale = min(max_w / crop_w, max_h / crop_h, 1.0)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    if (new_w, new_h) != crop.size:
        crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), (0, 255, 0))
    canvas.paste(crop, ((w - new_w) // 2, (h - new_h) // 2))
    canvas.save(path)
    print(f"[safe_pad] {path.name} margins={margins} -> min_margin={min_margin}, scale={scale:.3f}")


def step_choose(pet: str, stem: str):
    pet_dir = OUTPUT / pet
    pick = pet_dir / "candidates" / f"{stem}.jpeg"
    if not pick.exists():
        cands = sorted((pet_dir / "candidates").glob("*.jpeg"))
        sys.exit(f"候选 {stem} 不存在：{[c.stem for c in cands]}")
    normalize_bg(pick, pet_dir / "chosen.png")
    print(f"[choose] {pick.name} → 背景归一化标准亮绿 → chosen.png（动作片段库的统一首尾帧）")


def step_state_sheet(pet: str, style: str, model: str = MODEL_STYLIZE):
    """Generate one pose sheet, then crop idle/fast_walk/sleep first frames locally."""
    if style not in STATE_SHEET_PROMPTS:
        sys.exit(f"未知 state sheet style: {style}")
    photos = find_inputs(pet)
    pet_dir = OUTPUT / pet
    pet_dir.mkdir(parents=True, exist_ok=True)
    uris = [data_uri(p) for p in photos]
    sheet = pet_dir / f"state_sheet_{style}.jpeg"
    t0 = time.time()
    resp = None
    if sheet.exists():
        print(f"[state_sheet] skip existing {sheet.name}")
    else:
        body = {
            "model": model,
            "prompt": STATE_SHEET_PROMPTS[style],
            "image": uris if len(uris) > 1 else uris[0],
            "size": IMAGE_SIZE,
            "response_format": "url",
            "watermark": False,
        }
        resp = ark_request("/images/generations", body)
        download(resp["data"][0]["url"], sheet)
    crop_state_sheet(pet_dir, sheet)
    dt = round(time.time() - t0, 1)
    cell_bounds = {}
    for clip in STATE_SHEET_CLIPS:
        target = pet_dir / ("chosen.png" if clip == DEFAULT_CLIP else f"state_{clip}.png")
        ok, bounds = state_frame_safe(target)
        cell_bounds[clip] = {"safe": ok, "bounds": bounds}
        print(f"[state_sheet:{clip}] safe={ok} bounds={bounds}")
    record_metric(
        pet_dir,
        "state_sheet",
        {
            "model": model,
            "style": style,
            "file": sheet.name,
            "seconds": dt,
            "cells": list(STATE_SHEET_CLIPS),
            "bounds": cell_bounds,
            "usage": resp.get("usage") if resp else None,
        },
    )
    print(f"[state_sheet] {sheet.name} -> chosen/state_fast_walk/state_sleep done {dt}s")


def crop_state_sheet(pet_dir: Path, sheet: Path):
    from PIL import Image

    img = Image.open(sheet).convert("RGB")
    w, h = img.size
    mid_x, mid_y = w // 2, h // 2
    cells = {
        "idle": (0, 0, mid_x, mid_y),
        "fast_walk": (mid_x, 0, w, mid_y),
        "sleep": (0, mid_y, mid_x, h),
    }
    for clip, box in cells.items():
        crop = img.crop(box).resize((IMAGE_SIZE_PX, IMAGE_SIZE_PX), Image.Resampling.LANCZOS)
        tmp = pet_dir / f"state_{clip}_sheet_crop.jpeg"
        crop.save(tmp, quality=96)
        dest = pet_dir / ("chosen.png" if clip == DEFAULT_CLIP else f"state_{clip}.png")
        normalize_bg(tmp, dest)
        safe_pad_green_frame(dest)
        if clip == DEFAULT_CLIP:
            shutil.copyfile(dest, pet_dir / "state_idle.png")
        tmp.unlink(missing_ok=True)


def _legacy_step_state_frames(pet: str, clips: list[str], model: str = MODEL_STYLIZE):
    pet_dir = OUTPUT / pet
    chosen = pet_dir / "chosen.png"
    if not chosen.exists():
        sys.exit("缺 chosen.png：先跑 --step stylize 并 --choose")
    uri = data_uri(chosen)
    for clip in clips:
        if clip == DEFAULT_CLIP:
            state_dest = pet_dir / f"state_{clip}.png"
            if not state_dest.exists():
                shutil.copyfile(chosen, state_dest)
                print(f"[state:{clip}] 复用 chosen.png → {state_dest.name}")
            else:
                print(f"[state:{clip}] 跳过已存在 {state_dest.name}")
            continue
        if clip not in STATE_FRAME_PROMPTS:
            print(f"[state:{clip}] 无状态首帧提示词，继续使用 chosen.png")
            continue
        state_dest = pet_dir / f"state_{clip}.png"
        if state_dest.exists():
            print(f"[state:{clip}] 跳过已存在 {state_dest.name}（要重生先删该文件）")
            continue
        raw_dest = pet_dir / f"state_{clip}_raw.jpeg"
        t0 = time.time()
        body = {
            "model": model,
            "prompt": STATE_FRAME_PROMPTS[clip],
            "image": uri,
            "size": IMAGE_SIZE,
            "response_format": "url",
            "watermark": False,
        }
        resp = ark_request("/images/generations", body)
        download(resp["data"][0]["url"], raw_dest)
        normalize_bg(raw_dest, state_dest)
        safe_pad_green_frame(state_dest)
        dt = round(time.time() - t0, 1)
        print(f"[state:{clip}] {state_dest.name} 完成 {dt}s")
        record_metric(pet_dir, "state_frame", {"model": model, "clip": clip,
                                               "file": state_dest.name, "seconds": dt,
                                               "usage": resp.get("usage")})


# ---------- step 2: animate（动作片段库）----------

def step_state_frames(pet: str, clips: list[str], model: str = MODEL_STYLIZE):
    pet_dir = OUTPUT / pet
    chosen = pet_dir / "chosen.png"
    if not chosen.exists():
        sys.exit("missing chosen.png: run --step stylize and --choose first")
    uri = data_uri(chosen)
    for clip in clips:
        if clip == DEFAULT_CLIP:
            state_dest = pet_dir / f"state_{clip}.png"
            if not state_dest.exists():
                shutil.copyfile(chosen, state_dest)
                print(f"[state:{clip}] copied chosen.png -> {state_dest.name}")
            else:
                print(f"[state:{clip}] skip existing {state_dest.name}")
            continue
        if clip not in STATE_FRAME_PROMPTS:
            print(f"[state:{clip}] no state-frame prompt; keep using chosen.png")
            continue

        state_dest = pet_dir / f"state_{clip}.png"
        if state_dest.exists():
            ok, bounds = state_frame_safe(state_dest)
            if ok:
                print(f"[state:{clip}] skip existing {state_dest.name}; safe margins")
            else:
                print(f"[state:{clip}] skip existing {state_dest.name}; edge risk {bounds}; delete it to regenerate")
            continue

        image_input = uri
        ref_prompt = ""
        if clip == "sleep":
            pose_ref = pet_dir / "sleep_pose.png"
            if pose_ref.exists():
                image_input = [uri, data_uri(pose_ref)]
                ref_prompt = (
                    " Multiple input images are provided: image 1 controls identity, style, fur, markings, and face; "
                    "image 2 controls only the low prone sleeping pose and full-body composition. "
                    "Do not copy image 2 background or cropping; keep full tail and full body safely inside the green canvas."
                )

        raw_dest = pet_dir / f"state_{clip}_raw.jpeg"
        t0 = time.time()
        resp = None
        ok = False
        bounds = None
        prompt = STATE_FRAME_PROMPTS[clip] + ref_prompt
        attempt = 0
        for attempt in range(1, STATE_FRAME_ATTEMPTS + 1):
            attempt_dest = raw_dest if attempt == 1 else pet_dir / f"state_{clip}_raw_attempt{attempt}.jpeg"
            body = {
                "model": model,
                "prompt": prompt,
                "image": image_input,
                "size": IMAGE_SIZE,
                "response_format": "url",
                "watermark": False,
            }
            resp = ark_request("/images/generations", body)
            download(resp["data"][0]["url"], attempt_dest)
            normalize_bg(attempt_dest, state_dest)
            safe_pad_green_frame(state_dest)
            ok, bounds = state_frame_safe(state_dest)
            print(f"[state:{clip}] attempt {attempt}/{STATE_FRAME_ATTEMPTS} safe={ok} bounds={bounds}")
            if ok:
                break
            prompt = (
                STATE_FRAME_PROMPTS[clip]
                + " Retry constraint: pull the camera much farther back. The full pet body must stay inside the central safe area. "
                  "Ears, paws, body, and tail tip must not touch any image edge. Keep at least 15% pure green margin on every side. "
                  "The full tail must be visible, with the tail tip at least 15% away from the right edge. No cropping."
            )

        dt = round(time.time() - t0, 1)
        print(f"[state:{clip}] {state_dest.name} done {dt}s safe={ok}")
        record_metric(
            pet_dir,
            "state_frame",
            {
                "model": model,
                "clip": clip,
                "file": state_dest.name,
                "seconds": dt,
                "safe": ok,
                "bounds": bounds,
                "attempts": attempt,
                "usage": resp.get("usage") if resp else None,
            },
        )


def valid_raw_video(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def clip_first_frame(pet_dir: Path, clip: str) -> Path:
    state = pet_dir / f"state_{clip}.png"
    if state.exists():
        return state
    return pet_dir / "chosen.png"


def animate_request_body(uri: str, clip: str) -> dict:
    content = [
        {"type": "text", "text": CLIP_PROMPTS[clip]},
        {"type": "image_url", "image_url": {"url": uri}, "role": "first_frame"},
    ]
    if clip not in STATE_FRAME_CLIPS:
        content.append({"type": "image_url", "image_url": {"url": uri}, "role": "last_frame"})
    return {
        "model": MODEL_ANIMATE,
        "generate_audio": False,
        "content": content,
    }


SHEET_VIDEO_PROMPT = (
    "Generate a single 2x2 animated pet asset sheet video for a mobile app. "
    "Use the provided first frame exactly as the layout reference. Keep four equal quadrants, fixed camera, no zoom, no pan, no text labels, no borders, no props, no people, no extra animals. "
    "Each quadrant must stay inside its own cell and must not cross cell boundaries. The background in every cell stays pure bright green #00FF00. "
    "Top-left quadrant: idle loop, pet looks at the user with subtle breathing and blinking. "
    "Top-right quadrant: fast_walk loop, brisk in-place walking cycle with alternating legs, not running or jumping. "
    "Bottom-left quadrant: sleep loop, pet already lying down asleep, only gentle breathing, no wake-up or transition. "
    "Bottom-right quadrant: keep the duplicated idle reference stable and unobtrusive; it can move subtly but must not affect the other cells. "
    "The entire video must be seamless-loop friendly: no cut back to a different pose, no global camera movement, no black floor shadow, no contact shadow, no dark motion trail. "
    "--resolution 720p --duration 5 --camerafixed true --watermark false --generate_audio false"
)


def state_sheet_for_pet(pet_dir: Path) -> Path:
    sheets = sorted(pet_dir.glob("state_sheet_*.jpeg")) + sorted(pet_dir.glob("state_sheet_*.jpg")) + sorted(pet_dir.glob("state_sheet_*.png"))
    if sheets:
        return sheets[0]
    sys.exit(f"[animate_sheet] missing state_sheet_*.jpeg in {pet_dir}")


def sheet_video_request_body(uri: str) -> dict:
    return {
        "model": MODEL_ANIMATE,
        "generate_audio": False,
        "content": [
            {"type": "text", "text": SHEET_VIDEO_PROMPT},
            {"type": "image_url", "image_url": {"url": uri}, "role": "first_frame"},
        ],
    }


def crop_sheet_video(pet_dir: Path, sheet_video: Path, task_id: str | None, task_status: dict | None, seconds: float):
    crops = {
        "idle": "crop=iw/2:ih/2:0:0,scale=720:720:flags=lanczos",
        "fast_walk": "crop=iw/2:ih/2:iw/2:0,scale=720:720:flags=lanczos",
        "sleep": "crop=iw/2:ih/2:0:ih/2,scale=720:720:flags=lanczos",
    }
    usage = task_status.get("usage") if task_status else None
    for idx, (clip, vf) in enumerate(crops.items()):
        raw_video = pet_dir / f"raw_{clip}.mp4"
        subprocess.run(
            [
                ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(sheet_video),
                "-vf",
                vf,
                "-an",
                str(raw_video),
            ],
            check=True,
        )
        record_metric(
            pet_dir,
            "animate",
            {
                "model": MODEL_ANIMATE,
                "clip": clip,
                "task": task_id,
                "seconds": seconds,
                "generate_audio": False,
                "shared_sheet": True,
                "usage": usage if idx == 0 else None,
            },
        )
        print(f"[animate_sheet:{clip}] cropped -> {raw_video.name}")


def step_animate_sheet(pet: str):
    pet_dir = OUTPUT / pet
    sheet = state_sheet_for_pet(pet_dir)
    raw_sheet = pet_dir / "raw_sheet.mp4"
    task_status = None
    task_id = None
    dt = 0.0
    if valid_raw_video(raw_sheet):
        print("[animate_sheet] skip existing raw_sheet.mp4")
    else:
        if raw_sheet.exists():
            raw_sheet.unlink()
        t0 = time.time()
        task = ark_request("/contents/generations/tasks", sheet_video_request_body(data_uri(sheet)))
        task_id = task["id"]
        print(f"[animate_sheet] task {task_id} created")
        while True:
            time.sleep(15)
            st = ark_request(f"/contents/generations/tasks/{task_id}", method="GET")
            status = st.get("status")
            print(f"[animate_sheet] {status}")
            if status == "succeeded":
                task_status = st
                (pet_dir / "task_sheet.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
                download(st["content"]["video_url"], raw_sheet)
                strip_audio_track(raw_sheet)
                dt = round(time.time() - t0, 1)
                break
            if status in ("failed", "cancelled"):
                sys.exit(f"[animate_sheet] task failed: {json.dumps(st, ensure_ascii=False)[:500]}")
        record_metric(
            pet_dir,
            "animate_sheet",
            {
                "model": MODEL_ANIMATE,
                "file": raw_sheet.name,
                "task": task_id,
                "seconds": dt,
                "generate_audio": task_status.get("generate_audio") if task_status else False,
                "usage": task_status.get("usage") if task_status else None,
                "cells": list(STATE_SHEET_CLIPS),
            },
        )
        print(f"[animate_sheet] raw_sheet.mp4 saved and muted ({dt}s)")
    if task_status is None and (pet_dir / "task_sheet.json").exists():
        task_status = json.loads((pet_dir / "task_sheet.json").read_text(encoding="utf-8"))
        task_id = task_status.get("id")
        rows = json.loads((pet_dir / "metrics.json").read_text(encoding="utf-8")).get("animate_sheet", []) if (pet_dir / "metrics.json").exists() else []
        if rows:
            dt = rows[-1].get("seconds", 0.0)
    crop_sheet_video(pet_dir, raw_sheet, task_id, task_status, dt)


def finish_animate_task(pet_dir: Path, clip: str, task_id: str, task_status: dict, started_at: float):
    (pet_dir / f"task_{clip}.json").write_text(json.dumps(task_status, ensure_ascii=False, indent=2))
    raw_video = pet_dir / f"raw_{clip}.mp4"
    download(task_status["content"]["video_url"], raw_video)
    strip_audio_track(raw_video)
    dt = round(time.time() - started_at, 1)
    print(f"[animate:{clip}] raw_{clip}.mp4 已保存并去音轨（{dt}s）")
    record_metric(pet_dir, "animate", {"model": MODEL_ANIMATE, "clip": clip,
                                       "task": task_id, "seconds": dt,
                                       "generate_audio": task_status.get("generate_audio"),
                                       "usage": task_status.get("usage")})


def step_animate_many(pet: str, clips: list[str]):
    pet_dir = OUTPUT / pet
    pending = []
    for clip in clips:
        raw_video = pet_dir / f"raw_{clip}.mp4"
        if valid_raw_video(raw_video):
            print(f"[animate:{clip}] 跳过已存在 raw_{clip}.mp4（断点续跑；要重生先删该文件）")
            continue
        if raw_video.exists():
            print(f"[animate:{clip}] 删除空的 raw_{clip}.mp4 后重跑")
            raw_video.unlink()
        pending.append(clip)

    if not pending:
        return

    chosen = pet_dir / "chosen.png"
    if not chosen.exists():
        sys.exit("缺 chosen.png：先跑 --step stylize 并 --choose")

    tasks = {}
    for clip in pending:
        first_frame = clip_first_frame(pet_dir, clip)
        if not first_frame.exists():
            sys.exit(f"[animate:{clip}] 缺首帧图 {first_frame.name}")
        print(f"[animate:{clip}] first_frame={first_frame.name}")
        task = ark_request("/contents/generations/tasks", animate_request_body(data_uri(first_frame), clip))
        task_id = task["id"]
        tasks[clip] = {"id": task_id, "started_at": time.time()}
        print(f"[animate:{clip}] 任务 {task_id} 已创建，批量轮询中…")

    while tasks:
        time.sleep(15)
        for clip, info in list(tasks.items()):
            st = ark_request(f"/contents/generations/tasks/{info['id']}", method="GET")
            status = st.get("status")
            print(f"[animate:{clip}] {status}")
            if status == "succeeded":
                finish_animate_task(pet_dir, clip, info["id"], st, info["started_at"])
                del tasks[clip]
            elif status in ("failed", "cancelled"):
                sys.exit(f"[animate:{clip}] 任务失败：{json.dumps(st, ensure_ascii=False)[:500]}")


def step_animate(pet: str, clip: str):
    step_animate_many(pet, [clip])


# ---------- step 3: matte ----------

def sample_bg_color(video: Path) -> str:
    """从起始几帧的边缘区域取高置信绿色中位数作为 chromakey key 色。"""
    import numpy as np
    from PIL import Image

    tmp = video.parent / "_bg_sample"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                        "-vf", "fps=1,scale=320:-2", "-frames:v", "3",
                        str(tmp / "bg_%02d.png")], check=True)
        samples = []
        for frame in sorted(tmp.glob("bg_*.png")):
            arr = np.asarray(Image.open(frame).convert("RGB"))
            edge = np.concatenate([
                arr[:10, :, :].reshape(-1, 3),
                arr[-10:, :, :].reshape(-1, 3),
                arr[:, :10, :].reshape(-1, 3),
                arr[:, -10:, :].reshape(-1, 3),
            ], axis=0).astype(np.int16)
            r, g, b = edge[:, 0], edge[:, 1], edge[:, 2]
            high_conf = (g > r + 30) & (g > b + 30) & (g > 120)
            if high_conf.any():
                samples.append(edge[high_conf])
        if samples:
            rgb = np.median(np.concatenate(samples, axis=0), axis=0).astype(int)
            return f"0x{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ppm = video.parent / "_corner.ppm"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", "crop=8:8:4:4", "-frames:v", "1", str(ppm)], check=True)
    raw = ppm.read_bytes()
    parts = raw.split(b"\n", 3)
    px = parts[3]
    n = len(px) // 3
    r = sum(px[i * 3] for i in range(n)) // n
    g = sum(px[i * 3 + 1] for i in range(n)) // n
    b = sum(px[i * 3 + 2] for i in range(n)) // n
    ppm.unlink()
    return f"0x{r:02X}{g:02X}{b:02X}"


def assess_frames(frames_dir: Path):
    """抠图质检：返回 (绿残留最差占比, 中段帧可见像素占比)。
    残留高 = 容差不够（噪点/绿膜）；可见占比过低 = 抠过头（主体被吃，猫消失事故的熔断）。"""
    import numpy as np
    from PIL import Image
    pngs = sorted(frames_dir.glob("f_*.png"))
    take = pngs[::max(1, len(pngs) // 6)][:6]
    worst_green, mid_visible = 0.0, 1.0
    for idx, p in enumerate(take):
        a = np.asarray(Image.open(p).convert("RGBA")).astype(np.int16)
        vis = a[:, :, 3] > 16
        ratio = vis.mean()
        if idx == len(take) // 2:
            mid_visible = ratio
        if vis.sum() == 0:
            worst_green = 1.0
            continue
        greenish = ((a[:, :, 1] > a[:, :, 0] + 24) & (a[:, :, 1] > a[:, :, 2] + 24) & vis)
        worst_green = max(worst_green, greenish.sum() / vis.sum())
    return worst_green, mid_visible


def refine_rgba_frame(img):
    """压掉绿幕边缘残留。

    这里不做强力腐蚀，因为猫毛、胡须本来就是半透明细节；策略是只在 alpha 边缘区
    做 despill 和轻度收边，尽量把绿边藏掉，同时保留毛发的软边。
    """
    import cv2
    import numpy as np
    from PIL import Image

    arr = np.asarray(img.convert("RGBA")).astype(np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_rb = np.maximum(r, b)
    visible = alpha > 2

    # 全局去绿溢色：把明显高于红/蓝的绿色拉回到主体色附近。
    spill = visible & (g > max_rb + 8)
    g[spill] = max_rb[spill] + (g[spill] - max_rb[spill]) * 0.18

    # 只针对 alpha 边缘做轻度收边，避免把胡须和毛尖直接吃掉。
    mask = alpha > 6
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    edge = mask & (~eroded | (alpha < 232))
    green_rim = edge & (g > max_rb + 5)
    alpha[green_rim] *= 0.62
    g[green_rim] = max_rb[green_rim]

    # alpha 边缘轻微羽化，减少硬边框。
    soft_alpha = cv2.GaussianBlur(alpha, (3, 3), 0.55)
    alpha[edge] = alpha[edge] * 0.45 + soft_alpha[edge] * 0.55

    # Solidify / alpha bleed：半透明边缘 RGB 替换为最近主体核心色，减少黑边、脏边。
    solid = alpha > 220
    core = cv2.erode(solid.astype(np.uint8), kernel, iterations=2).astype(bool)
    if not core.any():
        core = solid
    fringe = (alpha > 0) & (alpha < 255)
    if core.any():
        filled = core.copy()
        bleed_rgb = np.where(core[:, :, None], rgb, 0)
        for _ in range(10):
            grow = cv2.dilate(filled.astype(np.uint8), kernel, iterations=1).astype(bool) & ~filled
            if not grow.any():
                break
            for c in range(3):
                channel = bleed_rgb[:, :, c]
                channel[grow] = cv2.dilate(channel, kernel, iterations=1)[grow]
                bleed_rgb[:, :, c] = channel
            filled |= grow
            if filled[fringe].all():
                break
        dark_edge = fringe & (rgb.max(axis=2) < 35)
        replace = fringe & (green_rim | dark_edge | (alpha < 210))
        rgb[replace] = bleed_rgb[replace]

        visible_mask = alpha > 8
        inner_visible = cv2.erode(visible_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
        silhouette = visible_mask & ~inner_visible
        dark_silhouette = silhouette & (rgb.mean(axis=2) < 82) & (alpha > 120)
        rgb[dark_silhouette] = bleed_rgb[dark_silhouette] * 0.72 + rgb[dark_silhouette] * 0.28
        alpha[dark_silhouette] *= 0.72

        # Remove dark high-alpha halos on the outer 1-3px contour. These are common after
        # chroma-keying Seedance output and are not caught by the semi-transparent fringe pass.
        outer1 = visible_mask & ~cv2.erode(visible_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        outer_visible = cv2.erode(visible_mask.astype(np.uint8), kernel, iterations=3).astype(bool)
        outer_band = visible_mask & ~outer_visible
        cur_luma = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
        edge_core = cv2.erode(solid.astype(np.uint8), kernel, iterations=5).astype(bool)
        if not edge_core.any():
            edge_core = core
        edge_filled = edge_core.copy()
        edge_bleed_rgb = np.where(edge_core[:, :, None], rgb, 0)
        for _ in range(20):
            grow = cv2.dilate(edge_filled.astype(np.uint8), kernel, iterations=1).astype(bool) & ~edge_filled
            if not grow.any():
                break
            for c in range(3):
                channel = edge_bleed_rgb[:, :, c]
                channel[grow] = cv2.dilate(channel, kernel, iterations=1)[grow]
                edge_bleed_rgb[:, :, c] = channel
            edge_filled |= grow
            if edge_filled[outer_band].all():
                break
        bleed_luma = edge_bleed_rgb[:, :, 0] * 0.299 + edge_bleed_rgb[:, :, 1] * 0.587 + edge_bleed_rgb[:, :, 2] * 0.114
        dark_outer = (
            outer_band
            & (alpha > 10)
            & (cur_luma < 210)
            & (
                (cur_luma < bleed_luma - 6)
                | ((bleed_luma > 130) & (cur_luma < 178))
            )
        )
        if dark_outer.any():
            rgb[dark_outer] = edge_bleed_rgb[dark_outer] * 0.98 + rgb[dark_outer] * 0.02
            alpha[dark_outer & outer1] *= 0.32
            alpha[dark_outer & ~outer1] *= 0.52

        very_dark_outer = outer1 & (alpha > 30) & (cur_luma < 90)
        if very_dark_outer.any():
            alpha[very_dark_outer] *= 0.35

        hard_outer = outer1 & (alpha > 205)
        if hard_outer.any():
            alpha[hard_outer] = alpha[hard_outer] * 0.65 + soft_alpha[hard_outer] * 0.35

        # Last-mile cleanup for light fur: remove one-pixel gray/dark burrs that survive
        # the halo pass. Limit it to bright nearby fur so interior shadows and dark pets stay intact.
        light_fur_burr = (
            outer1
            & (bleed_luma > 118)
            & (
                (cur_luma < 178)
                | ((bleed_luma - cur_luma) > 3)
                | ((alpha < 190) & (cur_luma < 218))
            )
        )
        if light_fur_burr.any():
            rgb[light_fur_burr] = edge_bleed_rgb[light_fur_burr]
            alpha[light_fur_burr] *= 0.04

    # Remove broad semi-transparent dark/gray background residue. Seedance sometimes leaves
    # a large low-alpha matte plate around fast motion; it is connected visually as a shadow,
    # but it does not contain any solid subject core.
    solid_subject = alpha > 120
    if solid_subject.any():
        medium_alpha = alpha > 28
        component_count, labels, _, _ = cv2.connectedComponentsWithStats(
            medium_alpha.astype(np.uint8), 8
        )
        for label in range(1, component_count):
            component = labels == label
            if not (component & solid_subject).any():
                alpha[component] = 0
                rgb[component] = 0

        subject_shell = cv2.dilate(
            solid_subject.astype(np.uint8), kernel, iterations=12
        ).astype(bool)
        luma = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
        low_alpha_residue = (alpha > 4) & (alpha <= 28) & ~subject_shell & (luma < 130)
        if low_alpha_residue.any():
            alpha[low_alpha_residue] = 0
            rgb[low_alpha_residue] = 0

        distance_from_solid = cv2.distanceTransform(
            (~solid_subject).astype(np.uint8), cv2.DIST_L2, 5
        )
        detached_dark_plate = (
            (alpha > 4)
            & (alpha < 125)
            & (luma < 150)
            & (distance_from_solid > 2.0)
        )
        if detached_dark_plate.any():
            alpha[detached_dark_plate] = 0
            rgb[detached_dark_plate] = 0

        # Seedance can invent gray/black contact shadows on locomotion clips even when the
        # first frame has a pure green background. For light-colored pets, remove only the
        # low-saturation dark plate in the lower frame so paws/fur edges are kept.
        visible_luma = luma[alpha > 80]
        if visible_luma.size and float(np.median(visible_luma)) > 115:
            h, w = alpha.shape
            yy = np.arange(h)[:, None]
            chroma = rgb.max(axis=2) - rgb.min(axis=2)
            lower_shadow = (
                (yy > int(h * 0.50))
                & (alpha > 10)
                & (luma < 102)
                & (chroma < 46)
            )
            if lower_shadow.any():
                shadow_count, shadow_labels, shadow_stats, _ = cv2.connectedComponentsWithStats(
                    lower_shadow.astype(np.uint8), 8
                )
                for shadow_label in range(1, shadow_count):
                    area = shadow_stats[shadow_label, cv2.CC_STAT_AREA]
                    if area < 4:
                        continue
                    component = shadow_labels == shadow_label
                    alpha[component] = 0
                    rgb[component] = 0

    # Drop isolated matte dust that is no longer connected to the pet after edge cleanup.
    component_mask = alpha > 4
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        component_mask.astype(np.uint8), 8
    )
    if component_count > 2:
        main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        for label in range(1, component_count):
            if label == main_label:
                continue
            component = labels == label
            alpha[component] = 0
            rgb[component] = 0

    # 完全透明区域清空 RGB，避免 WebP 编码漏色。
    alpha[alpha < 4] = 0
    rgb[alpha == 0] = 0

    final_mask = alpha > 4
    final_count, final_labels, final_stats, _ = cv2.connectedComponentsWithStats(
        final_mask.astype(np.uint8), 8
    )
    if final_count > 2:
        final_main = 1 + int(np.argmax(final_stats[1:, cv2.CC_STAT_AREA]))
        remove = final_mask & (final_labels != final_main)
        alpha[remove] = 0
        rgb[remove] = 0
        rgb[alpha == 0] = 0

    arr[:, :, :3] = np.clip(rgb, 0, 255)
    arr[:, :, 3] = np.clip(alpha, 0, 255)
    out_arr = arr.astype(np.uint8)
    out_alpha = out_arr[:, :, 3]
    out_alpha[out_alpha <= 4] = 0
    out_arr[out_alpha == 0, :3] = 0
    final_mask = out_alpha > 4
    final_count, final_labels, final_stats, _ = cv2.connectedComponentsWithStats(
        final_mask.astype(np.uint8), 8
    )
    if final_count > 2:
        final_main = 1 + int(np.argmax(final_stats[1:, cv2.CC_STAT_AREA]))
        remove = final_mask & (final_labels != final_main)
        out_arr[remove] = 0
    return Image.fromarray(out_arr, "RGBA")


def frame_loop_signature(frame):
    import numpy as np

    rgba = frame.convert("RGBA").resize((48, 48))
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    rgb = arr[:, :, :3] * alpha
    return np.concatenate([rgb, alpha], axis=2).reshape(-1)


def trim_to_best_loop_span(frames, min_frames=72, max_start=24):
    if len(frames) < min_frames + 2:
        return frames, {"mode": "too_short", "source_frames": len(frames)}
    signatures = [frame_loop_signature(frame) for frame in frames]
    n = len(frames)
    max_start = min(max_start, max(0, n - min_frames - 1))
    best = None
    for start in range(max_start + 1):
        for end in range(start + min_frames, n):
            diff = float(abs(signatures[start] - signatures[end]).mean())
            score = diff + (n - end) * 0.000025 + start * 0.00004
            if best is None or score < best[0]:
                best = (score, diff, start, end)
    if best is None:
        return frames, {"mode": "not_found", "source_frames": len(frames)}
    _, diff, start, end = best
    selected = frames[start:end + 1]
    return selected, {
        "mode": "best_span",
        "source_frames": len(frames),
        "selected": [start, end],
        "selected_frames": len(selected),
        "seam_diff": round(diff, 6),
    }


def build_loop_frames(pngs, close_frames=None, optimize_loop=False):
    """Read frames, clean mattes, and optionally trim to a smoother loop span."""
    from PIL import Image

    frames = [refine_rgba_frame(Image.open(p)) for p in pngs]
    loop_meta = None
    if optimize_loop:
        frames, loop_meta = trim_to_best_loop_span(frames)
    if close_frames is None:
        close_frames = LOOP_CLOSE_FRAMES
    if len(frames) < 2 or close_frames <= 0:
        return frames, 0, loop_meta
    first = frames[0]
    last = frames[-1]
    close = []
    for i in range(1, close_frames + 1):
        t = i / close_frames
        close.append(Image.blend(last, first, t))
    return frames + close, close_frames, loop_meta


def build_loop_frames(pngs, close_frames=None, optimize_loop=False):
    from PIL import Image

    frames = [refine_rgba_frame(Image.open(p)) for p in pngs]
    loop_meta = None
    if optimize_loop:
        frames, loop_meta = trim_to_best_loop_span(frames)
    if len(frames) < 2:
        return frames, 0, loop_meta
    trim_head = min(max(0, LOOP_TRIM_HEAD_FRAMES), max(0, len(frames) // 5))
    trim_tail = min(max(0, LOOP_TRIM_TAIL_FRAMES), max(0, len(frames) // 5))
    if trim_head or trim_tail:
        end = len(frames) - trim_tail if trim_tail else len(frames)
        stable = frames[trim_head:end]
        if len(stable) >= 24:
            frames = stable
    if close_frames is None:
        close_frames = LOOP_CLOSE_FRAMES
    close_n = min(max(0, close_frames), max(0, len(frames) // 5))
    if close_n <= 0:
        return frames, 0, loop_meta
    main = frames[:-close_n]
    for i in range(close_n):
        t = (i + 1) / (close_n + 1)
        main.append(Image.blend(frames[-close_n + i], frames[i], t))
    return main, close_n, loop_meta


def alpha_union_bbox(frames, threshold=4):
    import numpy as np

    boxes = []
    for frame in frames:
        alpha = np.asarray(frame.convert("RGBA"))[:, :, 3]
        ys, xs = np.where(alpha > threshold)
        if len(xs):
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def reframe_loop_frames(frames):
    """Normalize transparent WebP content to a shared bottom-aligned safe area."""
    from PIL import Image

    bbox = alpha_union_bbox(frames)
    if not bbox:
        return frames, None
    x0, y0, x1, y1 = bbox
    source_w, source_h = frames[0].size
    edge_risk = {
        "left": x0 <= WEBP_EDGE_RISK_MARGIN,
        "top": y0 <= WEBP_EDGE_RISK_MARGIN,
        "right": (source_w - x1) <= WEBP_EDGE_RISK_MARGIN,
        "bottom": (source_h - y1) <= WEBP_EDGE_RISK_MARGIN,
    }
    subject_w, subject_h = x1 - x0, y1 - y0
    pad = max(10, int(max(subject_w, subject_h) * 0.045))
    crop = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(source_w, x1 + pad),
        min(source_h, y1 + pad),
    )
    crop_w, crop_h = crop[2] - crop[0], crop[3] - crop[1]
    max_w = max(1, int(WEBP_WIDTH * WEBP_SUBJECT_WIDTH_RATIO))
    max_h = max(1, int(WEBP_WIDTH * WEBP_SUBJECT_HEIGHT_RATIO))
    scale = min(max_w / crop_w, max_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    paste_x = (WEBP_WIDTH - new_w) // 2
    paste_y = max(0, WEBP_WIDTH - WEBP_BOTTOM_MARGIN - new_h)
    out = []
    for frame in frames:
        cropped = frame.convert("RGBA").crop(crop)
        if (new_w, new_h) != cropped.size:
            cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (WEBP_WIDTH, WEBP_WIDTH), (0, 0, 0, 0))
        canvas.alpha_composite(cropped, (paste_x, paste_y))
        out.append(canvas)
    return out, {
        "source_bbox": [x0, y0, x1, y1],
        "source_margins": {
            "left": x0,
            "top": y0,
            "right": source_w - x1,
            "bottom": source_h - y1,
        },
        "source_edge_risk": edge_risk,
        "crop": list(crop),
        "scale": round(scale, 4),
        "paste": [paste_x, paste_y],
        "target_size": [new_w, new_h],
    }


def step_matte(pet: str, clip: str):
    """自适应抠图（2026-06-11 用户要求"部署到 app 抠图一定要做好"）：
    容差阶梯试抠 + 绿残留质检 + 主体消失熔断，取首个达标档。"""
    pet_dir = OUTPUT / pet
    video = pet_dir / f"raw_{clip}.mp4"
    if not video.exists():
        sys.exit(f"缺 raw_{clip}.mp4：先跑 --step animate --clip {clip}")
    t0 = time.time()
    key = sample_bg_color(video)
    frames = pet_dir / f"frames_{clip}"
    chosen_sim, residual = None, 1.0
    for sim in CHROMA_SIMILARITY_STEPS:
        if frames.exists():
            shutil.rmtree(frames)
        frames.mkdir()
        vf = (f"chromakey={key}:{sim}:{CHROMA_BLEND},"
              f"despill=type=green,scale={WEBP_WIDTH}:-2,fps={WEBP_FPS}")
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                        "-vf", vf, str(frames / "f_%04d.png")], check=True)
        worst_green, mid_visible = assess_frames(frames)
        print(f"[matte:{clip}] key={key} sim={sim} 绿残留 {worst_green:.1%} 可见 {mid_visible:.1%}")
        if mid_visible < 0.04:
            sys.exit(f"[matte:{clip}] 熔断：主体可见占比 {mid_visible:.1%} —— 抠过头（参照猫消失事故），"
                     f"检查背景归一化/key 色")
        chosen_sim, residual = sim, worst_green
        if worst_green < 0.015:
            break
    if residual >= 0.015:
        print(f"[matte:{clip}] ⚠️ 容差阶梯走完绿残留仍 {residual:.1%}（>1.5%），保留最后档，需人工复核")
    pngs = sorted(frames.glob("f_*.png"))
    if not pngs:
        sys.exit(f"[matte:{clip}] 没有抽出帧")
    duration_ms = round(1000 / WEBP_FPS)
    out = pet_dir / f"anim_{clip}.webp"
    print(f"[matte:{clip}] 后处理 {len(pngs)} 帧，准备写 WebP...", flush=True)
    state_loop = clip in STATE_FRAME_CLIPS or clip == "fast_walk"
    webp_frames, loop_added, loop_meta = build_loop_frames(
        pngs,
        close_frames=0 if state_loop else LOOP_CLOSE_FRAMES,
        optimize_loop=(clip in ("walk", "fast_walk")),
    )
    webp_frames, reframe = reframe_loop_frames(webp_frames)
    webp_frames[0].save(
        out,
        save_all=True,
        append_images=webp_frames[1:],
        duration=duration_ms,
        loop=0,
        lossless=False,
        quality=WEBP_QUALITY,
        alpha_quality=100,
        method=WEBP_METHOD,
        exact=True,
        minimize_size=False,
    )
    if clip == DEFAULT_CLIP:
        webp_frames[0].save(pet_dir / "preview.png")
    size_mb = round(out.stat().st_size / 1024 / 1024, 2)
    dt = round(time.time() - t0, 1)
    print(f"[matte:{clip}] loop_meta={loop_meta}")
    print(f"[matte:{clip}] reframe={reframe}")
    print(f"[matte:{clip}] {out.name} {size_mb}MB / {len(webp_frames)} 帧（源 {len(pngs)} + 闭环 {loop_added}）/ {dt}s")
    record_metric(pet_dir, "matte", {"clip": clip, "key_color": key, "frames": len(pngs),
                                     "webp_frames": len(webp_frames),
                                     "loop_close_frames": loop_added,
                                     "webp_fps": WEBP_FPS,
                                     "webp_width": WEBP_WIDTH,
                                     "webp_quality": WEBP_QUALITY,
                                     "webp_method": WEBP_METHOD,
                                     "webp_mb": size_mb, "seconds": dt,
                                     "similarity": chosen_sim, "green_residual": round(residual, 4),
                                     "loop_meta": loop_meta,
                                     "reframe": reframe})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pet", required=True, help="inputs/ 下的照片名（不含扩展名）")
    ap.add_argument("--step", default="", help="stylize|state_sheet|state_frames|animate|matte，逗号分隔可连跑")
    ap.add_argument("--style", default=DEFAULT_STYLE,
                    help=f"风格档位 {list(STYLE_PROMPTS)} 或 all（矩阵），默认 {DEFAULT_STYLE}")
    ap.add_argument("--clip", default=DEFAULT_CLIP,
                    help=f"动作片段 {list(CLIP_ORDER)} 或 all，逗号分隔可多段，默认 {DEFAULT_CLIP}")
    ap.add_argument("--choose", default="", help="把候选 <档位_序号>（如 cute_2）定为 chosen.png")
    ap.add_argument("--alt", action="store_true", help="stylize 用对照模型 Seedream-5.0")
    args = ap.parse_args()
    if args.choose:
        step_choose(args.pet, args.choose)
        return
    if args.style != "all" and args.style not in STYLE_PROMPTS:
        sys.exit(f"未知 style: {args.style}")
    clips = list(CLIP_ORDER) if args.clip == "all" else [c.strip() for c in args.clip.split(",") if c.strip()]
    for c in clips:
        if c not in CLIP_PROMPTS:
            sys.exit(f"未知 clip: {c}")
    steps = [x.strip() for x in args.step.split(",") if x.strip()]
    if any(s in ("stylize", "state_sheet", "state_frames", "animate", "animate_sheet") for s in steps) and "ARK_API_KEY" not in ENV:
        sys.exit("缺 .env 或环境变量 ARK_API_KEY")
    for s in steps:
        if s == "stylize":
            step_stylize(args.pet, args.style,
                         MODEL_STYLIZE_ALT if args.alt else MODEL_STYLIZE)
        elif s == "state_sheet":
            step_state_sheet(args.pet, args.style,
                             MODEL_STYLIZE_ALT if args.alt else MODEL_STYLIZE)
        elif s == "state_frames":
            step_state_frames(args.pet, clips, MODEL_STYLIZE_ALT if args.alt else MODEL_STYLIZE)
        elif s == "animate":
            step_animate_many(args.pet, clips)
        elif s == "animate_sheet":
            step_animate_sheet(args.pet)
        elif s == "matte":
            for c in clips:
                step_matte(args.pet, c)
        else:
            sys.exit(f"未知 step: {s}")


if __name__ == "__main__":
    main()
