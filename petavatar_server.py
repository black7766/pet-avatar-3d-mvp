#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local API server for the PetAvatar PoC.

Endpoints:
  POST /api/generate        multipart/form-data image upload
  POST /api/backfill        application/json missing clip backfill
  GET  /api/jobs/<job_id>   job status, logs, pet_id, page_url, metrics

Static files are served from poc_output/ with explicit WebP MIME support.
Generation is serialized through one background worker and delegates to poc.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import email
import email.policy
import functools
import html
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "poc_output"
POC_SCRIPT = ROOT / "poc.py"

CLIPS = ("idle", "fast_walk", "sleep")
ACTIVE_CLIP_SET = set(CLIPS)
CLIP_LABELS = {
    "idle": "静息",
    "run": "奔跑",
    "walk": "走动",
    "sleep": "睡觉",
}
CLIP_LABELS = {"idle": "静息", "fast_walk": "快走", "sleep": "睡觉"}
SOURCE_EXTS = ("jpg", "jpeg", "png", "webp", "heic")
VARIANTS = (
    {"key": "real", "style": "real", "choose": "real_1", "label": "实体版"},
    {"key": "paimomo", "style": "paimomo3d", "choose": "paimomo3d_1", "label": "萌宠版"},
)
VARIANTS = (
    {"key": "real", "style": "real", "choose": "real_1", "label": "实体版"},
    {"key": "paimomo", "style": "paimomo3d", "choose": "paimomo3d_1", "label": "萌宠版"},
)
VARIANT_BY_KEY = {item["key"]: item for item in VARIANTS}
MAX_UPLOAD_BYTES = int(os.environ.get("PETAVATAR_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_LOG_LINES = int(os.environ.get("PETAVATAR_MAX_LOG_LINES", "1200"))
PARALLEL_VARIANTS = max(1, int(os.environ.get("PETAVATAR_PARALLEL_VARIANTS", "2")))


mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/webm", ".webm")
mimetypes.add_type("application/json; charset=utf-8", ".json")


@dataclasses.dataclass
class Job:
    job_id: str
    pet_id: str
    input_path: str
    page_url: str
    mode: str = "generate"
    target_pets: list[str] | None = None
    clips: list[str] | None = None
    status: str = "queued"
    logs: list[str] = dataclasses.field(default_factory=list)
    metrics: Any = None
    error: str | None = None
    created_at: float = dataclasses.field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
job_queue: queue.Queue[str] = queue.Queue()
worker_started = False
worker_lock = threading.Lock()


def dotenv_api_key() -> str | None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.lstrip("\ufeff")
            if key.strip() == "ARK_API_KEY":
                value = value.strip().strip('"').strip("'")
                return value or None
    except OSError:
        return None
    return None


def get_api_key() -> str | None:
    return os.environ.get("ARK_API_KEY") or dotenv_api_key()


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_pet_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"pet_{stamp}_{uuid.uuid4().hex[:8]}"


def variant_pet_id(base_pet_id: str, variant_key: str) -> str:
    return f"{base_pet_id}_{variant_key}"


def auto_pet_info(pet_id: str) -> dict[str, str] | None:
    match = re.fullmatch(r"(pet_\d{8}_\d{6}_[0-9a-f]{8})(?:_(real|paimomo))?", pet_id)
    if not match:
        return None
    base_pet_id = match.group(1)
    variant_key = match.group(2) or "paimomo"
    variant = VARIANT_BY_KEY.get(variant_key, VARIANT_BY_KEY["paimomo"])
    return {
        "base_pet_id": base_pet_id,
        "variant": variant_key,
        "variant_label": variant["label"],
        "style": variant["style"],
    }


def append_log(job_id: str, line: str) -> None:
    api_key = get_api_key()
    if api_key:
        line = line.replace(api_key, "[redacted]")
    line = line.rstrip("\r\n")
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.logs.append(f"[{now_label()}] {line}")
        if len(job.logs) > MAX_LOG_LINES:
            del job.logs[: len(job.logs) - MAX_LOG_LINES]


def update_job(job_id: str, **fields: Any) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)


def load_metrics(pet_id: str) -> Any:
    metrics_file = OUTPUT / pet_id / "metrics.json"
    if not metrics_file.exists():
        return None
    try:
        return sanitize_metrics(json.loads(metrics_file.read_text(encoding="utf-8")))
    except Exception as exc:
        return {"_error": f"failed to read metrics.json: {exc}"}


def sanitize_metrics(metrics: Any) -> Any:
    if not isinstance(metrics, dict):
        return metrics
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if not isinstance(value, list):
            cleaned[key] = value
            continue
        rows: list[Any] = []
        for row in value:
            if not isinstance(row, dict):
                rows.append(row)
                continue
            clip = row.get("clip") or row.get("source_clip")
            if clip and clip not in ACTIVE_CLIP_SET:
                continue
            rows.append(row)
        cleaned[key] = rows
    return cleaned


def clip_files(pet_id: str) -> dict[str, bool]:
    pet_dir = OUTPUT / pet_id
    return {clip: (pet_dir / f"anim_{clip}.webp").exists() for clip in CLIPS}


def image_ext_from_upload(filename: str, image_bytes: bytes) -> str:
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    if suffix in SOURCE_EXTS:
        return "jpg" if suffix == "jpeg" else suffix
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"
    if len(image_bytes) > 12 and image_bytes[4:12] in (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftyphevx", b"ftypmif1"):
        return "heic"
    return "jpg"


def output_original_file_for_pet(pet_id: str) -> Path | None:
    pet_dir = OUTPUT / pet_id
    for ext in SOURCE_EXTS:
        candidate = pet_dir / f"uploaded_original.{ext}"
        if candidate.exists():
            return candidate
    return None


def save_original_copy(input_path: Path, pet_id: str, original_filename: str | None = None) -> Path:
    pet_dir = OUTPUT / pet_id
    pet_dir.mkdir(parents=True, exist_ok=True)
    ext = input_path.suffix.lower().lstrip(".") or "jpg"
    if ext == "jpeg":
        ext = "jpg"
    dest = pet_dir / f"uploaded_original.{ext}"
    shutil.copyfile(input_path, dest)
    meta = {
        "pet_id": pet_id,
        "input_file": input_path.name,
        "original_filename": original_filename,
        "saved_as": dest.name,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (pet_dir / "source.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def source_file_for_pet(pet_id: str) -> Path | None:
    names = [pet_id]
    info = auto_pet_info(pet_id)
    if info and info["base_pet_id"] not in names:
        names.append(info["base_pet_id"])
    for name in names:
        candidate = output_original_file_for_pet(name)
        if candidate:
            return candidate
    for name in names:
        for ext in SOURCE_EXTS:
            candidate = INPUTS / f"{name}.{ext}"
            if candidate.exists():
                return candidate
    return None


def source_url_for_pet(pet_id: str) -> str | None:
    source_file = source_file_for_pet(pet_id)
    if not source_file:
        return None
    try:
        rel = source_file.resolve().relative_to(OUTPUT.resolve())
        return "/" + quote(rel.as_posix())
    except ValueError:
        pass
    try:
        rel = source_file.resolve().relative_to(INPUTS.resolve())
        return "/inputs/" + quote(rel.as_posix())
    except ValueError:
        pass
    return None


def list_pets() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not OUTPUT.exists():
        return items
    for pet_dir in sorted(OUTPUT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not pet_dir.is_dir():
            continue
        pet_id = pet_dir.name
        metrics_file = pet_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        stat = pet_dir.stat()
        info = auto_pet_info(pet_id)
        if info and info["variant"] == "paimomo" and not pet_id.endswith("_paimomo"):
            if (OUTPUT / variant_pet_id(info["base_pet_id"], "paimomo")).exists():
                continue
        items.append({
            "pet_id": pet_id,
            "clips": clip_files(pet_id),
            "metrics": load_metrics(pet_id),
            "is_auto": bool(info),
            "base_pet_id": info["base_pet_id"] if info else None,
            "variant": info["variant"] if info else None,
            "variant_label": info["variant_label"] if info else None,
            "style": info["style"] if info else None,
            "page_url": f"/view/{quote(pet_id)}",
            "source_url": source_url_for_pet(pet_id),
            "updated_at": stat.st_mtime,
            "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
        })
    return items


def default_backfill_pet_ids() -> list[str]:
    ids: list[str] = []
    for pet_id in ("pet1", "pet_paimomo"):
        if (OUTPUT / pet_id / "chosen.png").exists():
            ids.append(pet_id)
    for item in list_pets():
        pet_id = item.get("pet_id")
        if not pet_id or pet_id in ids:
            continue
        if item.get("is_auto") and item.get("variant") in {"real", "paimomo"}:
            ids.append(pet_id)
    return ids


def pet_snapshot(pet_id: str) -> dict[str, Any]:
    info = auto_pet_info(pet_id)
    pet_dir = OUTPUT / pet_id
    stat_time = pet_dir.stat().st_mtime if pet_dir.exists() else None
    return {
        "pet_id": pet_id,
        "base_pet_id": info["base_pet_id"] if info else None,
        "variant": info["variant"] if info else None,
        "variant_label": info["variant_label"] if info else None,
        "style": info["style"] if info else None,
        "clips": clip_files(pet_id),
        "metrics": load_metrics(pet_id),
        "is_auto": bool(info),
        "page_url": f"/view/{quote(pet_id)}",
        "source_url": source_url_for_pet(pet_id),
        "updated_at": stat_time,
        "updated_at_iso": (
            time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat_time))
            if stat_time else None
        ),
    }


def variant_snapshots(base_pet_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for variant in VARIANTS:
        pet_id = variant_pet_id(base_pet_id, variant["key"])
        pet_dir = OUTPUT / pet_id
        stat_time = pet_dir.stat().st_mtime if pet_dir.exists() else None
        items.append({
            "pet_id": pet_id,
            "base_pet_id": base_pet_id,
            "variant": variant["key"],
            "variant_label": variant["label"],
            "style": variant["style"],
            "clips": clip_files(pet_id),
            "metrics": load_metrics(pet_id),
            "is_auto": True,
            "page_url": f"/view/{quote(pet_id)}",
            "source_url": source_url_for_pet(pet_id),
            "updated_at": stat_time,
            "updated_at_iso": (
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat_time))
                if stat_time else None
            ),
        })
    return items


def job_snapshot(job_id: str) -> dict[str, Any] | None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return None
        if job.mode == "backfill":
            pet_ids = job.target_pets or []
            pets = [pet_snapshot(pet_id) for pet_id in pet_ids]
            job.metrics = {"pets": pets}
            variants: list[dict[str, Any]] = []
        else:
            variants = variant_snapshots(job.pet_id)
            pets = []
            job.metrics = {"variants": variants}
        data = dataclasses.asdict(job)
    data["created_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(data["created_at"]))
    if data.get("started_at"):
        data["started_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(data["started_at"]))
    if data.get("finished_at"):
        data["finished_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(data["finished_at"]))
    data["source_url"] = source_url_for_pet(data["pet_id"])
    data["variants"] = variants
    data["pets"] = pets
    return data


def subprocess_env() -> dict[str, str]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("ARK_API_KEY is not set in the server environment or .env")
    env = os.environ.copy()
    env["ARK_API_KEY"] = api_key
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PETAVATAR_CANDIDATES", "1")
    env.setdefault("PETAVATAR_WEBP_FPS", "24")
    env.setdefault("PETAVATAR_WEBP_WIDTH", "640")
    env.setdefault("PETAVATAR_LOOP_CLOSE_FRAMES", "8")
    env.setdefault("PETAVATAR_WEBP_METHOD", "2")
    env.setdefault("PETAVATAR_WEBP_QUALITY", "90")
    return env


def run_poc(job_id: str, label: str, args: list[str], env: dict[str, str]) -> None:
    display = " ".join(quote_arg(x) for x in args)
    append_log(job_id, f"start {label}: {display}")
    proc = subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        append_log(job_id, line)
    code = proc.wait()
    append_log(job_id, f"finish {label}: exit {code}")
    if code != 0:
        raise RuntimeError(f"{label} failed with exit code {code}")


def quote_arg(value: str) -> str:
    if re.search(r"\s", value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def process_variant(job_id: str, base_pet_id: str, input_path: Path, variant: dict[str, str], env: dict[str, str], py: str) -> None:
    pet_id = variant_pet_id(base_pet_id, variant["key"])
    variant_input = INPUTS / f"{pet_id}{input_path.suffix or '.jpg'}"
    if not variant_input.exists():
        shutil.copyfile(input_path, variant_input)
    save_original_copy(input_path, pet_id, input_path.name)

    append_log(job_id, f"variant {variant['key']} started for {pet_id}")
    run_poc(
        job_id,
        f"{variant['key']} stylize {variant['style']}",
        [py, str(POC_SCRIPT), "--pet", pet_id, "--step", "stylize", "--style", variant["style"]],
        env,
    )
    run_poc(
        job_id,
        f"{variant['key']} choose {variant['choose']}",
        [py, str(POC_SCRIPT), "--pet", pet_id, "--choose", variant["choose"]],
        env,
    )
    update_job(job_id, metrics={"variants": variant_snapshots(base_pet_id)})
    run_poc(
        job_id,
        f"{variant['key']} animate {','.join(CLIPS)}",
        [py, str(POC_SCRIPT), "--pet", pet_id, "--step", "animate", "--clip", ",".join(CLIPS)],
        env,
    )
    update_job(job_id, metrics={"variants": variant_snapshots(base_pet_id)})
    run_poc(
        job_id,
        f"{variant['key']} matte {','.join(CLIPS)}",
        [py, str(POC_SCRIPT), "--pet", pet_id, "--step", "matte", "--clip", ",".join(CLIPS)],
        env,
    )
    update_job(job_id, metrics={"variants": variant_snapshots(base_pet_id)})
    append_log(job_id, f"variant {variant['key']} succeeded")


def process_backfill_job(job_id: str, snapshot: dict[str, Any]) -> None:
    target_pets = [pet_id for pet_id in snapshot.get("target_pets") or [] if re.fullmatch(r"[A-Za-z0-9_.-]+", pet_id)]
    clips = [clip for clip in snapshot.get("clips") or [] if clip in CLIPS]
    update_job(job_id, status="running", started_at=time.time())
    append_log(job_id, f"backfill started pets={len(target_pets)} clips={','.join(clips)}")
    try:
        if not target_pets:
            raise RuntimeError("no target pets to backfill")
        if not clips:
            raise RuntimeError("no valid clips to backfill")
        env = subprocess_env()
        py = sys.executable
        for pet_id in target_pets:
            pet_dir = OUTPUT / pet_id
            if not (pet_dir / "chosen.png").exists():
                append_log(job_id, f"skip {pet_id}: chosen.png not found")
                continue
            missing = [clip for clip in clips if not (pet_dir / f"anim_{clip}.webp").exists()]
            if not missing:
                append_log(job_id, f"skip {pet_id}: requested clips already exist")
                continue
            clip_csv = ",".join(missing)
            append_log(job_id, f"backfill {pet_id}: {clip_csv}")
            run_poc(
                job_id,
                f"{pet_id} animate {clip_csv}",
                [py, str(POC_SCRIPT), "--pet", pet_id, "--step", "animate", "--clip", clip_csv],
                env,
            )
            update_job(job_id, metrics={"pets": [pet_snapshot(pet) for pet in target_pets]})
            run_poc(
                job_id,
                f"{pet_id} matte {clip_csv}",
                [py, str(POC_SCRIPT), "--pet", pet_id, "--step", "matte", "--clip", clip_csv],
                env,
            )
            update_job(job_id, metrics={"pets": [pet_snapshot(pet) for pet in target_pets]})
        append_log(job_id, "backfill succeeded")
        update_job(job_id, status="succeeded", metrics={"pets": [pet_snapshot(pet) for pet in target_pets]}, finished_at=time.time())
    except Exception as exc:
        message = str(exc)
        append_log(job_id, f"backfill failed: {message}")
        update_job(
            job_id,
            status="failed",
            error=message,
            metrics={"pets": [pet_snapshot(pet) for pet in target_pets]},
            finished_at=time.time(),
        )


def process_job(job_id: str) -> None:
    snapshot = job_snapshot(job_id)
    if not snapshot:
        return
    if snapshot.get("mode") == "backfill":
        process_backfill_job(job_id, snapshot)
        return
    base_pet_id = snapshot["pet_id"]
    input_path = Path(snapshot["input_path"])
    update_job(job_id, status="running", started_at=time.time())
    append_log(job_id, f"job started for {base_pet_id}")
    try:
        env = subprocess_env()
        py = sys.executable
        append_log(job_id, f"parallel variants={PARALLEL_VARIANTS}; seedance clips are submitted in batch per variant")
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_VARIANTS) as executor:
            futures = [
                executor.submit(process_variant, job_id, base_pet_id, input_path, variant, env, py)
                for variant in VARIANTS
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        append_log(job_id, "job succeeded")
        update_job(job_id, status="succeeded", metrics={"variants": variant_snapshots(base_pet_id)}, finished_at=time.time())
    except Exception as exc:
        message = str(exc)
        append_log(job_id, f"job failed: {message}")
        update_job(
            job_id,
            status="failed",
            error=message,
            metrics={"variants": variant_snapshots(base_pet_id)},
            finished_at=time.time(),
        )


def worker_loop() -> None:
    while True:
        job_id = job_queue.get()
        try:
            process_job(job_id)
        finally:
            job_queue.task_done()


def ensure_worker() -> None:
    global worker_started
    with worker_lock:
        if worker_started:
            return
        thread = threading.Thread(target=worker_loop, name="petavatar-worker", daemon=True)
        thread.start()
        worker_started = True


def parse_multipart_image(headers: dict[str, str], body: bytes) -> tuple[bytes, str]:
    content_type = headers.get("content-type", "")
    if "multipart/form-data" not in content_type.lower():
        raise ValueError("expected multipart/form-data")
    raw = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = email.message_from_bytes(raw, policy=email.policy.default)
    if not message.is_multipart():
        raise ValueError("invalid multipart body")
    fallback: tuple[bytes, str] | None = None
    for part in message.iter_parts():
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        filename = part.get_filename() or "upload"
        content_main = part.get_content_maintype()
        disposition = part.get_content_disposition()
        if content_main == "image":
            return payload, filename
        if disposition == "form-data" and fallback is None and part.get_filename():
            fallback = (payload, filename)
    if fallback:
        return fallback
    raise ValueError("multipart body did not include an image file")


def json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".json": "application/json; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }

    server_version = "PetAvatarServer/1.0"

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/backfill":
            self.handle_backfill()
            return
        if path != "/api/generate":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
            return
        if length <= 0:
            self.send_json({"error": "empty request body"}, HTTPStatus.BAD_REQUEST)
            return
        if length > MAX_UPLOAD_BYTES:
            self.send_json({"error": f"upload too large; max {MAX_UPLOAD_BYTES} bytes"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            body = self.rfile.read(length)
            request_headers = {key.lower(): value for key, value in self.headers.items()}
            image_bytes, filename = parse_multipart_image(request_headers, body)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        INPUTS.mkdir(parents=True, exist_ok=True)
        pet_id = safe_pet_id()
        ext = image_ext_from_upload(filename, image_bytes)
        input_path = INPUTS / f"{pet_id}.{ext}"
        input_path.write_bytes(image_bytes)
        save_original_copy(input_path, pet_id, filename)

        job_id = uuid.uuid4().hex
        page_url = self.absolute_url(f"/view/{quote(pet_id)}")
        job = Job(job_id=job_id, pet_id=pet_id, input_path=str(input_path), page_url=page_url)
        job.logs.append(f"[{now_label()}] accepted upload {filename} as inputs/{pet_id}.{ext}")
        flow = "; ".join(
            f"{variant['key']}: stylize {variant['style']} -> choose {variant['choose']} -> animate/matte {','.join(CLIPS)}"
            for variant in VARIANTS
        )
        job.logs.append(f"[{now_label()}] queued flow: {flow}")
        with jobs_lock:
            jobs[job_id] = job
        ensure_worker()
        job_queue.put(job_id)
        self.send_json(job_snapshot(job_id), HTTPStatus.ACCEPTED)

    def handle_backfill(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
            return
        payload: dict[str, Any] = {}
        if length > 0:
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as exc:
                self.send_json({"error": f"invalid json: {exc}"}, HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(payload, dict):
                self.send_json({"error": "json body must be an object"}, HTTPStatus.BAD_REQUEST)
                return

        raw_clips = payload.get("clips") or payload.get("clip") or ["sleep"]
        if isinstance(raw_clips, str):
            raw_clips = [raw_clips]
        clips = [clip for clip in raw_clips if clip in CLIPS]
        if not clips:
            self.send_json({"error": f"clips must include one of: {', '.join(CLIPS)}"}, HTTPStatus.BAD_REQUEST)
            return

        raw_pet_ids = payload.get("pet_ids") or payload.get("pets") or default_backfill_pet_ids()
        if isinstance(raw_pet_ids, str):
            raw_pet_ids = [raw_pet_ids]
        pet_ids = []
        for pet_id in raw_pet_ids:
            if isinstance(pet_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", pet_id) and pet_id not in pet_ids:
                pet_ids.append(pet_id)
        if not pet_ids:
            self.send_json({"error": "no valid target pets"}, HTTPStatus.BAD_REQUEST)
            return

        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            pet_id="backfill",
            input_path="",
            page_url="",
            mode="backfill",
            target_pets=pet_ids,
            clips=clips,
        )
        job.logs.append(f"[{now_label()}] accepted backfill clips={','.join(clips)} pets={len(pet_ids)}")
        with jobs_lock:
            jobs[job_id] = job
        ensure_worker()
        job_queue.put(job_id)
        self.send_json(job_snapshot(job_id), HTTPStatus.ACCEPTED)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/pets":
            self.send_json({"pets": list_pets()})
            return
        if path.startswith("/api/jobs/"):
            job_id = unquote(path.removeprefix("/api/jobs/")).strip("/")
            data = job_snapshot(job_id)
            if not data:
                self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(data)
            return
        if path.startswith("/view/"):
            pet_id = unquote(path.removeprefix("/view/")).strip("/")
            self.send_view(pet_id)
            return
        if path.startswith("/inputs/"):
            filename = unquote(path.removeprefix("/inputs/")).strip("/")
            self.send_input_file(filename)
            return
        super().do_GET()

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json_bytes(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_input_file(self, filename: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid input filename")
            return
        target = (INPUTS / filename).resolve()
        try:
            target.relative_to(INPUTS.resolve())
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid input path")
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "input not found")
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        payload = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_view(self, pet_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", pet_id):
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid pet id")
            return
        pet_html = html.escape(pet_id, quote=True)
        info = auto_pet_info(pet_id)
        if info and pet_id == info["base_pet_id"]:
            view_pets = [
                (variant_pet_id(pet_id, "real"), "实体版"),
                (variant_pet_id(pet_id, "paimomo"), "萌宠版"),
            ]
        else:
            label = "萌宠版" if pet_id.endswith("_paimomo") else "实体版" if pet_id.endswith("_real") else "预览"
            view_pets = [(pet_id, label)]
        button_items = []
        for clip in CLIPS:
            active = ' class="on"' if clip == "idle" else ""
            label = html.escape(CLIP_LABELS.get(clip, clip), quote=True)
            button_items.append(f'<button data-clip="{clip}"{active}>{label}</button>')
        buttons = "\n".join(button_items)
        cards = []
        for view_pet, label in view_pets:
            exists = (OUTPUT / view_pet / "anim_idle.webp").exists()
            src = f"/{quote(view_pet)}/anim_idle.webp" if exists else ""
            label_html = html.escape(label, quote=True)
            view_pet_html = html.escape(view_pet, quote=True)
            if src:
                media = f'<img class="pet" data-pet="{view_pet_html}" src="{src}" alt="{label_html} animation">'
                links = (
                    f'<a href="/{quote(view_pet)}/metrics.json" target="_blank">metrics</a>'
                    f'<a href="/{quote(view_pet)}/preview.png" target="_blank">preview</a>'
                )
            else:
                media = '<div class="missing">待生成</div>'
                links = ""
            cards.append(f"""
    <article class="card">
      <header><strong>{label_html}</strong><code>{view_pet_html}</code></header>
      <section class="stage">{media}</section>
      <div class="links">{links}</div>
    </article>""")
        card_html = "\n".join(cards)
        body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PetAvatar {pet_html}</title>
  <style>
    body {{ margin: 0; min-height: 100vh; font-family: Arial, sans-serif; background: #f6f1ea; color: #241c18; }}
    main {{ width: min(1180px, 94vw); margin: 28px auto; display: grid; gap: 18px; }}
    .top {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; align-items: start; }}
    .card {{ background: #fffaf3; border: 1px solid rgba(40,30,20,.14); border-radius: 8px; overflow: hidden; }}
    .card header {{ display: flex; justify-content: space-between; gap: 12px; padding: 12px 14px; border-bottom: 1px solid rgba(40,30,20,.12); }}
    .stage {{ position: relative; width: 100%; aspect-ratio: 1; background: #f7f4ee; overflow: hidden; }}
    img.pet {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }}
    .missing {{ display: grid; place-items: center; width: 100%; height: 100%; color: #746960; }}
    .buttons {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }}
    .links {{ min-height: 42px; display: flex; gap: 8px; justify-content: center; align-items: center; padding: 10px; border-top: 1px solid rgba(40,30,20,.1); }}
    button, a {{ border: 1px solid rgba(40,30,20,.14); background: white; color: #241c18; border-radius: 8px; padding: 9px 13px; text-decoration: none; cursor: pointer; }}
    button.on {{ background: #187a69; color: white; border-color: #187a69; }}
    code {{ color: #695d54; font-size: 12px; }}
  </style>
</head>
<body>
<main>
  <div class="top">
    <div><strong>{pet_html}</strong> <code>{html.escape(' / '.join(CLIPS))}</code></div>
    <div class="buttons">{buttons}</div>
  </div>
  <section class="grid">{card_html}
  </section>
</main>
<script>
document.querySelectorAll("button[data-clip]").forEach((button) => {{
  button.addEventListener("click", () => {{
    document.querySelectorAll("button[data-clip]").forEach((b) => b.classList.toggle("on", b === button));
    document.querySelectorAll("img.pet[data-pet]").forEach((img) => {{
      img.src = "/" + encodeURIComponent(img.dataset.pet) + "/anim_" + button.dataset.clip + ".webp?t=" + Date.now();
    }});
  }});
}});
</script>
</body>
</html>
"""
        payload = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def absolute_url(self, path: str) -> str:
        host = self.headers.get("Host") or f"{self.server.server_address[0]}:{self.server.server_address[1]}"
        return f"http://{host}{path}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()

    ensure_worker()
    handler = functools.partial(Handler, directory=str(OUTPUT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving API and {OUTPUT} on http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
