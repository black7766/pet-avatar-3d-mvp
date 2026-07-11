"""Run a command while holding a cross-process benchmark GPU lock."""

from __future__ import annotations

import argparse
import errno
import os
import subprocess
import sys
import time
from pathlib import Path


def lock_file(handle, timeout_seconds: float, poll_seconds: float) -> None:
    if os.name == "nt":
        import msvcrt

        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EDEADLK, 13, 36}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for GPU lock after {timeout_seconds:.1f}s"
                    ) from exc
                time.sleep(poll_seconds)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def unlock_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path(".models/gpu_benchmark.lock"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        waiting_started = time.perf_counter()
        print(f"waiting for GPU benchmark lock: {args.lock_file}", flush=True)
        lock_file(handle, args.timeout_seconds, args.poll_seconds)
        waited = time.perf_counter() - waiting_started
        print(f"acquired GPU benchmark lock after {waited:.3f}s", flush=True)
        try:
            return subprocess.run(command, check=False).returncode
        finally:
            unlock_file(handle)
            print("released GPU benchmark lock", flush=True)


if __name__ == "__main__":
    sys.exit(main())
