#!/usr/bin/env python3
"""Parallel, resume-capable downloader for large model weight files.

Each byte range is fetched with retries and saved to ``<dest>.partNN`` so a
killed run can resume by reusing completed parts.  The final file is assembled
by concatenating the parts in order.

Usage:
    python scripts/download_weights.py URL DEST [NUM_PARTS]
"""
from __future__ import annotations

import concurrent.futures
import os
import sys
import time
import urllib.request


def _content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0))


def _fetch_range(url: str, start: int, end: int, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def _part_path(dest: str, i: int) -> str:
    return f"{dest}.part{i:02d}"


def download(url: str, dest: str, num_parts: int = 8) -> None:
    total = _content_length(url)
    if total <= 0:
        raise RuntimeError(f"cannot determine content length for {url}")

    if os.path.exists(dest) and os.path.getsize(dest) == total:
        print(f"[dl] {dest} already complete ({total} bytes)")
        return

    part_size = total // num_parts
    ranges = []
    for i in range(num_parts):
        s = i * part_size
        e = total - 1 if i == num_parts - 1 else (i + 1) * part_size - 1
        if s <= e:
            ranges.append((i, s, e))

    # Load already-finished parts (resume).
    todo = []
    for i, s, e in ranges:
        pp = _part_path(dest, i)
        want = e - s + 1
        if os.path.exists(pp) and os.path.getsize(pp) == want:
            print(f"[dl] part {i} already done ({want} bytes)", flush=True)
        else:
            todo.append((i, s, e))

    t0 = time.time()
    if todo:
        print(f"[dl] {os.path.basename(dest)}: {total} bytes, "
              f"{len(todo)}/{num_parts} parts to fetch", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as ex:
            def work(item):
                i, s, e = item
                data = _fetch_range(url, s, e)
                pp = _part_path(dest, i)
                with open(pp, "wb") as f:
                    f.write(data)
                return i, len(data)

            futures = [ex.submit(work, item) for item in todo]
            done_count = 0
            for fut in concurrent.futures.as_completed(futures):
                i, n = fut.result()
                done_count += 1
                downloaded = sum(
                    os.path.getsize(_part_path(dest, j))
                    for j in range(num_parts)
                    if os.path.exists(_part_path(dest, j))
                )
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    f"[dl] {done_count}/{len(todo)} parts | {downloaded}/{total} "
                    f"bytes ({downloaded/total*100:.1f}%) | "
                    f"{downloaded/elapsed/1024:.0f} KB/s",
                    flush=True,
                )

    # Assemble.
    with open(dest, "wb") as f:
        for i in range(num_parts):
            pp = _part_path(dest, i)
            with open(pp, "rb") as pf:
                f.write(pf.read())
    print(f"[dl] done {dest} ({os.path.getsize(dest)} bytes) in "
          f"{time.time()-t0:.1f}s", flush=True)


def main() -> int:
    url = sys.argv[1]
    dest = sys.argv[2]
    num_parts = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    download(url, dest, num_parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
