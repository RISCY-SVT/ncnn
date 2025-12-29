#!/usr/bin/env python3
"""
###############################################################################
# Version: 0.1.1
# Date:    2025-12-26
# Author:  Sergey Tyurin
# License: MIT
###############################################################################

Create a COCO2017 calibration image set for NCNN INT8 calibration.

What it does:
- Downloads COCO annotations archive (annotations_trainval2017.zip)
- Extracts instances_<split>.json (val2017 by default)
- Selects N random images (reproducible via --seed)
- Downloads ONLY those N images individually from images.cocodataset.org
- Produces:
    <out_dir>/images/*.jpg
    <out_dir>/imagelist.txt     (absolute paths, shuffled)
    <out_dir>/selection.json    (metadata)

This is tailored for ncnn/tools/k1x/convert_yolo_to_ncnn.py which accepts
--calib-dir or --imagelist for ncnn2table.

Notes:
- This script intentionally avoids heavy dependencies (FiftyOne, pycocotools).
- It does NOT download val2017.zip (full image archive). It downloads only N JPEGs.

------------------------------------------------------------------------------
Example usage:
python3 make_coco2017_calib_for_ncnn.py \
  --out-dir /data/datasets/coco_calib2K \
  --n 2000 \
  --seed 123 \
  --split val2017 \
  --jobs 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ANNOT_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

SPLITS = {
    "val2017": {
        "json_member": "annotations/instances_val2017.json",
        "fallback_base_url": "http://images.cocodataset.org/val2017/",
    },
    "train2017": {
        "json_member": "annotations/instances_train2017.json",
        "fallback_base_url": "http://images.cocodataset.org/train2017/",
    },
}


def _http_get_to_file(url: str, dst: Path, timeout_s: float, retries: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    headers = {
        "User-Agent": "coco-calib-downloader/1.0 (+https://images.cocodataset.org/)"
    }

    last_err: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                # Stream to disk
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)

            if not tmp.exists() or tmp.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file from {url}")

            tmp.replace(dst)
            return

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

            if attempt >= retries:
                break

            # Exponential backoff with a small cap
            sleep_s = min(10.0, 0.5 * (2 ** attempt))
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download {url} -> {dst}: {last_err}")


def _is_probably_jpeg(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 1024:
            return False
        with open(path, "rb") as f:
            head = f.read(2)
            f.seek(-2, os.SEEK_END)
            tail = f.read(2)
        return head == b"\xff\xd8" and tail == b"\xff\xd9"
    except Exception:
        return False


def _extract_member(zip_path: Path, member: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        if member not in zf.namelist():
            raise RuntimeError(f"ZIP does not contain '{member}'. Available: {len(zf.namelist())} members")
        with zf.open(member) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _load_instances_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"Invalid COCO JSON: missing/empty 'images' in {path}")

    # Normalize and sort for reproducibility
    norm: List[Dict] = []
    for im in images:
        if not isinstance(im, dict):
            continue
        file_name = im.get("file_name")
        image_id = im.get("id")
        if not file_name or image_id is None:
            continue
        norm.append(
            {
                "id": int(image_id),
                "file_name": str(file_name),
                "coco_url": str(im.get("coco_url") or ""),
            }
        )

    norm.sort(key=lambda x: x["id"])
    if not norm:
        raise RuntimeError(f"No valid image entries found in {path}")

    return norm


def _make_image_url(rec: Dict, fallback_base_url: str) -> str:
    # Prefer coco_url if present, otherwise build from base + filename
    url = rec.get("coco_url") or ""
    if url:
        url = url.strip()
    if not url:
        url = fallback_base_url.rstrip("/") + "/" + rec["file_name"].lstrip("/")
    # # Prefer https
    # if url.startswith("http://"):
    #     url = "https://" + url[len("http://") :]
    # Keep URL scheme as provided (COCO endpoints are commonly served over plain HTTP).
    return url


def _download_one_image(
    rec: Dict,
    out_images_dir: Path,
    fallback_base_url: str,
    timeout_s: float,
    retries: int,
) -> Tuple[bool, str]:
    """
    Returns (success, message). On success, message is the absolute path.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)

    file_name = rec["file_name"]
    dst = (out_images_dir / file_name).resolve()

    # If exists and looks ok, reuse
    if _is_probably_jpeg(dst):
        return True, str(dst)

    url = _make_image_url(rec, fallback_base_url)

    try:
        _http_get_to_file(url, dst, timeout_s=timeout_s, retries=retries)
        if not _is_probably_jpeg(dst):
            try:
                dst.unlink()
            except OSError:
                pass
            return False, f"Downloaded file is not a valid JPEG: {dst}"
        return True, str(dst)
    except Exception as e:
        return False, f"{file_name}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download a small COCO2017 split subset (images only) for NCNN INT8 calibration"
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory (will be created)")
    ap.add_argument("--n", type=int, default=500, help="Number of images to download (default: 500)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")
    ap.add_argument(
        "--split",
        choices=sorted(SPLITS.keys()),
        default="val2017",
        help="COCO split to sample from (default: val2017)",
    )
    ap.add_argument("--jobs", type=int, default=8, help="Parallel download threads (default: 8)")
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds (default: 30)")
    ap.add_argument("--retries", type=int, default=5, help="Retries per file (default: 5)")
    ap.add_argument(
        "--keep-annotations-zip",
        action="store_true",
        help="Keep annotations ZIP in out_dir (default: keep it anyway)",
    )
    args = ap.parse_args()

    if args.n <= 0:
        print("ERROR: --n must be > 0", file=sys.stderr)
        return 2
    if args.jobs <= 0:
        print("ERROR: --jobs must be > 0", file=sys.stderr)
        return 2

    out_dir = args.out_dir.resolve()
    out_images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download + extract instances JSON
    ann_zip = out_dir / "annotations_trainval2017.zip"
    instances_json = out_dir / f"instances_{args.split}.json"
    member = SPLITS[args.split]["json_member"]
    fallback_base_url = SPLITS[args.split]["fallback_base_url"]

    if not ann_zip.exists() or ann_zip.stat().st_size == 0:
        print(f"[INFO] Downloading annotations ZIP: {ANNOT_ZIP_URL}")
        _http_get_to_file(ANNOT_ZIP_URL, ann_zip, timeout_s=args.timeout, retries=args.retries)
    else:
        print(f"[INFO] Reusing existing ZIP: {ann_zip}")

    if not instances_json.exists() or instances_json.stat().st_size == 0:
        print(f"[INFO] Extracting {member} -> {instances_json}")
        _extract_member(ann_zip, member, instances_json)
    else:
        print(f"[INFO] Reusing existing JSON: {instances_json}")

    # Load image index and prepare deterministic order
    images = _load_instances_json(instances_json)
    rnd = random.Random(args.seed)

    idxs = list(range(len(images)))
    rnd.shuffle(idxs)

    # Download until we have N successes
    target = args.n
    downloaded_paths: List[str] = []
    selected_meta: List[Dict] = []

    print(f"[INFO] Sampling from {args.split}: available={len(images)}, target={target}")

    # We'll submit tasks in batches to avoid overscheduling
    next_pos = 0
    in_flight: List[cf.Future] = []

    def submit_one(i: int) -> cf.Future:
        rec = images[i]
        return executor.submit(
            _download_one_image,
            rec,
            out_images_dir,
            fallback_base_url,
            args.timeout,
            args.retries,
        )

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        # Prime the queue
        while next_pos < len(idxs) and len(in_flight) < args.jobs * 2 and len(downloaded_paths) < target:
            in_flight.append(submit_one(idxs[next_pos]))
            next_pos += 1

        done_count = 0
        fail_count = 0

        while in_flight and len(downloaded_paths) < target:
            done, pending = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
            in_flight = list(pending)

            for fut in done:
                ok, msg = fut.result()
                done_count += 1
                if ok:
                    downloaded_paths.append(msg)
                else:
                    fail_count += 1
                    print(f"[WARN] {msg}", file=sys.stderr)

            # Refill queue
            while next_pos < len(idxs) and len(in_flight) < args.jobs * 2 and len(downloaded_paths) < target:
                in_flight.append(submit_one(idxs[next_pos]))
                next_pos += 1

            # Progress
            if done_count % 25 == 0 or len(downloaded_paths) >= target:
                print(f"[INFO] progress: ok={len(downloaded_paths)}/{target} done={done_count} fail={fail_count}")

    if len(downloaded_paths) < target:
        print(f"[ERROR] Could not download enough images: got {len(downloaded_paths)} / {target}", file=sys.stderr)
        return 1

    # Keep exactly N and shuffle imagelist order (still reproducible)
    downloaded_paths = downloaded_paths[:target]
    rnd.shuffle(downloaded_paths)

    # selection.json (best-effort: map path -> record)
    # We'll rebuild selected_meta by filename lookup
    by_name = {im["file_name"]: im for im in images}
    for p in downloaded_paths:
        fn = Path(p).name
        rec = by_name.get(fn, {"file_name": fn, "id": -1, "coco_url": ""})
        selected_meta.append(
            {
                "id": rec.get("id", -1),
                "file_name": rec.get("file_name", fn),
                "url": _make_image_url(rec, fallback_base_url),
                "path": p,
            }
        )

    imagelist_path = (out_dir / "imagelist.txt").resolve()
    with open(imagelist_path, "w", encoding="utf-8") as f:
        for p in downloaded_paths:
            f.write(p + "\n")

    selection_path = (out_dir / "selection.json").resolve()
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "n": target,
                "seed": args.seed,
                "generated_at_unix": int(time.time()),
                "images": selected_meta,
            },
            f,
            indent=2,
        )

    print(f"[OK] Images dir: {out_images_dir}")
    print(f"[OK] Imagelist: {imagelist_path}")
    print(f"[OK] Selection: {selection_path}")
    print("[OK] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
