#!/usr/bin/env python3
"""Index the annotated front-camera (RGB) recordings into one file.

    python3 index_labelled.py

Scans the COCO label zips in every configured store, pairs each with its image
zip, and writes labelled_index.json holding every polygon. Training then only
ever opens image zips, never the label zips.

Two stores, disjoint and complementary:
  ML_Datasets/ml_labels   158 recordings, 13 offshore structure classes
  Yolo_Seg/ml_labels       16 recordings, body + anomaly

Every class that is actually annotated is kept. Nothing is filtered down to a
single class.

VERIFICATION IMAGES
--------------------
Also writes up to --verify_samples (default 100) sample episodes to
--verify_dir (default verify_labelled/): the reference frame and query frame
side by side, each with its class mask overlaid in red, exactly as an episode
would be built at training time (same group = same recording + class, two
distinct frames). Samples are spread round-robin across TRAINABLE classes, not
drawn in proportion to frame count, so a rare class (5 recordings) gets equal
inspection chances to `member` (136 recordings) rather than being drowned out.
Filenames carry the class, recording stem and frame numbers so you can trace
any image back to its source. Pass --verify_samples 0 to skip this.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import zipfile
from pathlib import Path

DEFAULT_STORES = [
    "/mnt/beex_datasets/Internal_Data/ML_Datasets",
    "/mnt/BeeX-Ashwin/Vision_Tools/Online_Registration/Yolo_Seg",
]


def index_store(store: str) -> list:
    """One entry per recording that has both labels and images."""
    label_dir, image_dir = Path(store) / "ml_labels", Path(store) / "ml_images"
    recordings = []
    for label_zip in sorted(label_dir.glob("*front_labels.zip")):
        stem = label_zip.name.replace("_labels.zip", "")
        image_zip = image_dir / f"{stem}.zip"
        if not image_zip.exists():
            continue
        try:
            archive = zipfile.ZipFile(label_zip)
        except Exception:
            continue

        # file_name in the COCO json must resolve to an entry in the image zip.
        try:
            members = {os.path.basename(n): n for n in zipfile.ZipFile(image_zip).namelist()
                       if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        except Exception:
            continue

        frames = {}
        for entry in archive.namelist():
            if not entry.lower().endswith(".json"):
                continue
            try:
                doc = json.loads(archive.read(entry))
            except Exception:
                continue
            categories = {c["id"]: c["name"] for c in doc.get("categories", [])}
            images = {im["id"]: im for im in doc.get("images", [])}

            by_image = collections.defaultdict(lambda: collections.defaultdict(list))
            for ann in doc.get("annotations", []):
                if ann.get("iscrowd") or ann.get("ignore"):
                    continue
                segmentation = ann.get("segmentation")
                if not isinstance(segmentation, list) or not segmentation:
                    continue           # RLE/bbox-only carries no polygon to rasterise
                name = categories.get(ann["category_id"])
                if not name:
                    continue
                polygons = [p for p in segmentation if isinstance(p, list) and len(p) >= 6]
                if polygons:
                    by_image[ann["image_id"]][name].extend(polygons)

            for image_id, per_class in by_image.items():
                meta = images.get(image_id)
                if not meta:
                    continue
                member = members.get(os.path.basename(meta["file_name"]))
                if not member:
                    continue
                # Same frame can appear in several json revisions; keep the richest.
                existing = frames.get(member)
                if existing and sum(len(v) for v in existing["anns"].values()) >= \
                        sum(len(v) for v in per_class.values()):
                    continue
                frames[member] = {"width": meta.get("width"), "height": meta.get("height"),
                                  "anns": {k: v for k, v in per_class.items()}}

        if frames:
            recordings.append({"stem": stem, "images_zip": str(image_zip), "frames": frames})
    return recordings



def render_episode_samples(recordings: list, out_dir: Path, n_samples: int, seed: int = 17) -> None:
    """Write n_samples (ref, query) episode pairs as side-by-side overlay jpgs.

    Mirrors CocoEpisodes' episode definition exactly: one group is (recording,
    class), an episode is two distinct frames from that group, the mask is the
    rasterised union of that class's polygons in each frame. This does not
    import underwater_dataset (keeps this script's dependencies light until
    verification is actually requested) -- it reimplements the same two lines
    of pairing logic instead.
    """
    import random
    import cv2
    import numpy as np

    groups = []  # (recording_idx, class_name, [frame_names])
    for idx, recording in enumerate(recordings):
        if not recording.get("classes"):
            continue
        per_class = collections.defaultdict(list)
        for frame_name, frame in recording["frames"].items():
            for name in frame["anns"]:
                if name in recording["classes"]:
                    per_class[name].append(frame_name)
        for name, frame_names in per_class.items():
            if len(frame_names) >= 2:
                groups.append((idx, name, sorted(frame_names)))

    if not groups:
        print("\nno trainable groups -- nothing to render for verification")
        return

    by_class = collections.defaultdict(list)
    for g in groups:
        by_class[g[1]].append(g)

    rng = random.Random(seed)
    for bucket in by_class.values():
        rng.shuffle(bucket)

    # Round-robin across classes so a rare class gets the same inspection
    # chances as `member`, instead of samples being handed out by frame count.
    class_names = sorted(by_class)
    order, cursors = [], {c: 0 for c in class_names}
    while len(order) < n_samples and any(cursors[c] < len(by_class[c]) for c in class_names):
        for c in class_names:
            if cursors[c] < len(by_class[c]):
                order.append(by_class[c][cursors[c]])
                cursors[c] += 1
            if len(order) >= n_samples:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_cache: dict = {}

    def load(images_zip, frame_name, class_name, polygons):
        if images_zip not in zip_cache:
            zip_cache[images_zip] = zipfile.ZipFile(images_zip)
        buf = np.frombuffer(zip_cache[images_zip].read(frame_name), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        for polygon in polygons:
            pts = np.array(polygon, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        overlay = bgr.copy()
        overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        frac = mask.mean()
        cv2.putText(overlay, "%s  %.1f%%" % (class_name, 100 * frac), (8, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return overlay

    written = 0
    for rec_idx, class_name, frame_names in order:
        recording = recordings[rec_idx]
        a, b = rng.sample(frame_names, 2)
        try:
            img_a = load(recording["images_zip"], a, class_name, recording["frames"][a]["anns"][class_name])
            img_b = load(recording["images_zip"], b, class_name, recording["frames"][b]["anns"][class_name])
        except Exception as exc:
            print("  skip (unreadable): %s / %s (%s)" % (recording["stem"], class_name, exc))
            continue
        h = min(img_a.shape[0], img_b.shape[0], 420)
        resize = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        pair = np.hstack([resize(img_a), np.full((h, 4, 3), 255, np.uint8), resize(img_b)])
        safe_stem = recording["stem"][:60].replace("/", "_")
        name = "%03d_%s_%s_ref-%s_qry-%s.jpg" % (
            written, class_name, safe_stem, Path(a).stem[-8:], Path(b).stem[-8:])
        cv2.imwrite(str(out_dir / name), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1

    classes_shown = sorted({g[1] for g in order[:written]})
    print(f"\nverification: wrote {written} episode pairs to {out_dir}")
    print(f"  classes covered: {', '.join(classes_shown)}")
    print("  (left = reference frame, right = query frame, same red mask convention)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stores", nargs="*", default=DEFAULT_STORES)
    parser.add_argument("--out", default=str(Path(__file__).parent / "labelled_index.json"))
    parser.add_argument("--min_frames_per_class", type=int, default=2,
                        help="an episode needs two frames of the same class")
    parser.add_argument("--verify_dir", default=str(Path(__file__).parent / "verify_labelled"),
                        help="where to write sample episode images; '' disables the dir default")
    parser.add_argument("--verify_samples", type=int, default=100,
                        help="sample episode pairs to render for visual inspection; 0 to skip")
    parser.add_argument("--verify_seed", type=int, default=17)
    args = parser.parse_args()

    recordings = []
    for store in args.stores:
        if not Path(store).is_dir():
            print(f"  skip (missing): {store}")
            continue
        found = index_store(store)
        print(f"  {store}: {len(found)} recordings")
        recordings.extend(found)

    class_counts = collections.Counter()
    groups = 0
    for recording in recordings:
        per_class = collections.Counter()
        for frame in recording["frames"].values():
            for name in frame["anns"]:
                per_class[name] += 1
                class_counts[name] += 1
        recording["classes"] = {k: v for k, v in per_class.items()
                                if v >= args.min_frames_per_class}
        groups += len(recording["classes"])

    index = {"recordings": recordings, "classes": dict(class_counts.most_common()),
             "min_frames_per_class": args.min_frames_per_class}
    Path(args.out).write_text(json.dumps(index))

    frames = sum(len(r["frames"]) for r in recordings)
    # A class is only trainable if some ONE recording holds >= min_frames_per_class
    # frames of it -- an episode pairs two frames of the same class from the same
    # recording. A class spread one-frame-per-recording is indexed but unusable.
    usable = collections.Counter()
    for recording in recordings:
        for name in recording["classes"]:
            usable[name] += 1

    print(f"\nrecordings : {len(recordings)}")
    print(f"frames     : {frames:,}")
    print(f"episode groups (recording x class, >= {args.min_frames_per_class} frames): {groups}")
    print(f"classes    : {len(class_counts)} indexed, {len(usable)} trainable")
    print(f"\n  {'class':<20} {'frames':>7} {'recordings':>11}  status")
    for name, count in class_counts.most_common():
        recs = usable.get(name, 0)
        status = "trainable" if recs else f"NOT trainable (no recording has >= {args.min_frames_per_class})"
        print(f"  {name:<20} {count:7d} {recs:11d}  {status}")
    skipped = [n for n in class_counts if n not in usable]
    if skipped:
        print(f"\n  excluded from training: {', '.join(skipped)}")
    print(f"\nwrote      : {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    if args.verify_samples > 0 and args.verify_dir:
        render_episode_samples(recordings, Path(args.verify_dir), args.verify_samples, args.verify_seed)


if __name__ == "__main__":
    main()
