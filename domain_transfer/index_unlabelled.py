#!/usr/bin/env python3
"""Index the unlabelled RGB_FRAMES corpus once, so training can sample it fast.

    python3 index_unlabelled.py                      # index everything (~2.66 M frames)
    python3 index_unlabelled.py --max_recordings 300 # a quick subset

Walking 2.66 M files takes minutes, so the result is cached to
unlabelled_index.json and reused. Only recordings with enough frames to form a
temporal pair are kept.

The index stores directories and frame counts, not file lists -- frames are
named frame_NNNNNN.jpg contiguously, so the dataset rebuilds paths by number.
That keeps the index small (a few hundred KB instead of ~200 MB).

VERIFICATION IMAGES
--------------------
Also writes up to --verify_samples (default 100) sample (frame_a, frame_b)
pairs to --verify_dir (default verify_unlabelled/), side by side with the
frame numbers and gap labelled -- the exact pairing UnlabelledPairs draws for
cycle-consistency training (same recording, frame_gap frames apart, default
3-15). Recordings are sampled spread across the whole corpus, not just the
first ones found. Pass --verify_samples 0 to skip this.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

FRAME_RE = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)


def index_corpus(root: str, min_frames: int, max_recordings: int = 0) -> dict:
    recordings = []
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        numbers = []
        for name in filenames:
            match = FRAME_RE.match(name)
            if match:
                numbers.append(int(match.group(1)))
        if len(numbers) < min_frames:
            continue
        numbers.sort()
        # Contiguity matters: the dataset reconstructs paths from frame numbers,
        # and a recording with gaps would produce missing files.
        contiguous = numbers[-1] - numbers[0] + 1 == len(numbers)
        recordings.append({
            "dir": os.path.relpath(dirpath, root),
            "first": numbers[0],
            "count": len(numbers),
            "contiguous": contiguous,
        })
        total += len(numbers)
        if max_recordings and len(recordings) >= max_recordings:
            break
    return {"root": root, "min_frames": min_frames,
            "recordings": recordings, "total_frames": total}



def render_pair_samples(index: dict, out_dir: Path, n_samples: int,
                        frame_gap: tuple = (3, 15), seed: int = 17) -> None:
    """Write n_samples (frame_a, frame_b) temporal pairs for visual inspection.

    Mirrors UnlabelledPairs' sampling exactly: one recording, two frames
    frame_gap apart. Recordings are sampled evenly across the whole indexed
    list (not just the first N found) so the sample reflects the corpus, not
    whichever project os.walk happened to reach first.
    """
    import random
    import cv2

    recordings = [r for r in index["recordings"]
                  if r["contiguous"] and r["count"] > frame_gap[1] + 1]
    if not recordings:
        print("\nno recording has enough contiguous frames -- nothing to render")
        return

    rng = random.Random(seed)
    root = Path(index["root"])
    # Evenly spaced picks across the indexed order, not a fully random sample,
    # so the result spans the corpus (many different projects) rather than
    # clustering wherever os.walk listed recordings first.
    step = max(1, len(recordings) // n_samples)
    picks = recordings[::step][:n_samples]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for recording in picks:
        gap = rng.randint(frame_gap[0], frame_gap[1])
        start = recording["first"]
        last = start + recording["count"] - 1 - gap
        if last <= start:
            continue
        number = rng.randint(start, last)
        path_a = root / recording["dir"] / f"frame_{number:06d}.jpg"
        path_b = root / recording["dir"] / f"frame_{number + gap:06d}.jpg"
        img_a, img_b = cv2.imread(str(path_a)), cv2.imread(str(path_b))
        if img_a is None or img_b is None:
            print("  skip (unreadable): %s" % recording["dir"])
            continue
        h = min(img_a.shape[0], img_b.shape[0], 420)
        resize = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        for im, tag in ((img_a, "t=%06d" % number), (img_b, "t=%06d  (+%d)" % (number + gap, gap))):
            cv2.putText(im, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        import numpy as np
        pair = np.hstack([resize(img_a), np.full((h, 4, 3), 255, np.uint8), resize(img_b)])
        safe_dir = recording["dir"].replace("/", "_")[:70]
        name = "%03d_%s_t%06d_gap%d.jpg" % (written, safe_dir, number, gap)
        cv2.imwrite(str(out_dir / name), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1

    print(f"\nverification: wrote {written} temporal pairs to {out_dir}")
    print(f"  recordings sampled: {len(picks)} (spread across {len(recordings)} usable recordings)")
    print("  (left = frame t, right = frame t+gap, gap drawn from frame_gap=%s)" % (frame_gap,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/mnt/beex_datasets/Internal_Data/Datasets_VIT/RGB_FRAMES")
    parser.add_argument("--out", default=str(Path(__file__).parent / "unlabelled_index.json"))
    parser.add_argument("--min_frames", type=int, default=30)
    parser.add_argument("--max_recordings", type=int, default=0)
    parser.add_argument("--verify_dir", default=str(Path(__file__).parent / "verify_unlabelled"),
                        help="where to write sample frame-pair images; '' disables the dir default")
    parser.add_argument("--verify_samples", type=int, default=100,
                        help="sample temporal pairs to render for visual inspection; 0 to skip")
    parser.add_argument("--verify_frame_gap", type=int, nargs=2, default=[3, 15],
                        help="must match semisup.frame_gap in the training config to verify the real thing")
    parser.add_argument("--verify_seed", type=int, default=17)
    args = parser.parse_args()

    print(f"walking {args.root} ...")
    index = index_corpus(args.root, args.min_frames, args.max_recordings)
    gaps = sum(1 for r in index["recordings"] if not r["contiguous"])
    Path(args.out).write_text(json.dumps(index))
    print(f"recordings : {len(index['recordings'])}  ({gaps} with numbering gaps)")
    print(f"frames     : {index['total_frames']:,}")
    print(f"wrote      : {args.out}")

    if args.verify_samples > 0 and args.verify_dir:
        render_pair_samples(index, Path(args.verify_dir), args.verify_samples,
                            tuple(args.verify_frame_gap), args.verify_seed)


if __name__ == "__main__":
    main()
