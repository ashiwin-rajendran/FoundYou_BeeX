#!/usr/bin/env python3
"""Visual spot-check: run the model on real episodes and save overlay images.

    python3 predict.py                                    # 20 val episodes, all classes
    python3 predict.py --checkpoint checkpoints/underwater_drone_adapters/adapters_best.pth
    python3 predict.py --classes body --n 10               # just the drone
    python3 predict.py --split train                       # sanity check on train data

For each sampled episode, saves one image: reference (with its prompt mask) |
query with GROUND TRUTH (green) | query with PREDICTION (red). Mismatches are
visible directly, not just as a number -- a 70% IoU on a thin cable can look
very different from a 70% IoU on a compact anode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from engine import build_labelled_datasets, build_model, forward_episode

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def to_bgr(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return np.clip(img, 0, 1)[:, :, ::-1] * 255


def overlay(bgr: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    out = bgr.copy()
    out[mask > 0] = 0.45 * out[mask > 0] + 0.55 * np.array(color)
    return out.astype(np.uint8)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = ((pred > 0) | (gt > 0)).sum()
    return float(((pred > 0) & (gt > 0)).sum() / union) if union else 1.0


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/underwater_drone.yaml"))
    parser.add_argument("--checkpoint",
                        default="checkpoints/underwater_drone_adapters/adapters_best.pth")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--classes", nargs="*", default=None,
                        help="restrict to these classes, e.g. --classes body cable")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out_dir", default="predict_out")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.classes:
        cfg["data"]["classes"] = args.classes
    device = torch.device(args.device)

    dataset = build_labelled_datasets(cfg, splits=(args.split,))[args.split]
    model = build_model(cfg["paths"]["model_config"], cfg["paths"]["checkpoint"], device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("model", state)
        model.load_state_dict(state, strict=False)
        print(f"loaded {args.checkpoint}")
    model.eval()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n, len(dataset)), replace=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch1 = lambda t: t[None]

    rows = []
    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        batch = {k: batch1(v) if torch.is_tensor(v) else [v] for k, v in sample.items()}
        with torch.no_grad():
            logits, target = forward_episode(model, batch, 0, device)
        pred = (logits.sigmoid() > 0.5).cpu().numpy()
        gt = target.cpu().numpy() > 0.5
        score = iou(pred, gt)

        ref_bgr = to_bgr(sample["ref_image"])
        ref_mask = sample["ref_mask"].numpy()
        qry_bgr = to_bgr(sample["query_image"])
        gt_img = overlay(qry_bgr, gt, (0, 200, 0))
        pred_img = overlay(qry_bgr, pred, (0, 0, 255))
        ref_img = overlay(ref_bgr, ref_mask, (0, 200, 0))

        h = 260
        resize = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        for im, label in ((ref_img, "reference (prompt)"), (gt_img, "query: ground truth"),
                          (pred_img, "query: prediction")):
            cv2.putText(im, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        row = np.hstack([resize(ref_img), np.full((h, 3, 3), 255, np.uint8),
                         resize(gt_img), np.full((h, 3, 3), 255, np.uint8), resize(pred_img)])
        cv2.putText(row, f"{sample['class_name']}  IoU={score*100:.1f}%", (6, h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        name = f"{i:03d}_{sample['class_name']}_iou{score*100:.0f}.jpg"
        cv2.imwrite(str(out_dir / name), row, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append((sample["class_name"], score))

    print(f"\nwrote {len(rows)} images to {out_dir}/")
    by_class: dict = {}
    for name, score in rows:
        by_class.setdefault(name, []).append(score)
    for name, scores in sorted(by_class.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {name:<20} {sum(scores)/len(scores)*100:6.1f}%  (n={len(scores)})")


if __name__ == "__main__":
    main()
