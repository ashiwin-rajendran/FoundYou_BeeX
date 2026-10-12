#!/usr/bin/env python3
"""Measure FoundYou personalized-segmentation quality on the underwater set.

Run it before training to get the zero-shot baseline, and after training with
--checkpoint pointing at a fine-tuned adapter file, to see what moved:

    python3 evaluate.py                                     # zero-shot baseline
    python3 evaluate.py --checkpoint checkpoints/<run>/adapters_best.pth
    python3 evaluate.py --split train                       # sanity check

Reported on the temporal validation tail by default -- the frames held out
from the END of the sequence, which are not near-duplicates of anything the
model trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from engine import (REPO, build_labelled_datasets, build_model, describe_dataset,
                    evaluate, trainable_summary)


def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/underwater_drone.yaml"))
    parser.add_argument("--checkpoint", default=None,
                        help="adapter .pth from training; default = the released FoundYou weights")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--prompt", default=None, choices=["mask", "box", "point"])
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    prompt = args.prompt or cfg["eval"]["prompt"]
    device = torch.device(args.device)

    dataset = build_labelled_datasets(cfg, splits=(args.split,))[args.split]
    print(describe_dataset(dataset, args.split))
    if getattr(dataset, "class_names", None):
        print(f"classes: {', '.join(dataset.class_names)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(paths["model_config"], paths["checkpoint"], device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("model", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        print(f"loaded {loaded}/{len(state)} fine-tuned tensors from {args.checkpoint}")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
        if loaded == 0:
            raise SystemExit("No tensors matched the model -- wrong checkpoint file?")

    trainable, total = trainable_summary(model)
    print(f"params: {total/1e6:.1f}M total, {trainable/1e6:.2f}M trainable")

    metrics = evaluate(model, loader, device, prompt_type=prompt,
                       threshold=cfg["eval"]["threshold"], max_episodes=args.max_episodes)
    print(f"\n{args.split}  prompt={prompt}")
    print(f"  mIoU = {metrics['mIoU']*100:.2f}%")
    print(f"  bIoU = {metrics['bIoU']*100:.2f}%")
    print(f"  episodes = {metrics['episodes']}")
    if len(metrics.get("per_class", {})) > 1:
        print("\n  per class:")
        for name, iou in metrics["per_class"].items():
            print(f"    {name:<20} {iou*100:6.2f}%")

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(
            {"split": args.split, "prompt": prompt, "checkpoint": args.checkpoint, **metrics}, indent=2))
        print(f"wrote {args.save_json}")


if __name__ == "__main__":
    main()
