#!/usr/bin/env python3
"""Fine-tune FoundYou's AdaptFormer adapters on BeeX underwater footage.

    python3 train.py                                  # uses configs/underwater_drone.yaml
    python3 train.py --config configs/my_run.yaml
    python3 train.py --resume checkpoints/<run>/adapters_last.pth

SAM 2 stays frozen; only the adapters (2.45 M of 52.4 M) move, which is the
same surface the published FoundYou checkpoint contains for the segmentation
path. Checkpoints hold ONLY the trained tensors (~10 MB), so they layer on top
of the released weights rather than duplicating all 52 M parameters.

Validation is the temporal tail of the sequence with fixed episodes, so the
number is comparable across epochs and is not measured on near-duplicates of
training frames.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import numpy as np

from engine import (build_labelled_datasets, build_model, cycle_consistency_loss,
                    describe_dataset, evaluate, forward_episode, segmentation_loss,
                    trainable_summary)
from underwater_dataset import UnlabelledPairs


def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def trainable_state(model: torch.nn.Module) -> dict:
    return {name: param.detach().cpu()
            for name, param in model.named_parameters() if param.requires_grad}


def lr_at(step: int, total: int, base_lr: float, min_lr: float, warmup: int) -> float:
    """Linear warmup then cosine decay."""
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/underwater_drone.yaml"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, data_cfg, train_cfg = cfg["paths"], cfg["data"], cfg["training"]
    loss_cfg, eval_cfg = cfg["loss"], cfg["eval"]
    device = torch.device(args.device)

    torch.manual_seed(train_cfg["seed"])

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(paths["output_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run        : {cfg['run_name']}")
    print(f"output     : {out_dir}")

    sets = build_labelled_datasets(cfg)
    train_set, val_set = sets["train"], sets["val"]
    print(describe_dataset(train_set, "train"))
    print(describe_dataset(val_set, "val"))
    if getattr(train_set, "class_names", None):
        print(f"classes    : {', '.join(train_set.class_names)}")

    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=train_cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=min(2, train_cfg["num_workers"]))

    # Unlabelled branch (optional).
    semisup_cfg = cfg.get("semisup", {}) or {}
    use_semisup = bool(semisup_cfg.get("enabled"))
    unlabelled_loader = None
    if use_semisup:
        index_path = Path(semisup_cfg["index_path"])
        if not index_path.is_absolute():
            index_path = Path(__file__).resolve().parent / index_path
        if not index_path.exists():
            raise SystemExit(
                f"semisup is on but {index_path} is missing.\n"
                f"Build it first:  python3 index_unlabelled.py")
        unlabelled_set = UnlabelledPairs(
            index_path=str(index_path),
            img_size=data_cfg["img_size"],
            frame_gap=tuple(semisup_cfg["frame_gap"]),
            length=semisup_cfg["pairs_per_epoch"],
            seed=train_cfg["seed"],
        )
        unlabelled_loader = DataLoader(unlabelled_set, batch_size=1, shuffle=False,
                                       num_workers=train_cfg["num_workers"])
        print(f"unlabelled : {len(unlabelled_set.recordings)} recordings -> "
              f"{len(unlabelled_set)} pairs/epoch, weight {semisup_cfg['weight']} "
              f"(ramped over {semisup_cfg['warmup_epochs']} epochs)")
    else:
        print("unlabelled : disabled (supervised only)")

    model = build_model(paths["model_config"], paths["checkpoint"], device,
                        train_mask_decoder=train_cfg["train_mask_decoder"])
    trainable, total = trainable_summary(model)
    print(f"params     : {total/1e6:.1f}M total, {trainable/1e6:.2f}M trainable "
          f"({100*trainable/total:.1f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_miou = 0, -1.0
    baseline_miou = None
    resume_path = args.resume or paths.get("resume_from") or ""
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        best_miou = ckpt.get("best_miou", -1.0)
        print(f"resumed    : {resume_path} @ epoch {start_epoch} (best mIoU {best_miou:.4f})")

    accum = max(1, train_cfg["accum_steps"])
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * train_cfg["epochs"]
    history = []

    # Pre-training reference point, so any later number has something honest
    # to be compared against.
    if start_epoch == 0:
        baseline = evaluate(model, val_loader, device, eval_cfg["prompt"],
                            eval_cfg["threshold"], train_cfg["val_max_episodes"])
        baseline_miou = baseline["mIoU"]
        # best_miou stays at -1 so the first epoch always writes a checkpoint.
        # Seeding it with the baseline instead meant that a run which never beat
        # zero-shot produced no adapters_best.pth at all, while still printing
        # the command to evaluate that file.
        print(f"\nzero-shot baseline: mIoU {baseline['mIoU']*100:.2f}%  "
              f"bIoU {baseline['bIoU']*100:.2f}%  ({baseline['episodes']} episodes)\n")
        history.append({"epoch": 0, "stage": "baseline", **baseline})

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, train_cfg["epochs"]):
        # Must happen BEFORE the DataLoader iterator is created: workers fork
        # at that moment and carry whatever epoch value the dataset holds.
        # Without it every epoch replays the same episodes.
        train_set.set_epoch(epoch)
        if unlabelled_loader is not None:
            unlabelled_loader.dataset.set_epoch(epoch)

        model.train()
        epoch_start = time.time()
        running = {"loss": 0.0, "bce": 0.0, "dice": 0.0}
        counted = 0
        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader):
            for i in range(batch["query_image"].shape[0]):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, target = forward_episode(model, batch, i, device, eval_cfg["prompt"])
                    loss, parts = segmentation_loss(
                        logits.float(), target.float(),
                        loss_cfg["bce_weight"], loss_cfg["dice_weight"])
                scaler.scale(loss / accum).backward()
                for key in running:
                    running[key] += parts[key]
                counted += 1

            if (it + 1) % accum == 0:
                lr = lr_at(global_step, total_steps, train_cfg["lr"],
                           train_cfg["min_lr"], train_cfg["warmup_steps"])
                for group in optimizer.param_groups:
                    group["lr"] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, train_cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    denom = max(1, counted)
                    print(f"  epoch {epoch+1} step {global_step}/{total_steps}  "
                          f"loss {running['loss']/denom:.4f} "
                          f"(bce {running['bce']/denom:.4f} dice {running['dice']/denom:.4f})  "
                          f"lr {lr:.2e}")

        denom = max(1, counted)
        train_loss = running["loss"] / denom
        entry = {"epoch": epoch + 1, "train_loss": train_loss}

        # ---- unlabelled cycle-consistency phase ----
        if unlabelled_loader is not None:
            warmup_epochs = max(1, semisup_cfg["warmup_epochs"])
            # Ramp in: the labelled anchor should be established before the
            # model starts learning from its own predictions.
            ramp = min(1.0, (epoch + 1) / warmup_epochs)
            weight = semisup_cfg["weight"] * ramp
            rng = np.random.default_rng(train_cfg["seed"] + epoch)

            cyc_loss_sum = cyc_iou_sum = 0.0
            used = skipped = 0
            optimizer.zero_grad(set_to_none=True)

            for it, pair in enumerate(unlabelled_loader):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss_u, stats = cycle_consistency_loss(
                        model, pair["frame_a"][0], pair["frame_b"][0], rng, device,
                        min_area=semisup_cfg["min_area"], max_area=semisup_cfg["max_area"],
                        bce_weight=loss_cfg["bce_weight"], dice_weight=loss_cfg["dice_weight"])
                if loss_u is None:
                    skipped += 1
                    continue
                scaler.scale(weight * loss_u / accum).backward()
                cyc_loss_sum += stats["loss"]
                cyc_iou_sum += stats["cycle_iou"]
                used += 1

                if used and used % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, train_cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            if used:
                entry.update({"cycle_loss": cyc_loss_sum / used,
                              "cycle_iou": cyc_iou_sum / used,
                              "cycle_used": used, "cycle_skipped": skipped,
                              "cycle_weight": weight})
                print(f"  unlabelled: {used} used / {skipped} skipped  "
                      f"cycle_loss {cyc_loss_sum/used:.4f}  "
                      f"cycle_IoU {cyc_iou_sum/used:.3f}  w={weight:.2f}")
            else:
                # Everything degenerate is a real signal, not a hiccup: the
                # seeds are producing empty or frame-filling masks.
                print(f"  unlabelled: ALL {skipped} pairs skipped as degenerate "
                      f"-- check min_area/max_area or frame_gap")
                entry.update({"cycle_used": 0, "cycle_skipped": skipped})

        entry["minutes"] = (time.time() - epoch_start) / 60.0

        if (epoch + 1) % train_cfg["val_every"] == 0:
            metrics = evaluate(model, val_loader, device, eval_cfg["prompt"],
                               eval_cfg["threshold"], train_cfg["val_max_episodes"])
            entry.update(metrics)
            print(f"epoch {epoch+1}/{train_cfg['epochs']}  loss {train_loss:.4f}  "
                  f"val mIoU {metrics['mIoU']*100:.2f}%  bIoU {metrics['bIoU']*100:.2f}%  "
                  f"({entry['minutes']:.1f} min)")

            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                torch.save({"model": trainable_state(model), "epoch": epoch + 1,
                            "best_miou": best_miou, "config": cfg},
                           out_dir / "adapters_best.pth")
                print(f"  new best ({best_miou*100:.2f}%) -> adapters_best.pth")
        else:
            print(f"epoch {epoch+1}/{train_cfg['epochs']}  loss {train_loss:.4f}  "
                  f"({entry['minutes']:.1f} min)")

        history.append(entry)
        if (epoch + 1) % train_cfg["checkpoint_every"] == 0:
            torch.save({"model": trainable_state(model),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch + 1, "best_miou": best_miou, "config": cfg},
                       out_dir / "adapters_last.pth")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\ndone. best val mIoU {best_miou*100:.2f}%")
    if baseline_miou is not None:
        delta = (best_miou - baseline_miou) * 100
        if delta > 0:
            print(f"  zero-shot was {baseline_miou*100:.2f}%  ->  +{delta:.2f} points")
        else:
            print(f"  WARNING: never beat the {baseline_miou*100:.2f}% zero-shot baseline "
                  f"({delta:+.2f}). The released weights are the better model here.")
    print(f"  {out_dir/'adapters_best.pth'}")
    print(f"  evaluate with: python3 evaluate.py --checkpoint {out_dir/'adapters_best.pth'}")


if __name__ == "__main__":
    main()
