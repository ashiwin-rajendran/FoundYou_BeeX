"""Shared pieces for FoundYou underwater domain transfer.

train.py and evaluate.py both import from here so the forward pass, the
freezing rules and the metric can never drift apart between the two.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

# Make the FoundYou repo importable when run from this subdirectory.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.foundyou import build_foundyou                 # noqa: E402
from util.commons import load_checkpoint                   # noqa: E402
from util.promptable_utils import build_prompt_dict        # noqa: E402
from metrics.segmentation import (                         # noqa: E402
    BinarySegmentationEvaluator,
    BinarySegmentationMeter,
)

# Parameter-name prefixes of what the published FoundYou checkpoint actually
# contains (167 tensors, 5.88M params) -- confirmed by reading the released
# model.safetensors:
#
#   sam.image_encoder.trunk.*.adapter.*       2.453M   AdaptFormer adapters
#   sam.retrieval_decoder.*                   3.423M   retrieval branch only
#
# sam_mask_decoder is NOT in it: the segmentation head stays at stock SAM2
# weights, and the adapters are the only thing adapting the segmentation
# path. That is why adapters are the default trainable set here -- it is the
# same surface the authors trained, and the one the architecture is designed
# to move.
ADAPTER_KEY = "adapter"
MASK_DECODER_PREFIX = "sam.sam_mask_decoder"


@contextlib.contextmanager
def _in_repo_root():
    """Run with the FoundYou repo as CWD.

    SAM2_PATHS_CONFIG stores the backbone weights at the RELATIVE path
    'pretrain/sam2_hiera_small.pt', so build_foundyou() resolves it against
    whatever the current directory happens to be -- and silently re-downloads
    184 MB into a second pretrain/ when called from this subdirectory.
    """
    previous = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(previous)


def build_model(
    model_config: str,
    checkpoint: str,
    device: torch.device,
    train_mask_decoder: bool = False,
) -> torch.nn.Module:
    """Build FoundYou, load weights, and freeze everything except the adapters."""
    config_path = model_config if Path(model_config).is_absolute() else str(REPO / model_config)
    with _in_repo_root():
        model = build_foundyou(config_path)
    if checkpoint:
        load_checkpoint(checkpoint, model)
    model.to(device)

    for name, param in model.named_parameters():
        trainable = ADAPTER_KEY in name.lower()
        if train_mask_decoder and name.startswith(MASK_DECODER_PREFIX):
            trainable = True
        param.requires_grad = trainable
    return model


def trainable_summary(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def forward_episode(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    index: int,
    device: torch.device,
    prompt_type: str = "mask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one episode. Returns (logits[H,W], ground-truth[H,W]).

    Episodes are processed one at a time rather than as a batched tensor:
    encode_references() builds a memory bank per episode, so references from
    different episodes cannot share a forward pass. With ~2.9 GB per episode
    at 518px this is also the only thing that fits -- use gradient
    accumulation (train.accum_steps) for a larger effective batch.
    """
    ref_image = batch["ref_image"][index].to(device)
    ref_mask = batch["ref_mask"][index].to(device)[None, None]     # [1,1,S,S]
    query_image = batch["query_image"][index].to(device)[None]     # [1,3,S,S]
    query_mask = batch["query_mask"][index].to(device)             # [S,S]

    prompt = build_prompt_dict(ref_mask, prompt_type, device)
    context = model.encode_references([ref_image], [prompt])
    logits = model.segment_candidates(query_image, context)        # [1,S,S]
    return logits[0], query_mask


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BCE + soft Dice.

    Dice carries the overlap signal and is insensitive to how much of the
    frame the drone occupies (here: 0 to 38%, varying shot to shot); BCE keeps
    per-pixel calibration from drifting. Using BCE alone on a small, mostly
    background object biases towards predicting empty masks.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)

    probs = logits.sigmoid()
    intersection = (probs * target).sum()
    denominator = probs.sum() + target.sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

    loss = bce_weight * bce + dice_weight * dice
    return loss, {"bce": bce.item(), "dice": dice.item(), "loss": loss.item()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    prompt_type: str = "mask",
    threshold: float = 0.5,
    max_episodes: int = 0,
) -> Dict[str, float]:
    """mIoU / boundary-IoU over episodes, using the repo's own metric classes."""
    model.eval()
    evaluator = BinarySegmentationEvaluator()
    meter = BinarySegmentationMeter()

    seen = 0
    for batch in loader:
        for i in range(batch["query_image"].shape[0]):
            logits, target = forward_episode(model, batch, i, device, prompt_type)
            pred = (logits.sigmoid() > threshold).cpu()
            gt = target.cpu() > 0.5
            inter, union, b_inter, b_union = evaluator.classify_prediction(pred, gt)
            # Group by class so mIoU is a mean over CLASSES, not over episodes.
            # Without this, `member` (6.6k frames) would drown the 13 smaller
            # classes and the headline number would describe one of them.
            names = batch.get("class_name")
            group = names[i] if isinstance(names, (list, tuple)) and i < len(names) else "object"
            meter.update(group, inter.item(), union.item(), b_inter.item(), b_union.item())
            seen += 1
            if max_episodes and seen >= max_episodes:
                break
        if max_episodes and seen >= max_episodes:
            break

    miou, mbiou = meter.compute()
    # Per-class IoU as well as the mean. With `member` at 6604 frames against
    # 31 for iccp_anode_face, the mean alone hides which classes actually work.
    per_class = {name: stats["intersection"] / (stats["union"] + 1e-10)
                 for name, stats in meter.groups.items()}
    return {"mIoU": miou, "bIoU": mbiou, "episodes": seen,
            "per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1]))}


def random_box_prompt(size: int, rng, device: torch.device) -> torch.Tensor:
    """A random box mask used to seed an unlabelled cycle.

    The unlabelled frames have no annotations, so something has to tell the
    model which object to track. A randomly placed box lets SAM 2 turn
    whatever is inside it into an object mask, and randomising position and
    scale stops the cycle term from only ever learning about centred objects.
    """
    side = int(size * rng.uniform(0.25, 0.6))
    top = int(rng.uniform(0, size - side))
    left = int(rng.uniform(0, size - side))
    mask = torch.zeros(size, size, device=device)
    mask[top:top + side, left:left + side] = 1.0
    return mask


def _segment(model, ref_image, ref_mask, query_image, prompt_type="mask"):
    prompt = build_prompt_dict(ref_mask[None, None], prompt_type, ref_image.device)
    context = model.encode_references([ref_image], [prompt])
    return model.segment_candidates(query_image[None], context)[0]


def cycle_consistency_loss(
    model: torch.nn.Module,
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
    rng,
    device: torch.device,
    min_area: float = 0.02,
    max_area: float = 0.90,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
):
    """Label-free objective: a -> b -> a must come back to where it started.

    Three passes:
      1. no_grad  seed a random box on frame A      -> mask_a  (pseudo-target)
      2. no_grad  propagate mask_a to frame B       -> mask_b
      3. GRAD     propagate mask_b back to frame A  -> compared against mask_a

    Only the backward hop carries gradient. Steps 1-2 are detached on purpose:
    if every hop were trainable, the cheapest way to satisfy the objective is
    to make all three outputs identical and empty. Freezing the target turns
    this into self-training against the model's own most-confident view (on the
    frame it was prompted from, so no propagation error has accumulated yet).

    Returns (loss, stats) or (None, stats) when the episode is unusable --
    a seed that collapses to nothing or swallows the frame carries no signal,
    so it is skipped rather than trained on.
    """
    frame_a, frame_b = frame_a.to(device), frame_b.to(device)
    size = frame_a.shape[-1]

    with torch.no_grad():
        seed = random_box_prompt(size, rng, device)
        mask_a = (_segment(model, frame_a, seed, frame_a).sigmoid() > 0.5).float()
        area_a = mask_a.mean().item()
        if not (min_area < area_a < max_area):
            return None, {"skipped": "seed_area", "area_a": area_a}

        mask_b = (_segment(model, frame_a, mask_a, frame_b).sigmoid() > 0.5).float()
        area_b = mask_b.mean().item()
        if not (min_area < area_b < max_area):
            return None, {"skipped": "propagated_area", "area_a": area_a, "area_b": area_b}

    logits_back = _segment(model, frame_b, mask_b, frame_a)
    loss, parts = segmentation_loss(logits_back.float(), mask_a, bce_weight, dice_weight)

    with torch.no_grad():
        back = (logits_back.sigmoid() > 0.5).float()
        union = ((back + mask_a) > 0).sum()
        cycle_iou = (((back * mask_a) > 0).sum() / union).item() if union > 0 else 1.0

    parts.update({"area_a": area_a, "area_b": area_b, "cycle_iou": cycle_iou, "skipped": None})
    return loss, parts


def build_labelled_datasets(cfg, splits=("train", "val")):
    """Build labelled episode datasets for whichever source the config names.

    coco       -> CocoEpisodes, all annotated classes, split by RECORDING
    front_mask -> UnderwaterEpisodes, the single `body` class, split temporally
    """
    from underwater_dataset import CocoEpisodes, UnderwaterEpisodes, verify_polarity

    data_cfg, paths, train_cfg = cfg["data"], cfg["paths"], cfg["training"]
    source = data_cfg.get("source", "coco")
    out = {}

    if source == "coco":
        index_path = Path(paths["labelled_index"])
        if not index_path.is_absolute():
            index_path = Path(__file__).resolve().parent / index_path
        if not index_path.exists():
            raise SystemExit(
                f"{index_path} is missing.\nBuild it first:  python3 index_labelled.py")
        for split in splits:
            out[split] = CocoEpisodes(
                index_path=str(index_path),
                split=split,
                img_size=data_cfg["img_size"],
                val_fraction=data_cfg["val_fraction"],
                episodes_per_epoch=data_cfg["episodes_per_epoch"],
                classes=data_cfg.get("classes") or None,
                min_mask_fraction=data_cfg.get("min_mask_fraction", 0.001),
                single_instance_only=data_cfg.get("single_instance_only", False),
                augment=data_cfg["augment"] and split == "train",
                seed=train_cfg["seed"],
            )
    elif source == "front_mask":
        print("polarity   :", verify_polarity(paths["data_root"], data_cfg["mask_invert"]))
        for split in splits:
            out[split] = UnderwaterEpisodes(
                root=paths["data_root"],
                split=split,
                img_size=data_cfg["img_size"],
                val_fraction=data_cfg["val_fraction"],
                temporal_split=data_cfg["temporal_split"],
                min_frame_gap=data_cfg["min_frame_gap"],
                episodes_per_epoch=data_cfg["episodes_per_epoch"],
                mask_invert=data_cfg["mask_invert"],
                augment=data_cfg["augment"] and split == "train",
                seed=train_cfg["seed"],
            )
    else:
        raise SystemExit(f"unknown data.source: {source!r} (use 'coco' or 'front_mask')")
    return out


def describe_dataset(dataset, label: str) -> str:
    if hasattr(dataset, "groups"):
        return (f"{label:<10} {len(dataset.groups)} groups, "
                f"{len(dataset.class_names)} classes -> {len(dataset)} episodes")
    return f"{label:<10} {len(dataset.stems)} frames -> {len(dataset)} episodes"
