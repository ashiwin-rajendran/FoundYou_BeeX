# FoundYou domain transfer — land SAM 2 → BeeX underwater

Adapts FoundYou for personalized segmentation on BeeX drone footage.
Trains on 14 annotated classes plus unlabelled survey video.

**Conda env needed?** No. Runs on system Python 3.8.10 / torch 2.4.1.

## Quick start

```bash
cd FoundYou_BeeX/domain_transfer

python3 index_labelled.py      # index the annotations   (once)
python3 index_unlabelled.py    # index RGB_FRAMES         (once)

python3 evaluate.py            # baseline
python3 train.py               # ~35 min
python3 evaluate.py --checkpoint checkpoints/multiclass_semisup/adapters_best.pth
```

Both indexers also save 100 sample images so you can check the data before
training — see [Verify the data](#verify-the-data).

## Data

165 recordings, 8,809 frames, **14 trainable classes**, from two stores:

| class | frames | recordings | class | frames | recordings |
|---|---:|---:|---|---:|---:|
| member | 6604 | 136 | jacket_jtube | 148 | 3 |
| cable | 2238 | 46 | iccp_anode | 123 | 14 |
| anode | 1866 | 38 | grout_pipe_riser | 84 | 11 |
| boatlanding | 664 | 29 | grout_skirt | 70 | 12 |
| jacket_leg | 320 | 4 | iccp_conduit | 45 | 8 |
| jacket_brace | 272 | 4 | iccp_anode_face | 31 | 5 |
| **body** (drone) | 258 | 13 | ~~anomaly~~ | 2 | — |

- `body` comes from the Yolo_Seg store; the other 13 come from beex_datasets.
  No recordings are shared between the two.
- `anomaly` is indexed but can't train — its 2 frames live in 2 different
  recordings, and an episode needs 2 frames of the same class in one recording.
- Unlabelled: 2.66M frames, 158 GB, 55 projects, no annotations.

## How episodes are built

An episode = 2 frames of the same class from the same recording. Reference
frame prompts the model, query frame is the target.

- **Split by recording**, not frame — frames in one recording are the same
  structure seconds apart, so a frame-level split would validate on
  near-duplicates.
- **Stratified** — rare classes (3–4 recordings) would otherwise land entirely
  in train. All 14 classes now appear in both splits.
- **Union masks** — COCO has no instance identity across frames, so with
  several objects in view the mask is their union. Affects 7.6% of episodes;
  set `single_instance_only: true` to avoid them.
- **mIoU is averaged per class**, not per episode, so `member` doesn't drown
  the other 13. `evaluate.py` prints the full breakdown.

## Verify the data

Both indexers save sample images by default — check these before a full run:

- `verify_labelled/` — 100 reference/query pairs, mask overlaid in red, spread
  evenly across all 14 classes (not by frame count).
- `verify_unlabelled/` — 100 `(frame t, frame t+gap)` pairs from the unlabelled
  corpus, spread across all 8,288 recordings.

```bash
python3 index_labelled.py --verify_samples 200   # more samples
python3 index_labelled.py --verify_samples 0     # skip rendering
```

Both folders are gitignored; delete anytime, they regenerate from the index.

## Semi-supervised

The unlabelled corpus has no masks, so it trains on cycle-consistency instead:

```
seed a random box on frame t  ->  mask_t      (no grad)
propagate to frame t+k        ->  mask_t+k    (no grad)
propagate back to frame t     ->  compare to mask_t   (trains)
```

Only the last hop trains. If all three did, the cheapest fix is three empty
masks — freezing the first two makes it self-training against the model's
own best guess, with the labelled loss alongside to stop collapse.

Watch `cycle_IoU` in the logs: rising = working, many `skipped` = degenerate
seeds (widen `semisup.min_area`/`max_area`).

## What trains

SAM 2 is frozen. Only the **AdaptFormer adapters** move — 2.45M of 52.4M
params (4.7%), a 9.8 MB checkpoint. This matches the released FoundYou
checkpoint itself, which also ships adapters-only for segmentation.

## Environment

| | needed | you have |
|---|---|---|
| Python | 3.10 suggested | 3.8.10 ✅ |
| PyTorch | >2.0 | 2.4.1+cu121 ✅ |

`requirements.txt` pins a newer torch that isn't necessary. If you do hit a
dependency issue:

```bash
conda create --name foundyou python=3.10 -y && conda activate foundyou
pip install -r requirements.txt
```

## Files

```
configs/underwater_drone.yaml   all settings
index_labelled.py               COCO zips  -> labelled_index.json
index_unlabelled.py             RGB_FRAMES -> unlabelled_index.json
underwater_dataset.py           episode datasets
engine.py                       model, losses, metric
train.py / evaluate.py
checkpoints/<run>/              adapters_best.pth (9.8 MB), _last, history.json
```

Checkpoints hold only the trained tensors — `evaluate.py` loads the released
FoundYou weights first, then layers yours on top.

## Tuning

| problem | fix |
|---|---|
| focus on fewer classes | `data.classes: [member, anode, cable]` |
| ambiguous multi-instance episodes | `data.single_instance_only: true` |
| val mIoU flat | raise `training.lr` to `3e-4` |
| peaks then drops | normal — best checkpoint is always kept |
| sloppy boundaries | raise `loss.dice_weight` |
| most unlabelled pairs skipped | widen `semisup.min_area` / `max_area` |
| cycle_IoU not rising | raise `semisup.weight`, lower `frame_gap` |
| too slow | lower `semisup.pairs_per_epoch` |

Legacy single-class path: `data.source: front_mask`.

## Limits

- `member` dominates frame count; per-class mIoU matters more than the mean.
- `anomaly` is indexed but excluded from training.
- Masks are class unions, not instance identities.
- Retrieval branch is untouched — adapting it needs gallery pairs this data
  doesn't have.
- Frozen SAM 2 sets the ceiling; if adapters saturate, that's the limit.
