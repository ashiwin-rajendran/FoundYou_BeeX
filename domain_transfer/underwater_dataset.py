"""Episodic underwater dataset for FoundYou adapter domain transfer.

FoundYou is a *personalized* segmenter: it is given one prompted reference of
an object and must segment that same object in a different image. Training it
therefore needs episodes, not plain (image, mask) pairs:

    episode = (reference image + prompt, query image + ground-truth mask)

This builds those episodes from the BeeX front-camera drone set
(`ml_datasets/front_all_20260727`), which is consecutive video of the drone in
turbid water -- the exact domain gap the adapters have to close.

TWO THINGS HERE ARE EASY TO GET WRONG, SO BOTH ARE GUARDED
-----------------------------------------------------------
1. MASK POLARITY. `labels_mask/*.png` is INVERTED: 1 = not-the-drone,
   0 = the drone. Measured over the 258 labelled frames:

       IoU(labels_mask > 0, rasterised polygon) = 0.0001
       IoU(labels_mask ==0, rasterised polygon) = 0.9974

   Training on the wrong polarity does not crash -- it silently teaches the
   model to segment the entire background and still reports a plausible-looking
   loss curve. `verify_polarity()` re-measures this against the YOLO polygons
   at startup and refuses to run if the convention ever changes.

2. TEMPORAL LEAKAGE AND TRIVIAL EPISODES. These are consecutive video frames
   (223 of 257 index steps are +1). Two consequences:

   * A random train/val split puts near-duplicates of training frames into
     validation, producing a number that looks good and means nothing. The
     split here is temporal by default.
   * An episode whose reference and query are adjacent frames is nearly
     self-referential -- the model can copy rather than generalise. Episodes
     enforce `min_frame_gap` between reference and query.
"""

from __future__ import annotations

import collections
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(img_size: int) -> transforms.Compose:
    """Matches datasets/__init__.py so training and inference agree."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def rasterise_polygons(label_path: Path, height: int, width: int) -> np.ndarray:
    """Rasterise a YOLO-segmentation .txt into a binary mask (1 = object)."""
    mask = np.zeros((height, width), np.uint8)
    if not label_path.exists():
        return mask
    text = label_path.read_text().strip()
    if not text:
        return mask
    for line in text.split("\n"):
        parts = line.split()
        if len(parts) < 7:          # need class + >=3 xy pairs
            continue
        pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def _stratified_val_recordings(recordings, val_fraction: float, seed: int) -> set:
    """Pick validation recordings so rare classes are still represented.

    A plain random split put jacket_leg, jacket_brace and jacket_jtube entirely
    in train -- each lives in only 3-4 recordings -- so they were never
    validated and the reported mIoU silently covered 11 of 14 classes.

    So: walk the classes rarest-first and give each one a validation recording
    where that can be done without emptying it out of train (a class needs >= 2
    usable recordings to be split at all), then top up at random to reach
    val_fraction. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    per_class = {}
    for idx, recording in enumerate(recordings):
        for name in (recording.get("classes") or {}):
            per_class.setdefault(name, []).append(idx)

    target = max(1, int(round(len(recordings) * val_fraction)))
    val_ids: set = set()

    for name in sorted(per_class, key=lambda k: len(per_class[k])):
        owners = per_class[name]
        if len(owners) < 2:
            continue                      # splitting it would remove it from train
        if any(i in val_ids for i in owners):
            continue                      # already covered
        val_ids.add(int(rng.choice(owners)))

    remaining = [i for i in range(len(recordings)) if i not in val_ids]
    rng.shuffle(remaining)
    for idx in remaining:
        if len(val_ids) >= target:
            break
        val_ids.add(int(idx))
    return val_ids


class EpisodeSampling:
    """Per-worker, per-epoch RNG for datasets that sample randomly.

    Two failure modes this exists to prevent, both measured here before it was
    added, and both silent -- the loss curve looks entirely normal:

    1. DataLoader forks workers with a COPY of the dataset, so a single
       self.rng gives every worker the same seed and the same draw sequence.
       With num_workers=4 that turned 24 requested episodes into 6 distinct
       ones, each repeated 4x.
    2. Workers are re-forked every epoch from the same unused parent RNG, so
       epoch 2 replayed epoch 1 exactly.

    Together those reduced a 300-episode epoch to ~75 distinct episodes, frozen
    for the whole run -- training memorises a fixed subset while the loss falls
    convincingly.

    Seeding from (base seed, epoch, worker id) fixes both and stays
    reproducible. train.py must call set_epoch() before each epoch so the new
    value is pickled into the workers at fork time.
    """

    def _init_sampling(self, seed: int) -> None:
        self._base_seed = int(seed)
        self._epoch = 0
        self._rng_key = None
        self._rng = np.random.default_rng(self._base_seed)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def rng(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        key = (self._epoch, worker_id)
        if self._rng_key != key:
            self._rng = np.random.default_rng(
                self._base_seed + 1_000_003 * self._epoch + 7919 * worker_id)
            self._rng_key = key
        return self._rng


class UnderwaterEpisodes(EpisodeSampling, Dataset):
    """(reference, query) episodes of the same object in underwater footage.

    Args:
        root: dataset dir holding images/, labels_mask/, labels_yolo/.
        split: 'train' | 'val'.
        img_size: side length fed to the model (SAM2 re-scales to 1024 inside).
        val_fraction: held out from the END of the sequence when temporal.
        temporal_split: see the module docstring. Keep True for video.
        min_frame_gap: minimum |ref - query| frame distance in an episode.
        episodes_per_epoch: train episodes sampled per epoch (val is fixed).
        mask_invert: True for this dataset. Checked by verify_polarity().
        augment: photometric jitter on the query only.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 518,
        val_fraction: float = 0.2,
        temporal_split: bool = True,
        min_frame_gap: int = 10,
        episodes_per_epoch: int = 400,
        mask_invert: bool = True,
        augment: bool = True,
        seed: int = 17,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.min_frame_gap = min_frame_gap
        self.episodes_per_epoch = episodes_per_epoch
        self.mask_invert = mask_invert
        self.augment = augment and split == "train"
        self.transform = build_transform(img_size)
        self._init_sampling(seed + (0 if split == "train" else 1))

        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "labels_mask"
        self.label_dir = self.root / "labels_yolo"
        for d in (self.image_dir, self.mask_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"missing {d}")

        stems = self._labelled_stems()
        if len(stems) < 2:
            raise RuntimeError(f"need >=2 labelled frames, found {len(stems)}")

        # Temporal split: validation is the TAIL of the sequence, so no val
        # frame is a near-duplicate of a training frame.
        cut = int(round(len(stems) * (1.0 - val_fraction)))
        cut = max(1, min(cut, len(stems) - 1))
        if temporal_split:
            self.stems = stems[:cut] if split == "train" else stems[cut:]
        else:
            shuffled = list(stems)
            np.random.default_rng(seed).shuffle(shuffled)
            self.stems = shuffled[:cut] if split == "train" else shuffled[cut:]

        self.frame_ids = np.array([self._frame_id(s) for s in self.stems])

        # Validation episodes are fixed so the metric is comparable run to run.
        self.fixed_episodes: Optional[List[Tuple[int, int]]] = None
        if split != "train":
            self.fixed_episodes = self._build_fixed_episodes()

    # ---------------------------------------------------------------- setup

    def _labelled_stems(self) -> List[str]:
        """Frames that actually contain the object, sorted by frame index."""
        stems = []
        for mask_path in sorted(self.mask_dir.glob("*.png")):
            stem = mask_path.stem
            if not (self.image_dir / f"{stem}.jpg").exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if self._object_from_mask(mask).sum() > 0:
                stems.append(stem)
        return sorted(stems, key=self._frame_id)

    @staticmethod
    def _frame_id(stem: str) -> int:
        digits = "".join(c for c in stem if c.isdigit())
        return int(digits) if digits else 0

    def _object_from_mask(self, mask: np.ndarray) -> np.ndarray:
        return (mask == 0) if self.mask_invert else (mask > 0)

    def _build_fixed_episodes(self) -> List[Tuple[int, int]]:
        """One episode per val frame, paired with the furthest valid reference.

        Furthest, not random: it is the hardest honest pairing available and it
        is deterministic, so val mIoU moves only when the model moves.
        """
        episodes = []
        for q in range(len(self.stems)):
            gaps = np.abs(self.frame_ids - self.frame_ids[q])
            valid = np.where(gaps >= self.min_frame_gap)[0]
            if len(valid) == 0:                     # short split: take furthest
                valid = np.array([int(np.argmax(gaps))])
            ref = int(valid[np.argmax(gaps[valid])])
            episodes.append((ref, q))
        return episodes

    # ----------------------------------------------------------------- load

    def _load(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[index]
        bgr = cv2.imread(str(self.image_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(str(self.mask_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask_raw is None:
            raise RuntimeError(f"could not read frame {stem}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        obj = self._object_from_mask(mask_raw).astype(np.uint8)

        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # NEAREST only: bilinear on a label map invents in-between values.
        obj = cv2.resize(obj, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image = self.transform(rgb)
        mask = torch.from_numpy(obj).float()
        return image, mask

    def _augment_query(self, image: torch.Tensor) -> torch.Tensor:
        """Mild photometric jitter, applied post-normalisation.

        Underwater colour cast and backscatter vary shot to shot; geometric
        augmentation is deliberately NOT applied, because the reference and
        query must stay consistent views of one physical object.
        """
        if self.rng.random() < 0.5:
            image = image * float(self.rng.uniform(0.85, 1.15))     # brightness
        if self.rng.random() < 0.5:
            mean = image.mean()
            image = (image - mean) * float(self.rng.uniform(0.85, 1.15)) + mean
        return image

    # ------------------------------------------------------------- protocol

    def __len__(self) -> int:
        if self.fixed_episodes is not None:
            return len(self.fixed_episodes)
        return self.episodes_per_epoch

    def __getitem__(self, idx: int):
        if self.fixed_episodes is not None:
            ref_idx, query_idx = self.fixed_episodes[idx]
        else:
            query_idx = int(self.rng.integers(len(self.stems)))
            gaps = np.abs(self.frame_ids - self.frame_ids[query_idx])
            valid = np.where(gaps >= self.min_frame_gap)[0]
            if len(valid) == 0:
                valid = np.array([i for i in range(len(self.stems)) if i != query_idx])
            ref_idx = int(self.rng.choice(valid))

        ref_image, ref_mask = self._load(ref_idx)
        query_image, query_mask = self._load(query_idx)
        if self.augment:
            query_image = self._augment_query(query_image)

        return {
            "ref_image": ref_image,            # [3,S,S] normalised
            "ref_mask": ref_mask,              # [S,S] {0,1}
            "query_image": query_image,        # [3,S,S]
            "query_mask": query_mask,          # [S,S] {0,1}
            "ref_name": self.stems[ref_idx],
            "query_name": self.stems[query_idx],
        }


def verify_polarity(root: str, mask_invert: bool, sample: int = 60) -> dict:
    """Re-measure mask polarity against the YOLO polygons.

    Called at startup by train.py and evaluate.py. A flipped mask convention is
    the one error here that produces a confident, entirely wrong model without
    raising anything, so it is checked rather than assumed.
    """
    root_path = Path(root)
    mask_dir, label_dir = root_path / "labels_mask", root_path / "labels_yolo"
    if not label_dir.is_dir():
        return {"checked": 0, "skipped": "no labels_yolo/ to cross-check against"}

    stems = []
    for label_path in sorted(label_dir.glob("*.txt")):
        if label_path.read_text().strip():
            stems.append(label_path.stem)
    stems = stems[:sample]

    direct, inverted = [], []
    for stem in stems:
        mask_raw = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            continue
        height, width = mask_raw.shape[:2]
        polygon = rasterise_polygons(label_dir / f"{stem}.txt", height, width) > 0
        for store, candidate in ((direct, mask_raw > 0), (inverted, mask_raw == 0)):
            union = (candidate | polygon).sum()
            store.append(float((candidate & polygon).sum() / union) if union else 1.0)

    if not direct:
        return {"checked": 0, "skipped": "no readable masks"}

    iou_direct, iou_inverted = float(np.mean(direct)), float(np.mean(inverted))
    chosen = iou_inverted if mask_invert else iou_direct
    other = iou_direct if mask_invert else iou_inverted
    if chosen < other:
        raise RuntimeError(
            "Mask polarity looks wrong. With mask_invert=%s the object masks "
            "agree with the polygon labels at IoU %.4f, but the opposite "
            "convention gives %.4f. Flip `data.mask_invert` in the config.\n"
            "Training through this would not crash -- it would teach the model "
            "to segment the background."
            % (mask_invert, chosen, other)
        )
    return {"checked": len(direct), "iou_with_polygons": chosen, "iou_if_flipped": other}


class UnlabelledPairs(EpisodeSampling, Dataset):
    """Temporal frame pairs from the unlabelled RGB_FRAMES corpus.

    No masks exist for this data, so it carries no supervised target. It is
    used for the cycle-consistency term: segment frame t, propagate the mask to
    frame t+k, propagate it back, and require the round trip to agree. That
    objective needs only two frames of the same scene.

    Pairs come from inside ONE recording, so the two frames really are the same
    scene a moment apart. `frame_gap` controls how far apart: too small and the
    round trip is trivially satisfied, too large and the object may have left
    the view entirely.

    Build the index first with index_unlabelled.py.
    """

    def __init__(
        self,
        index_path: str,
        img_size: int = 518,
        frame_gap: Tuple[int, int] = (3, 15),
        length: int = 300,
        seed: int = 17,
    ) -> None:
        import json

        with open(index_path) as handle:
            index = json.load(handle)
        self.root = Path(index["root"])
        self.recordings = [r for r in index["recordings"]
                           if r["contiguous"] and r["count"] > frame_gap[1] + 1]
        if not self.recordings:
            raise RuntimeError(
                f"no usable recordings in {index_path} "
                f"(need contiguous numbering and > {frame_gap[1]+1} frames)")
        self.img_size = img_size
        self.frame_gap = frame_gap
        self.length = length
        self.transform = build_transform(img_size)
        self._init_sampling(seed)

    def __len__(self) -> int:
        return self.length

    def _load(self, recording: dict, number: int) -> torch.Tensor:
        path = self.root / recording["dir"] / f"frame_{number:06d}.jpg"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return self.transform(rgb)

    def __getitem__(self, idx: int):
        for _ in range(8):                       # a few tries on unreadable files
            recording = self.recordings[int(self.rng.integers(len(self.recordings)))]
            gap = int(self.rng.integers(self.frame_gap[0], self.frame_gap[1] + 1))
            start = recording["first"]
            last = start + recording["count"] - 1 - gap
            if last <= start:
                continue
            number = int(self.rng.integers(start, last + 1))
            try:
                return {
                    "frame_a": self._load(recording, number),
                    "frame_b": self._load(recording, number + gap),
                    "recording": recording["dir"],
                }
            except RuntimeError:
                continue
        raise RuntimeError("could not sample a readable unlabelled pair")


class CocoEpisodes(EpisodeSampling, Dataset):
    """Multi-class episodes from the annotated front-camera recordings.

    Built from labelled_index.json (see index_labelled.py): 165 recordings,
    ~8.8k annotated frames, 15 classes -- member, cable, anode, boatlanding,
    jacket_leg/brace/jtube, weh, iccp_*, grout_*, body, anomaly.

    An episode is one (recording, class) group: two different frames of the
    same recording that both contain that class. The reference is prompted with
    that class's mask in frame A, and the target is that class's mask in frame
    B. Pairing inside one recording keeps it the same physical structure under
    a moving camera, which is the relationship FoundYou is built to exploit.

    CLASS-UNION MASKS
    -----------------
    COCO gives no instance identity across frames, so when a frame holds
    several objects of one class (7.6% of frame-class pairs here) there is no
    way to know which anode in frame B is the anode prompted in frame A. The
    mask is therefore the UNION of that class in the frame, making the task
    "segment this and things like it" rather than strict instance identity.
    With a single instance present -- the other 92% -- the two are the same
    thing. Set `single_instance_only=True` to train on unambiguous episodes.

    SPLIT BY RECORDING
    ------------------
    Validation holds out whole recordings, not frames. Frames inside one
    recording are the same structure seconds apart, so splitting within a
    recording would validate on near-duplicates. Holding out recordings asks
    the harder and more useful question: does this work on a structure the
    model has never seen?
    """

    def __init__(
        self,
        index_path: str,
        split: str = "train",
        img_size: int = 518,
        val_fraction: float = 0.2,
        episodes_per_epoch: int = 300,
        classes: Optional[List[str]] = None,
        min_mask_fraction: float = 0.001,
        single_instance_only: bool = False,
        augment: bool = True,
        seed: int = 17,
    ) -> None:
        import json

        with open(index_path) as handle:
            index = json.load(handle)

        self.img_size = img_size
        self.split = split
        self.augment = augment and split == "train"
        self.episodes_per_epoch = episodes_per_epoch
        self.min_mask_fraction = min_mask_fraction
        self.transform = build_transform(img_size)
        self._init_sampling(seed + (0 if split == "train" else 1))
        self._zip_cache: dict = {}

        recordings = index["recordings"]
        val_ids = _stratified_val_recordings(recordings, val_fraction, seed)
        chosen = ({i for i in range(len(recordings)) if i not in val_ids}
                  if split == "train" else val_ids)

        wanted = set(classes) if classes else None
        self.groups: List[Tuple[int, str, List[str]]] = []
        self.recordings = recordings
        for idx, recording in enumerate(recordings):
            if idx not in chosen:
                continue
            per_class = collections.defaultdict(list)
            for frame_name, frame in recording["frames"].items():
                for name, polygons in frame["anns"].items():
                    if wanted and name not in wanted:
                        continue
                    if single_instance_only and len(polygons) > 1:
                        continue
                    per_class[name].append(frame_name)
            for name, frame_names in per_class.items():
                if len(frame_names) >= 2:
                    self.groups.append((idx, name, sorted(frame_names)))

        if not self.groups:
            raise RuntimeError(f"no usable episode groups for split '{split}'")

        self.class_names = sorted({name for _, name, _ in self.groups})
        # Fixed episodes make the validation number comparable across epochs.
        self.fixed_episodes = None
        if split != "train":
            rng = np.random.default_rng(seed + 99)
            episodes = []
            for group_idx, (_, _, frame_names) in enumerate(self.groups):
                take = min(4, len(frame_names) - 1)
                for step in range(take):
                    a = step % len(frame_names)
                    b = (step + len(frame_names) // 2) % len(frame_names)
                    if a != b:
                        episodes.append((group_idx, a, b))
            self.fixed_episodes = episodes

    def _archive(self, path: str):
        """One open handle per zip per worker -- reopening per frame is slow."""
        import zipfile
        if path not in self._zip_cache:
            self._zip_cache[path] = zipfile.ZipFile(path)
        return self._zip_cache[path]

    def _load(self, recording: dict, frame_name: str, class_name: str):
        archive = self._archive(recording["images_zip"])
        buffer = np.frombuffer(archive.read(frame_name), np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode {frame_name}")
        height, width = bgr.shape[:2]

        mask = np.zeros((height, width), np.uint8)
        for polygon in recording["frames"][frame_name]["anns"].get(class_name, []):
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return self.transform(rgb), torch.from_numpy(mask).float()

    def _augment_query(self, image: torch.Tensor) -> torch.Tensor:
        if self.rng.random() < 0.5:
            image = image * float(self.rng.uniform(0.85, 1.15))
        if self.rng.random() < 0.5:
            mean = image.mean()
            image = (image - mean) * float(self.rng.uniform(0.85, 1.15)) + mean
        return image

    def __len__(self) -> int:
        if self.fixed_episodes is not None:
            return len(self.fixed_episodes)
        return self.episodes_per_epoch

    def _next_index(self, idx: int) -> int:
        """Where to look after an unusable episode.

        Validation walks to the next index instead of drawing a random one:
        the fixed episode list only makes val comparable between epochs if a
        skipped episode is replaced the same way every time.
        """
        if self.fixed_episodes is not None:
            return (idx + 1) % len(self)
        return int(self.rng.integers(len(self)))

    def __getitem__(self, idx: int):
        for _ in range(10):
            if self.fixed_episodes is not None:
                group_idx, a, b = self.fixed_episodes[idx]
            else:
                group_idx = int(self.rng.integers(len(self.groups)))
                frames = self.groups[group_idx][2]
                a, b = self.rng.choice(len(frames), size=2, replace=False)

            rec_idx, class_name, frame_names = self.groups[group_idx]
            recording = self.recordings[rec_idx]
            try:
                ref_image, ref_mask = self._load(recording, frame_names[a], class_name)
                query_image, query_mask = self._load(recording, frame_names[b], class_name)
            except Exception:
                idx = self._next_index(idx)
                continue

            # An empty mask after rasterisation gives the model nothing to
            # point at and no target to hit.
            if ref_mask.mean() < self.min_mask_fraction or \
                    query_mask.mean() < self.min_mask_fraction:
                idx = self._next_index(idx)
                continue

            if self.augment:
                query_image = self._augment_query(query_image)

            return {
                "ref_image": ref_image,
                "ref_mask": ref_mask,
                "query_image": query_image,
                "query_mask": query_mask,
                "class_name": class_name,
                "recording": recording["stem"],
            }
        raise RuntimeError("could not sample a usable episode")
