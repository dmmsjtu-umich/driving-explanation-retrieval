"""
Dataset and dataloaders for Vision-Language Transformer training.

Loads raw video frames from TSV files and tokenizes action narrations on the fly.
"""

from __future__ import annotations

import base64
import json
import random
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

# Dataset split sizes
TRAIN_SIZE = 21143
VAL_SIZE = 2519


class VideoTransform:
    """Video transform with optional training augmentation.

    Training: consistent random crop + color jitter across all frames
    Validation/Test: simple resize
    """

    def __init__(self, size: int = 224, train: bool = False):
        self.size = size
        self.train = train
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if train:
            self.scale = (0.7, 1.0)
            self.ratio = (3/4, 4/3)
            # Color jitter parameters
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.05

    def _apply_color_jitter(self, img: Image.Image, brightness_factor, contrast_factor,
                            saturation_factor, hue_factor) -> Image.Image:
        """Apply color jitter with given factors."""
        img = transforms.functional.adjust_brightness(img, brightness_factor)
        img = transforms.functional.adjust_contrast(img, contrast_factor)
        img = transforms.functional.adjust_saturation(img, saturation_factor)
        img = transforms.functional.adjust_hue(img, hue_factor)
        return img

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        """Transform list of PIL images to (C, T, H, W) tensor.

        For training: applies consistent random crop and color jitter to all frames.
        """
        if self.train and len(frames) > 0:
            # Get consistent crop parameters for all frames
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                frames[0], scale=self.scale, ratio=self.ratio
            )
            # Get consistent color jitter parameters
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            hue_factor = random.uniform(-self.hue, self.hue)

        processed = []
        for frame in frames:
            if self.train:
                # Apply consistent augmentation
                frame = self._apply_color_jitter(frame, brightness_factor, contrast_factor,
                                                  saturation_factor, hue_factor)
                frame = transforms.functional.resized_crop(frame, i, j, h, w, (self.size, self.size))
            else:
                frame = transforms.functional.resize(frame, (self.size, self.size))
            processed.append(self.normalize(transforms.functional.to_tensor(frame)))
        return torch.stack(processed, dim=1)  # (C, T, H, W)


class FrameTSVReader:
    """
    TSV reader for pre-extracted frame archives.

    Each row: <key> \t {"width": W, "height": H} \t <frame1_b64> ... <frameN_b64>
    """

    def __init__(self, tsv_path: Path):
        self.tsv_path = Path(tsv_path)
        self.lineidx_path = self.tsv_path.with_suffix(".lineidx")

        if not self.tsv_path.exists():
            raise FileNotFoundError(f"Frame TSV not found: {tsv_path}")
        if not self.lineidx_path.exists():
            raise FileNotFoundError(f"Lineidx not found: {self.lineidx_path}")

        with self.lineidx_path.open("r") as f:
            self.offsets = [int(line.strip()) for line in f]
        self._fp = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None  # Don't pickle file handles
        return state

    def _ensure_open(self):
        if self._fp is None or self._fp.closed:
            self._fp = self.tsv_path.open("r")

    def get_frames(self, idx: int, num_frames: int, mode: str = "uniform") -> List[Image.Image]:
        """Decode frames with various temporal sampling strategies.

        Args:
            idx: Sample index
            num_frames: Number of frames to sample
            mode: Sampling mode
                - "uniform": evenly spaced frames (for val/test)
                - "random_clip": random contiguous clip (for train)
                - "random_stride": random start with random stride (for train)
        """
        self._ensure_open()
        self._fp.seek(self.offsets[idx])
        row = self._fp.readline().rstrip("\n").split("\t")

        meta = json.loads(row[1]) if len(row) > 1 else {}
        w, h = int(meta.get("width", 224)), int(meta.get("height", 224))
        b64_frames = row[2:]
        total = len(b64_frames)

        if total == 0:
            return [Image.new("RGB", (w, h)) for _ in range(num_frames)]

        if total >= num_frames:
            if mode == "uniform":
                # Evenly spaced sampling
                step = (total - 1) / float(num_frames - 1) if num_frames > 1 else 0
                indices = [int(round(i * step)) for i in range(num_frames)]

            elif mode == "random_clip":
                # Random contiguous clip
                start = random.randint(0, total - num_frames)
                indices = list(range(start, start + num_frames))

            elif mode == "random_stride":
                # Random start with random stride (1, 2, or 3)
                stride = random.choice([1, 2, 3])
                max_start = total - stride * (num_frames - 1) - 1
                if max_start < 0:
                    # Fallback to stride=1 if video too short
                    stride = 1
                    max_start = max(0, total - num_frames)
                start = random.randint(0, max(0, max_start))
                indices = [min(start + i * stride, total - 1) for i in range(num_frames)]

            else:
                raise ValueError(f"Unknown sampling mode: {mode}")
        else:
            # Not enough frames: repeat last frame
            indices = list(range(total)) + [total - 1] * (num_frames - total)

        frames = []
        for i in indices:
            try:
                img = Image.open(BytesIO(base64.b64decode(b64_frames[i]))).convert("RGB")
            except Exception:
                img = Image.new("RGB", (w, h))
            frames.append(img)

        return frames

    def close(self):
        if self._fp is not None and not self._fp.closed:
            self._fp.close()


class DrivingExplanationDataset(Dataset):
    """Dataset that loads raw frames + action text for classification."""

    def __init__(
        self,
        samples: List[dict],
        frame_tsv_path: Path,
        tokenizer: AutoTokenizer,
        label_to_id: Dict[int, int],
        num_frames: int = 32,
        max_text_len: int = 48,
        img_size: int = 224,
        sampling_mode: str = "uniform",
        train_augment: bool = False,
    ):
        """
        Args:
            sampling_mode: "uniform" for val/test, "random_stride" for train
            train_augment: Whether to apply random crop + color jitter
        """
        self.samples = samples
        self.frame_reader = FrameTSVReader(frame_tsv_path)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.num_frames = num_frames
        self.max_text_len = max_text_len
        self.sampling_mode = sampling_mode
        self.transform = VideoTransform(size=img_size, train=train_augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        frames = self.frame_reader.get_frames(idx, self.num_frames, mode=self.sampling_mode)
        video_tensor = self.transform(frames)

        tokens = self.tokenizer(
            sample["action"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        token_type_ids = tokens.get("token_type_ids", torch.zeros_like(tokens["input_ids"]))

        return {
            "video": video_tensor,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "label": torch.tensor(self.label_to_id[sample["cluster"]], dtype=torch.long),
        }


def _load_cluster_json(path: Path) -> List[dict]:
    """Load and normalize cluster JSON data."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Assign splits based on global index if missing
    for item in data:
        if item.get("split") in {"training", "validation", "testing"}:
            continue
        gidx = item.get("global_index", item.get("index", 0))
        if gidx < TRAIN_SIZE:
            item["split"] = "training"
        elif gidx < TRAIN_SIZE + VAL_SIZE:
            item["split"] = "validation"
        else:
            item["split"] = "testing"
    return data


def create_frame_dataloaders(
    data_json: Path,
    frame_tsv_root: Path,
    tokenizer_name: str = "bert-base-uncased",
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 32,
    img_size: int = 224,
    frame_size: int = 256,
    max_text_len: int = 48,
    train_sampling: str = "random_stride",
    train_augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build dataloaders that read raw frames and tokenize actions on the fly.

    Args:
        data_json: Path to reasons_with_clusters.json
        frame_tsv_root: Directory containing split TSVs
        tokenizer_name: HuggingFace tokenizer name
        batch_size: Batch size
        num_workers: DataLoader workers
        num_frames: Frames to sample per clip
        img_size: Resize target size
        frame_size: TSV filename resolution suffix
        max_text_len: Max tokenized text length
        train_sampling: Sampling mode for training ("uniform", "random_stride", "random_clip")
        train_augment: Whether to apply augmentation during training

    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    data = _load_cluster_json(data_json)

    # Build label mapping
    label_classes = sorted({int(item["cluster"]) for item in data})
    label_to_id = {c: i for i, c in enumerate(label_classes)}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Split samples
    train_samples = [item for item in data if item["split"] == "training"]
    val_samples = [item for item in data if item["split"] == "validation"]
    test_samples = [item for item in data if item["split"] == "testing"]

    def tsv_path(split: str) -> Path:
        return frame_tsv_root / f"{split}_32frames_img_size{frame_size}.img.tsv"

    # Create datasets with different sampling strategies
    train_ds = DrivingExplanationDataset(
        train_samples, tsv_path("training"), tokenizer, label_to_id,
        num_frames, max_text_len, img_size,
        sampling_mode=train_sampling,
        train_augment=train_augment,
    )
    val_ds = DrivingExplanationDataset(
        val_samples, tsv_path("validation"), tokenizer, label_to_id,
        num_frames, max_text_len, img_size,
        sampling_mode="uniform",
        train_augment=False,
    )
    test_ds = DrivingExplanationDataset(
        test_samples, tsv_path("testing"), tokenizer, label_to_id,
        num_frames, max_text_len, img_size,
        sampling_mode="uniform",
        train_augment=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Metadata
    metadata = {
        "num_classes": len(label_classes),
        "label_classes": label_classes,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "num_frames": num_frames,
        "img_size": img_size,
    }

    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    # Quick test
    print("Testing dataset loading...")
    data_json = Path("driving-explanation-retrieval/data_analysis/clustering_results/reasons_with_clusters.json")
    frame_root = Path("/home/dmmsjtu/Desktop/cse595/datasets/BDDX/frame_tsv")

    if data_json.exists() and frame_root.exists():
        train_loader, val_loader, test_loader, meta = create_frame_dataloaders(
            data_json, frame_root, batch_size=2, num_workers=0,
        )
        print(f"Train: {meta['train_size']} | Val: {meta['val_size']} | Test: {meta['test_size']}")
        print(f"Classes: {meta['num_classes']} | Frames: {meta['num_frames']}")

        for batch in train_loader:
            print(f"Video: {batch['video'].shape} | Label: {batch['label'].shape}")
            break
    else:
        print("Data files not found.")
