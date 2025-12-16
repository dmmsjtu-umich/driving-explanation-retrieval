#!/usr/bin/env python
"""
Precompute video-frame embeddings for Driving Explanation Retrieval baselines.

This script mirrors prepare_action_features.py but extracts CNN features from
the 32-frame clips stored in BDD-X frame TSV files. The resulting cache can
be combined with action text embeddings to build multimodal baselines.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for video feature extraction") from exc

try:
    from torchvision import models, transforms
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for video feature extraction") from exc

try:
    import open_clip  # type: ignore
except ImportError:
    open_clip = None

from sklearn.preprocessing import LabelEncoder

TRAIN_SIZE = 21143
VAL_SIZE = 2519

ImageType = Image.Image


@dataclass
class LineListEntry:
    row_idx: int
    cap_idx: int


class TSVFile:
    """Random access loader for ADAPT TSV files using the accompanying lineidx."""

    def __init__(self, tsv_path: Path):
        self.tsv_path = tsv_path
        base = str(tsv_path).rsplit(".", 1)[0]
        li8 = Path(base + ".lineidx.8b")
        li = Path(base + ".lineidx")

        if li8.exists():
            with li8.open("rb") as f:
                buf = f.read()
            self.offsets = [
                int.from_bytes(buf[i : i + 8], "little")
                for i in range(0, len(buf), 8)
            ]
        elif li.exists():
            with li.open("r") as f:
                self.offsets = [int(x.strip()) for x in f]
        else:
            raise FileNotFoundError(f"Missing index file: {li8} or {li}")

        self.fp = tsv_path.open("rb")

    def __len__(self) -> int:
        return len(self.offsets)

    def get_row(self, idx: int) -> List[str]:
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError(f"Index {idx} out of range [0, {len(self.offsets)-1}]")
        self.fp.seek(self.offsets[idx])
        line = self.fp.readline().rstrip(b"\n")
        return [part.decode("utf-8") for part in line.split(b"\t")]

    def close(self) -> None:
        if hasattr(self, "fp") and not self.fp.closed:
            self.fp.close()

    def __del__(self):
        self.close()


def decode_b64_image(data: str) -> ImageType:
    """Decode a base64-encoded JPEG frame into a PIL image."""

    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def load_linelist(path: Path) -> List[LineListEntry]:
    with path.open("r", encoding="utf-8") as f:
        pairs = [tuple(map(int, line.strip().split("\t"))) for line in f]
    return [LineListEntry(row_idx=p[0], cap_idx=p[1]) for p in pairs]


def default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_backbone(model_name: str, pretrained: bool):
    model_name = model_name.lower()

    def _resnet_constructor(name, weights_enum):
        weights = weights_enum.DEFAULT if pretrained else None
        model = getattr(models, name)(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        transform = weights.transforms() if weights else default_transform()
        return model, feature_dim, transform

    registry = {
        "resnet18": lambda: _resnet_constructor("resnet18", models.ResNet18_Weights),
        "resnet34": lambda: _resnet_constructor("resnet34", models.ResNet34_Weights),
        "resnet50": lambda: _resnet_constructor("resnet50", models.ResNet50_Weights),
        "resnet101": lambda: _resnet_constructor("resnet101", models.ResNet101_Weights),
        "mobilenet_v2": lambda: _mobilenet_v2(pretrained),
        "efficientnet_b0": lambda: _efficientnet_b0(pretrained),
        "clip_vit_b16": lambda: _clip_vit_b16(pretrained),
    }

    if model_name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported model {model_name}, available: {available}")

    return registry[model_name]()


def _mobilenet_v2(pretrained: bool):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    feature_dim = model.classifier[1].in_features
    model.classifier = nn.Identity()
    transform = weights.transforms() if weights else default_transform()
    return model, feature_dim, transform


def _efficientnet_b0(pretrained: bool):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    feature_dim = model.classifier[1].in_features
    model.classifier = nn.Identity()
    transform = weights.transforms() if weights else default_transform()
    return model, feature_dim, transform


def _clip_vit_b16(pretrained: bool):
    if open_clip is None:
        raise ImportError("open_clip is required for CLIP ViT-B/16")

    if not pretrained:
        raise ValueError("clip_vit_b16 only supports pretrained weights, do not use --no-pretrained")

    pretrained_tag = "openai"

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=pretrained_tag
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to download or load CLIP pretrained weights. "
            "Network may be unable to access huggingface.co. "
            f"Original error: {exc}"
        ) from exc
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    class ClipEncoder(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model

        def forward(self, x):
            return self.clip_model.encode_image(x)

    clip_encoder = ClipEncoder(model)
    feature_dim = model.visual.output_dim
    transform = preprocess
    return clip_encoder, feature_dim, transform


@torch.no_grad()
def encode_frames(
    images: Sequence[ImageType],
    model: nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for img in images:
        tensors.append(transform(img))
    stacked = torch.stack(tensors, dim=0)
    if stacked.dtype != torch.float32:
        stacked = stacked.float()
    stacked = stacked.to(device)
    features = []
    for start in range(0, stacked.size(0), batch_size):
        end = start + batch_size
        feats = model(stacked[start:end])
        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)
        features.append(feats)
    return torch.cat(features, dim=0)


def aggregate_feature(frames: torch.Tensor, mode: str) -> torch.Tensor:
    """Aggregate frame features.

    Args:
        frames: (T, D) tensor of per-frame features
        mode: aggregation mode
            - "mean": temporal mean → (D,)
            - "max": temporal max → (D,)
            - "mean_max": concat mean and max → (2D,)
            - "mean_std": concat mean and std → (2D,)
            - "none": no aggregation, preserve temporal → (T, D)

    Returns:
        Aggregated feature tensor
    """
    mode = mode.lower()
    if mode == "none":
        return frames  # Keep (T, D) shape, preserve temporal information
    if mode == "mean":
        return frames.mean(dim=0)
    if mode == "max":
        return frames.max(dim=0).values
    if mode == "mean_max":
        mean = frames.mean(dim=0)
        max_ = frames.max(dim=0).values
        return torch.cat([mean, max_], dim=0)
    if mode == "mean_std":
        mean = frames.mean(dim=0)
        std = frames.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0)
    raise ValueError(f"Unknown aggregation mode: {mode}")


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        gidx = item.get("global_index", item.get("index", 0))
        split = item.get("split")
        if split in {"training", "validation", "testing"}:
            continue
        if gidx < TRAIN_SIZE:
            item["split"] = "training"
        elif gidx < TRAIN_SIZE + VAL_SIZE:
            item["split"] = "validation"
        else:
            item["split"] = "testing"
    return data


def build_split_mappings(
    dataset: Sequence[Dict],
) -> Dict[str, List[Tuple[int, int]]]:
    split_offsets = {
        "training": 0,
        "validation": TRAIN_SIZE,
        "testing": TRAIN_SIZE + VAL_SIZE,
    }
    mappings: Dict[str, List[Tuple[int, int]]] = {"training": [], "validation": [], "testing": []}

    for data_idx, item in enumerate(dataset):
        split = item["split"]
        if split not in mappings:
            continue
        gidx = item.get("global_index", data_idx)
        offset = split_offsets[split]
        local_idx = gidx - offset
        mappings[split].append((data_idx, local_idx))
    return mappings


class FrameProvider:
    """Helper that maps split-local indices to decoded frame images."""

    def __init__(self, split: str, bddx_dir: Path):
        self.split = split
        self.bddx_dir = bddx_dir
        frame_tsv = bddx_dir / "frame_tsv" / f"{split}_32frames_img_size256.img.tsv"
        if not frame_tsv.exists():
            raise FileNotFoundError(f"Frame TSV not found: {frame_tsv}")
        self.tsv = TSVFile(frame_tsv)

        linelist_path = bddx_dir / f"{split}.caption.linelist.tsv"
        if not linelist_path.exists():
            raise FileNotFoundError(f"Linelist not found: {linelist_path}")
        self.linelist = load_linelist(linelist_path)

    def __len__(self) -> int:
        return len(self.linelist)

    @lru_cache(maxsize=None)
    def get_row_index(self, local_idx: int) -> int:
        return self.linelist[local_idx].row_idx

    def decode_frames_for_row(self, row_idx: int) -> List[ImageType]:
        record = self.tsv.get_row(row_idx)
        frame_b64_list = record[2:]
        return [decode_b64_image(b64) for b64 in frame_b64_list]

    def get_images(self, local_idx: int) -> List[ImageType]:
        row_idx = self.get_row_index(local_idx)
        return self.decode_frames_for_row(row_idx)

    def close(self) -> None:
        self.tsv.close()

    def __del__(self):
        self.close()


class FrameDataset(torch.utils.data.Dataset):
    """Dataset for multi-process frame loading."""

    def __init__(
        self,
        pairs: List[Tuple[int, int]],  # (data_idx, local_idx)
        split: str,
        bddx_dir: Path,
        transform,
    ):
        self.pairs = pairs
        self.split = split
        self.bddx_dir = bddx_dir
        self.transform = transform
        self._provider = None  # Lazy init for multiprocessing

    def _get_provider(self):
        if self._provider is None:
            self._provider = FrameProvider(self.split, self.bddx_dir)
        return self._provider

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        data_idx, local_idx = self.pairs[idx]
        provider = self._get_provider()
        images = provider.get_images(local_idx)

        # Apply transform to each frame and stack
        tensors = [self.transform(img) for img in images]
        stacked = torch.stack(tensors, dim=0)  # (T, C, H, W)
        return data_idx, stacked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("driving-explanation-retrieval/data_analysis/clustering_results/reasons_with_clusters.json"),
        help="Path to clustered reasons JSON",
    )
    parser.add_argument(
        "--bddx-dir",
        type=Path,
        default=Path("datasets/BDDX"),
        help="BDD-X preprocessed data directory (containing frame_tsv/ and *.linelist.tsv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("driving-explanation-retrieval/data/baseline/video_features_resnet18_mean.npz"),
        help="Output video feature cache path",
    )
    parser.add_argument(
        "--model",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "mobilenet_v2",
            "efficientnet_b0",
            "clip_vit_b16",
        ],
        default="resnet18",
        help="Feature extraction backbone",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights (enabled by default)",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max", "mean_max", "mean_std", "none"],
        default="mean",
        help="Frame feature aggregation strategy (none=preserve temporal, output 3D tensor)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Maximum frame batch size for forward pass",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. cuda or cpu (auto-detect by default)",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (0=single process)",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(args.data)
    if not dataset:
        raise ValueError(f"Dataset is empty: {args.data}")

    pretrained = not args.no_pretrained
    model, base_dim, transform = build_backbone(args.model, pretrained)

    # Determine output shape based on aggregation mode
    num_frames = 32  # BDD-X uses 32 frames per clip
    temporal_mode = args.aggregation.lower() == "none"

    if temporal_mode:
        # No aggregation: preserve (T, D) per sample → output shape (N, T, D)
        feature_dim = base_dim
        feature_shape = (len(dataset), num_frames, feature_dim)
    else:
        # Aggregation: (D,) or (2D,) per sample → output shape (N, feature_dim)
        probe = torch.zeros(1, base_dim)
        agg_output = aggregate_feature(probe, args.aggregation)
        feature_dim = agg_output.numel()
        feature_shape = (len(dataset), feature_dim)

    model.to(device)
    model.eval()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform([item["cluster"] for item in dataset])
    splits = np.array([item["split"] for item in dataset])

    features = np.zeros(feature_shape, dtype=np.float32)

    split_map = build_split_mappings(dataset)

    # Use DataLoader with multiple workers for faster data loading
    for split, pairs in split_map.items():
        if not pairs:
            continue

        frame_dataset = FrameDataset(
            pairs=pairs,
            split=split,
            bddx_dir=args.bddx_dir,
            transform=transform,
        )

        # DataLoader with multi-process loading
        loader = torch.utils.data.DataLoader(
            frame_dataset,
            batch_size=1,  # Process one sample at a time (32 frames)
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        iterable = loader
        if not args.no_tqdm and tqdm is not None:
            iterable = tqdm(loader, desc=f"{split} features", unit="sample")

        for batch in iterable:
            data_idx, frame_batch = batch
            data_idx = data_idx.item()  # (1,) -> scalar
            frame_batch = frame_batch.squeeze(0)  # (1, T, C, H, W) -> (T, C, H, W)

            # Move to GPU and encode
            frame_batch = frame_batch.to(device)
            with torch.no_grad():
                # Encode all frames in batches
                frame_features = []
                for start in range(0, frame_batch.size(0), args.batch_size):
                    end = start + args.batch_size
                    feats = model(frame_batch[start:end])
                    if feats.ndim > 2:
                        feats = torch.flatten(feats, start_dim=1)
                    frame_features.append(feats)
                frame_tensor = torch.cat(frame_features, dim=0)  # (T, D)

            aggregated = aggregate_feature(frame_tensor, args.aggregation)
            features[data_idx] = aggregated.cpu().numpy().astype(np.float32)

    mask_train = splits == "training"
    mask_val = splits == "validation"
    mask_test = splits == "testing"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "modality": "video",
        "model": args.model,
        "pretrained": str(pretrained),
        "aggregation": args.aggregation,
        "feature_dim": feature_dim,
        "X_train": features[mask_train],
        "y_train": labels[mask_train],
        "X_val": features[mask_val],
        "y_val": labels[mask_val],
        "X_test": features[mask_test],
        "y_test": labels[mask_test],
        "label_classes": label_encoder.classes_,
    }

    # Add temporal metadata if using none aggregation
    if temporal_mode:
        save_dict["temporal"] = True
        save_dict["num_frames"] = num_frames
        shape_str = f"({len(dataset)}, {num_frames}, {feature_dim})"
    else:
        save_dict["temporal"] = False
        shape_str = f"({len(dataset)}, {feature_dim})"

    np.savez(args.output, **save_dict)
    print(f"✅ Saved video features to {args.output}")
    print(f"   Shape: {shape_str}, Model: {args.model}, Aggregation: {args.aggregation}")


if __name__ == "__main__":
    main()
