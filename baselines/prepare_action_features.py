#!/usr/bin/env python
"""Precompute SBERT action embeddings for reason-cluster classification baselines.

This script now only supports SBERT for text encoding (simplified pipeline).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

TRAIN_SIZE = 21143
VAL_SIZE = 2519


def load_dataset(path: Path):
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

def encode_texts_sbert(texts, model_name=None):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed, cannot use S-BERT encoding")
    model_name = model_name or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, model_name


def topk_accuracy(probs, labels, k=3):
    topk = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    correct = sum(int(true in topk_row) for true, topk_row in zip(labels, topk))
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("driving-explanation-retrieval/data_analysis/clustering_results/reasons_with_clusters.json"),
        help="Path to clustered reasons JSON"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SBERT model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("driving-explanation-retrieval/data/baseline/action_features_sbert.npz"),
        help="Where to store SBERT encoded features"
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    actions = [item["action"] for item in dataset]
    labels = [item["cluster"] for item in dataset]
    splits = np.array([item["split"] for item in dataset])

    matrix, model_name = encode_texts_sbert(actions, model_name=args.model_name)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    mask_train = splits == "training"
    mask_val = splits == "validation"
    mask_test = splits == "testing"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        encoder="sbert",
        model=model_name,
        X_train=matrix[mask_train],
        y_train=encoded_labels[mask_train],
        X_val=matrix[mask_val],
        y_val=encoded_labels[mask_val],
        X_test=matrix[mask_test],
        y_test=encoded_labels[mask_test],
        label_classes=label_encoder.classes_
    )
    print(f"✅ Saved SBERT feature cache to {args.output}")


if __name__ == "__main__":
    main()
