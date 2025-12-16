#!/usr/bin/env python
"""
Concatenate video features + action text features for reason-cluster classification.

Inputs:
  - --action-cache: NPZ produced by prepare_action_features.py
  - --video-cache:  NPZ produced by prepare_video_features.py

Both caches must be created from the same JSON (reasons_with_clusters.json)
and therefore share the same train/val/test ordering.

Output:
  - JSON metrics with Top-1 / Top-3 / Macro-F1 under result/baseline/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    correct = sum(int(true in row) for true, row in zip(labels, topk))
    return correct / len(labels)


def load_cache(path: Path):
    cache = np.load(path, allow_pickle=True)
    X_train = cache["X_train"]
    X_val = cache["X_val"]
    X_test = cache["X_test"]
    y_train = cache["y_train"]
    y_val = cache["y_val"]
    y_test = cache["y_test"]
    label_classes = cache["label_classes"]
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_classes": label_classes,
    }


def ensure_same_labels(base_labels: np.ndarray, other_labels: np.ndarray, base_classes: np.ndarray, other_classes: np.ndarray) -> np.ndarray:
    """Remap other_labels to base label indices if class order differs."""
    if np.array_equal(base_classes, other_classes):
        return other_labels
    mapping = {cls: i for i, cls in enumerate(other_classes)}
    remap = {i: np.where(base_classes == cls)[0][0] for cls, i in mapping.items() if cls in set(base_classes)}
    # Vectorized remap
    out = np.array([remap[i] for i in other_labels], dtype=base_labels.dtype)
    return out


def hstack_and_align(A: dict, B: dict, w_a: float, w_b: float, standardize: bool):
    # Verify sample counts match per split
    for split in ("train", "val", "test"):
        Xa = A[f"X_{split}"]
        Xb = B[f"X_{split}"]
        if Xa.shape[0] != Xb.shape[0]:
            raise ValueError(f"Split {split} size mismatch: action {Xa.shape[0]} vs video {Xb.shape[0]}")

    # Align labels (use action cache as base)
    y_train = A["y_train"]
    y_val = A["y_val"]
    y_test = A["y_test"]

    B["y_train"] = ensure_same_labels(y_train, B["y_train"], A["label_classes"], B["label_classes"])
    B["y_val"] = ensure_same_labels(y_val, B["y_val"], A["label_classes"], B["label_classes"])
    B["y_test"] = ensure_same_labels(y_test, B["y_test"], A["label_classes"], B["label_classes"])

    # Sanity check that per-split labels match (ordering should be consistent)
    for split in ("train", "val", "test"):
        if not np.array_equal(A[f"y_{split}"], B[f"y_{split}"]):
            # If labels differ, we still proceed using action labels, but warn via exception text
            raise ValueError(f"Label misalignment detected in {split}. Ensure both caches came from same dataset order.")

    # Optionally standardize each modality on train split, then scale by weights and concat
    def process_block(block: dict, prefix: str, weight: float):
        Xtr, Xva, Xte = block["X_train"], block["X_val"], block["X_test"]
        if standardize:
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr = scaler.fit_transform(Xtr)
            Xva = scaler.transform(Xva)
            Xte = scaler.transform(Xte)
        if weight != 1.0:
            Xtr = Xtr * weight
            Xva = Xva * weight
            Xte = Xte * weight
        return Xtr, Xva, Xte

    A_tr, A_va, A_te = process_block(A, "A", w_a)
    B_tr, B_va, B_te = process_block(B, "B", w_b)

    X_train = np.hstack([A_tr, B_tr])
    X_val = np.hstack([A_va, B_va])
    X_test = np.hstack([A_te, B_te])

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action-cache",
        type=Path,
        default=Path("driving-explanation-retrieval/data/baseline/action_features_sbert.npz"),
        help="NPZ from prepare_action_features.py (SBERT only)",
    )
    parser.add_argument(
        "--video-cache",
        type=Path,
        default=Path("driving-explanation-retrieval/data/baseline/video_features_resnet18_mean.npz"),
        help="NPZ from prepare_video_features.py (default: data/baseline/video_features_resnet18_mean.npz)",
    )
    parser.add_argument("--output", type=Path, default=Path("driving-explanation-retrieval/result/baseline/video_action_concat_results.json"))
    parser.add_argument("--video-weight", type=float, default=1.0, help="Scale for video feature block before concat")
    parser.add_argument("--text-weight", type=float, default=1.0, help="Scale for text/action feature block before concat")
    parser.add_argument("--standardize", action="store_true", help="Z-score each block on train split before concat")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    # SGD training options (with tqdm progress)
    parser.add_argument("--sgd", action="store_true", help="Use SGDClassifier with per-epoch tqdm progress (default)")
    parser.add_argument("--logreg", action="store_true", help="Force LogisticRegression instead of SGD (overrides default)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for SGDClassifier when --sgd is set")
    parser.add_argument("--log-every", type=int, default=1, help="Log metrics every N epochs (SGD mode)")
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization for SGDClassifier")
    parser.add_argument(
        "--learning-rate",
        choices=["optimal", "constant", "invscaling", "adaptive"],
        default="optimal",
        dest="learning_rate",
        help="Learning rate schedule for SGDClassifier",
    )
    parser.add_argument("--eta0", type=float, default=0.0, help="Initial LR for SGDClassifier when applicable")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars (SGD mode)")
    parser.add_argument("--verbose", action="store_true", help="Verbose solver logs (LogisticRegression)")
    # Default to SGD unless --logreg is explicitly provided
    parser.set_defaults(sgd=True)
    args = parser.parse_args()
    if args.logreg:
        args.sgd = False

    # Friendly existence checks with guidance
    if not args.action_cache.exists() or not args.video_cache.exists():
        missing = []
        if not args.action_cache.exists():
            missing.append(f"action-cache: {args.action_cache}")
        if not args.video_cache.exists():
            missing.append(f"video-cache: {args.video_cache}")
        raise FileNotFoundError(
            "\n".join([
                "Missing feature cache: " + ", ".join(missing),
                "Please generate first: ",
                "  - Action (SBERT): python driving-explanation-retrieval/baselines/prepare_action_features.py --model-name all-MiniLM-L6-v2 --output driving-explanation-retrieval/data/baseline/action_features_sbert.npz",
                "  - Video:          python driving-explanation-retrieval/baselines/prepare_video_features.py --device cuda",
            ])
        )

    A = load_cache(args.action_cache)
    V = load_cache(args.video_cache)

    X_train, X_val, X_test, y_train, y_val, y_test = hstack_and_align(
        A, V, w_a=args.text_weight, w_b=args.video_weight, standardize=args.standardize
    )

    def evaluate(model, X, y_true, name):
        probs = model.predict_proba(X)
        pred = probs.argmax(axis=1)
        return {
            "top1": float(accuracy_score(y_true, pred)),
            "top3": float(topk_accuracy(probs, y_true, k=3)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
        }

    if args.sgd:
        if tqdm is None and not args.no_tqdm:
            raise ImportError("tqdm is not installed, cannot display training progress. Please install tqdm or use --no-tqdm")
        classes = np.unique(y_train)
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=args.alpha,
            learning_rate=args.learning_rate,
            eta0=args.eta0,
            max_iter=1,
            tol=None,
            random_state=args.random_state,
        )
        epoch_iterable = range(args.epochs)
        if tqdm is not None and not args.no_tqdm:
            epoch_iterable = tqdm(epoch_iterable, desc="SGD epochs", unit="epoch")
        for epoch in epoch_iterable:
            X_epoch, y_epoch = sklearn_shuffle(
                X_train,
                y_train,
                random_state=args.random_state + epoch,
            )
            clf.partial_fit(X_epoch, y_epoch, classes=classes)
            if (epoch + 1) % max(1, args.log_every) == 0:
                tr = evaluate(clf, X_train, y_train, f"Train@{epoch+1}")
                va = evaluate(clf, X_val, y_val, f"Val@{epoch+1}")
                if tqdm is None or args.no_tqdm:
                    print(
                        f"Epoch {epoch+1}/{args.epochs} - Train Top1 {tr['top1']:.4f} Val Top1 {va['top1']:.4f}"
                    )
        trainer = "sgd"
    else:
        # Dense LR with lbfgs
        logreg_kwargs = dict(
            max_iter=args.max_iter,
            solver="lbfgs",
            random_state=args.random_state,
            n_jobs=None,
        )
        if args.verbose:
            logreg_kwargs["verbose"] = 1
        clf = LogisticRegression(**logreg_kwargs)
        clf.fit(X_train, y_train)
        trainer = "logreg(lbfgs)"

    metrics = {
        "fusion": "concat",
        "video_weight": args.video_weight,
        "text_weight": args.text_weight,
        "standardize": bool(args.standardize),
        "trainer": trainer,
        "train": evaluate(clf, X_train, y_train, "train"),
        "validation": evaluate(clf, X_val, y_val, "validation"),
        "test": evaluate(clf, X_test, y_test, "test"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
