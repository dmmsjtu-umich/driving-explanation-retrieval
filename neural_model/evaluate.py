#!/usr/bin/env python
"""
Evaluation script for Vision-Language Transformer classifier.

Computes: Top-k accuracy, Macro/Micro F1, per-class F1, confusion matrix, ECE.

Usage:
    python neural_model/evaluate.py \
        --checkpoint result/vl_transformer/best_model.pt \
        --output result/vl_transformer/evaluation/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from dataset import create_frame_dataloaders
from model import build_model


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Compute top-k accuracy."""
    topk = logits.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE)."""
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = i / n_bins, (i + 1) / n_bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].float().mean()
            ece += torch.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece.item()


def find_optimal_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find optimal temperature for calibration via grid search."""
    best_ece, best_temp = float('inf'), 1.0
    for temp in np.linspace(0.5, 3.0, 51):
        ece = compute_ece(logits / temp, labels)
        if ece < best_ece:
            best_ece, best_temp = ece, temp
    return best_temp


@torch.no_grad()
def evaluate_model(model, dataloader, device, label_classes=None) -> dict:
    """Comprehensive model evaluation."""
    model.eval()
    all_logits, all_labels = [], []

    iterator = tqdm(dataloader, desc="Evaluating") if tqdm else dataloader
    for batch in iterator:
        video = batch["video"].to(device)
        text_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "token_type_ids": batch["token_type_ids"].to(device),
        }
        labels = batch["label"].to(device)
        logits = model(video, text_inputs)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    predictions = all_logits.argmax(dim=1)

    # Metrics
    metrics = {
        'top1': (predictions == all_labels).float().mean().item(),
        'top3': topk_accuracy(all_logits, all_labels, k=3),
        'top5': topk_accuracy(all_logits, all_labels, k=5),
        'macro_f1': f1_score(all_labels.numpy(), predictions.numpy(), average='macro'),
        'micro_f1': f1_score(all_labels.numpy(), predictions.numpy(), average='micro'),
        'weighted_f1': f1_score(all_labels.numpy(), predictions.numpy(), average='weighted'),
    }

    # Calibration
    metrics['ece_before'] = compute_ece(all_logits, all_labels)
    optimal_temp = find_optimal_temperature(all_logits, all_labels)
    metrics['optimal_temperature'] = optimal_temp
    metrics['ece_after'] = compute_ece(all_logits / optimal_temp, all_labels)

    # Per-class F1
    per_class_f1 = f1_score(all_labels.numpy(), predictions.numpy(), average=None)
    metrics['per_class_f1'] = per_class_f1.tolist()

    sorted_idx = np.argsort(per_class_f1)
    metrics['worst_classes'] = sorted_idx[:5].tolist()
    metrics['best_classes'] = sorted_idx[-5:][::-1].tolist()

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(all_labels.numpy(), predictions.numpy()).tolist()

    if label_classes is not None:
        metrics['classification_report'] = classification_report(
            all_labels.numpy(), predictions.numpy(),
            target_names=[str(c) for c in label_classes], output_dict=True,
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VL Transformer classifier")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--data-json", type=Path,
        default=Path("driving-explanation-retrieval/data_analysis/clustering_results/reasons_with_clusters.json"))
    parser.add_argument("--frame-tsv-root", type=Path,
        default=Path("/home/dmmsjtu/Desktop/cse595/datasets/BDDX/frame_tsv"))
    parser.add_argument("--output", type=Path,
        default=Path("driving-explanation-retrieval/result/vl_transformer/evaluation/"))
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint.get('args', {})

    tokenizer_name = model_args.get("tokenizer_name", "bert-base-uncased")

    # Load data
    print("Loading data...")
    _, val_loader, test_loader, metadata = create_frame_dataloaders(
        data_json=args.data_json,
        frame_tsv_root=args.frame_tsv_root,
        tokenizer_name=tokenizer_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        img_size=args.img_size,
        frame_size=args.frame_size,
    )

    # Build model
    print("Building model...")
    model = build_model(
        num_classes=metadata['num_classes'],
        hidden_dim=model_args.get('hidden_dim', 256),
        num_transformer_layers=model_args.get('num_layers', 4),
        num_attention_heads=model_args.get('num_heads', 4),
        dropout=model_args.get('dropout', 0.1),
        text_model_name=tokenizer_name,
        freeze_early_layers=False,  # No freezing during eval
        num_trainable_text_layers=model_args.get('num_trainable_text_layers', 2),
        max_temporal_tokens=args.num_frames // 8 + 2,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Evaluate
    results = {'checkpoint': str(args.checkpoint), 'model_args': model_args}

    if args.split in ["val", "both"]:
        print("\nEvaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device, metadata['label_classes'])
        results['validation'] = val_metrics
        print(f"  Top-1: {val_metrics['top1']:.4f} | Top-3: {val_metrics['top3']:.4f} | "
              f"Macro-F1: {val_metrics['macro_f1']:.4f} | ECE: {val_metrics['ece_after']:.4f}")

    if args.split in ["test", "both"]:
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(model, test_loader, device, metadata['label_classes'])
        results['test'] = test_metrics
        print(f"  Top-1: {test_metrics['top1']:.4f} | Top-3: {test_metrics['top3']:.4f} | "
              f"Macro-F1: {test_metrics['macro_f1']:.4f} | ECE: {test_metrics['ece_after']:.4f}")

    # Save results
    results_path = args.output / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Split':<12} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'Macro-F1':<10}")
    print("-" * 60)
    if 'validation' in results:
        v = results['validation']
        print(f"{'Validation':<12} {v['top1']:.4f}     {v['top3']:.4f}     {v['top5']:.4f}     {v['macro_f1']:.4f}")
    if 'test' in results:
        t = results['test']
        print(f"{'Test':<12} {t['top1']:.4f}     {t['top3']:.4f}     {t['top5']:.4f}     {t['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
