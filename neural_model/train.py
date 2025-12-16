#!/usr/bin/env python
"""
Training script for Vision-Language Transformer classifier.

Usage:
    python neural_model/train.py \
        --data-json data_analysis/clustering_results/reasons_with_clusters.json \
        --frame-tsv-root /path/to/frame_tsv \
        --output result/vl_transformer/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from dataset import create_frame_dataloaders
from model import build_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    """Compute top-k accuracy."""
    topk = logits.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def compute_metrics(model: nn.Module, dataloader, device: torch.device, use_amp: bool = False) -> dict:
    """Compute evaluation metrics on a dataset."""
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            video = batch["video"].to(device)
            text_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "token_type_ids": batch["token_type_ids"].to(device),
            }
            labels = batch["label"].to(device)
            with autocast(enabled=use_amp):
                logits = model(video, text_inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    preds = all_logits.argmax(dim=1)

    # Metrics
    top1 = (preds == all_labels).float().mean().item()
    top3 = topk_accuracy(all_logits, all_labels, k=3)
    top5 = topk_accuracy(all_logits, all_labels, k=5)

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels.numpy(), preds.numpy(), average='macro')

    return {'top1': top1, 'top3': top3, 'top5': top5, 'macro_f1': macro_f1}


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    use_tqdm: bool = True,
    scaler: GradScaler | None = None,
    use_wandb: bool = False,
    log_every: int = 100,
    entropy_lambda: float = 0.0,
) -> dict:
    """Train for one epoch.

    Args:
        entropy_lambda: Weight for entropy regularization (confidence penalty).
            Positive values encourage less confident predictions, reducing overfitting.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    iterator = dataloader
    if use_tqdm and tqdm is not None:
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for step, batch in enumerate(iterator):
        video = batch["video"].to(device)
        text_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "token_type_ids": batch["token_type_ids"].to(device),
        }
        labels = batch["label"].to(device)

        # Forward + loss
        with autocast(enabled=scaler is not None):
            logits = model(video, text_inputs)
            ce_loss = criterion(logits, labels)

            # Entropy regularization: penalize overconfident predictions
            if entropy_lambda > 0:
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()
                loss = ce_loss - entropy_lambda * entropy
            else:
                loss = ce_loss

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if use_tqdm and tqdm is not None:
            iterator.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log to W&B every N steps
        if use_wandb and log_every > 0 and (step + 1) % log_every == 0:
            global_step = epoch * len(dataloader) + step
            wandb.log({"step": global_step, "train/step_loss": loss.item()})

    return {'loss': total_loss / max(num_batches, 1)}


def main():
    parser = argparse.ArgumentParser(description="Train VL Transformer classifier")

    # Data paths
    parser.add_argument("--data-json", type=Path,
        default=Path("driving-explanation-retrieval/data_analysis/clustering_results/reasons_with_clusters.json"))
    parser.add_argument("--frame-tsv-root", type=Path,
        default=Path("/home/dmmsjtu/Desktop/cse595/datasets/BDDX/frame_tsv"))
    parser.add_argument("--output", type=Path,
        default=Path("driving-explanation-retrieval/result/vl_transformer/"))

    # Data parameters
    parser.add_argument("--num-frames", type=int, default=32, help="Frames per clip")
    parser.add_argument("--img-size", type=int, default=224, help="Image crop size")
    parser.add_argument("--frame-size", type=int, default=256, help="Frame resolution in TSV")
    parser.add_argument("--max-text-len", type=int, default=48, help="Max tokenized text length")
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")

    # Model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-early-layers", action="store_true", default=True,
        help="Freeze early encoder layers")
    parser.add_argument("--no-freeze-early-layers", action="store_true",
        help="Train all encoder layers")
    parser.add_argument("--num-trainable-text-layers", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=2,
        help="Video spatial grid size (1=global, 2=2x2, 3=3x3)")
    parser.add_argument("--video-token-drop", type=float, default=0.1,
        help="Dropout rate for video tokens during training")
    parser.add_argument("--modality-drop", type=float, default=0.1,
        help="Probability of dropping action token during training")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100, help="Log to W&B every N steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="LR for transformer head")
    parser.add_argument("--lr-backbone", type=float, default=3e-5, help="LR for encoders")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Mixed precision training")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
        help="Label smoothing for cross-entropy loss (0=none, 0.1=recommended)")
    parser.add_argument("--entropy-lambda", type=float, default=0.02,
        help="Entropy regularization weight (0=none, 0.02=recommended)")
    parser.add_argument("--train-sampling", type=str, default="random_stride",
        choices=["uniform", "random_stride", "random_clip"],
        help="Temporal sampling mode for training")
    parser.add_argument("--no-train-augment", action="store_true",
        help="Disable training augmentation (random crop + color jitter)")

    # Ablation
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="driving-explanation-retrieval")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    # Validate
    if args.video_only and args.text_only:
        parser.error("Cannot specify both --video-only and --text-only")

    ablation_mode = "video_only" if args.video_only else ("text_only" if args.text_only else "both")

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    # W&B
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"vl_h{args.hidden_dim}_l{args.num_layers}",
            config=vars(args),
        )

    # Load data
    print("Loading data...")
    train_augment = not args.no_train_augment
    train_loader, val_loader, test_loader, metadata = create_frame_dataloaders(
        data_json=args.data_json,
        frame_tsv_root=args.frame_tsv_root,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        img_size=args.img_size,
        frame_size=args.frame_size,
        max_text_len=args.max_text_len,
        train_sampling=args.train_sampling,
        train_augment=train_augment,
    )
    print(f"  Train: {metadata['train_size']} | Val: {metadata['val_size']} | Test: {metadata['test_size']}")
    print(f"  Classes: {metadata['num_classes']} | Frames: {metadata['num_frames']}")
    print(f"  Train sampling: {args.train_sampling} | Augment: {train_augment}")

    # Build model
    print("\nBuilding model...")
    freeze_early = args.freeze_early_layers and not args.no_freeze_early_layers

    # Calculate max video tokens: (num_frames / 8) * grid_size^2
    max_video_tokens = (args.num_frames // 8) * (args.grid_size ** 2)

    model = build_model(
        num_classes=metadata["num_classes"],
        hidden_dim=args.hidden_dim,
        num_transformer_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        dropout=args.dropout,
        text_model_name=args.tokenizer_name,
        freeze_early_layers=freeze_early,
        num_trainable_text_layers=args.num_trainable_text_layers,
        max_video_tokens=max_video_tokens,
        ablation_mode=ablation_mode,
        grid_size=args.grid_size,
        video_token_drop=args.video_token_drop,
        modality_drop=args.modality_drop,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {trainable:,} / {num_params:,} trainable ({100*trainable/num_params:.1f}%)")
    print(f"  Grid size: {args.grid_size}x{args.grid_size} -> {max_video_tokens} video tokens")

    # Mixed precision
    scaler = GradScaler() if args.amp and device.type == "cuda" else None
    if scaler:
        print("  Using mixed precision (AMP)")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.label_smoothing > 0:
        print(f"  Using label smoothing: {args.label_smoothing}")

    backbone_params = list(model.text_encoder.parameters()) + list(model.video_encoder.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("text_encoder") and not n.startswith("video_encoder")]
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    num_warmup = int(args.epochs * args.warmup_ratio)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, total_iters=max(num_warmup, 1)),
        CosineAnnealingLR(optimizer, T_max=max(args.epochs - num_warmup, 1), eta_min=1e-6),
    ], milestones=[num_warmup])

    # Resume
    start_epoch = 0
    best_val_top1 = 0.0
    if args.resume and args.resume.exists():
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_top1 = ckpt.get('best_val_top1', 0.0)

    # Training loop
    print("\nStarting training...")
    if args.entropy_lambda > 0:
        print(f"  Using entropy regularization: {args.entropy_lambda}")
    use_tqdm = not args.no_tqdm and tqdm is not None
    history = []

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, use_tqdm=use_tqdm, scaler=scaler,
            use_wandb=use_wandb, log_every=args.log_every,
            entropy_lambda=args.entropy_lambda,
        )
        scheduler.step()

        # Clear CUDA cache before evaluation to free training memory
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Evaluate (use AMP to match training memory usage)
        val_metrics = compute_metrics(model, val_loader, device, use_amp=scaler is not None)
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics})

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Val Top-1: {val_metrics['top1']:.4f} | "
              f"Top-3: {val_metrics['top3']:.4f} | "
              f"F1: {val_metrics['macro_f1']:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_metrics['loss'],
                "val/top1": val_metrics['top1'],
                "val/top3": val_metrics['top3'],
                "val/top5": val_metrics['top5'],
                "val/macro_f1": val_metrics['macro_f1'],
            })

        # Save best model
        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_top1': best_val_top1,
                'val_metrics': val_metrics,
            }, args.output / "best_model.pt")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Free optimizer memory before final evaluation
    use_amp_eval = args.amp and device.type == "cuda"
    del optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if (args.output / "best_model.pt").exists():
        ckpt = torch.load(args.output / "best_model.pt", weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = compute_metrics(model, test_loader, device, use_amp=use_amp_eval)
    print(f"Test - Top-1: {test_metrics['top1']:.4f} | "
          f"Top-3: {test_metrics['top3']:.4f} | "
          f"Top-5: {test_metrics['top5']:.4f} | "
          f"F1: {test_metrics['macro_f1']:.4f}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'test': test_metrics,
        'history': history,
    }
    with open(args.output / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    if use_wandb:
        wandb.log({"test/top1": test_metrics['top1'], "test/macro_f1": test_metrics['macro_f1']})
        wandb.finish()

    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
